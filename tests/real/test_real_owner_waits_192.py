"""Actual natural/pass exits precede durable waiting and original continuation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime import timer_waits as waits
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_notifications as notices
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("ending", ["natural", "pass_turn"])
async def test_real_model_exit_precedes_durable_wait_without_a_provider_gate_and_original_task_resumes(tmp_path, record_property, ending):
    store = await SessionStore.open(str(tmp_path / "owner-wait.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-owner-wait", principal_id="host", executor_id="executor", boot_id="boot")
    registry, registrations = ToolRegistry(), []
    register_spawn_agent(registry)
    register_wait_handoff_tool(registry)
    provider = BoundedProvider(limit=14)
    continuing, consuming = False, False

    async def arm():
        source, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, runtime.actor(sid), caller=source.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 50, "Review the original finite task.", "Continue this same calculation and report 42."),
            policy=waits.WaitPolicy(now + 120000))
        registrations.append(registration)
        return "Original wait accepted. Finish this turn without polling."

    registry.register(ToolDefinition(name="arm_wait", description="Register one original task wait exactly once.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False}), arm)
    ending_prompt = ("After arm_wait acceptance, write READY-42 as final assistant text without tools or pass_turn."
        if ending == "natural" else "After arm_wait acceptance, call pass_turn exactly once to end this turn without any other tool.")
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Initially create exactly one requested child, then write WAITING without pass_turn. "
                "When a wait_handoff notification arrives, decide continue using its original event_id, then write DECISION-RECORDED. "
                "Never create replacement work. Later consume the original child result."))
    parent = await loop.new_session()

    async def text_policy(ctx):
        # Restrict actual response shape before request fingerprinting. There
        # is no gate/event/sleep preventing the child model from completing.
        if get_session_id() == parent and consuming:
            ctx.tools = []
        if get_session_id() != parent and registrations and (ending == "natural" or continuing):
            ctx.tools = []

    loop.hooks.register(HookPoint.LLM_BEFORE, text_policy, name="test.original-owner-response-shape")
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(150):
            await loop.send("Use spawn_agent create once with tools=['arm_wait'], max_rounds=4, task='Compute 19+23', "
                "system_prompt='Call arm_wait exactly once. " + ending_prompt + "'. Then reply WAITING.", parent)
            owner = None
            for _ in range(500):
                if registrations:
                    [registration] = registrations
                    owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
                    if owner.status == "waiting":
                        break
                    assert owner.outcome is None, owner
                await asyncio.sleep(.02)
            assert owner is not None and owner.status == "waiting" and owner.resource_state == "running"
            [source] = await manager.db.fetchall(f"SELECT job_id,status,resource_state FROM {manager.jobs} WHERE control_owner_job_id=? AND kind='send'", (owner.job_id,))
            assert source["status"] == "model_closed" and source["resource_state"] == "stopped"
            [entry] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='owner_waiting' AND job_id=?", (owner.job_id,))
            assert json.loads(entry["payload_json"])["raw"]["status"] == "completed"
            child_messages = await store.load_all_messages(owner.session_id)
            actual_pass = any(m.role == "assistant" and any(c.get("function", {}).get("name") == "pass_turn" for c in (m.tool_calls or [])) for m in child_messages)
            assert actual_pass == (ending == "pass_turn")
            calls = provider.calls
            await asyncio.sleep(.03)
            assert provider.calls == calls
            assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='result'", (owner.job_id,))
            due = await waits.prepare_due(manager, runtime.actor(parent), registration.job.job_id)
            await notices.enqueue_notification(loop, runtime.actor(parent), due["event_id"])
            await loop.flush_follow_ups(parent)
            continuing = True
            await effects.apply(manager, runtime.actor(parent), due["event_id"])
            tasks = set(delegation._TASKS) - before
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 30)
            owner = await manager.get_job(runtime.actor(parent), owner.job_id)
            assert owner.outcome == "succeeded" and owner.resource_state == "stopped", owner
            assert len(await store.list_children(parent)) == 1
            assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='owner_wait_resumed'")) == 1
            consuming = True
            await loop.flush_follow_ups(parent)
            rows = await manager.db.fetchall(f"SELECT kind,status,supervision_status FROM {manager.jobs}")
            assert all(r["supervision_status"] == "closed" for r in rows), rows
            record_property("original_waiting_entry", entry["payload_json"])
            record_property("original_owner_result", json.dumps(owner.result))
            record_property("supervision_tree", json.dumps(rows))
    finally:
        for registration in registrations:
            owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
            if owner.outcome is None:
                await manager.request_cancel(runtime.actor(parent), owner.job_id, system_key="real-owner-wait-cleanup", reason="test cleanup")
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)
        for session in [parent, *(s.session_id for s in await store.list_children(parent))]:
            record_property("messages_" + session, json.dumps([asdict(m) for m in await store.load_all_messages(session)]))
        provider.report(record_property)
        await provider.close()
        await store.close()
