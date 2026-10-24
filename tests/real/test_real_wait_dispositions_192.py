"""Real decisions which never executed, followed by original negative closure."""
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
from power_loop.runtime import wait_reviews as reviews
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("choice", ["continue", "defer", "cancel"])
async def test_real_unapplied_main_intent_closes_with_original_negative_proof(tmp_path, record_property, choice):
    path = str(tmp_path / "unapplied-original.db")
    store = await SessionStore.open(path, schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-negative-handoff", principal_id="host", executor_id="executor", boot_id="boot")
    provider, registry, registrations = BoundedProvider(limit=14), ToolRegistry(), []
    register_spawn_agent(registry)
    register_wait_handoff_tool(registry)
    consuming = False

    async def arm():
        source, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registered = await waits.register_wait(manager, runtime.actor(sid), caller=source.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 300, "Review the original finite arithmetic task.", "Continue the original arithmetic task."),
            policy=waits.WaitPolicy(now + 120000))
        registrations.append(registered)
        return "Original wait accepted. Finish this turn without polling."

    registry.register(ToolDefinition(name="arm_wait", description="Register this task's required main-agent review dependency once; this is application work, not polling or an unnecessary delay.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False}), arm)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Initially create exactly the requested child, then reply WAITING without pass_turn. "
                "When a wait_handoff notification arrives, choose " + choice + " using its original event_id. "
                "For defer choose next_review_at_ms equal to deadline_at_ms minus 1000. "
                "After its acceptance reply DECISION-RECORDED, without further tools. "
                "A decision is intent only; do not claim it executed. Never create replacement work. "
                "When the original child result is later available, consume it and report that result."))
    parent = await loop.new_session()

    async def response_shape(ctx):
        if get_session_id() == parent and consuming:
            ctx.tools = []
        elif get_session_id() != parent and registrations and choice != "defer":
            ctx.tools = []

    loop.hooks.register(HookPoint.LLM_BEFORE, response_shape, name="test.unapplied-response-shape")
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(150):
            ending = ("After acceptance call pass_turn once." if choice == "defer" else
                "After acceptance return READY-42 as final assistant text, without tools or pass_turn.")
            await loop.send("Use spawn_agent create exactly once with tools=['arm_wait'], max_rounds=4, "
                "task='Compute 19+23; main-agent review is required before releasing the result', "
                "system_prompt='This task requires main-agent review before result release. First call arm_wait exactly once "
                "to register that required dependency. This is required application work, not sleeping, polling, or an optional alarm. "
                "Do not omit the review dependency even though the arithmetic is simple. " + ending + "'. Then reply WAITING.", parent)
            owner = None
            for _ in range(750):
                if registrations:
                    [registration] = registrations
                    owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
                    if owner.status == "waiting":
                        break
                    assert owner.outcome is None, owner
                await asyncio.sleep(.02)
            assert owner is not None and owner.status == "waiting" and owner.resource_state == "running"
            child_rows = await store.load_all_messages(owner.session_id)
            passed = any(tc.get("function", {}).get("name") == "pass_turn" for m in child_rows for tc in (m.tool_calls or []))
            assert passed == (choice == "defer")
            original = await waits.read_wait(manager, runtime.actor(parent), registration.job.job_id)
            remaining = original["request"]["request"]["due_at_ms"] - manager.now()
            if remaining > 0:
                await asyncio.sleep(remaining / 1000)
            due = await waits.prepare_due(manager, runtime.actor(parent), registration.job.job_id)
            await notices.enqueue_notification(loop, runtime.actor(parent), due["event_id"])
            assert (await loop.flush_follow_ups(parent)).status == "completed"
            [decision] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='wait_decision'")
            assert json.loads(decision["payload_json"])["request"]["decision"] == choice
            assert decision["delivery_status"] == "pending_delivery"
            calls = provider.calls
            await manager.request_cancel(runtime.actor(parent), owner.job_id, system_key="external-original-owner-stop", reason="external cancellation after decision")
            tasks = set(delegation._TASKS) - before
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 30)
            owner = await manager.get_job(runtime.actor(parent), owner.job_id)
            assert owner.outcome == "cancelled" and owner.resource_state == "stopped"
            reply = await reviews.drive(loop, runtime.actor(parent), registration.job.job_id)
            assert reply["status"] == "owner_terminated_without_effect" and not reply["continuation_dispatched"]
            assert not reply["execution_authority"] and not reply["billing_settled"]
            assert provider.calls == calls
            assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='wait_effect' OR action='nudge'")
            assert await waits.read_wait(manager, runtime.actor(parent), registration.job.job_id) == original
            consuming = True
            await loop.flush_follow_ups(parent)
            rows = await manager.db.fetchall(f"SELECT kind,status,supervision_status FROM {manager.jobs}")
            assert all(r["supervision_status"] == "closed" for r in rows), rows
            reopened = await SessionStore.open(path, schema="tasks")
            try:
                assert await effects.apply(JobManager(reopened), runtime.actor(parent), due["event_id"]) == reply
            finally:
                await reopened.close()
            assert len(await store.list_children(parent)) == 1
            record_property("negative_disposition", json.dumps(reply))
            record_property("supervision_tree", json.dumps(rows))
            record_property("original_decision", decision["payload_json"])
    finally:
        for registration in registrations:
            owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
            if owner.outcome is None:
                await manager.request_cancel(runtime.actor(parent), owner.job_id, system_key="real-disposition-cleanup", reason="test cleanup")
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 25)
        for session in [parent, *(s.session_id for s in await store.list_children(parent))]:
            record_property("messages_" + session, json.dumps([asdict(m) for m in await store.load_all_messages(session)]))
        provider.report(record_property)
        await provider.close()
        await store.close()
