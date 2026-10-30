"""Real main decisions produce original-task effects without replacement work."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime import timer_waits as waits
from power_loop.runtime import wait_decisions as decisions
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_notifications as notices
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("choice", ["continue", "defer", "cancel"])
async def test_real_original_child_effects_reuse_its_task_and_actual_input_processing(
    tmp_path, record_property, choice
):
    store = await SessionStore.open(str(tmp_path / "wait-handoff.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-handoff", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=12)
    registry, registrations = ToolRegistry(), []
    register_spawn_agent(registry)
    register_wait_handoff_tool(registry)
    ready, release = asyncio.Event(), asyncio.Event()

    async def arm():
        binding, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, runtime.actor(sid), caller=binding.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 250, "Please review this original child wait.", "Continue this same arithmetic task."),
            policy=waits.WaitPolicy(now + 120000))
        registrations.append(registration)
        return json.dumps({"status": "accepted", "job_id": registration.job.job_id})

    registry.register(ToolDefinition(name="arm_wait", description="Register the original child wait exactly once.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False}, pricing=Free("test_fixture")), arm)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt=("Follow the requested original-task sequence. Initially spawn one child, then reply WAITING "
                "without pass_turn. When a later wait_handoff notification arrives, record decision " + choice +
                " using the supplied wait_handoff decide action and event_id. For defer, choose next_review_at_ms "
                "equal to deadline_at_ms minus 1000. After acceptance reply DECISION-RECORDED, without further tools. "
                "A decision records intent only; do not claim the child resumed, stopped, or completed. "
                "Never create a replacement child or workflow.")))
    parent = await loop.new_session()

    async def hold_after_receipt(ctx):
        if get_session_id() != parent and registrations:
            ready.set()
            await release.wait()
            # This acceptance fixture has no further business/tool work. Keep
            # the real provider's response as plain text; an extra pass_turn
            # on its last allowed round exercises the separate round-limit
            # failure path rather than original-input success settlement.
            ctx.tools = []

    loop.hooks.register(HookPoint.LLM_BEFORE, hold_after_receipt, name="test.wait-original-child")
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(150):
            await loop.send("Use spawn_agent create exactly once, tools=['arm_wait'], max_rounds=3, task='Compute 19+23', "
                "system_prompt='Call arm_wait exactly once with no arguments. After its acceptance, reply READY-42 "
                "as final assistant text without more tools or pass_turn.'. Then reply WAITING.", parent)
            await asyncio.wait_for(ready.wait(), 30)
            [registration] = registrations
            saved = await waits.read_wait(manager, runtime.actor(parent), registration.job.job_id)
            await asyncio.sleep(max(0, (saved["request"]["request"]["due_at_ms"] - manager.now()) / 1000) + .01)
            event = await waits.prepare_due(manager, runtime.actor(parent), registration.job.job_id)
            calls_before = provider.calls
            first = await notices.enqueue_notification(loop, runtime.actor(parent), event["event_id"])
            assert first == await notices.enqueue_notification(loop, runtime.actor(parent), event["event_id"])
            assert provider.calls == calls_before  # Enqueueing is not a paid wake.
            assert (await decisions.inspect(manager, runtime.actor(parent), event["event_id"]))["decision_status"] == "pending"
            result = await loop.flush_follow_ups(parent)
            assert result.status == "completed"
            [intent] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='wait_decision'")
            data = json.loads(intent["payload_json"])
            assert data["request"]["decision"] == choice and intent["delivery_status"] == "pending_delivery"
            assert data["observation"]["model_attempt_ids"]
            operation = await manager.get_operation(runtime.actor(parent), data["operation_id"])
            assert operation.call_ref["id"] != operation.receipt_ref["id"]
            assert saved["call_ref"]["id"] != saved["receipt_ref"]["id"]
            messages = await store.load_all_messages(parent)
            assert any(m.role == "user" and notices.DEFAULT_CONSTRAINTS in m.content for m in messages)
            receipts = [json.loads(m.content) for m in messages if m.role == "tool" and m.name == "wait_handoff"]
            assert any(r.get("decision") == choice and r.get("status") == "accepted" for r in receipts)
            [inbox] = await manager.db.fetchall(f"SELECT content FROM {store.t.follow_up_queue} WHERE kind='wait_handoff'")
            assert '"executor_id"' not in inbox["content"] and '"boot_id"' not in inbox["content"]
            owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
            assert owner.outcome is None and owner.resource_state == "running"
            timer = await manager.get_job(runtime.actor(parent), registration.job.job_id)
            assert timer.outcome is None and timer.generation == 0
            assert len(await store.list_children(parent)) == 1
            first_effect = await effects.apply(manager, runtime.actor(parent), event["event_id"])
            assert first_effect == await effects.apply(manager, runtime.actor(parent), event["event_id"])
            assert not first_effect["execution_authority"] and not first_effect["physical_stop_confirmed"]
            if choice == "defer":
                assert first_effect["status"] == "review_scheduled"
                assert first_effect["next_review_at_ms"] == data["request"]["next_review_at_ms"]
                assert (await manager.get_job(runtime.actor(parent), registration.job.job_id)).outcome is None
            else:
                release.set()
                tasks = set(delegation._TASKS) - before
                if tasks:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 30)
                owner = await manager.get_job(runtime.actor(parent), owner.job_id)
                assert owner.outcome == ("succeeded" if choice == "continue" else "cancelled")
                assert owner.resource_state == "stopped"
                if choice == "continue":
                    timer = await manager.get_job(runtime.actor(parent), registration.job.job_id)
                    assert timer.outcome == "succeeded" and timer.resource_state == "stopped" and timer.generation == 0
                    [input_event] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE action='nudge'")
                    assert input_event["delivery_status"] == "handled"
                    assert input_event["call_ref_json"] is None and input_event["receipt_ref_json"] is None
                    assert (await effects.apply(manager, runtime.actor(parent), event["event_id"]))["effect_status"] == "handled"
                else:
                    assert owner.cancel_requested_at is not None
                    assert (await manager.get_job(runtime.actor(parent), registration.job.job_id)).outcome is None
            assert len(await store.list_children(parent)) == 1
            record_property("system_effect", json.dumps(first_effect))
            record_property("original_source_refs", json.dumps({k: saved[k] for k in ("call_ref", "receipt_ref")}))
            record_property("decision_intent", intent["payload_json"])
            record_property("decision_refs", json.dumps({"call": operation.call_ref, "receipt": operation.receipt_ref}))
    finally:
        # Review-cycle driving and cancelled-wait finalization are separate gates.
        # Mechanical cleanup below is not evidence of those effects.
        for registration in registrations:
            timer = await manager.get_job(runtime.actor(parent), registration.job.job_id)
            if timer.outcome is not None:
                continue
            await manager.request_cancel(runtime.actor(parent), registration.job.job_id,
                system_key="handoff-test-cleanup", reason="test cleanup")
            await manager.record_unstarted_result(runtime.actor(registration.job.created_by_session_id),
                registration.job.job_id, outcome="cancelled", result={"reason": "test cleanup"})
        release.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)
        for session in [parent, *(s.session_id for s in await store.list_children(parent))]:
            record_property("messages_" + session, json.dumps([asdict(m) for m in await store.load_all_messages(session)]))
        provider.report(record_property)
        await provider.close()
        await store.close()
