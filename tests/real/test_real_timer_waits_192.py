"""Real source call/receipt admit one required wait; only evidence becomes due."""

from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime import timer_waits as waits
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_child_original_call_and_receipt_produce_one_due_event_without_root_delivery(
    tmp_path, record_property
):
    store = await SessionStore.open(str(tmp_path / "wait-admission.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-wait", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=6)
    registry, registrations = ToolRegistry(), []
    register_spawn_agent(registry)
    ready, release = asyncio.Event(), asyncio.Event()

    async def arm():
        binding = _SEND.get()
        sid = get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, runtime.actor(sid), caller=binding.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 250, "Original child wait", "Continue this same arithmetic task."),
            policy=waits.WaitPolicy(now + 20000))
        registrations.append(registration)
        return json.dumps({"status": "accepted", "job_id": registration.job.job_id})

    registry.register(ToolDefinition(name="arm_wait", description="Request one deferred decision from the owner.",
        input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), arm)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Follow the user's sequence exactly. Do not create replacements. After child acceptance "
                          "reply WAITING in final assistant text without more tools or pass_turn."))
    parent = await loop.new_session()

    async def hold_after_receipt(ctx):
        if get_session_id() != parent and registrations:
            ready.set()
            await release.wait()

    loop.hooks.register(HookPoint.LLM_BEFORE, hold_after_receipt, name="test.wait-after-receipt")
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(120):
            await loop.send("Use spawn_agent create exactly once, tools=['arm_wait'], max_rounds=3, task='Compute 19+23', "
                "system_prompt='Call arm_wait exactly once with no arguments. After its acceptance, reply READY-42 "
                "as final assistant text without more tools or pass_turn.'. Then reply WAITING.", parent)
            await asyncio.wait_for(ready.wait(), 30)
            [registration] = registrations
            saved = await waits.read_wait(manager, runtime.actor(parent), registration.job.job_id)
            await asyncio.sleep(max(0, (saved["request"]["request"]["due_at_ms"] - manager.now()) / 1000) + .01)
            first = await waits.prepare_due(manager, runtime.actor(parent), registration.job.job_id)
            second = await waits.prepare_due(manager, runtime.actor(parent), registration.job.job_id)
            assert first == second and first["status"] == "decision_pending"
            saved = first["payload"]["wait"]
            assert saved["call_ref"]["id"] != saved["receipt_ref"]["id"]
            assert saved["binding"]["main_session_id"] == parent
            owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
            assert owner.outcome is None and owner.resource_state == "running"
            assert (await manager.get_job(runtime.actor(parent), registration.job.job_id)).outcome is None
            assert (await store.inbox_counts(parent))["pending"] == 0
            assert len(await store.list_children(parent)) == 1
            record_property("wait_due_event_id", first["event_id"])
            record_property("source_refs", json.dumps({k: saved[k] for k in ("call_ref", "receipt_ref")}))
    finally:
        # This test exercises admission/due evidence, not the future root model
        # decision adapter. Mechanical cancellation explicitly resolves its wait.
        for registration in registrations:
            await manager.request_cancel(runtime.actor(parent), registration.job.job_id,
                                          system_key="test-cleanup", reason="test cleanup")
            await manager.record_unstarted_result(runtime.actor(registration.job.created_by_session_id),
                registration.job.job_id, outcome="cancelled", result={"reason": "test cleanup"})
        release.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
