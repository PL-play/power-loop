"""An actual provider request is interrupted by the original finite policy."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime import timer_waits as waits
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("boot", [False, True], ids=["local-runtime", "automatic-boot"])
async def test_real_provider_short_policy_stops_original_request_and_keeps_unknown_fee(tmp_path, record_property, boot):
    store = await SessionStore.open(str(tmp_path / "inflight-policy.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-policy", principal_id="host", executor_id="executor", boot_id="boot",
        boot_contract_hash="a" * 64 if boot else None)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    provider = BoundedProvider(limit=7)
    registrations = []
    started, cancelled = asyncio.Event(), asyncio.Event()
    complete = provider.complete
    child_requests = []
    parent = None

    async def observed(request, **kwargs):
        sid = get_session_id()
        bounded = sid != parent and bool(registrations)
        if bounded:
            child_requests.append(sid)
            started.set()
        try:
            return await complete(request, **kwargs)
        except asyncio.CancelledError:
            if bounded:
                cancelled.set()
            raise

    provider.complete = observed

    async def arm():
        source, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, runtime.actor(sid), caller=source.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 250, "Review original computation", "Continue original computation"),
            policy=waits.WaitPolicy(now + 1500))
        registrations.append(registration)
        return ("Policy armed. Now give a rigorous, very detailed derivation of the prime number theorem, "
                "at least 2000 words. Use prose with all intermediate arguments, and no further tools.")

    registry.register(ToolDefinition(name="arm_wait", description="Arm the original finite task policy once",
        input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), arm)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048))
    parent = await loop.new_session()

    async def prose_after_arm(ctx):
        if get_session_id() != parent and registrations:
            ctx.tools = []

    loop.hooks.register(HookPoint.LLM_BEFORE, prose_after_arm, name="test.policy-prose")
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(90):
            await loop.send("Call spawn_agent once with tools=['arm_wait'], max_rounds=3, "
                "system_prompt='Call arm_wait exactly once immediately, then follow the next instruction in its result.', "
                "task='Call arm_wait immediately'. Then reply WAITING without more tools.", parent)
            await asyncio.wait_for(started.wait(), 20)
            await asyncio.wait_for(cancelled.wait(), 5)
            tasks = set(delegation._TASKS) - before
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
            [registration] = registrations
            owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
            assert owner.outcome == "timed_out" and owner.resource_state == "unknown", owner
            assert owner.supervision_status == "open" and owner.cancel_requested_at is None
            assert len(child_requests) == 1
            [source] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='send' AND control_owner_job_id=?", (owner.job_id,))
            items = (await list_attempts(manager, runtime.actor(parent), source["job_id"]))["items"]
            assert sorted(item["status"] for item in items) == ["returned", "unknown"]
            unknown = next(item for item in items if item["status"] == "unknown")
            assert unknown["estimated"] and unknown["observed_usage"]["prompt_tokens"] > 0
            record_property("original_owner_result", owner.result)
            record_property("original_attempts", items)
    finally:
        provider.report(record_property)
        for registration in registrations:
            owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
            if owner.outcome is None:
                await manager.request_cancel(runtime.actor(parent), owner.job_id, system_key="real-policy-cleanup", reason="test cleanup")
        pending = set(delegation._TASKS) - before
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 10)
        if boot:
            stopped = await runtime.shutdown(timeout_s=5)
            assert stopped["durable_drain_confirmed"] and not stopped["pending_handles"]
            assert not stopped["physical_stop_confirmed"] and not stopped["billing_settled"]
        await loop.aclose()
        await provider.close()
        await store.close()
