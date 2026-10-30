"""A short original wait policy must also stop a request already in progress."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation, model_attempts
from power_loop.runtime import timer_waits as waits
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND

from .test_job_registry_192 import registry_store
from .test_owner_waits_192 import original
from .test_workflow_task_runtime_192 import leaf, managed_workflow

__all__ = ["registry_store", "original", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("late_response", [False, True])
async def test_short_policy_interrupts_original_inflight_request_before_lease_renewal(original, monkeypatch, late_response):
    c = original
    c.state.wait_ms = 1200
    entered, stopped, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    complete = c.model.complete

    async def in_progress(request, **kwargs):
        response = await complete(request, **kwargs)
        if get_session_id() != c.state.parent and c.model.calls["child"] == 2:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                stopped.set()
                if not late_response:
                    raise
                # A cancellation-suppressing provider can still return usage.
                # This gate models lateness, never a falsely quiescent owner.
                await release.wait()
        return response

    monkeypatch.setattr(c.model, "complete", in_progress)
    before = set(delegation._TASKS)
    try:
        await c.loop.send("Start original task with a short deadline", c.state.parent)
        await asyncio.wait_for(entered.wait(), 3)
        assert c.runtime.lease_ms == 30000
        # The policy is much shorter than the ten-second lease heartbeat.
        # No fake wall clock, external cancel, replacement request or model retry.
        await asyncio.wait_for(stopped.wait(), 2.5)
        release.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)

        [registration] = c.state.registrations
        owner = await c.manager.get_job(c.runtime.actor(c.state.parent), registration.job.control_owner_job_id)
        assert c.model.calls["child"] == 2
        assert owner.cancel_requested_at is None
        assert owner.outcome == "timed_out", owner
        assert owner.result["status"] == "timed_out"
        assert owner.resource_state == ("stopped" if late_response else "unknown")
        assert owner.supervision_status == "open"
        [send] = await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='send' AND control_owner_job_id=?", (owner.job_id,))
        items = (await list_attempts(c.manager, c.runtime.actor(c.state.parent), send["job_id"]))["items"]
        assert sorted(item["status"] for item in items) == (["returned", "returned"] if late_response else ["returned", "unknown"])
        if late_response:
            assert all(item["observed_usage"]["total_tokens"] == 10 for item in items)
        else:
            [unknown] = [item for item in items if item["status"] == "unknown"]
            assert unknown["estimated"] is True and unknown["observed_usage"]["prompt_tokens"] > 0
    finally:
        release.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)

async def test_uncooperative_provider_does_not_hold_original_controller_past_deadline(original, monkeypatch, *, shutdown=False):
    c = original
    c.state.wait_ms = 1200
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    complete = c.model.complete

    async def ignore_cancel(request, **kwargs):
        response = await complete(request, **kwargs)
        if get_session_id() != c.state.parent and c.model.calls["child"] == 2:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
        return response

    monkeypatch.setattr(c.model, "complete", ignore_cancel)
    before = set(delegation._TASKS)
    controller_tasks = set()
    try:
        await c.loop.send("Start original task", c.state.parent)
        await asyncio.wait_for(started.wait(), 3)
        controller_tasks = set(delegation._TASKS) - before
        await asyncio.wait_for(cancelled.wait(), 2.5)
        done, pending = await asyncio.wait(controller_tasks, timeout=.5)
        assert done and not pending, "an uncooperative request must not hold its original controller"
        [registration] = c.state.registrations
        owner = await c.manager.get_job(c.runtime.actor(c.state.parent), registration.job.control_owner_job_id)
        assert owner.outcome == "timed_out" and owner.resource_state == "unknown", owner
        assert owner.supervision_status == "open" and not release.is_set()
        assert c.runtime._adapter_tasks, "the original in-flight request handle must remain reachable"
        if shutdown:
            snapshot = await c.runtime.shutdown(timeout_s=.2, grace_s=.05)
            assert snapshot["pending_handles"] and not snapshot["store_safe_to_close"]
            assert not snapshot["physical_stop_confirmed"] and not snapshot["billing_settled"]
    finally:
        release.set()
        pending = controller_tasks | set(c.runtime._adapter_tasks)
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 5)
    assert c.model.calls["child"] == 2
    assert not c.runtime._adapter_tasks
    assert (await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)).result == owner.result
    [send] = await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='send' AND control_owner_job_id=?", (owner.job_id,))
    items = (await list_attempts(c.manager, c.runtime.actor(c.state.parent), send["job_id"]))["items"]
    assert [item["status"] for item in items] == ["returned", "returned"]
    assert sum(item["observed_usage"]["total_tokens"] for item in items) == 20
    # execution_usage is the immutable model-exit snapshot, not the late billing ledger.


@pytest.mark.parametrize("original", ["boot"], indirect=True)
async def test_deadline_then_shutdown_retains_the_same_uncooperative_request(original, monkeypatch):
    assert original.runtime._lifecycle is not None
    try:
        await test_uncooperative_provider_does_not_hold_original_controller_past_deadline(original, monkeypatch, shutdown=True)
        current = await original.runtime.executor_status()
        assert not current["pending_handles"] and current["store_safe_to_close"]
        assert not current["physical_stop_confirmed"] and not current["billing_settled"]
    finally:
        await original.runtime.shutdown(timeout_s=2)


async def test_native_node_short_policy_does_not_run_downstream_or_retry_business(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, parent, state, _ = managed_workflow
    model.hold = set()
    runtime.lease_ms = 30000
    state["on_complete"] = None
    state["spec"] = {"name": "short-original-node", "root": {"type": "sequence",
        "steps": [leaf("BUILD", tools=["arm_wait"]), leaf("REVIEW")]}}
    registrations = []
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    complete = model.complete

    async def arm():
        source, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registrations.append(await waits.register_wait(manager, runtime.actor(sid), caller=source.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 250, "Review original node", "Continue original node"),
            policy=waits.WaitPolicy(now + 1200)))
        return "Original policy armed."

    loop.tool_registry.register(ToolDefinition(name="arm_wait", description="Arm original node policy",
        input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), arm)

    async def in_progress(request, **kwargs):
        response = await complete(request, **kwargs)
        if "ROLE-BUILD" in (request.system_prompt or ""):
            if model.counts["BUILD"] == 1:
                response.raw_text = ""
                response.tool_calls = [{"id": "node-policy", "type": "function", "function": {
                    "name": "arm_wait", "arguments": json.dumps({})}}]
            else:
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
        return response

    monkeypatch.setattr(model, "complete", in_progress)
    try:
        await loop.send("Run original workflow", parent)
        await asyncio.wait_for(started.wait(), 3)
        await asyncio.wait_for(cancelled.wait(), 2.5)
        [handle] = state["handles"]
        await asyncio.wait_for(asyncio.shield(handle.task), 5)
        [registration] = registrations
        owner = await manager.get_job(runtime.actor(parent), registration.job.control_owner_job_id)
        assert owner.outcome == "timed_out" and owner.resource_state == "unknown", owner
        assert owner.supervision_status == "open" and owner.cancel_requested_at is None
        assert model.counts["BUILD"] == 2 and model.counts["REVIEW"] == 0
        workflow = await manager.get_job(runtime.actor(parent), handle.job_id)
        assert workflow.outcome == "failed" and workflow.resource_state == "unknown"
        assert workflow.result["results"]["BUILD"]["status"] == "timed_out"
    finally:
        release.set()


async def test_lost_original_dispatch_ack_keeps_one_request_and_original_deadline(original, monkeypatch):
    c = original
    dispatch = model_attempts.dispatch
    lost = []

    async def lose_ack(attempt):
        deadline = await dispatch(attempt)
        if attempt.binding.actor.session_id != c.state.parent and c.state.registrations and not lost:
            lost.append((attempt.attempt_id, deadline))
            raise OSError("original dispatch committed, acknowledgement lost")
        return deadline

    monkeypatch.setattr(model_attempts, "dispatch", lose_ack)
    await test_short_policy_interrupts_original_inflight_request_before_lease_renewal(c, monkeypatch, False)
    assert len(lost) == 1 and lost[0][1] is not None


async def test_original_dispatch_bound_does_not_need_another_store_round_trip(original, monkeypatch):
    c = original
    deadline = model_attempts.provider_deadline
    checked = []

    async def use_grant(attempt):
        write = c.manager._write

        def unexpected_read():
            pytest.fail("original dispatch already checked the policy; do not open another transaction")

        monkeypatch.setattr(c.manager, "_write", unexpected_read)
        try:
            bound = await deadline(attempt)
        finally:
            monkeypatch.setattr(c.manager, "_write", write)
        if attempt.binding.actor.session_id != c.state.parent and c.state.registrations:
            checked.append(bound)
        return bound

    monkeypatch.setattr(model_attempts, "provider_deadline", use_grant)
    await test_short_policy_interrupts_original_inflight_request_before_lease_renewal(c, monkeypatch, False)
    assert len(checked) == 1 and checked[0] is not None


async def test_dispatch_commit_delay_cannot_start_transport_after_original_deadline(original, monkeypatch):
    c = original
    c.state.wait_ms = 1200
    dispatch = model_attempts.dispatch
    delayed = []

    async def slow_commit_ack(attempt):
        deadline = await dispatch(attempt)
        if attempt.binding.actor.session_id != c.state.parent and c.state.registrations:
            assert deadline is not None
            delayed.append(attempt.attempt_id)
            await asyncio.sleep(max(0, deadline - asyncio.get_running_loop().time()) + .02)
        return deadline

    monkeypatch.setattr(model_attempts, "dispatch", slow_commit_ack)
    before = set(delegation._TASKS)
    await c.loop.send("Start original finite task", c.state.parent)
    tasks = set(delegation._TASKS) - before
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 6)
    assert len(delayed) == 1
    assert c.model.calls["child"] == 1, "a stale original grant must not start another paid transport"
    [registration] = c.state.registrations
    owner = await c.manager.get_job(c.runtime.actor(c.state.parent), registration.job.control_owner_job_id)
    assert owner.outcome == "timed_out" and owner.supervision_status == "open"
    [send] = await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='send' AND control_owner_job_id=?", (owner.job_id,))
    items = (await list_attempts(c.manager, c.runtime.actor(c.state.parent), send["job_id"]))["items"]
    [unconfirmed] = [item for item in items if item["status"] == "unknown"]
    assert unconfirmed["observed_usage"]["total_tokens"] == 0 and unconfirmed["estimated"] is True
    assert unconfirmed["provider_reported_cost"] is None  # no fabricated final zero fee/refund
