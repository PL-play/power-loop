"""Transferring a timed-out request must not double-signal ordinary shutdown."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, CancellationToken, StatefulAgentLoop
from power_loop.runtime.model_attempt_query import list_attempts

from .test_executor_lifecycle_192 import host
from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import AdapterModel

__all__ = ["host", "registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_ordinary_shutdown_keeps_one_request_cancellation_and_usage_cleanup(host, monkeypatch):
    store, manager, runtime = host
    model = AdapterModel()
    started, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    signals = []

    async def inflight(request, **kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            signals.append("request-cancelled")
            try:
                await asyncio.sleep(.02)  # original provider cancellation cleanup
                cleaned.set()
            except asyncio.CancelledError:
                signals.append("cleanup-cancelled-again")
                raise
            raise
        raise AssertionError("must not turn shutdown into a completed answer")

    monkeypatch.setattr(model, "complete", inflight)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2))
    sid = await loop.new_session()
    sending = asyncio.create_task(loop.send("Compute original answer", sid))
    try:
        await asyncio.wait_for(started.wait(), 3)
        stopped = await runtime.shutdown(timeout_s=2, grace_s=.3)
        result = await sending
        assert result.status == "cancelled"
        assert signals == ["request-cancelled"] and cleaned.is_set()
        assert stopped["durable_drain_confirmed"] and not stopped["pending_handles"]
        assert not runtime._adapter_tasks
        [job] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
        [attempt] = (await list_attempts(manager, runtime.actor(sid), job["job_id"]))["items"]
        assert attempt["status"] == "unknown" and attempt["estimated"]
        assert job["resource_state"] == "unknown" and job["supervision_status"] == "open"
    finally:
        release.set()
        await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.parametrize("stop_kind", ["shutdown", "token", "caller"])
async def test_original_request_stays_reachable_after_controller_exits(host, monkeypatch, stop_kind):
    store, manager, runtime = host
    runtime.lease_ms = 30000
    model = AdapterModel()
    model.release.set()
    original = model.complete
    started, signalled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    handles = set()
    async def uncooperative(request, **kwargs):
        handles.add(asyncio.current_task())
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            signalled.set()
            await release.wait()
        return await original(request, **kwargs)
    monkeypatch.setattr(model, "complete", uncooperative)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2))
    sid = await loop.new_session()
    token = CancellationToken()
    sending = asyncio.create_task(loop.send("original request", sid, stop_event=token if stop_kind == "token" else None))
    try:
        await asyncio.wait_for(started.wait(), 3)
        if stop_kind == "shutdown":
            stopped = await runtime.shutdown(timeout_s=.2, grace_s=.05)
            assert stopped["pending_handles"] and not stopped["store_safe_to_close"], {"snapshot": stopped, "requests": [repr(t) for t in handles], "adapters": [repr(t) for t in runtime._adapter_tasks]}
        elif stop_kind == "token":
            token.cancel("original explicit stop")
        else:
            sending.cancel("original caller stop")
        await asyncio.wait_for(signalled.wait(), 1)
        done, pending = await asyncio.wait({sending}, timeout=6)
        assert done and not pending, "controller cleanup did not finish its original bounded wait"
        [request_task] = handles
        assert not request_task.done()
        current = await runtime.executor_status()
        assert current["pending_handles"] > 0 and not current["store_safe_to_close"], current
        assert request_task in runtime._adapter_tasks
        [job] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
        original_result = job["result_json"]
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(sending, *handles, return_exceptions=True), 3)

    assert not runtime._adapter_tasks and all(t.done() for t in handles)
    [attempt] = (await list_attempts(manager, runtime.actor(sid), job["job_id"]))["items"]
    assert attempt["status"] == "returned" and not attempt["estimated"]
    assert attempt["observed_usage"]["total_tokens"] == 10
    [after] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert after["result_json"] == original_result
    assert model.child_calls == 1
