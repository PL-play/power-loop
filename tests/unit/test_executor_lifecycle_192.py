"""Automatic boot admission and local stop never imply remote settlement."""

from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, CancellationToken, StatefulAgentLoop
from power_loop.runtime import executor_registry as registry
from power_loop.runtime.executor_lifecycle import signal_token
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import AdapterModel

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def host(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(
        manager,
        tenant_id="lifecycle",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        lease_ms=3000,
        boot_contract_hash="a" * 64,
    )
    try:
        yield store, manager, runtime
    finally:
        await runtime.shutdown(timeout_s=0.5, grace_s=0.05)


async def test_concurrent_start_is_one_registration_and_cancelled_waiter_does_not_revoke_it(host, monkeypatch):
    _, manager, runtime = host
    entered, release = asyncio.Event(), asyncio.Event()
    original = registry.register
    calls = []

    async def delayed(*args, **kwargs):
        calls.append(kwargs)
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(registry, "register", delayed)
    first = asyncio.create_task(runtime.start())
    await entered.wait()
    others = [asyncio.create_task(runtime.start()) for _ in range(5)]
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    release.set()
    await asyncio.gather(*others)
    assert len(calls) == 1
    current = await registry.get(manager, runtime._lifecycle.identity)
    assert current["admission_open"] and current["version"] == 2
    events = await manager.db.fetchall(f"SELECT action FROM {manager.prefix}task_executor_events")
    assert sorted(e["action"] for e in events) == ["activate", "register"]


@pytest.mark.parametrize("phase", ["register", "activate", "heartbeat", "drain"])
async def test_lost_ack_recalls_same_boot_operation_without_duplicate_actions(host, monkeypatch, phase):
    _, manager, runtime = host
    original = getattr(registry, phase)
    accepted, lost = asyncio.Event(), False

    async def uncertain(*args, **kwargs):
        nonlocal lost
        result = await original(*args, **kwargs)
        if not lost:
            lost = True
            accepted.set()
            raise RuntimeError("committed before ACK")
        return result

    monkeypatch.setattr(registry, phase, uncertain)
    await runtime.start()
    if phase == "heartbeat":
        await asyncio.wait_for(accepted.wait(), 5)
        while runtime._lifecycle._version == 2:
            await asyncio.sleep(0.01)
    result = await runtime.shutdown(timeout_s=1)
    assert lost and result["durable_drain_confirmed"] and not result["pending_handles"]
    assert not result["physical_stop_confirmed"] and not result["billing_settled"]
    actions = await manager.db.fetchall(f"SELECT action,operation_key FROM {manager.prefix}task_executor_events")
    assert len(actions) == len({a["operation_key"] for a in actions})
    assert sum(a["action"] == phase for a in actions) == 1


async def test_idle_runtime_renews_boot_without_any_model_send(host):
    _, manager, runtime = host
    await runtime.start()
    first = await registry.get(manager, runtime._lifecycle.identity)

    async def renewed():
        while True:
            current = await registry.get(manager, runtime._lifecycle.identity)
            if current["version"] > first["version"]:
                return current
            await asyncio.sleep(0.02)

    current = await asyncio.wait_for(renewed(), 5)
    assert current["lease_until_ms"] > first["lease_until_ms"] and current["admission_open"]
    assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")


async def test_boot_heartbeat_failure_closes_local_admission_and_drains_without_model(host, monkeypatch):
    _, manager, runtime = host

    async def unavailable(*args, **kwargs):
        raise RuntimeError("PRIVATE-STORE-ERROR")

    monkeypatch.setattr(registry, "heartbeat", unavailable)
    await runtime.start()

    async def closed():
        while runtime._lifecycle._shutdown_task is None:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(closed(), 5)
    with pytest.raises(JobError, match="executor_supervision_lost"):
        await runtime.start()
    result = await runtime.shutdown()
    assert result["durable_drain_confirmed"] and result["pending_handles"] == 0
    assert not (await registry.get(manager, runtime._lifecycle.identity))["admission_open"]
    assert "PRIVATE" not in str(result)


@pytest.mark.parametrize("fault", ["exception", "hang"])
async def test_store_failure_still_signals_owned_work_and_retains_uncertain_control(host, monkeypatch, fault):
    _, manager, runtime = host
    await runtime.start()
    release, exited = asyncio.Event(), asyncio.Event()
    original = registry.drain

    async def broken(*args, **kwargs):
        if fault == "exception":
            raise RuntimeError("PRIVATE-DRAIN-ERROR")
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(registry, "drain", broken)

    async def owned():
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    task = asyncio.create_task(owned())
    await asyncio.sleep(0)
    runtime.track_task(task, cancel_token=CancellationToken.never())
    try:
        start = asyncio.get_running_loop().time()
        result = await runtime.shutdown(timeout_s=0.1, grace_s=0.01)
        assert asyncio.get_running_loop().time() - start < 0.5
        assert result["admission_open"] is False and not result["durable_drain_confirmed"]
        assert not result["store_safe_to_close"] and not result["physical_stop_confirmed"]
        assert result["control_pending"] == (fault == "hang")
        assert "PRIVATE" not in str(result)
        await asyncio.wait_for(exited.wait(), 1)
        with pytest.raises(JobError, match="executor_not_accepting"):
            runtime.check_local_admission()
    finally:
        release.set()
        await asyncio.gather(task, runtime._lifecycle._control_task, return_exceptions=True)


async def test_one_shutdown_deadline_for_noncooperative_tasks_and_repeated_calls(host):
    _, _, runtime = host
    await runtime.start()
    release = asyncio.Event()
    cancelled = []

    async def stubborn(index):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(index)
            await release.wait()

    tasks = [asyncio.create_task(stubborn(i)) for i in range(3)]
    for task in tasks:
        runtime.track_task(task)
    await asyncio.sleep(0)
    try:
        start = asyncio.get_running_loop().time()
        result = await runtime.shutdown(timeout_s=0.1, grace_s=0.01)
        assert asyncio.get_running_loop().time() - start < 0.5
        assert result["pending_handles"] == result["local_tasks_pending"] == 3
        assert not result["store_safe_to_close"] and result["durable_drain_confirmed"]
        second = await runtime.shutdown(timeout_s=20, reason="another caller")
        assert second is result and sorted(cancelled) == [0, 1, 2]
        assert all(task.cancelling() == 1 for task in tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks)


async def test_shutdown_from_inside_owned_host_task_does_not_cancel_or_wait_on_itself(host):
    _, _, runtime = host
    await runtime.start()
    current = asyncio.current_task()
    runtime.track_task(current)
    result = await runtime.shutdown(timeout_s=0.2)
    assert current.cancelling() == 0
    assert result["local_tasks_pending"] == 1 and not result["store_safe_to_close"]
    runtime._lifecycle._owned_tasks.pop(current)


@pytest.mark.parametrize("token_kind", ["never", "callable"])
async def test_lease_loss_with_readonly_token_stops_instead_of_hanging(host, monkeypatch, token_kind):
    store, manager, runtime = host
    sid = await store.create_session()
    actor = runtime.actor(sid)
    reg = await manager.register_job(actor, kind="send", session_id=sid, system_key="root")
    binding = await runtime.claim(actor, reg)
    runtime.lease_ms = 300
    token = CancellationToken.never() if token_kind == "never" else CancellationToken.from_any(lambda: False)
    renewed = asyncio.Event()

    async def unavailable(*args, **kwargs):
        renewed.set()
        raise JobError("injected_lease_failure")

    monkeypatch.setattr(manager, "renew_execution", unavailable)

    async def execute():
        async with runtime.execution(binding, cancel_token=token):
            await asyncio.Event().wait()

    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(renewed.wait(), 3)
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done
        with pytest.raises(JobError, match="task_supervision_lost"):
            await task
        assert not token.is_cancelled()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_automatic_start_and_shutdown_cover_queued_child_and_original_handles(host, monkeypatch):
    store, manager, runtime = host
    from power_loop.runtime import delegation

    model = AdapterModel()
    tools = ToolRegistry()
    register_spawn_agent(tools)
    loop = StatefulAgentLoop(
        llm=model,
        store=store,
        task_runtime=runtime,
        tool_registry=tools,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3),
    )
    sid = await loop.new_session()
    model.root = sid
    original = delegation.start
    semaphore = asyncio.Semaphore(0)

    async def queued(*args, **kwargs):
        return await original(*args, **kwargs, admission=semaphore)

    monkeypatch.setattr(delegation, "start", queued)
    await loop.send("delegate", sid)
    assert runtime._lifecycle._owned_tasks and model.child_calls == 0
    result = await runtime.shutdown(timeout_s=2, grace_s=0.05)
    assert result["durable_drain_confirmed"] and not result["pending_handles"]
    assert not runtime._local_executions and not runtime._model_executions and not runtime._lifecycle._owned_tasks
    assert model.child_calls == 0
    [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='subagent'")
    assert child["outcome"] == "cancelled" and child["cancel_requested_at"] is not None
    before = model.root_calls
    with pytest.raises(JobError, match="executor_not_accepting"):
        await loop.send("must not create paid work", sid)
    assert model.root_calls == before


async def test_legacy_runtime_does_not_silently_enable_boot_lifecycle(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="old", principal_id="host", executor_id="e", boot_id="b")
    await runtime.ensure_started()
    assert not await manager.db.fetchall(f"SELECT * FROM {manager.prefix}task_executors")
    with pytest.raises(JobError, match="executor_lifecycle_not_configured"):
        await runtime.start()
    with pytest.raises(JobError, match="executor_lifecycle_not_configured"):
        await runtime.shutdown()


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 301, "1"])
async def test_invalid_shutdown_does_not_close_admission(host, value):
    _, _, runtime = host
    await runtime.start()
    with pytest.raises(JobError, match="invalid_executor_shutdown_timeout"):
        await runtime.shutdown(timeout_s=value)
    runtime.check_local_admission()


async def test_token_helper_does_not_count_noop_or_faulty_signals():
    class Broken:
        def cancel(self, reason):
            raise ValueError("private")

    assert not signal_token(None, "stop")
    assert not signal_token(Broken(), "stop")
    assert not signal_token(CancellationToken.never(), "stop")
    owned = CancellationToken()
    assert signal_token(owned, "stop") and owned.is_cancelled()


@pytest.mark.parametrize("phase", ["register", "activate"])
async def test_shutdown_during_startup_retains_late_control_and_cannot_reopen_boot(host, monkeypatch, phase):
    _, manager, runtime = host
    entered, release = asyncio.Event(), asyncio.Event()
    original = getattr(registry, phase)

    async def delayed(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(registry, phase, delayed)
    starting = asyncio.create_task(runtime.start())
    await entered.wait()
    try:
        result = await runtime.shutdown(timeout_s=0.05)
        assert result["control_pending"] and not result["store_safe_to_close"]
        assert result["admission_open"] is False
    finally:
        release.set()
        await asyncio.gather(starting, runtime._lifecycle._control_task, return_exceptions=True)
    assert not (await registry.get(manager, runtime._lifecycle.identity))["admission_open"]
    assert runtime._lifecycle._heartbeat_task is None
    with pytest.raises(JobError, match="executor_startup_failed"):
        await runtime.start()


@pytest.mark.parametrize("phase", ["register", "activate"])
async def test_startup_failure_does_not_automatically_retry_a_new_boot_or_claim_jobs(host, monkeypatch, phase):
    _, manager, runtime = host
    called = []

    async def rejected(*args, **kwargs):
        called.append(phase)
        raise JobError("startup_reconciliation_required")

    monkeypatch.setattr(registry, phase, rejected)
    with pytest.raises(JobError, match="startup_reconciliation_required"):
        await runtime.start()
    with pytest.raises(JobError, match="executor_startup_failed"):
        await runtime.start()
    assert called == [phase]
    assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")


async def test_cancelled_shutdown_waiter_does_not_cancel_durable_control_or_stop_other_waiters(host, monkeypatch):
    _, manager, runtime = host
    await runtime.start()
    entered, release = asyncio.Event(), asyncio.Event()
    original = registry.drain

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(registry, "drain", delayed)
    first = asyncio.create_task(runtime.shutdown(timeout_s=1))
    await entered.wait()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    assert not runtime._lifecycle._control_task.cancelling()
    release.set()
    result = await runtime.shutdown(timeout_s=10)
    assert result["durable_drain_confirmed"] and result["pending_handles"] == 0
    rows = await manager.db.fetchall(f"SELECT action FROM {manager.prefix}task_executor_events WHERE action='drain'")
    assert len(rows) == 1


@pytest.mark.parametrize("field", ["tenant_id", "principal_id", "executor_id", "boot_id", "manager"])
async def test_mutable_runtime_cannot_downgrade_or_drain_a_different_identity(host, field):
    _, manager, runtime = host
    await runtime.start()
    setattr(runtime, field, JobManager(manager.store) if field == "manager" else "another")
    with pytest.raises(JobError, match="executor_binding_mismatch"):
        await runtime.start()
    stopped = await runtime.shutdown(timeout_s=1)
    assert stopped["durable_drain_confirmed"]
    original = await registry.get(manager, runtime._lifecycle.identity)
    assert original["state"] == "draining"
    [row] = await manager.db.fetchall(
        f"SELECT tenant_id,principal_id,executor_id,boot_id FROM {manager.prefix}task_executors"
    )
    assert row == {"tenant_id": "lifecycle", "principal_id": "host", "executor_id": "executor", "boot_id": "boot"}


async def test_late_control_ack_can_be_reobserved_without_restarting_shutdown(host, monkeypatch):
    _, manager, runtime = host
    await runtime.start()
    release = asyncio.Event()
    original = registry.drain

    async def delayed(*args, **kwargs):
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(registry, "drain", delayed)
    snapshot = await runtime.shutdown(timeout_s=0.05)
    assert snapshot["control_pending"] and not snapshot["store_safe_to_close"]
    release.set()
    await runtime._lifecycle._control_task
    current = await runtime.executor_status()
    assert current["durable_drain_confirmed"] and current["store_safe_to_close"]
    assert current["pending_handles"] == 0 and not current["admission_open"]
    assert not current["physical_stop_confirmed"] and not current["billing_settled"]
    assert await runtime.shutdown(timeout_s=30) is snapshot
    [row] = await manager.db.fetchall(
        f"SELECT count(*) AS n FROM {manager.prefix}task_executor_events WHERE action='drain'"
    )
    assert row["n"] == 1


@pytest.mark.parametrize("field", ["store", "db", "prefix", "jobs", "events"])
async def test_mutating_bound_manager_cannot_redirect_control(host, field, monkeypatch):
    _, manager, runtime = host
    await runtime.start()
    value = object() if field in ("store", "db") else "foreign"
    original = getattr(manager, field)
    monkeypatch.setattr(manager, field, value)
    try:
        result = await runtime.shutdown(timeout_s=0.2)
        assert not result["durable_drain_confirmed"]
        assert result["control_error"] == "executor_binding_mismatch"
        status = await runtime.executor_status()
        assert status["control_error"] == "executor_binding_mismatch" and not status["store_safe_to_close"]
    finally:
        monkeypatch.setattr(manager, field, original)
    # No redirected operation was issued. The original boot can still be
    # mechanically drained with its original identity for test cleanup.
    await registry.drain(
        manager, runtime._lifecycle.identity, system_key="repair-original", reason="original binding restored"
    )


async def test_job_heartbeat_task_allocation_failure_restores_all_execution_contexts(host, monkeypatch):
    from power_loop.runtime.task_runtime import _OWNER, _SEND, _TASK_OWNERS

    store, manager, runtime = host
    await runtime.start()
    sid = await store.create_session()
    actor = runtime.actor(sid)
    registration = await manager.register_job(actor, kind="send", session_id=sid, system_key="factory", request={})
    binding = await runtime.claim(actor, registration)
    before = (_SEND.get(), _OWNER.get(), _TASK_OWNERS.get())
    create = asyncio.create_task
    rejected = []

    def unavailable(coroutine, *, name=None, **kwargs):
        if name and name.startswith("job-lease-"):
            rejected.append(coroutine)
            raise RuntimeError("task allocation failed")
        return create(coroutine, name=name, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_task", unavailable)
        with pytest.raises(RuntimeError, match="task allocation failed"):
            async with runtime.execution(binding):
                pytest.fail("No execution body may start without its heartbeat")
    assert (_SEND.get(), _OWNER.get(), _TASK_OWNERS.get()) == before
    assert len(rejected) == 1 and rejected[0].cr_frame is None
    assert not runtime._local_executions
    # Allocation failure is not a fabricated result/physical stop receipt.
    assert (await manager.get_job(actor, registration.job.job_id)).result is None


@pytest.mark.parametrize("phase", ["start", "heartbeat", "shutdown", "drain"])
async def test_lifecycle_task_allocation_failure_stops_owned_work_and_keeps_recovery_available(
    host, monkeypatch, phase
):
    _, manager, runtime = host
    entered = asyncio.Event()

    async def original_work():
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(original_work())
    runtime.track_task(task)
    await entered.wait()
    create, rejected = asyncio.create_task, []
    if phase in ("shutdown", "drain"):
        await runtime.start()

    def unavailable(coroutine, *, name=None, **kwargs):
        if name and name.startswith("executor-" + phase + "-"):
            rejected.append(coroutine)
            raise RuntimeError("task allocation failed")
        return create(coroutine, name=name, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "create_task", unavailable)
            with pytest.raises(RuntimeError, match="task allocation failed"):
                if phase in ("start", "heartbeat"):
                    await runtime.start()
                else:
                    await runtime.shutdown(timeout_s=1, grace_s=0.01, reason="original stop")
        assert len(rejected) == 1 and rejected[0].cr_frame is None
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
        assert task.cancelled() and runtime._lifecycle.closed
        with pytest.raises(JobError, match="executor_(startup|shutdown)_failed"):
            runtime.check_local_admission()
        deadline = runtime._lifecycle._shutdown_deadline
        await runtime.shutdown(timeout_s=2, reason="must not replace original stop")
        if deadline is not None:
            assert runtime._lifecycle._shutdown_deadline == deadline
            assert runtime._lifecycle._shutdown_reason == "original stop"
        control = runtime._lifecycle._control_task
        if control is not None:
            await asyncio.gather(control, return_exceptions=True)
        state = await runtime.executor_status()
        assert not state["admission_open"] and not state["physical_stop_confirmed"]
        if phase != "start":
            assert state["durable_drain_confirmed"]
            rows = await manager.db.fetchall(
                f"SELECT action FROM {manager.prefix}task_executor_events WHERE action='drain'"
            )
            assert len(rows) == 1
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
