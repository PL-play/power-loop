"""Exercise original continuations with registered boot bindings, not legacy mode."""

from __future__ import annotations

import asyncio

import pytest

from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_subagent_resume_192 import ready
from .test_subagent_resume_192 import (
    test_original_job_and_session_resume_once_with_distinct_results_and_usage as verify_agent,
)
from .test_task_runtime_192 import finish, managed

__all__ = ['registry_store', 'ready', 'managed']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
async def registered_host(monkeypatch):
    # Test-host startup through the real persistent ports. It runs before the
    # original scenario fixtures' first control writer; no model response is
    # altered. Boot proof must finish before the joined send admission opens.
    ensure_started = TaskRuntime.ensure_started
    seen = set()

    async def registered(runtime):
        await ensure_started(runtime)
        identity = registry.ExecutorIdentity(
            runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id
        )
        key = (runtime.manager.store.store_id, identity)
        if key not in seen:
            assert runtime.manager._joined_write.get() is None
            await registry.register(runtime.manager, identity, contract_hash="a" * 64, startup_lease_ms=300000)
            await registry.activate(
                runtime.manager, identity, expected_version=1, system_key="startup", lease_ms=300000
            )
            seen.add(key)

    monkeypatch.setattr(TaskRuntime, "ensure_started", registered)


async def assert_registered_scopes(manager, actor):
    jobs = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE generation>0")
    assert jobs and all(j["executor_key"] is not None for j in jobs)
    for job in jobs:
        for generation in range(1, job["generation"] + 1):
            saved = await manager.get_execution_membership(actor, job["job_id"], generation)
            assert saved["scope"]["version"] == 2 and saved["scope"]["executor_binding"]


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack"])
async def test_registered_agent_original_resume_keeps_identity_and_canonical_usage(ready, monkeypatch, fault):
    await verify_agent(ready, monkeypatch, fault)
    await assert_registered_scopes(ready.manager, ready.runtime.actor(ready.parent))








async def test_registered_runtime_drain_stops_original_child_and_rejects_new_paid_send(managed):
    _, manager, runtime, loop, model, sid, _ = managed
    await loop.send("delegate", sid)
    await asyncio.wait_for(model.entered.wait(), 3)
    identity = registry.ExecutorIdentity(runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id)
    before = model.root_calls, model.child_calls
    stopped = await registry.drain(manager, identity, system_key="shutdown", reason="host shutdown")
    assert stopped["effects"]["executions"] and not stopped["physical_stop_confirmed"]
    with pytest.raises(JobError, match="executor_not_accepting"):
        await loop.send("new paid work must not start", sid)
    await finish(managed)
    assert (model.root_calls, model.child_calls) == before
    rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='subagent'")
    assert len(rows) == 1
    child = await manager.get_job(runtime.actor(sid), rows[0]["job_id"])
    assert child.cancel_requested_at is not None and child.outcome == "cancelled"
    assert child.supervision_status == "open"




@pytest.mark.parametrize("cause", ["executor_lease_expired", "lease_expired", "cancel_requested"])
async def test_caught_heartbeat_cancellation_preserves_failure_cause_before_result_commit(
    registry_store, monkeypatch, cause
):
    from power_loop.runtime.jobs import JobManager

    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(
        manager, tenant_id="heartbeat", principal_id="host", executor_id="executor", boot_id="boot", lease_ms=300
    )
    sid = await store.create_session()
    actor = runtime.actor(sid)
    reg = await manager.register_job(actor, kind="send", session_id=sid, system_key="original", request={})
    binding = await runtime.claim(actor, reg)
    calls = []

    async def failed_renewal(*args, **kwargs):
        calls.append(binding.lease.job_id)
        raise JobError(cause)

    monkeypatch.setattr(manager, "renew_execution", failed_renewal)

    async def body():
        async with runtime.execution(binding):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Mirrors a workflow's caught cancellation before its context
                # exits. Completed output and usage must remain available.
                return await runtime.record_scope(binding, {"status": "cancelled", "text": "REAL-COMPLETED-OUTPUT"})

    task = asyncio.create_task(body())
    try:
        if cause == "cancel_requested":
            result = await asyncio.wait_for(task, 5)
            assert result["status"] == "cancelled"
        else:
            with pytest.raises(JobError, match="task_supervision_lost"):
                await asyncio.wait_for(task, 5)
        job = await manager.get_job(actor, reg.job.job_id)
        assert job.result["text"] == "REAL-COMPLETED-OUTPUT" and calls == [reg.job.job_id]
        assert job.outcome == ("cancelled" if cause == "cancel_requested" else "failed")
        if cause != "cancel_requested":
            assert job.result["supervision_error"] == cause and job.result["execution_status"] == "cancelled"
        assert not runtime._local_executions
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
