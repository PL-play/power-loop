"""Exercise original continuations with registered boot bindings, not legacy mode."""

from __future__ import annotations

import asyncio

import pytest

from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_node_resume_192 import ready_node
from .test_node_resume_192 import test_finished_leaf_resumes_same_job_sid_before_downstream as verify_node
from .test_subagent_resume_192 import ready
from .test_subagent_resume_192 import (
    test_original_job_and_session_resume_once_with_distinct_results_and_usage as verify_agent,
)
from .test_task_runtime_192 import finish, managed
from .test_workflow_bound_restore_192 import bound_restore
from .test_workflow_bound_restore_192 import (
    test_original_pool_and_node_identities_survive_two_restorations_without_replay_charge as verify_bound,
)
from .test_workflow_restore_runtime_192 import native_restore
from .test_workflow_restore_runtime_192 import (
    test_original_workflow_and_nodes_recompute_and_main_consumes_only_new_revision as verify_workflow,
)
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "ready_node", "ready", "managed", "managed_workflow", "native_restore", "bound_restore"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
async def registered_host(monkeypatch):
    # Test-host startup through the real persistent ports. It runs before the
    # original scenario fixtures' first claim; no model response is altered.
    claim = TaskRuntime.claim
    seen = set()

    async def registered(runtime, *args, **kwargs):
        identity = registry.ExecutorIdentity(
            runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id
        )
        key = (runtime.manager.store.store_id, identity)
        if key not in seen:
            await registry.register(runtime.manager, identity, contract_hash="a" * 64, startup_lease_ms=300000)
            await registry.activate(
                runtime.manager, identity, expected_version=1, system_key="startup", lease_ms=300000
            )
            seen.add(key)
        return await claim(runtime, *args, **kwargs)

    monkeypatch.setattr(TaskRuntime, "claim", registered)


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


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack"])
async def test_registered_live_node_resume_keeps_owner_and_scope_history(
    ready_node, managed_workflow, monkeypatch, fault
):
    await verify_node(ready_node, managed_workflow, monkeypatch, fault)
    await assert_registered_scopes(ready_node.manager, ready_node.runtime.actor(ready_node.parent))


async def test_registered_whole_workflow_restores_original_scopes(native_restore):
    await verify_workflow(native_restore)
    await assert_registered_scopes(native_restore.manager, native_restore.runtime.actor(native_restore.parent))


async def test_registered_bound_workflow_retains_original_pool_through_two_restorations(bound_restore):
    await verify_bound(bound_restore)
    await assert_registered_scopes(bound_restore.manager, bound_restore.runtime.actor(bound_restore.parent))


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


@pytest.mark.parametrize("during_failure_write", [False, True, "before_commit"])
async def test_boot_expiry_before_node_commit_cannot_advance_review_or_lose_completed_output(
    managed_workflow, monkeypatch, during_failure_write
):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = set()
    record = runtime.record_scope
    usage = manager.execution_usage
    first_read = []

    async def pause_first_read(actor, job_id, **kwargs):
        local = next((v for v in runtime._local_executions.values() if v["binding"].lease.job_id == job_id), None)
        if (
            during_failure_write == "before_commit"
            and not first_read
            and local
            and local["binding"].stop_renewal.is_set()
        ):
            row = await manager._row(manager.db, job_id)
            if row["kind"] == "workflow_node":
                first_read.append(job_id)
                await asyncio.sleep(1.1)
        return await usage(actor, job_id, **kwargs)

    monkeypatch.setattr(manager, "execution_usage", pause_first_read)

    async def expire_at_commit(binding, result):
        if result.get("node_id") == "BUILD" and result.get("status") == "completed":
            identity = registry.ExecutorIdentity(
                runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id
            )
            boot = await registry.get(manager, identity)
            monkeypatch.setattr(manager, "now", lambda: boot["lease_until_ms"] + 1)
        return await record(binding, result)

    monkeypatch.setattr(runtime, "record_scope", expire_at_commit)
    record_result = manager.record_execution_result
    failed_writes = []

    async def slow_failure_write(actor, lease, **kwargs):
        if (
            during_failure_write is True
            and kwargs["outcome"] == "failed"
            and kwargs["result"].get("node_id") == "BUILD"
        ):
            failed_writes.append(lease.job_id)
            # Allow the original workflow heartbeat to observe the expired
            # clock while the node's original negative result write is in flight.
            await asyncio.sleep(1)
        return await record_result(actor, lease, **kwargs)

    monkeypatch.setattr(manager, "record_execution_result", slow_failure_write)
    await loop.send("run", sid)
    await asyncio.wait_for(state["handles"][0].task, 8)
    assert model.counts["BUILD"] == 1 and model.counts["REVIEW"] == 0
    [row] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
    build = await manager.get_job(runtime.actor(sid), row["job_id"])
    assert build.outcome == "failed" and build.result["text"] == "PROOF-42"
    assert build.result["usage"]["total_tokens"] == 10
    workflow = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert workflow.outcome == "failed" and workflow.result["usage"]["total_tokens"] == 10
    assert workflow.result["results"]["BUILD"]["text"] == "PROOF-42"
    assert workflow.result["results"]["BUILD"]["usage"]["total_tokens"] == 10
    assert not any("KeyError" in error for error in workflow.result["errors"])
    if during_failure_write is True:
        assert failed_writes == [build.job_id]
        assert workflow.result["supervision_error"] in ("execution_lease_expired", "executor_lease_expired")
        assert workflow.result["execution_status"] == "cancelled"
    if during_failure_write == "before_commit":
        assert first_read == [build.job_id]
        assert build.result["supervision_error"] == workflow.result["supervision_error"]
        assert build.result["supervision_error"] in ("execution_lease_expired", "executor_lease_expired")
    assert not runtime._adapter_tasks


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
