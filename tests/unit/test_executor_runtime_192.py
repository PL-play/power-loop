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
        identity = registry.ExecutorIdentity(runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id)
        key = (runtime.manager.store.store_id, identity)
        if key not in seen:
            await registry.register(runtime.manager, identity, contract_hash="a"*64, startup_lease_ms=300000)
            await registry.activate(runtime.manager, identity, expected_version=1, system_key="startup", lease_ms=300000)
            seen.add(key)
        return await claim(runtime, *args, **kwargs)
    monkeypatch.setattr(TaskRuntime, "claim", registered)


async def assert_registered_scopes(manager, actor):
    jobs = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE generation>0")
    assert jobs and all(j["executor_key"] is not None for j in jobs)
    for job in jobs:
        for generation in range(1, job["generation"]+1):
            saved = await manager.get_execution_membership(actor, job["job_id"], generation)
            assert saved["scope"]["version"] == 2 and saved["scope"]["executor_binding"]


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack"])
async def test_registered_agent_original_resume_keeps_identity_and_canonical_usage(ready, monkeypatch, fault):
    await verify_agent(ready, monkeypatch, fault)
    await assert_registered_scopes(ready.manager, ready.runtime.actor(ready.parent))


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack"])
async def test_registered_live_node_resume_keeps_owner_and_scope_history(ready_node, managed_workflow, monkeypatch, fault):
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


async def test_boot_expiry_before_node_commit_cannot_advance_review_or_lose_completed_output(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = set()
    record = runtime.record_scope
    async def expire_at_commit(binding, result):
        if result.get("node_id") == "BUILD" and result.get("status") == "completed":
            identity = registry.ExecutorIdentity(runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id)
            boot = await registry.get(manager, identity)
            monkeypatch.setattr(manager, "now", lambda: boot["lease_until_ms"]+1)
        return await record(binding, result)
    monkeypatch.setattr(runtime, "record_scope", expire_at_commit)
    await loop.send("run", sid)
    await asyncio.wait_for(state["handles"][0].task, 8)
    assert model.counts["BUILD"] == 1 and model.counts["REVIEW"] == 0
    [row] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
    build = await manager.get_job(runtime.actor(sid), row["job_id"])
    assert build.outcome == "failed" and build.result["text"] == "PROOF-42"
    assert build.result["usage"]["total_tokens"] == 10
    workflow = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert workflow.outcome == "failed" and workflow.result["usage"]["total_tokens"] == 10
