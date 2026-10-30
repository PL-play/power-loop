"""Original bound resources survive actual same-workflow/SID restoration."""
from __future__ import annotations

import asyncio
import copy

import pytest

from power_loop.runtime import workflow_restore_execution, workflow_resume
from power_loop.workflow import InProcessExecutor, SharedBudget
from power_loop.workflow.resume import resume_detached

from .test_job_registry_192 import registry_store
from .test_node_nudge_runtime_192 import client
from .test_workflow_resource_profile_192 import install
from .test_workflow_restore_plan_192 import finished
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def bound_restore(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"]["steps"][-1]["inputs_from"] = ["BUILD"]
    c.binding, c.pool, c.registry, c.track = install(managed_workflow)
    c.initial = await finished(managed_workflow)
    c.actor = c.runtime.actor(c.parent)
    c.original = await c.manager.get_result_revision(c.actor, c.initial.job_id, 1)
    c.nodes = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='workflow_node'")
    c.counts = c.model.counts.copy()

    async def restore(*, node="BUILD", repeat=False):
        handles = []
        async def handler():
            handle = await resume_detached(c.loop, c.parent, c.initial.run_id,
                continuation=(node, "NUDGED-43", 3))
            handles.append(handle)
            c.state["handles"].append(handle)
            if repeat:
                assert await resume_detached(c.loop, c.parent, c.initial.run_id,
                    continuation=(node, "NUDGED-43", 3)) is handle
            return {"job_id": handle.job_id, "operation_id": handle.operation_id, "status": "accepted"}
        await c.issue(handler=handler)
        if handles:
            await asyncio.wait_for(handles[0].task, 10)
            return handles[0]
        return None
    c.restore = restore
    return c


async def test_original_pool_and_node_identities_survive_two_restorations_without_replay_charge(bound_restore):
    c = bound_restore
    assert c.pool.spent == 27
    c.pool.commit({"total_tokens": 13})  # another scope, invisible to this W's ledger
    handle = await c.restore(repeat=True)
    assert handle.job_id == c.initial.job_id and handle.run_id == c.initial.run_id
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "succeeded" and current.generation == 2, current.result
    assert current.result["final"]["text"] == "NUDGED-43"
    assert c.pool.spent == 60  # 7 before creation + 20 + 13 + 20; no refill/prior re-charge
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 1) == c.original
    for nid, result in current.result["results"].items():
        assert result["session_id"] == c.original["payload"]["result"]["results"][nid]["session_id"]
    assert {n["job_id"] for n in await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow_node'")} == {n["job_id"] for n in c.nodes}
    op = await c.manager.get_operation(c.actor, handle.operation_id)
    assert op.result["request"]["resource_evidence"]["budget_spent"] == 40
    assert op.call_ref["id"] != op.receipt_ref["id"]
    assert c.track["resolve"] == 3  # duplicate exact tool call does not resolve again
    await c.loop.flush_follow_ups(c.parent)
    next_handle = await c.restore(node="REVIEW")
    current = await c.manager.get_job(c.actor, next_handle.job_id)
    assert current.outcome == "succeeded" and current.generation == 3, current.result
    assert c.model.counts["BUILD"] == 2 and c.model.counts["REVIEW"] == 3
    assert current.result["usage"]["total_tokens"] == 10 and c.pool.spent == 70


@pytest.mark.parametrize("fault", ["denied", "revoke", "capacity", "floor", "rewound", "registry"])
async def test_invalid_original_resources_fail_before_new_generation_or_any_child_request(bound_restore, fault):
    c = bound_restore
    if fault == "denied":
        c.track["allow"] = False
    elif fault == "revoke":
        c.registry.unregister("host", "1.0.0")
    elif fault == "capacity":
        c.pool.max_tokens += 1000
    elif fault == "floor":
        c.pool._floor += 1
    elif fault == "rewound":
        c.pool._spent -= 1
    else:
        c.runtime.workflow_resources = None
    assert await c.restore() is None
    current = await c.manager.get_job(c.actor, c.initial.job_id)
    assert current.generation == 1 and current.outcome == "succeeded"
    assert c.model.counts["BUILD"] == c.counts["BUILD"] and c.model.counts["REVIEW"] == c.counts["REVIEW"]
    assert await c.manager.get_result_revision(c.actor, current.job_id, 1) == c.original


@pytest.mark.parametrize("port", ["admit", "claim_dispatch"])
async def test_committed_ack_loss_keeps_one_operation_and_original_pool(bound_restore, monkeypatch, port):
    c = bound_restore
    original = getattr(workflow_resume, port)
    async def ack_lost(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("committed, ACK lost")
    monkeypatch.setattr(workflow_resume, port, ack_lost)
    handle = await c.restore(repeat=True)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "succeeded" and current.generation == 2, current.result
    assert c.pool.spent == 47
    operations = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND action='nudge'", (handle.job_id,))
    assert len(operations) == 1 and c.track["resolve"] == 3


@pytest.mark.parametrize("fault", ["pool", "executor", "preparation", "profile"])
async def test_engine_cannot_substitute_prepared_resources_before_provider_dispatch(bound_restore, monkeypatch, fault):
    c = bound_restore
    original = workflow_restore_execution.validate_engine
    async def substitute(runtime, binding, engine, spec):
        if fault == "pool":
            engine._budget = SharedBudget(c.pool.max_tokens)
            engine._budget.commit({"total_tokens": c.pool.spent})
        elif fault == "executor":
            engine._executor = InProcessExecutor()
        elif fault == "preparation":
            engine._restore_resources = None
        else:
            engine._restore_resources.profile["resource_binding"]["resource_id"] = "substitute"
        return await original(runtime, binding, engine, spec)
    monkeypatch.setattr(workflow_restore_execution, "validate_engine", substitute)
    handle = await c.restore()
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.generation == 2 and current.outcome != "succeeded"
    assert c.model.counts["BUILD"] == c.counts["BUILD"] and c.model.counts["REVIEW"] == c.counts["REVIEW"]
    assert c.pool.spent == 27


@pytest.mark.parametrize("fault", ["denied", "aba", "rewound"])
async def test_permission_or_spend_change_after_claim_cannot_start_a_child(bound_restore, monkeypatch, fault):
    c = bound_restore
    original = workflow_resume.claim_dispatch
    async def changed(*args, **kwargs):
        result = await original(*args, **kwargs)
        if fault == "denied":
            c.track["allow"] = False
        elif fault == "aba":
            adapter = c.registry._entry("host", "1.0.0")[1]
            c.registry.unregister("host", "1.0.0")
            c.registry.register("host", "1.0.0", adapter)
        else:
            c.pool._spent -= 1
        return result
    monkeypatch.setattr(workflow_resume, "claim_dispatch", changed)
    handle = await c.restore()
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.generation == 2 and current.outcome != "succeeded"
    assert c.model.counts["BUILD"] == c.counts["BUILD"] and c.model.counts["REVIEW"] == c.counts["REVIEW"]


async def test_preview_does_not_call_resolver_even_when_registry_is_revoked(bound_restore):
    c = bound_restore
    before = copy.deepcopy(c.track)
    c.registry.unregister("host", "1.0.0")
    plan = await c.manager.get_workflow_restore_plan(c.actor, c.initial.job_id,
        generation=1, node_id="BUILD", message="NUDGED-43")
    assert plan["profile"]["resource_binding"] == c.binding.to_dict() and c.track == before
    assert (await c.manager.get_job(c.actor, c.initial.job_id)).generation == 1


async def test_bound_profile_without_prepared_resources_cannot_admit(bound_restore, monkeypatch):
    c = bound_restore
    from power_loop.runtime import workflow_restore_resources
    async def absent(*args):
        return None
    monkeypatch.setattr(workflow_restore_resources, "prepare", absent)
    assert await c.restore() is None
    assert (await c.manager.get_job(c.actor, c.initial.job_id)).generation == 1
    assert c.model.counts["BUILD"] == c.counts["BUILD"]


async def test_denial_after_admission_keeps_cancellation_responsibility_and_starts_no_child(bound_restore, monkeypatch):
    c = bound_restore
    original = workflow_resume.admit
    async def denied(*args, **kwargs):
        result = await original(*args, **kwargs)
        c.track["allow"] = False
        return result
    monkeypatch.setattr(workflow_resume, "admit", denied)
    assert await c.restore() is None
    current = await c.manager.get_job(c.actor, c.initial.job_id)
    assert current.generation == 2 and current.cancel_requested_at is not None
    assert current.resource_state != "stopped"  # no invented physical exit
    assert c.model.counts["BUILD"] == c.counts["BUILD"] and c.model.counts["REVIEW"] == c.counts["REVIEW"]
    assert c.pool.spent == 27


async def test_original_spec_budget_charges_only_new_generation_usage(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    # This checks restored usage accounting, not the two-second fixture lease.
    c.runtime.lease_ms = 30000
    c.state["spec"]["budget"] = {"max_tokens": 50, "stop_at_remaining_pct": 0}
    initial = await finished(managed_workflow)
    pools = []
    original = workflow_restore_execution.validate_engine
    async def observe(runtime, binding, engine, spec):
        await original(runtime, binding, engine, spec)
        pools.append((engine, engine._budget))
    monkeypatch.setattr(workflow_restore_execution, "validate_engine", observe)
    async def restore():
        handle = await resume_detached(c.loop, c.parent, initial.run_id,
            continuation=("BUILD", "NUDGED-43", 3))
        c.state["handles"].append(handle)
        return {"operation_id": handle.operation_id}
    await c.issue(handler=restore)
    assert len(c.state["handles"]) == 2
    await asyncio.wait_for(c.state["handles"][-1].task, 10)
    [ (engine, pool) ] = pools
    current = await c.manager.get_job(c.runtime.actor(c.parent), initial.job_id)
    assert current.outcome == "succeeded" and current.result["usage"]["total_tokens"] == 20, current.result
    assert pool.spent == 40 and engine._total_usage()["total_tokens"] == 20


async def test_recreated_pool_with_same_id_and_workflow_minimum_cannot_erase_other_scope_spend(bound_restore):
    c = bound_restore
    c.pool.commit({"total_tokens": 13})
    replacement = SharedBudget(c.pool.max_tokens)
    replacement.commit({"total_tokens": 27})  # sufficient W minimum, but loses the other 13
    original_adapter = c.registry._entry("host", "1.0.0")[1]
    original_resources = await original_adapter.resolve(c.binding)
    async def resolve(*args):
        from dataclasses import replace
        return replace(original_resources, budget=replacement)
    from power_loop.runtime.workflow_resources import WorkflowResourceAdapter
    c.registry.unregister("host", "1.0.0")
    c.registry.register("host", "1.0.0", WorkflowResourceAdapter(resolve, resolve, original_adapter.authorize))
    assert await c.restore() is None
    assert (await c.manager.get_job(c.actor, c.initial.job_id)).generation == 1
    assert c.pool.spent == 40 and replacement.spent == 27
    assert c.model.counts["BUILD"] == c.counts["BUILD"]


async def test_create_dispatch_cannot_switch_to_new_pool_with_same_binding(managed_workflow):
    from dataclasses import replace

    from power_loop.runtime.workflow_resources import WorkflowResourceAdapter

    from .test_workflow_task_runtime_192 import settle
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    binding, pool, registry, track = install(managed_workflow)
    adapter = registry._entry("host", "1.0.0")[1]
    replacement = SharedBudget(pool.max_tokens)
    replacement.commit({"total_tokens": pool.spent})
    async def resolve(*args):
        resources = await adapter.resolve(*args)
        return replace(resources, budget=replacement) if track["resolve"] == 2 else resources
    registry.unregister("host", "1.0.0")
    registry.register("host", "1.0.0", WorkflowResourceAdapter(adapter.prepare, resolve, adapter.authorize))
    await loop.send("launch", sid)
    await settle(managed_workflow)
    [handle] = state["handles"]
    current = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert current.outcome == "failed" and "budget_substituted" in str(current.result)
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    assert pool.spent == replacement.spent == 7
