"""Pure preparation checks; these tests do not claim native restore admission."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace

import pytest

from power_loop.runtime import workflow_restore_resources as ports
from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.workflow_profile import capture
from power_loop.runtime.workflow_resources import (
    WorkflowResourceAdapter,
    WorkflowResourceBinding,
    WorkflowResourceRegistry,
    WorkflowResources,
    execution_contract,
)
from power_loop.workflow import InProcessExecutor, SharedBudget, WorkflowSpec

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def host():
    spec = WorkflowSpec.from_json(
        {
            "name": "restore",
            "root": {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "work", "tools": []}},
        }
    )
    registry = WorkflowResourceRegistry()
    runtime = TaskRuntime(
        object(),
        tenant_id="tenant",
        principal_id="host",
        executor_id="exec",
        boot_id="boot",
        workflow_resources=registry,
    )
    actor = runtime.actor("original-main")
    pool = SharedBudget(1000)
    pool.commit({"total_tokens": 7})
    binding = WorkflowResourceBinding(
        "host",
        "1.0.0",
        "original-resource",
        1,
        actor.tenant_id,
        actor.principal_id,
        actor.session_id,
        "a" * 64,
        execution_contract(spec, []),
        budget_scope_id="original-pool",
    )
    resources = WorkflowResources(binding, InProcessExecutor(), budget=pool)
    state = {"allow": True, "calls": 0, "wait": False}

    async def prepare(*args):
        return resources

    async def resolve(*args):
        return resources

    async def authorize(*args):
        state["calls"] += 1
        if state["wait"]:
            await asyncio.Event().wait()
        return state["allow"]

    adapter = WorkflowResourceAdapter(prepare, resolve, authorize)
    registry.register("host", "1.0.0", adapter)
    registry.retain_soft_pool(binding, pool)
    profile = capture(spec, executor=resources.executor, budget=pool, resource_binding=binding, allowed_tools=[])
    plan = {"spec": spec.to_dict(), "allowed_tools": [], "profile": profile}
    return runtime, actor, pool, resources, state, adapter, plan


async def test_preparation_preserves_original_pool_and_accepts_additional_scope_spend():
    runtime, actor, pool, resources, state, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    assert prepared.resources is resources and state["calls"] == 1
    pool.commit({"total_tokens": 25})  # includes 20 by this W and 5 by another scope
    evidence = ports.confirm(prepared, runtime, actor, plan["profile"], prior_usage={"total_tokens": 20})
    assert evidence["budget_spent"] == 32 and pool.spent == 32
    assert evidence["binding_hash"] == resources.binding.digest
    assert (
        await ports.reauthorize(prepared, runtime, actor, plan["profile"], prior_usage={"total_tokens": 20})
    ) == evidence
    assert state["calls"] == 2
    pool.commit({"total_tokens": 3})
    assert ports.confirm(prepared, runtime, actor, plan["profile"], minimum_spent=32)["budget_spent"] == 35


@pytest.mark.parametrize(
    "fault",
    ["scope", "registry", "aba", "revoke", "profile", "prepared_profile", "capacity", "floor", "spent", "ledger"],
)
async def test_preparation_cannot_be_substituted_or_rewind_spend(fault):
    runtime, actor, pool, _, _, adapter, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    profile = copy.deepcopy(plan["profile"])
    options = {}
    if fault == "scope":
        actor = replace(actor, session_id="other")
    elif fault == "registry":
        runtime.workflow_resources = WorkflowResourceRegistry()
    elif fault == "aba":
        runtime.workflow_resources.unregister("host", "1.0.0")
        runtime.workflow_resources.register("host", "1.0.0", adapter)
    elif fault == "revoke":
        runtime.workflow_resources.unregister("host", "1.0.0")
    elif fault == "profile":
        profile["resource_binding"]["resource_id"] = "other"
    elif fault == "prepared_profile":
        prepared.profile["resource_binding"]["resource_id"] = "other"
    elif fault == "capacity":
        pool.max_tokens = 2000
    elif fault == "floor":
        pool._floor = 500
    elif fault == "spent":
        pool._spent = 6
    else:
        options["prior_usage"] = {"total_tokens": 1}
    with pytest.raises(JobError):
        ports.confirm(prepared, runtime, actor, profile, **options)


async def test_admission_spend_proof_rejects_rewind_even_above_original_workflow_minimum():
    runtime, actor, pool, _, _, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    pool.commit({"total_tokens": 50})
    evidence = ports.confirm(prepared, runtime, actor, plan["profile"], prior_usage={"total_tokens": 20})
    pool._spent = 50  # still >= 7+20, but lower than observed 57
    with pytest.raises(JobError, match="budget_rewound"):
        ports.confirm(
            prepared,
            runtime,
            actor,
            plan["profile"],
            prior_usage={"total_tokens": 20},
            minimum_spent=evidence["budget_spent"],
        )


@pytest.mark.parametrize("value", [True, -1, 1.0, "1", 2**63])
async def test_ambiguous_usage_proof_is_not_converted_to_an_allowance(value):
    runtime, actor, _, _, _, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    with pytest.raises(JobError, match="usage_unknown"):
        ports.confirm(prepared, runtime, actor, plan["profile"], prior_usage={"total_tokens": value})


async def test_dispatch_reauthorization_can_deny_the_actual_objects():
    runtime, actor, _, _, state, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    state["allow"] = False
    with pytest.raises(JobError, match="access_denied"):
        await ports.reauthorize(prepared, runtime, actor, plan["profile"])
    assert state["calls"] == 2


async def test_reauthorization_is_bounded():
    runtime, actor, _, _, state, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    runtime.workflow_resources._timeout_s = 0.02
    state["wait"] = True
    with pytest.raises(JobError, match="timeout"):
        await ports.reauthorize(prepared, runtime, actor, plan["profile"])


async def test_original_profile_v1_has_no_implicit_host_preparation():
    runtime, actor, _, _, state, _, plan = host()
    plan["profile"] = capture(WorkflowSpec.from_json(plan["spec"]))
    assert await ports.prepare(runtime, actor, plan) is None
    assert state["calls"] == 0


@pytest.mark.parametrize("fault", ["missing", "version", "binding", "profile", "extra", "spent", "bool_spent"])
async def test_durable_preparation_evidence_requires_exact_original_profile_and_spend(fault):
    runtime, actor, _, _, _, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    evidence = ports.admission(prepared, runtime.manager, actor, plan["profile"], prior_usage={})
    if fault == "missing":
        evidence = None
    elif fault == "version":
        evidence["version"] = True
    elif fault == "binding":
        evidence["binding_hash"] = "b"*64
    elif fault == "profile":
        evidence["profile_hash"] = "b"*64
    elif fault == "extra":
        evidence["current_config"] = True
    elif fault == "spent":
        evidence["budget_spent"] = 6
    else:
        evidence["budget_spent"] = True
    with pytest.raises(JobError):
        ports.validate_evidence(plan["profile"], evidence)


async def test_admission_uses_exact_runtime_manager_and_observed_spend():
    runtime, actor, pool, _, _, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    with pytest.raises(JobError, match="resources_required"):
        ports.admission(prepared, object(), actor, plan["profile"], prior_usage={})
    with pytest.raises(JobError, match="resources_required"):
        ports.admission(None, runtime.manager, actor, plan["profile"], prior_usage={})
    pool.commit({"total_tokens": 10})
    evidence = ports.admission(prepared, runtime.manager, actor, plan["profile"], prior_usage={})
    pool._spent -= 1
    with pytest.raises(JobError, match="budget_rewound"):
        ports.admission(prepared, runtime.manager, actor, plan["profile"], prior_usage={}, previous=evidence)


async def test_native_profile_cannot_accept_host_preparation_or_evidence():
    runtime, actor, _, _, _, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    profile = capture(WorkflowSpec.from_json(plan["spec"]))
    assert ports.admission(None, runtime.manager, actor, profile, prior_usage={}) is None
    ports.validate_evidence(profile, None)
    with pytest.raises(JobError, match="binding_mismatch"):
        ports.admission(prepared, runtime.manager, actor, profile, prior_usage={})
    with pytest.raises(JobError, match="binding_mismatch"):
        ports.validate_evidence(profile, {})


async def test_bound_unbounded_profile_cannot_gain_a_runtime_pool():
    runtime, actor, _, resources, _, adapter, plan = host()
    binding = replace(resources.binding, budget_scope_id=None)
    resources = replace(resources, binding=binding, budget=None)
    async def resolved(*args):
        return resources
    runtime.workflow_resources.unregister("host", "1.0.0")
    runtime.workflow_resources.register("host", "1.0.0", WorkflowResourceAdapter(resolved, resolved, adapter.authorize))
    plan["profile"] = capture(WorkflowSpec.from_json(plan["spec"]), executor=resources.executor,
        resource_binding=binding, allowed_tools=[])
    prepared = await ports.prepare(runtime, actor, plan)
    evidence = ports.admission(prepared, runtime.manager, actor, plan["profile"], prior_usage={})
    assert evidence["budget_spent"] is None
    ports.validate_evidence(plan["profile"], evidence)
    with pytest.raises(JobError, match="binding_mismatch"):
        ports.validate_evidence(plan["profile"], {**evidence, "budget_spent": 0})
    with pytest.raises(JobError, match="binding_mismatch"):
        ports.confirm(prepared, runtime, actor, plan["profile"], minimum_spent=0)


async def test_cancelled_host_authorization_does_not_become_a_successful_preparation():
    runtime, actor, _, _, state, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    state["wait"] = True
    task = asyncio.create_task(ports.reauthorize(prepared, runtime, actor, plan["profile"]))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("fault", ["missing", "substitute"])
async def test_original_soft_pool_requires_actual_continuity_in_addition_to_saved_minimum(fault):
    runtime, actor, pool, resources, _, _, plan = host()
    if fault == "missing":
        runtime.workflow_resources._soft_pools.clear()
        with pytest.raises(JobError, match="requires_reconciliation"):
            await ports.prepare(runtime, actor, plan)
    else:
        replacement = SharedBudget(pool.max_tokens)
        replacement.commit({"total_tokens": pool.spent})
        with pytest.raises(JobError, match="budget_substituted"):
            runtime.workflow_resources.retain_soft_pool(resources.binding, replacement)
        with pytest.raises(JobError, match="budget_substituted"):
            runtime.workflow_resources.require_soft_pool(resources.binding, replacement)


@pytest.mark.parametrize("fault", ["binding", "pool"])
async def test_soft_pool_anchor_rejects_untyped_or_unscoped_resources(fault):
    runtime, _, pool, resources, _, _, _ = host()
    with pytest.raises(JobError):
        runtime.workflow_resources.retain_soft_pool(
            replace(resources.binding, budget_scope_id=None) if fault == "binding" else resources.binding,
            pool if fault == "binding" else object())


@pytest.mark.parametrize("fault", ["runtime", "actor", "absent", "foreign_preparation", "binding", "usage", "tool_string", "empty_tool", "non_text_tool"])
async def test_untrusted_preparation_inputs_cannot_authorize_original_resources(fault):
    runtime, actor, _, resources, _, _, plan = host()
    prepared = await ports.prepare(runtime, actor, plan)
    with pytest.raises(JobError):
        if fault == "runtime":
            await ports.prepare(object(), actor, plan)
        elif fault == "actor":
            await ports.prepare(runtime, replace(actor, tenant_id="foreign"), plan)
        elif fault == "absent":
            ports.confirm(None, runtime, actor, plan["profile"])
        elif fault == "foreign_preparation":
            ports.confirm(replace(prepared, runtime=object()), runtime, actor, plan["profile"])
        elif fault == "binding":
            different = replace(resources, binding=replace(resources.binding, resource_id="different"))
            ports.confirm(replace(prepared, resources=different), runtime, actor, plan["profile"])
        elif fault == "usage":
            ports.confirm(prepared, runtime, actor, plan["profile"], prior_usage=[])
        else:
            plan["allowed_tools"] = {"tool_string": "all", "empty_tool": [""], "non_text_tool": [7]}[fault]
            await ports.prepare(runtime, actor, plan)
