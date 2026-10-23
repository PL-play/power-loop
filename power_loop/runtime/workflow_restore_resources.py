"""Trusted resource preparation for original-workflow restoration.

No resolver or host authorization runs in a database transaction. The caller
must compare this preparation with the freshly read original profile during
admission and reauthorize the actual engine resources before dispatch.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field

from .jobs import JobActor, JobError, _hash
from .workflow_resources import (
    WorkflowResourceBinding,
    WorkflowResourceScope,
    _host_call,
    execution_contract,
    require_managed_creation,
)


@dataclass(frozen=True)
class ResourcePreparation:
    runtime: object = field(repr=False)
    registry: object = field(repr=False)
    registration: object = field(repr=False)
    resources: object = field(repr=False)
    scope: WorkflowResourceScope
    binding: WorkflowResourceBinding
    spec: object = field(repr=False)
    allowed_tools: tuple[str, ...] | None
    profile: dict = field(repr=False)
    profile_hash: str


def _scope(runtime, actor):
    from .task_runtime import TaskRuntime
    if type(runtime) is not TaskRuntime or type(actor) is not JobActor or runtime.actor(actor.session_id) != actor:
        raise JobError("workflow_resource_scope_mismatch")
    return WorkflowResourceScope(actor.tenant_id, actor.principal_id, actor.session_id)


def _tokens(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise JobError("workflow_restore_usage_unknown")
    return value


def confirm(prepared, runtime, actor, profile, *, prior_usage=None, minimum_spent=None):
    """Synchronous checks suitable inside admission's existing transaction.

    This evidence is not dispatch authority or durable shared-pool accounting.
    It cannot reconstruct a pool or establish the spend of other scopes.
    """
    from .workflow_profile import capture
    if type(prepared) is not ResourcePreparation or prepared.runtime is not runtime:
        raise JobError("workflow_restore_resources_required")
    scope = _scope(runtime, actor)
    if prepared.scope != scope or prepared.registry is not runtime.workflow_resources:
        raise JobError("workflow_resource_scope_mismatch")
    binding = prepared.binding
    if prepared.registry._entry(binding.adapter_key, binding.adapter_version) is not prepared.registration:
        raise JobError("workflow_resource_adapter_changed")
    if _hash(prepared.profile) != prepared.profile_hash or _hash(profile) != prepared.profile_hash:
        raise JobError("workflow_resource_binding_mismatch")
    binding.require(scope, execution_contract(prepared.spec, prepared.allowed_tools))
    resources = require_managed_creation(prepared.resources)
    if resources.binding != binding:
        raise JobError("workflow_resource_binding_mismatch")
    if resources.budget is not None:
        prepared.registry.require_soft_pool(binding, resources.budget)
    actual = capture(prepared.spec, executor=resources.executor, budget=resources.budget,
        file_io=resources.file_io, resource_binding=binding, allowed_tools=prepared.allowed_tools)
    spent = None
    if profile["budget"]["kind"] == "runtime_soft" and actual["budget"]["kind"] == "runtime_soft":
        if prior_usage is not None and not isinstance(prior_usage, dict):
            raise JobError("workflow_restore_usage_unknown")
        initial = _tokens(profile["budget"]["spent_at_admission"])
        spent = _tokens(actual["budget"]["spent_at_admission"])
        minimum = initial + _tokens((prior_usage or {}).get("total_tokens", 0))
        if minimum_spent is not None:
            minimum = max(minimum, _tokens(minimum_spent))
        if spent < minimum:
            raise JobError("workflow_resource_budget_rewound")
        actual["budget"]["spent_at_admission"] = initial
    elif minimum_spent is not None:
        raise JobError("workflow_resource_binding_mismatch")
    if _hash(actual) != prepared.profile_hash:
        raise JobError("workflow_resource_binding_mismatch")
    return {"version": 1, "profile_hash": prepared.profile_hash, "binding_hash": binding.digest,
        "budget_spent": spent}


def admission(prepared, manager, actor, profile, *, prior_usage, previous=None):
    """Check selected objects without running host code under the DB lock."""
    if profile["version"] == 1:
        if prepared is not None or previous is not None:
            raise JobError("workflow_resource_binding_mismatch")
        return None
    if type(prepared) is not ResourcePreparation or prepared.runtime.manager is not manager:
        raise JobError("workflow_restore_resources_required")
    minimum = None
    if previous is not None:
        validate_evidence(profile, previous)
        minimum = previous["budget_spent"]
    return confirm(prepared, prepared.runtime, actor, profile,
        prior_usage=prior_usage, minimum_spent=minimum)


def validate_evidence(profile, evidence):
    if profile["version"] == 1:
        if evidence is not None:
            raise JobError("workflow_resource_binding_mismatch")
        return
    binding = WorkflowResourceBinding.from_dict(profile["resource_binding"])
    if (not isinstance(evidence, dict) or set(evidence) != {"version", "profile_hash", "binding_hash", "budget_spent"} or
            type(evidence["version"]) is not int or evidence["version"] != 1 or
            evidence["profile_hash"] != _hash(profile) or evidence["binding_hash"] != binding.digest):
        raise JobError("workflow_resource_binding_mismatch")
    if profile["budget"]["kind"] == "runtime_soft":
        if _tokens(evidence["budget_spent"]) < _tokens(profile["budget"]["spent_at_admission"]):
            raise JobError("workflow_resource_budget_rewound")
    elif evidence["budget_spent"] is not None:
        raise JobError("workflow_resource_binding_mismatch")


async def prepare(runtime, actor, plan):
    """Resolve a validated original plan, without issuing any execution lease."""
    from power_loop.workflow.spec import WorkflowSpec

    from .workflow_profile import require_native, validate
    scope = _scope(runtime, actor)
    spec = WorkflowSpec.from_json(copy.deepcopy(plan["spec"]))
    allowed = plan["allowed_tools"]
    if allowed is not None:
        if not isinstance(allowed, list) or any(type(name) is not str or not name for name in allowed):
            raise JobError("workflow_resource_binding_mismatch")
        allowed = tuple(sorted(allowed))
    profile = validate(copy.deepcopy(plan["profile"]), spec=spec, completion_callback=False, allowed_tools=allowed)
    if profile["version"] == 1:
        require_native(profile)
        return None
    binding = WorkflowResourceBinding.from_dict(profile["resource_binding"])
    binding.require(scope, execution_contract(spec, allowed))
    registry = runtime.workflow_resources
    if registry is None:
        raise JobError("workflow_resource_registry_required")
    registration = registry._entry(binding.adapter_key, binding.adapter_version)
    resources = await registry.resolve(binding, scope=scope, contract_sha256=binding.contract_sha256)
    result = ResourcePreparation(runtime, registry, registration, resources, scope, binding, spec,
        allowed, profile, _hash(profile))
    confirm(result, runtime, actor, profile)
    return result


async def reauthorize(prepared, runtime, actor, profile, *, prior_usage=None, minimum_spent=None):
    """Reauthorize the actual selected objects outside a transaction."""
    options = {"prior_usage": prior_usage, "minimum_spent": minimum_spent}
    confirm(prepared, runtime, actor, profile, **options)
    try:
        async with asyncio.timeout(prepared.registry._timeout_s):
            adapter = prepared.registration[1]
            if await _host_call(adapter.authorize, prepared.binding, prepared.resources, prepared.scope) is not True:
                raise JobError("workflow_resource_access_denied")
    except TimeoutError as exc:
        raise JobError("workflow_resource_timeout") from exc
    return confirm(prepared, runtime, actor, profile, **options)
