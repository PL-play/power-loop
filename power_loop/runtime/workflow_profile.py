"""Frozen creation capabilities; a profile is evidence, never dispatch permission."""
from __future__ import annotations

import json
import math

from .jobs import JobError, _hash, _json


def _external(value):
    cls = type(value)
    return {"kind": "external", "class": cls.__module__ + "." + cls.__qualname__}


def _budget(spec, budget):
    from power_loop.workflow.result import SharedBudget
    if budget is None:
        if spec.budget is None:
            return {"kind": "unbounded"}
        return {"kind": "spec_soft", "max_tokens": spec.budget.max_tokens,
            "stop_at_remaining_pct": spec.budget.stop_at_remaining_pct,
            "continuation_policy": "logical_run_cumulative"}
    if type(budget) is not SharedBudget:
        return _external(budget)
    # Snapshot the actual soft pool in one lock, including pre-existing spend.
    # This is not a portable reservation or permission to reconstruct the pool.
    with budget._lock:
        return {"kind": "runtime_soft", "max_tokens": budget.max_tokens,
            "floor_tokens": budget._floor, "spent_at_admission": budget._spent,
            "continuation_policy": "host_adapter_required"}


def capture(spec, *, executor=None, budget=None, file_io=None, completion_callback=False, resource_binding=None, allowed_tools=None):
    from power_loop.workflow.engine import InProcessExecutor
    if type(completion_callback) is not bool:
        raise JobError("workflow_profile_mismatch")
    data = {"version": 1, "protocol": "task-workflow-profile-v1",
        "executor": {"kind": "native_in_process"} if executor is None or type(executor) is InProcessExecutor else _external(executor),
        "budget": _budget(spec, budget),
        "file_io": {"kind": "none"} if file_io is None else _external(file_io),
        "notification": {"kind": "opaque_callback", "settlement": "library_exit_only"} if completion_callback else {"kind": "durable_inbox"}}
    if resource_binding is not None:
        from .workflow_resources import WorkflowResourceBinding
        if type(resource_binding) is not WorkflowResourceBinding:
            raise JobError("workflow_profile_mismatch")
        data.update(version=2, protocol="task-workflow-profile-v2", resource_binding=resource_binding.to_dict())
    return validate(data, spec=spec, completion_callback=completion_callback, allowed_tools=allowed_tools)


def validate(data, *, spec, completion_callback, allowed_tools=None):
    """Strict types and complete shape; absent historical fields stay unknown."""
    if data is None:
        raise JobError("workflow_profile_unknown")
    try:
        if isinstance(data, dict) and type(data.get("version")) is int and data["version"] == 2:
            from .workflow_resources import WorkflowResourceBinding, execution_contract
            if (set(data) != {"version", "protocol", "executor", "budget", "file_io", "notification", "resource_binding"} or
                    data["protocol"] != "task-workflow-profile-v2"):
                raise ValueError("incomplete resource profile")
            binding = WorkflowResourceBinding.from_dict(data["resource_binding"])
            base = {key: value for key, value in data.items() if key != "resource_binding"}
            base.update(version=1, protocol="task-workflow-profile-v1")
            base = validate(base, spec=spec, completion_callback=completion_callback)
            if (binding.contract_sha256 != execution_contract(spec, allowed_tools) or
                    (binding.workspace_id is not None) != (base["file_io"]["kind"] != "none") or
                    (binding.budget_scope_id is not None) != (base["budget"]["kind"] in ("runtime_soft", "external")) or
                    (binding.notification_mode == "host_receipt") != completion_callback):
                raise ValueError("resource capability mismatch")
            return json.loads(_json(data))
        if (not isinstance(data, dict) or set(data) != {"version", "protocol", "executor", "budget", "file_io", "notification"} or
                type(data["version"]) is not int or data["version"] != 1 or data["protocol"] != "task-workflow-profile-v1" or
                type(completion_callback) is not bool):
            raise ValueError("incomplete profile")
        for key, native in (("executor", "native_in_process"), ("file_io", "none")):
            value = data[key]
            if value != {"kind": native}:
                if (not isinstance(value, dict) or set(value) != {"kind", "class"} or value["kind"] != "external" or
                        not isinstance(value["class"], str) or not value["class"].strip()):
                    raise ValueError("invalid adapter")
        budget = data["budget"]
        if not isinstance(budget, dict):
            raise ValueError("invalid budget")
        kind = budget.get("kind")
        if kind in ("unbounded", "spec_soft"):
            if _hash(budget) != _hash(_budget(spec, None)):
                raise ValueError("spec budget differs")
        elif kind == "runtime_soft":
            if (set(budget) != {"kind", "max_tokens", "floor_tokens", "spent_at_admission", "continuation_policy"} or
                    type(budget["max_tokens"]) is not int or budget["max_tokens"] < 0 or
                    type(budget["spent_at_admission"]) is not int or budget["spent_at_admission"] < 0 or
                    type(budget["floor_tokens"]) not in (int, float) or not math.isfinite(budget["floor_tokens"]) or
                    not 0 <= budget["floor_tokens"] <= budget["max_tokens"] or
                    budget["continuation_policy"] != "host_adapter_required"):
                raise ValueError("invalid soft pool")
        elif kind == "external":
            if set(budget) != {"kind", "class"} or not isinstance(budget["class"], str) or not budget["class"].strip():
                raise ValueError("invalid host budget")
        else:
            raise ValueError("unknown budget")
        expected = {"kind": "opaque_callback", "settlement": "library_exit_only"} if completion_callback else {"kind": "durable_inbox"}
        if _hash(data["notification"]) != _hash(expected):
            raise ValueError("callback contract differs")
        return json.loads(_json(data))
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise JobError("workflow_profile_mismatch") from exc


def require_native(data):
    """First restore adapter eligibility, not restored execution authority.

    Call only with a validated profile. Host objects/callbacks need explicit
    adapters; class names and callback return cannot prove portable effects.
    """
    if data.get("version") == 2:
        raise JobError("workflow_profile_adapter_required:resource_binding")
    for key, allowed in (("executor", {"native_in_process"}), ("budget", {"unbounded", "spec_soft"}),
                         ("file_io", {"none"}), ("notification", {"durable_inbox"})):
        if data[key]["kind"] not in allowed:
            raise JobError("workflow_profile_adapter_required:" + key)
    return data


def creation_request(spec, allowed_tools, *, executor=None, budget=None, file_io=None, completion_callback=False, resource_binding=None):
    # A detached task must never reread a mutated Workflow after admission awaits.
    return json.loads(_json({"spec": spec.to_dict(), "allowed_tools": sorted(allowed_tools) if allowed_tools is not None else None,
        "completion_callback": completion_callback,
        "execution_profile": capture(spec, executor=executor, budget=budget, file_io=file_io,
            completion_callback=completion_callback, resource_binding=resource_binding, allowed_tools=allowed_tools)}))


async def read(manager, reader, row):
    """Validate original operation fingerprint, including the complete profile."""
    from power_loop.workflow.spec import WorkflowSpec, WorkflowSpecError

    from .workflow_controller import creation
    if row["kind"] != "workflow" or row["execution_role"] != "workflow":
        raise JobError("workflow_profile_target_mismatch")
    operation_id, _ = await creation(manager, reader, row)
    event = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation_id,))
    try:
        payload = json.loads(event["payload_json"])
        request = payload.get("request", {})
        call_ref = json.loads(event["call_ref_json"]) if event["call_ref_json"] else None
        fingerprint = _hash(["workflow", row["title"], None, row["parent_job_id"], True, request, call_ref])
        if event["request_hash"] != fingerprint:
            raise JobError("workflow_profile_creation_mismatch")
        spec = WorkflowSpec.from_json(request["spec"])
        data = validate(request.get("execution_profile"), spec=spec,
            completion_callback=request.get("completion_callback"), allowed_tools=request["allowed_tools"])
        if data["version"] == 2:
            from .workflow_resources import WorkflowResourceBinding, WorkflowResourceScope, execution_contract
            WorkflowResourceBinding.from_dict(data["resource_binding"]).require(
                WorkflowResourceScope(row["tenant_id"], row["principal_id"], row["created_by_session_id"]),
                execution_contract(spec, request["allowed_tools"]))
        return data
    except (TypeError, ValueError, KeyError, WorkflowSpecError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_profile_creation_mismatch") from exc


async def get(manager, actor, job_id):
    async with manager._write() as tx:
        return await read(manager, tx, await manager._authorize(tx, actor, job_id))
