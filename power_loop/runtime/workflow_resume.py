"""Atomic restored-execution admission. Dispatch claims do not run an engine.

The native adapter must retain owned work before admission, then claim dispatch
and bind the reserved driver before any node/model execution. Until that adapter
is wired, legacy resume/force and ordinary later-generation binds stay closed.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import JobError, JobLease, _hash
from .jobs import verify_stored as _verify_stored
from .result_revisions import execution_id


def reservation_id(manager, job_id, generation, node_id):
    return "wf_res_" + _hash([manager.store.store_id, job_id, generation, node_id])[:48]


def dispatch_id(operation_id):
    return "wf_disp_" + _hash(operation_id)[:48]


async def _source(manager, tx, actor, caller, ref):
    row = await manager._fence(tx, actor, caller)
    origin = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (caller.job_id,))
    index = json.loads(origin["payload_json"]).get("request", {}).get("send_index") if origin else None
    if (row["kind"] != "send" or row["session_id"] != actor.session_id or row["execution_role"] != "main" or
            type(index) is not int or index < 1 or index != ref.send_index):
        raise JobError("resume_caller_mismatch")
    return row


@_verify_stored
async def admit(manager, actor, job_id, *, caller, call_ref, tool_call_id, generation, node_id, message,
                executor_id, boot_id, plan_hash=None, lease_ms=30000, rounds=None, resources=None):
    """All new authority is reserved in one transaction, not physical execution."""
    if type(generation) is not int or not 1 <= generation < 2**63-1:
        raise JobError("invalid_result_generation")
    if not isinstance(node_id, str) or not node_id or len(node_id) > 255:
        raise JobError("workflow_restore_target_mismatch")
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if not isinstance(caller, JobLease):
        raise JobError("execution_lease_required")
    if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
        raise JobError("invalid_executor")
    if type(lease_ms) is not int or not 1 <= lease_ms <= 300000:
        raise JobError("invalid_lease_duration")
    if plan_hash is not None and (not isinstance(plan_hash, str) or len(plan_hash) != 64 or any(c not in "0123456789abcdef" for c in plan_hash)):
        raise JobError("workflow_restore_plan_mismatch")
    if rounds is not None and (type(rounds) is not int or rounds < 1):
        raise JobError("invalid_continuation_rounds")
    target = JobLease(job_id, executor_id, boot_id, generation+1)
    intent = {"caller": asdict(caller), "target": asdict(target), "previous_generation": generation,
        "node_id": node_id, "message": message, "expected_plan_hash": plan_hash, "lease_ms": lease_ms, "requested_rounds": rounds}
    from .workflow_admission import admit as admit_workflow
    return await admit_workflow(manager, actor, intent, call_ref, tool_call_id, resources=resources)


async def execution(manager, tx, row, generation=None, *, expected_operation=None):
    """Read immutable admission evidence. Historical reads do not grant a lease."""
    generation = row["generation"] if generation is None else generation
    if row["kind"] != "workflow" or row["execution_role"] != "workflow":
        raise JobError("workflow_restore_execution_mismatch")
    record = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (execution_id(row["job_id"], generation),))
    try:
        data = json.loads(record["payload_json"]) if record else {}
        if (record is None or record["event_type"] != "execution" or record["job_id"] != row["job_id"] or
                record["request_hash"] != _hash(data) or type(data.get("version")) is not int or data["version"] != 1 or
                record["delivery_status"] != "not_applicable" or data["target"]["generation"] != generation):
            raise JobError("workflow_restore_execution_mismatch")
        op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (data["operation_id"],))
        payload = json.loads(op["payload_json"]) if op else {}
        request = payload["request"]
        from .execution_membership import _lease
        _lease(request["caller"])
        _lease(request["target"])
        _lease(request["plan"]["workflow"])
        intent = {k: request[k] for k in ("caller", "target", "previous_generation", "node_id", "message", "expected_plan_hash", "lease_ms", "requested_rounds")}
        if (op["event_type"] != "operation" or op["action"] != "nudge" or op["job_id"] != row["job_id"] or
                (expected_operation is not None and op["event_id"] != expected_operation) or
                op["request_hash"] != _hash([intent, json.loads(op["call_ref_json"])]) or
                payload.get("mode") != "workflow_continue" or type(payload["generation"]) is not int or payload["generation"] != generation or
                type(request["previous_generation"]) is not int or request["previous_generation"]+1 != generation or
                request["plan"]["workflow"]["generation"] != request["previous_generation"] or
                request["target"]["job_id"] != row["job_id"] or record["job_version"] != op["job_version"] or
                _hash(request) != data["request_hash"] or _hash(request["plan"]) != request["plan_hash"] or
                request["driver_sid"] != data["driver_sid"] or _hash(request["target"]) != _hash(data["target"]) or
                payload["run_id"] != data["run_id"] or data["run_id"] != request["plan"]["run_id"] or
                payload["execution_id"] != op["event_id"][3:]):
            raise JobError("workflow_restore_execution_mismatch")
        from .workflow_restore_resources import validate_evidence
        validate_evidence(request["plan"]["profile"], request.get("resource_evidence"))
        return op, request
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_restore_execution_mismatch") from exc


async def claim_dispatch(manager, actor, operation_id, *, caller, token, resources=None):
    if not isinstance(token, str) or not 0 < len(token) <= 64 or not isinstance(caller, JobLease):
        raise JobError("invalid_dispatch_token")
    from .resume_dispatch import claim
    return await claim(manager, actor, operation_id, caller=caller, token=token, resources=resources)
