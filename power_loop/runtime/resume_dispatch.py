"""A retained caller grants one dispatch token before its target records it.

Recovery may complete these original facts. It never constructs a callback,
model request, executor, new generation or replacement dispatch token.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from . import control_requests as requests
from .jobs import JobError, JobLease, _hash, _json, verify_stored
from .managed_effects import _ref

ACTION = "dispatch"
TARGET_BARRIERS = frozenset({
    "stale_execution", "cancel_requested", "job_intake_closed", "execution_lease_expired",
    "execution_not_running", "unresolved_prior_attempt", "stale_parent_execution",
    "owner_completion_deadline_exceeded", "owner_wait_deadline_exceeded", "executor_not_accepting",
})


async def _operation(manager, reader, actor, operation_id, caller):
    op = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND action='nudge' AND event_type='operation'",
        (operation_id,))
    if op is None or op["caller_session_id"] != actor.session_id:
        raise JobError("resume_dispatch_mismatch")
    job = await manager._authorize(reader, actor, op["job_id"])
    payload = json.loads(op["payload_json"])
    if payload.get("mode") not in ("continue", "workflow_continue") or payload["request"].get("caller") != asdict(caller):
        raise JobError("resume_dispatch_mismatch")
    if payload["mode"] == "workflow_continue":
        from .workflow_resume import execution
        await execution(manager, reader, job, payload["generation"], expected_operation=operation_id)
    return op, job, payload["request"]


def _intent(op, caller, token, target):
    payload = json.loads(op["payload_json"])
    request = {"job_id": op["job_id"], "caller": asdict(caller), "call_ref": json.loads(op["call_ref_json"]),
        "tool_call_id": op["tool_call_id"], "operation_id": op["event_id"], "operation": requests.reference(op),
        "target": target, "token": token, "mode": payload["mode"]}
    if payload["mode"] == "workflow_continue":
        request["driver_sid"] = payload["request"]["driver_sid"]
    return request


def _evidence(request, request_ref):
    if request["mode"] == "workflow_continue":
        from .workflow_resume import dispatch_id
        return dispatch_id(request["operation_id"]), "workflow_dispatch", {
            "version": 1, "operation_id": request["operation_id"], "target": request["target"],
            "driver_sid": request["driver_sid"], "token": request["token"], "request_ref": request_ref}
    return "dispatch_"+request["operation_id"], "dispatch", {
        "token": request["token"], "operation_id": request["operation_id"], "request_ref": request_ref}


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None):
    op, _, original = await _operation(manager, reader, actor, request["operation_id"], JobLease(**request["caller"]))
    if _intent(op, JobLease(**request["caller"]), request["token"], original["target"]) != request:
        raise JobError("resume_dispatch_request_corrupt")
    accepted = await requests.stored(manager, reader, actor, event, request, action=ACTION, completion=completion)
    if (accepted.data["target"]["generation"] != request["target"]["generation"]
            or accepted.data["source"]["execution_role"] != "main"):
        raise JobError("resume_dispatch_request_corrupt")
    return accepted


def _receipt(manager, event, request, request_ref):
    event_id, kind, value = _evidence(request, request_ref)
    if (event["event_id"] != event_id or event["event_type"] != kind
            or event["job_id"] != request["job_id"] or event["delivery_status"] != "claimed"
            or event["request_hash"] != _hash(value) or json.loads(event["payload_json"]) != value
            or event["caller_session_id"] != request["call_ref"]["session_id"]
            or event["tool_call_id"] != request["tool_call_id"] or json.loads(event["call_ref_json"]) != request["call_ref"]):
        raise JobError("workflow_restore_dispatch_mismatch" if request["mode"] == "workflow_continue"
            else "resume_dispatch_receipt_mismatch")
    return True


async def _source(manager, tx, actor, request):
    caller, ref = JobLease(**request["caller"]), _ref(request["call_ref"])
    source = await manager._fence(tx, actor, caller)
    if source["kind"] != "send" or source["session_id"] != actor.session_id or source["execution_role"] != "main":
        raise JobError("resume_caller_mismatch")
    create = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (caller.job_id,))
    if json.loads(create["payload_json"])["request"]["send_index"] != ref.send_index:
        raise JobError("resume_caller_mismatch")
    # Admission already bound this immutable raw call. Historical request
    # verification checks it in the read-only view, outside this source writer.
    return source


async def _publish(manager, actor, accepted, outcome):
    if outcome is not None:
        return requests.read_outcome(manager, accepted, outcome), outcome
    request = accepted.data["request"]
    event_id, kind, value = _evidence(request, requests.reference(accepted.event))
    key = manager._origin_key(actor, ACTION, _ref(request["call_ref"]), request["tool_call_id"], None)
    async with manager.tree_write(accepted.data['target']['job_id']) as tx:
        old = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE event_id IN (?,?)",
            (event_id, requests.rejection_id(accepted.event["event_id"])))
        if old:
            return requests.read_outcome(manager, accepted, old[0]), old[0]
        row = await manager._authorize(tx, actor, request["job_id"])
        try:
            job = await manager._fence(tx, actor, JobLease(**request["target"]))
        except JobError as exc:
            if exc.code not in TARGET_BARRIERS:
                raise
            return None, await requests.reject(manager, tx, actor, accepted.event, accepted.data, row, exc.code)
        event = {"event_id": event_id, "job_id": job["job_id"], "event_type": kind, "job_version": job["version"],
            "operation_key": key, "caller_session_id": actor.session_id, "tool_call_id": request["tool_call_id"],
            "call_ref_json": _json(request["call_ref"]), "request_hash": _hash(value), "payload_json": _json(value),
            "delivery_status": "claimed", "created_at": manager.now()}
        if await tx.append_event_if_current(event, job) != 1:
            raise JobError("resume_dispatch_target_changed")
    return True, event


async def _finish(manager, actor, accepted, outcome):
    requests.remember_event(manager, accepted.event)
    result, outcome = await _publish(manager, actor, accepted, outcome)
    await requests.complete(manager, actor, accepted, outcome)
    if result is None:
        raise JobError(json.loads(outcome["payload_json"])["error"])
    return result


async def claim(manager, actor, operation_id, *, caller, token, resources=None):
    while True:
        async with manager._read() as reader:
            op, job, original = await _operation(manager, reader, actor, operation_id, caller)
            request = _intent(op, caller, token, original["target"])
            key = manager._origin_key(actor, ACTION, _ref(request["call_ref"]), request["tool_call_id"], None)
            permission = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?",
                (requests.identity(ACTION, key),))
            if permission is not None:
                stored = json.loads(permission["payload_json"])["request"]
                accepted, outcome = await requests.discover(manager, reader, actor, stored, action=ACTION)
                if stored["token"] != token:
                    return False
            else:
                accepted, outcome = None, None
                if job["kind"] == "workflow_node":
                    from .node_resume import validate_execution
                    await validate_execution(manager, reader, job, original["session_id"])
                if request["mode"] == "workflow_continue":
                    from .workflow_restore_resources import admission
                    admission(resources, manager, actor, original["plan"]["profile"],
                        prior_usage=original["prior_usage"], previous=original.get("resource_evidence"))
                frozen = {"job_id": job["job_id"], "root_job_id": job["root_job_id"],
                    "generation": original["target"]["generation"]}
        if accepted is not None:
            return await _finish(manager, actor, accepted, outcome)
        try:
            async def authorize(tx, request=request):
                return await _source(manager, tx, actor, request)
            accepted, _ = await requests.accept(manager, actor, request, frozen, action=ACTION, authorize=authorize)
        except JobError as exc:
            if exc.code != "operation_conflict":
                raise
            # Another token may win the source permission after discovery.
            # Re-read that original; only its exact token can finish the claim.
            continue
        return await _finish(manager, actor, accepted, None)


@verify_stored
async def recover_claim(manager, actor, operation_id, *, caller, token):
    """Finish this owner's retained permission after an uncertain commit ACK."""
    async with manager._read() as reader:
        op, _, original = await _operation(manager, reader, actor, operation_id, caller)
        request = _intent(op, caller, token, original["target"])
        key = manager._origin_key(actor, ACTION, _ref(request["call_ref"]), request["tool_call_id"], None)
        permission = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (requests.identity(ACTION, key),))
        if permission is None:
            return None
        stored = json.loads(permission["payload_json"])["request"]
        accepted, outcome = await requests.discover(manager, reader, actor, stored, action=ACTION)
        if stored["token"] != token:
            return False
    return await _finish(manager, actor, accepted, outcome)


@verify_stored
async def reconcile_request(manager, actor, request_id):
    from .managed_completion import _host_only
    _host_only()
    async with manager._read() as reader:
        event = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='task_control_request'", (request_id,))
        if event is None:
            raise JobError("resume_dispatch_request_original_missing")
        request = json.loads(event["payload_json"])["request"]
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
    return await _finish(manager, actor, accepted, outcome)
