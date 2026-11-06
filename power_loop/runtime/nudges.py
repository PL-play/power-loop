"""Atomic active-subagent steering; no terminal-task replay or platform imports."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

from . import control_requests as requests
from .invocation import current_invocation
from .jobs import JobError, JobLease, _hash, _operation, verify_stored
from .message_ref import MessageRef

log = logging.getLogger(__name__)


TARGET_BARRIERS = frozenset({
    "stale_execution", "cancel_requested", "job_intake_closed", "execution_lease_expired",
    "execution_not_running", "unresolved_prior_attempt", "stale_parent_execution",
    "owner_completion_deadline_exceeded", "owner_wait_deadline_exceeded", "executor_not_accepting",
    "node_input_closed", "nudge_target_unavailable",
})


def _intent(target, caller, ref, call_id, sid, message, node):
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if not isinstance(target, JobLease) or not isinstance(caller, JobLease):
        raise JobError("execution_lease_required")
    if not isinstance(ref, MessageRef) or not isinstance(call_id, str) or not call_id:
        raise JobError("persisted_tool_call_required")
    from .execution_membership import _lease
    _lease(asdict(target))
    _lease(asdict(caller))
    command = {"target": asdict(target), "session_id": sid, "message": message}
    if node is not None:
        command["node"] = json.loads(json.dumps(node))
    return {"mode": "steer", "job_id": target.job_id, "caller": asdict(caller),
        "call_ref": ref.to_dict(), "tool_call_id": call_id, "input": command}


async def _source(manager, tx, actor, request):
    from .managed_effects import _ref
    caller, ref = JobLease(**request["caller"]), _ref(request["call_ref"])
    source = await manager._fence(tx, actor, caller)
    if source["kind"] != "send" or source["session_id"] != actor.session_id:
        raise JobError("nudge_caller_mismatch")
    if source["execution_role"] not in ("main", "workflow_node"):
        raise JobError("role_forbids_delegation")
    creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (caller.job_id,))
    send_index = json.loads(creation["payload_json"]).get("request", {}).get("send_index") if creation else None
    if type(send_index) is not int or send_index < 1 or send_index != ref.send_index:
        raise JobError("nudge_caller_mismatch")
    await manager._validate_call(tx, actor, ref, request["tool_call_id"])
    for action in ("create", "steer"):
        key = manager._origin_key(actor, action, ref, request["tool_call_id"], None)
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=?", (key,)):
            raise JobError("operation_conflict")
    return source


async def _target(manager, tx, actor, request, source_role):
    command = request["input"]
    target = JobLease(**command["target"])
    job = await manager._fence(tx, actor, target)
    sid = command["session_id"]
    if "node" in command:
        from .node_inputs import descriptor, is_closed
        workflow = await manager._row(tx, job["parent_job_id"])
        if (job["kind"] != "workflow_node" or source_role != "main" or workflow is None
                or workflow["kind"] != "workflow" or workflow["created_by_session_id"] != actor.session_id):
            raise JobError("nudge_target_mismatch")
        from .workflow_controller import for_node
        await for_node(manager, tx, job)
        if _hash(command["node"]) != _hash(await descriptor(manager, tx, job, sid)):
            raise JobError("node_input_binding_mismatch")
        if await is_closed(manager, tx, job, sid):
            raise JobError("node_input_closed")
    else:
        if job["kind"] != "subagent" or job["created_by_session_id"] != actor.session_id or job["session_id"] != sid:
            raise JobError("nudge_target_mismatch")
        session = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
        metadata = json.loads(session["metadata_json"] or "{}") if session else {}
        if (session is None or session["parent_session_id"] != actor.session_id
                or metadata.get("execution_role") != "subagent" or not metadata.get("standalone_subagent")):
            raise JobError("nudge_target_unavailable")
    return job


async def _live_target(manager, tx, actor, request):
    """Only mutable target-tree barriers remain after source acceptance.

    The accepted generation freezes its original route and binding. In
    particular, verifying a restored workflow's original driver may inspect a
    different tree and belongs in the preceding read-only preparation.
    """
    command = request["input"]
    job = await manager._fence(tx, actor, JobLease(**command["target"]))
    if "node" in command:
        from .node_inputs import is_closed
        if await is_closed(manager, tx, job, command["session_id"]):
            raise JobError("node_input_closed")
    elif await tx.fetchone(f"SELECT session_id FROM {manager.store.t.sessions} WHERE session_id=?", (command["session_id"],)) is None:
        raise JobError("nudge_target_unavailable")
    return job


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None):
    from .managed_effects import _ref
    command = request["input"]
    expected = _intent(JobLease(**command["target"]), JobLease(**request["caller"]),
        _ref(request["call_ref"]), request["tool_call_id"], command["session_id"], command["message"], command.get("node"))
    if expected != request:
        raise JobError("nudge_request_corrupt")
    accepted = await requests.stored(manager, reader, actor, event, request, action="nudge", completion=completion)
    if (accepted.data["target"]["generation"] != command["target"]["generation"]
            or accepted.data["source"]["execution_role"] not in ("main", "workflow_node")):
        raise JobError("nudge_request_corrupt")
    return accepted


def _receipt(manager, event, request, request_ref):
    from .result_revisions import result_id
    command = request["input"]
    try:
        value = json.loads(event["payload_json"])
        required = {"job_id", "session_id", "run_id", "mode", "status", "request", "request_ref"}
        if command["target"]["generation"] > 1:
            required.update(("generation", "result_event_id"))
        if (event["event_type"] != "operation" or event["action"] != "nudge" or event["job_id"] != request["job_id"]
                or event["caller_session_id"] != request["call_ref"]["session_id"] or event["tool_call_id"] != request["tool_call_id"]
                or json.loads(event["call_ref_json"]) != request["call_ref"]
                or event["request_hash"] != _hash([command, request["call_ref"]]) or set(value) != required
                or value["request"] != command or value["request_ref"] != request_ref or value["mode"] != "steer"
                or value["status"] != "accepted" or value["job_id"] != request["job_id"]
                or value["session_id"] != command["session_id"]):
            raise JobError("nudge_receipt_mismatch")
        if command["target"]["generation"] > 1 and (value["generation"] != command["target"]["generation"]
                or value["result_event_id"] != result_id(request["job_id"], command["target"]["generation"])):
            raise JobError("nudge_receipt_mismatch")
        if "node" in command and value["run_id"] != command["node"]["workflow_run_id"]:
            raise JobError("nudge_receipt_mismatch")
        return _operation(event)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("nudge_receipt_mismatch") from exc


async def _apply_nudge_request(manager, actor, event, data):
    from .managed_effects import _ref
    from .result_revisions import delivery_id, result_id

    request, command = data["request"], data["request"]["input"]
    target = JobLease(**command["target"])
    ref = _ref(request["call_ref"])
    key = manager._origin_key(actor, "nudge", ref, request["tool_call_id"], None)
    async with manager.tree_write(target.job_id) as tx:
        old = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE operation_key=? OR event_id=?",
            (key, requests.rejection_id(event["event_id"])))
        if old:
            await manager._authorize(tx, actor, old[0]["job_id"])
            return requests.read_outcome(manager, requests.AcceptedRequest(event, data, {}), old[0]), old[0]
        row = await manager._authorize(tx, actor, target.job_id)
        try:
            job = await _live_target(manager, tx, actor, request)
        except JobError as exc:
            if exc.code not in TARGET_BARRIERS:
                raise
            return None, await requests.reject(manager, tx, actor, event, data, row, exc.code)
        if job["root_job_id"] != data["target"]["root_job_id"]:
            raise JobError("nudge_target_mismatch")
        run_id = command["node"]["workflow_run_id"] if "node" in command else (await delivery_id(manager, tx, job))[9:]
        payload = {"job_id": target.job_id, "session_id": command["session_id"], "run_id": run_id,
            "mode": "steer", "status": "accepted", "request": command, "request_ref": requests.reference(event)}
        if target.generation > 1:
            payload.update(generation=target.generation, result_event_id=result_id(target.job_id, target.generation))
        now = manager.now()
        if await tx.write_job(target.job_id, f"UPDATE {manager.jobs} SET version=version+1,updated_at=? "
                "WHERE job_id=? AND root_job_id=? AND version=? AND generation=? AND executor_id=? AND boot_id=? "
                "AND intake='open' AND cancel_requested_at IS NULL",
                (now, target.job_id, data["target"]["root_job_id"], job["version"], target.generation,
                 target.executor_id, target.boot_id)) != 1:
            raise JobError("nudge_write_mismatch")
        operation = await manager._insert_operation(tx, job_id=target.job_id, version=job["version"]+1,
            actor=actor, action="nudge", key=key, fingerprint=_hash([command, ref.to_dict()]),
            call_ref=ref, tool_call_id=request["tool_call_id"], payload=payload)
        await tx.update_event(operation.job_id, f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?",
            (operation.operation_id,), notify=True)
        cols = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
        await manager.store.signal_changes(tx, command["session_id"], "inbox")
        if not await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, cols),
                (command["session_id"], "nudge:"+operation.operation_id, "relay", "steer", "pending", command["message"], now)):
            raise JobError("nudge_inbox_conflict")
        outcome = {"event_id": operation.operation_id, "job_id": target.job_id, "event_type": "operation",
            "job_version": job["version"]+1, "request_hash": _hash([command, ref.to_dict()]), "payload_json": json.dumps(payload)}
    return operation, outcome


async def _finish(manager, actor, accepted, outcome):
    requests.remember_event(manager, accepted.event)
    if outcome is None:
        operation, outcome = await _apply_nudge_request(manager, actor, accepted.event, accepted.data)
    else:
        operation = requests.read_outcome(manager, accepted, outcome)
    await requests.complete(manager, actor, accepted, outcome)
    if operation is None:
        raise JobError(json.loads(outcome["payload_json"])["error"])
    return operation


async def accept(manager, actor, target, *, caller, call_ref, tool_call_id, session_id, message, node=None):
    request = _intent(target, caller, call_ref, tool_call_id, session_id, message, node)
    async with manager._read() as reader:
        accepted, outcome = await requests.discover(manager, reader, actor, request, action="nudge")
        if accepted is None:
            source = await manager._authorize(reader, actor, caller.job_id)
            manager._check_proof(source, caller)
            if source["kind"] != "send" or source["session_id"] != actor.session_id:
                raise JobError("nudge_caller_mismatch")
            if source["execution_role"] not in ("main", "workflow_node"):
                raise JobError("role_forbids_delegation")
            job = await _target(manager, reader, actor, request, source["execution_role"])
            frozen = {key: job[key] for key in ("job_id", "root_job_id", "generation")}
    replayed = accepted is not None
    if accepted is None:
        async def authorize(tx):
            return await _source(manager, tx, actor, request)
        accepted, replayed = await requests.accept(manager, actor, request, frozen, action="nudge", authorize=authorize)
    return await _finish(manager, actor, accepted, outcome), replayed


@verify_stored
async def reconcile_request(manager, actor, request_id):
    from .managed_completion import _host_only
    _host_only()
    async with manager._read() as reader:
        event = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='task_control_request'", (request_id,))
        if event is None:
            raise JobError("nudge_request_original_missing")
        request = json.loads(event["payload_json"])["request"]
        accepted, outcome = await requests.discover(manager, reader, actor, request, action="nudge")
    return await _finish(manager, actor, accepted, outcome)



@verify_stored
async def recover_call(manager, actor, ref, call_id, *, session_id, message, node_key=None, expected_job=None):
    """Finish an existing permission before an adapter considers a fresh resume.

    No permission is manufactured for a missing source event. This also closes
    the source receipt after an uncertain target/source commit acknowledgement.
    """
    key = manager._origin_key(actor, "nudge", ref, call_id, None)
    async with manager._read() as reader:
        event = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='task_control_request'",
            (requests.identity("nudge", key),))
        if event is None:
            return None
        request = json.loads(event["payload_json"])["request"]
        command = request["input"]
        node = command.get("node")
        if (command["session_id"] != session_id or command["message"] != message
                or node_key is None and node is not None
                or node_key is not None and (node is None or (node["workflow_run_id"], node["node_id"]) != node_key)
                or expected_job is not None and (request["job_id"], command["target"]["generation"]) != expected_job):
            raise JobError("operation_conflict")
        accepted, outcome = await requests.discover(manager, reader, actor, request, action="nudge")
    return await _finish(manager, actor, accepted, outcome)


def receipt(operation, replayed):
    return {key: operation.result[key] for key in ("job_id", "session_id", "run_id", "mode", "status")} | {
        "operation_id": operation.operation_id, "replayed": replayed} | {
            key: operation.result[key] for key in ("generation", "result_event_id") if key in operation.result}


async def steer(loop, session_id, message, *, on_complete=None, finalize=None, admission=None):
    """Only the original local executor may receive a newly admitted nudge."""
    from . import delegation
    from .task_runtime import _SEND

    runtime = loop.task_runtime
    execution = _SEND.get()
    parent = execution.actor.session_id if execution is not None else None
    if execution is None or execution.runtime is not runtime:
        raise JobError("task_send_required")
    assistant_seq, call_id = current_invocation(parent)
    if assistant_seq is None or not call_id:
        raise JobError("persisted_tool_call_required")
    store = runtime.manager.store
    call_ref = await store.message_ref(parent, assistant_seq)

    async def old_operation():
        operation = await runtime.manager.operation_for_call(execution.actor, call_ref, call_id)
        if operation is not None and operation.action != "nudge":
            raise JobError("operation_conflict")
        return operation

    async def replay(operation):
        request = operation.result["request"]
        if request["session_id"] != session_id or request["message"] != message or operation.call_ref != call_ref.to_dict():
            raise JobError("operation_conflict")
        if operation.result["mode"] == "steer":
            recovered = await recover_call(runtime.manager, execution.actor, call_ref, call_id,
                session_id=session_id, message=message)
            if recovered is None:
                raise JobError("nudge_request_original_missing")
            operation = recovered
        elif operation.result["mode"] == "continue":
            from .subagent_admission import recover_call as recover_resume
            recovered = await recover_resume(runtime.manager, execution.actor, call_ref, call_id,
                session_id=session_id, message=message)
            if recovered is None:
                raise JobError("resume_request_original_missing")
            operation = recovered
        await notify(operation)
        return receipt(operation, True)

    async def notify(operation):
        try:
            row = await runtime.manager.db.fetchone(f"SELECT status FROM {store.t.follow_up_queue} "
                "WHERE session_id=? AND item_id=?", (session_id, "nudge:" + operation.operation_id))
            if row is not None and row["status"] == "pending":
                loop.steer_event(session_id).set()
        except Exception:
            log.exception("nudge accepted; local signal failed: %s", operation.operation_id)

    async with delegation._gate(store, session_id):
        old = await old_operation()
        if old is not None:
            return await replay(old)
        running = delegation._RUNNING.get((id(store), session_id))
        if (running is None or running.task_binding is None or running.task_binding.runtime is not runtime or
                running.task is None or running.task.done()):
            recovered = await recover_call(runtime.manager, execution.actor, call_ref, call_id,
                session_id=session_id, message=message)
            if recovered is not None:
                await notify(recovered)
                return receipt(recovered, True)
            from .subagent_admission import recover_call as recover_resume
            recovered = await recover_resume(runtime.manager, execution.actor, call_ref, call_id,
                session_id=session_id, message=message)
            if recovered is not None:
                await notify(recovered)
                return receipt(recovered, True)
            from .subagent_resume import resume_locked
            return await resume_locked(loop, execution, call_ref, call_id, session_id, message,
                                       on_complete=on_complete, finalize=finalize, admission=admission)
        if running.parent != parent:
            raise JobError("nudge_target_mismatch")
        args = {"caller": execution.lease, "call_ref": call_ref, "tool_call_id": call_id,
                "session_id": session_id, "message": message}
        try:
            operation, repeated = await runtime.manager.accept_nudge(execution.actor, running.task_binding.lease, **args)
        except Exception:
            recovered = await recover_call(runtime.manager, execution.actor, call_ref, call_id,
                session_id=session_id, message=message)
            if recovered is None:
                raise
            await notify(recovered)
            return receipt(recovered, True)
        # The event is an optimization; the original executor also drains the
        # durable inbox at boundaries and before committing its canonical result.
        await notify(operation)
        return receipt(operation, repeated)
