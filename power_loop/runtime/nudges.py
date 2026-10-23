"""Atomic active-subagent steering; no terminal-task replay or platform imports."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

from .invocation import current_invocation
from .jobs import JobError, JobLease, _hash

log = logging.getLogger(__name__)


async def accept(manager, actor, target, *, caller, call_ref, tool_call_id, session_id, message, node=None):
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if not isinstance(target, JobLease):
        raise JobError("execution_lease_required")
    request = {"target": asdict(target), "session_id": session_id, "message": message}
    if node is not None:
        request["node"] = json.loads(json.dumps(node))
    key = manager._origin_key(actor, "nudge", call_ref, tool_call_id, None)
    fingerprint = _hash([request, call_ref.to_dict()])
    async with manager._write() as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        if old:
            from .jobs import _operation
            await manager._authorize(tx, actor, old["job_id"])
            if old["request_hash"] != fingerprint:
                raise JobError("operation_conflict")
            return _operation(old), True
        await manager._validate_call(tx, actor, call_ref, tool_call_id)
        for action in ("create", "steer"):
            other_key = manager._origin_key(actor, action, call_ref, tool_call_id, None)
            if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=?", (other_key,)):
                raise JobError("operation_conflict")
        source = await manager._fence(tx, actor, caller)
        if source["kind"] != "send" or source["session_id"] != actor.session_id:
            raise JobError("nudge_caller_mismatch")
        if source["execution_role"] not in ("main", "workflow_node"):
            raise JobError("role_forbids_delegation")
        source_create = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (caller.job_id,))
        send_index = json.loads(source_create["payload_json"]).get("request", {}).get("send_index") if source_create else None
        if isinstance(send_index, bool) or not isinstance(send_index, int) or send_index < 1 or send_index != call_ref.send_index:
            raise JobError("nudge_caller_mismatch")
        job = await manager._fence(tx, actor, target)
        if node is not None:
            from .node_inputs import descriptor, is_closed
            workflow = await manager._row(tx, job["parent_job_id"])
            if (job["kind"] != "workflow_node" or source["execution_role"] != "main" or
                    workflow is None or workflow["kind"] != "workflow" or workflow["created_by_session_id"] != actor.session_id):
                raise JobError("nudge_target_mismatch")
            from .workflow_controller import for_node
            await for_node(manager, tx, job)
            if _hash(request["node"]) != _hash(await descriptor(manager, tx, job, session_id)):
                raise JobError("node_input_binding_mismatch")
            if await is_closed(manager, tx, job, session_id):
                raise JobError("node_input_closed")
        else:
            if job["kind"] != "subagent" or job["created_by_session_id"] != actor.session_id or job["session_id"] != session_id:
                raise JobError("nudge_target_mismatch")
            session = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (session_id,))
            metadata = json.loads(session["metadata_json"] or "{}") if session else {}
            if (session is None or session["parent_session_id"] != actor.session_id or
                    metadata.get("execution_role") != "subagent" or not metadata.get("standalone_subagent")):
                raise JobError("nudge_target_unavailable")
        from .result_revisions import delivery_id, result_id
        run_id = request["node"]["workflow_run_id"] if node is not None else (await delivery_id(manager, tx, job))[9:]
        payload = {"job_id": target.job_id, "session_id": session_id, "run_id": run_id,
                   "mode": "steer", "status": "accepted", "request": request}
        if job["generation"] > 1:
            payload.update(generation=job["generation"], result_event_id=result_id(target.job_id, job["generation"]))
        await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (manager.now(), target.job_id))
        operation = await manager._insert_operation(tx, job_id=target.job_id, version=job["version"]+1,
            actor=actor, action="nudge", key=key, fingerprint=fingerprint,
            call_ref=call_ref, tool_call_id=tool_call_id, payload=payload)
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (operation.operation_id,))
        cols = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
        inserted = await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, cols),
            (session_id, "nudge:" + operation.operation_id, "relay", "steer", "pending", message, manager.now()))
        if not inserted:
            raise JobError("nudge_inbox_conflict")
        return operation, False


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
            # A lost commit ACK is a lookup, not another enqueue/dispatch.
            old = await old_operation()
            if old is None:
                raise
            return await replay(old)
        # The event is an optimization; the original executor also drains the
        # durable inbox at boundaries and before committing its canonical result.
        await notify(operation)
        return receipt(operation, repeated)
