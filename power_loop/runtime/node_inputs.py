"""Exact physical workflow-node input bindings; logical nodes have no SID."""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import JobError, _hash


def event_id(manager, prefix, job_id, sid, generation=1):
    if isinstance(generation, bool) or not isinstance(generation, int) or not 1 <= generation < 2**63:
        raise JobError("invalid_execution_generation")
    key = [manager.store.store_id, job_id, sid]
    if generation > 1:
        key.append(generation)
    return prefix + _hash(key)[:32]


def closed_id(manager, job_id, sid, generation=1):
    return event_id(manager, "input_closed_", job_id, sid, generation)


def _object(raw):
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise JobError("node_input_binding_mismatch") from exc
    if not isinstance(data, dict):
        raise JobError("node_input_binding_mismatch")
    return data


async def session_data(manager, reader, job, sid, *, historical=False):
    """Validate physical metadata against the frozen admission, not itself."""
    session = await reader.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
    meta = _object(session["metadata_json"] or "{}") if session else {}
    authorized = False
    if not historical and job["generation"] > 1:
        from .workflow_restore_execution import authority, has_authority
        if await has_authority(manager, reader, job):
            await authority(manager, reader, job, sid)
            authorized = True
    if (job["kind"] != "workflow_node" or not session or
            (session["parent_session_id"] != job["created_by_session_id"] and
             not ((historical or authorized) and session["parent_session_id"] is None)) or
            meta.get("execution_role") != "workflow_node" or meta.get("task_owner_job_id") != job["job_id"]):
        raise JobError("session_ownership_mismatch")
    workflow_op = await reader.fetchone(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (job["parent_job_id"],))
    node_op = await reader.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (job["job_id"],))
    if workflow_op is None or node_op is None:
        raise JobError("node_route_mismatch")
    request = _object(node_op["payload_json"]).get("request", {})
    if not isinstance(request, dict) or not isinstance(request.get("node", {}), dict):
        raise JobError("node_route_mismatch")
    node = request.get("node", {})
    retry = node.get("retry", {})
    if not isinstance(retry, dict):
        raise JobError("node_route_mismatch")
    expected_node, expected_route = node.get("id"), request.get("route")
    attempt, max_attempts = meta.get("attempt"), retry.get("max_attempts", 1)
    if (meta.get("workflow_run_id") != workflow_op["event_id"][3:] or
            (expected_node is not None and meta.get("workflow_node_id") != expected_node) or
            (expected_route is not None and _hash(meta.get("task_node_route")) != _hash(expected_route)) or
            (expected_node is not None and (isinstance(attempt, bool) or not isinstance(attempt, int) or
                isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= attempt <= max_attempts))):
        raise JobError("node_route_mismatch")
    data = {"session_id": sid, "workflow_run_id": meta.get("workflow_run_id"),
            "node_id": meta.get("workflow_node_id"), "route": meta.get("task_node_route"), "attempt": attempt}
    if job["generation"] > 1:
        data["generation"] = job["generation"]
    return data


async def descriptor(manager, reader, job, sid, *, historical=False):
    event = await reader.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='node_session' AND job_id=?",
        (manager._node_session_event_id(job["job_id"], sid, job["generation"]), job["job_id"]))
    if event is None:
        raise JobError("node_input_binding_mismatch")
    data = _object(event["payload_json"])
    try:
        expected = await session_data(manager, reader, job, sid, historical=historical)
    except JobError as exc:
        raise JobError("node_input_binding_mismatch") from exc
    if _hash(data) != _hash(expected):
        raise JobError("node_input_binding_mismatch")
    return data


def _closed_data(job, sid):
    data = {"session_id": sid}
    if job["generation"] > 1:
        data["generation"] = job["generation"]
    return data


async def is_closed(manager, reader, job, sid):
    row = await reader.fetchone(f"SELECT job_id,event_type,payload_json FROM {manager.events} WHERE event_id=?",
        (closed_id(manager, job["job_id"], sid, job["generation"]),))
    if row is None:
        return False
    try:
        data = _object(row["payload_json"])
    except JobError as exc:
        raise JobError("node_input_closure_mismatch") from exc
    if (row["job_id"] != job["job_id"] or row["event_type"] != "node_input_closed" or
            _hash(data) != _hash(_closed_data(job, sid))):
        raise JobError("node_input_closure_mismatch")
    return True


async def input_target(manager, reader, actor, send, event):
    """The fenced actual send may access only its own durably bound node input."""
    job = await manager._row(reader, event["job_id"])
    if job is not None and job["kind"] == "workflow_node":
        if (job["job_id"] != send["control_owner_job_id"] or
                (job["tenant_id"], job["principal_id"]) != (actor.tenant_id, actor.principal_id)):
            raise JobError("processing_input_mismatch")
        data = await descriptor(manager, reader, job, actor.session_id)
        request = json.loads(event["payload_json"])["request"]
        if _hash(request.get("node")) != _hash(data):
            raise JobError("processing_input_mismatch")
        return job
    return await manager._authorize(reader, actor, event["job_id"])


async def close(runtime, binding, sid):
    """Serialize persistent input closure with admission; model sends stay open."""
    manager = runtime.manager
    async with manager._write() as tx:
        try:
            job = await manager._fence(tx, binding.actor, binding.lease)
        except JobError as exc:
            if exc.code != "cancel_requested":
                raise
            # Closing this exact attempt is exit bookkeeping, not permission
            # to dispatch another send. A durable cancellation must not block
            # its input-closure marker and escape into the parent workflow.
            job = await manager._authorize(tx, binding.actor, binding.lease.job_id)
            manager._check_proof(job, binding.lease)
            if job["kind"] != "workflow_node" or job["resource_state"] not in ("running", "unknown"):
                raise JobError("node_input_closure_not_running") from None
        await descriptor(manager, tx, job, sid)
        marker_id = closed_id(manager, job["job_id"], sid, job["generation"])
        if await is_closed(manager, tx, job, sid):
            return
        await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (manager.now(), job["job_id"]))
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
            (marker_id, job["job_id"], "node_input_closed", job["version"]+1, json.dumps(_closed_data(job, sid)), "not_applicable", manager.now()))


async def finish_attempt(runtime, engine, raw):
    """Close this attempt's input intake before the final durable inbox drain."""
    from . import delegation
    sid = raw.get("session_id")
    target = engine._task_nodes.get(sid)
    if target is None:
        return
    async with delegation._gate(runtime.manager.store, sid):
        await close(runtime, target, sid)
        # A failed provider cannot process accepted input by starting a fresh
        # business attempt. Preserve the original failed scope and its inbox.
        if raw.get("status") != "completed":
            pending = await runtime.manager.db.fetchone(f"SELECT event_id FROM {runtime.manager.events} "
                "WHERE job_id=? AND action='nudge' AND delivery_status<>'handled'", (target.lease.job_id,))
            if pending is not None:
                error = JobError("required_input_unhandled")
                error.execution_result = dict(raw)
                raise error


async def steer(loop, engine, run_id, node_id, sid, message, *, expected_job=None):
    """Admit into the original live engine, or recall an exact prior admission."""
    import logging

    from . import delegation
    from .invocation import current_invocation
    from .nudges import receipt
    from .task_runtime import _SEND

    runtime = loop.task_runtime
    if expected_job is not None and (not isinstance(expected_job, tuple) or len(expected_job) != 2 or
            not isinstance(expected_job[0], str) or not 0 < len(expected_job[0]) <= 64 or
            type(expected_job[1]) is not int or not 1 <= expected_job[1] < 2**63):
        raise JobError("invalid_node_control_expectation")
    source = _SEND.get()
    if source is None or source.runtime is not runtime:
        raise JobError("task_send_required")
    seq, call_id = current_invocation(source.actor.session_id)
    if seq is None or not call_id:
        raise JobError("persisted_tool_call_required")
    manager = runtime.manager
    ref = await manager.store.message_ref(source.actor.session_id, seq)

    async def previous():
        operation = await manager.operation_for_call(source.actor, ref, call_id)
        if operation is not None:
            request = operation.result.get("request", {})
            data = request.get("node", {})
            if (operation.action != "nudge" or request.get("session_id") != sid or request.get("message") != message or
                    data.get("workflow_run_id") != run_id or data.get("node_id") != node_id or operation.call_ref != ref.to_dict() or
                    expected_job is not None and (operation.job_id, request.get("previous_generation", data.get("generation", 1))) != expected_job):
                raise JobError("operation_conflict")
        return operation

    async with delegation._gate(manager.store, sid):
        old = await previous()
        if old is not None:
            return receipt(old, True)
        binding = engine._task_nodes.get(sid) if engine is not None else None
        if (binding is None or binding.runtime is not runtime or engine._run_id != run_id or binding.stop_renewal.is_set()):
            if engine is None or engine._run_id != run_id:
                raise JobError("task_adapter_unavailable:workflow_node_resume")
            import asyncio

            from .node_resume import reserve
            return await asyncio.shield(reserve(loop, engine, source, ref, call_id, node_id, sid, message, expected_job=expected_job))
        if expected_job is not None and (binding.lease.job_id, binding.lease.generation) != expected_job:
            raise JobError("workflow_node_generation_changed")
        job = await manager.get_job(source.actor, binding.lease.job_id)
        data = await descriptor(manager, manager.db, asdict(job), sid)
        if data["node_id"] != node_id or data["workflow_run_id"] != run_id:
            raise JobError("node_input_binding_mismatch")
        try:
            operation, replayed = await manager.accept_nudge(source.actor, binding.lease, caller=source.lease,
                call_ref=ref, tool_call_id=call_id, session_id=sid, message=message, node=data)
        except Exception:
            old = await previous()
            if old is None:
                raise
            return receipt(old, True)
        try:
            loop.steer_event(sid).set()
        except Exception:
            logging.getLogger(__name__).exception("node nudge accepted; local signal failed: %s", operation.operation_id)
        return receipt(operation, replayed)
