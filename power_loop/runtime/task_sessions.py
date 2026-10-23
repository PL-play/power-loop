"""Read-only task/session authority for audit tools, independent of liveness.

A session ID or workflow metadata match is not authority. Logical workflow nodes
own durable node_session descriptors; their synthetic driver may already be gone.
This module grants historical reads only, never continuation or dispatch.
"""
from __future__ import annotations

import json

from .jobs import JobError, _hash

_MAX_BINDINGS = 100


def _object(raw):
    def nonfinite(_):
        raise ValueError("non-finite binding")
    try:
        value = json.loads(raw, parse_constant=nonfinite)
    except (ValueError, TypeError) as exc:
        raise JobError("task_session_binding_corrupt") from exc
    if not isinstance(value, dict):
        raise JobError("task_session_binding_corrupt")
    return value


async def _bindings(manager, tx, job):
    """Resolve actual physical sessions without inspecting unrelated metadata."""
    if job["generation"] > 0:
        from .execution_membership import scope
        saved = await scope(manager, tx, job["job_id"], job["generation"])
        keys = ("tenant_id", "principal_id", "kind", "execution_role", "parent_job_id", "created_by_session_id")
        if job["kind"] in ("send", "subagent"):
            keys += ("session_id",)
        if _hash({key: saved[key] for key in keys}) != _hash({key: job[key] for key in keys}):
            raise JobError("task_session_binding_corrupt")
    if job["kind"] == "workflow_node":
        from .node_inputs import descriptor
        rows = await tx.fetchall(f"SELECT event_id,payload_json FROM {manager.events} "
            "WHERE job_id=? AND event_type='node_session' ORDER BY created_at,event_id LIMIT ?",
            (job["job_id"], _MAX_BINDINGS + 1))
        if len(rows) > _MAX_BINDINGS:
            raise JobError("task_session_scope_too_large")
        bindings = []
        seen = set()
        for event in rows:
            data = _object(event["payload_json"])
            sid, generation = data.get("session_id"), data.get("generation", 1)
            if (not isinstance(sid, str) or not 0 < len(sid) <= 255 or type(generation) is not int or
                    not 1 <= generation <= job["generation"] or (sid, generation) in seen or
                    event["event_id"] != manager._node_session_event_id(job["job_id"], sid, generation)):
                raise JobError("task_session_binding_corrupt")
            expected_keys = {"session_id", "workflow_run_id", "node_id", "route", "attempt"}
            if generation > 1:
                expected_keys.add("generation")
            creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (job["job_id"],))
            workflow = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (job["parent_job_id"],))
            payload = _object(creation["payload_json"]) if creation else {}
            request = payload.get("request")
            node = request.get("node") if isinstance(request, dict) else None
            retry = node.get("retry", {}) if isinstance(node, dict) else None
            attempts = retry.get("max_attempts", 1) if isinstance(retry, dict) else None
            if (set(data) != expected_keys or workflow is None or not isinstance(node, dict) or
                    data["workflow_run_id"] != workflow["event_id"][3:] or data["node_id"] != node.get("id") or
                    _hash(data["route"]) != _hash(request.get("route")) or
                    type(data["attempt"]) is not int or type(attempts) is not int or not 1 <= data["attempt"] <= attempts):
                raise JobError("task_session_binding_corrupt")
            seen.add((sid, generation))
            session = await tx.fetchone(f"SELECT session_id FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
            # A deleted physical session stays an unavailable historical binding,
            # never permission to find a substitute with similar metadata.
            if session is not None:
                historical_job = {**job, "generation": generation}
                await descriptor(manager, tx, historical_job, sid, historical=True)
            bindings.append({"session_id": sid, "generation": generation,
                "binding_event_id": event["event_id"], "node_id": data.get("node_id"),
                "route": data.get("route"), "attempt": data.get("attempt"),
                "available": session is not None, "reason": None if session else "session_deleted",
                "history_scope": "physical_session", "execution_authority": False})
        return bindings
    if job["kind"] not in ("send", "subagent"):
        # Background/timer session_id is the caller; a workflow's SID is a
        # synthetic driver. Neither is its own model transcript.
        return []
    sid = job["session_id"]
    if sid is None:
        return []
    if not isinstance(sid, str) or not 0 < len(sid) <= 255:
        raise JobError("task_session_binding_corrupt")
    session = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
    send_index = None
    if job["kind"] == "send":
        creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? "
            "AND event_type='operation' AND action='create'", (job["job_id"],))
        payload = _object(creation["payload_json"]) if creation else {}
        request = payload.get("request")
        send_index = request.get("send_index") if isinstance(request, dict) else None
        if type(send_index) is not int or not 1 <= send_index < 2**63:
            raise JobError("task_send_binding_corrupt")
    elif session is not None:
        metadata = _object(session["metadata_json"] or "{}")
        if (session["parent_session_id"] != job["created_by_session_id"] or
                metadata.get("execution_role") != "subagent"):
            raise JobError("task_session_binding_corrupt")
    return [{"session_id": sid, "generation": None, "binding_event_id": None,
        "available": session is not None, "reason": None if session else "session_deleted",
        "history_scope": "send" if send_index else "physical_session", "send_index": send_index,
        "execution_authority": False}]


async def list_sessions(manager, actor, job_id):
    """Return bounded owned bindings; no model, repair or write is performed.

    A physical node SID can appear in several execution generations. Each
    descriptor is checked; this does not claim the whole SID is one generation.
    """
    if not isinstance(job_id, str) or not 0 < len(job_id) <= 64:
        raise JobError("invalid_task_session_job")
    async with manager._write() as tx:
        job = await manager._authorize(tx, actor, job_id)
        bindings = await _bindings(manager, tx, job)
        return {"version": 1, "store_id": manager.store.store_id, "job_id": job_id,
            "kind": job["kind"], "job_generation": job["generation"], "bindings": bindings,
            "reason": None if bindings else ("not_created" if job["kind"] in ("send", "subagent", "workflow_node") else "logical_task"),
            "execution_authority": False}


async def authorize_session(manager, actor, job_id, session_id):
    """Authorize one exact task/SID pair, including readable retained history."""
    if not isinstance(session_id, str) or not 0 < len(session_id) <= 255:
        raise JobError("invalid_task_session_id")
    result = await list_sessions(manager, actor, job_id)
    matched = [binding for binding in result["bindings"] if binding["session_id"] == session_id]
    if not matched:
        raise JobError("task_session_access_denied")
    return {key: value for key, value in result.items() if key != "bindings"} | {"bindings": matched}
