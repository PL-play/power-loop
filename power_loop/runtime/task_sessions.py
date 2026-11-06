"""Read-only task/session authority for audit tools, independent of liveness.

A session ID or workflow metadata match is not authority. Logical workflow nodes
their own durable session and frozen start input.
This module grants historical reads only, never continuation or dispatch.
"""
from __future__ import annotations

import json

from .jobs import JobError

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
    if job["kind"] not in ("send", "subagent", "workflow_node"):
        return []
    sid = job["session_id"]
    session = await tx.fetchone(f"SELECT session_id FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
    send_index = None
    if job["kind"] == "send":
        creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (job["job_id"],))
        request = _object(creation["payload_json"])["request"]
        reference = request["input_ref"]
        source = await tx.fetchone(f"SELECT delivered_send_index FROM {manager.store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?", (reference["session_id"], reference["message_id"]))
        send_index = source["delivered_send_index"]
    return [{"session_id": sid, "generation": job["generation"], "binding_event_id": None,
        "node_id": job["title"] if job["kind"] == "workflow_node" else None,
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
    async with manager._read() as tx:
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
