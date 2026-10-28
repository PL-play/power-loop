"""Authorized bounded model-attempt metadata; no prompts, credentials or ACKs."""
from __future__ import annotations

from .jobs import JobError, _hash, _json
from .model_attempts import _all, _object, _read
from .task_events import _MAX_EVENTS, _cursor, _decode, _key, _query
from .task_history import _snapshot


async def _summary(manager, tx, job, raw, phases):
    data = _object(raw["payload_json"])
    identity = data["attempt_id"]
    prepared = phases["prepared"][identity]
    if (prepared[0]["event_id"] != raw["event_id"] or prepared[1] != data or data["scope"]["job_id"] != job["job_id"] or
            data["session_id"] != job["session_id"]):
        raise JobError("model_attempt_evidence_corrupt")
    tracking = await _read(manager, tx, "tracking", data["scope"])
    if tracking[1] != {"version": 1, "scope": data["scope"], "session_id": data["session_id"]}:
        raise JobError("model_attempt_evidence_corrupt")
    dispatched = phases["dispatched"].get(identity)
    finished = phases["finished"].get(identity)
    state = "prepared"
    observed = {}
    if dispatched:
        if dispatched[1] != {"version": 1, "scope": data["scope"], "attempt_id": identity, "prepared_hash": _hash(data)}:
            raise JobError("model_attempt_evidence_corrupt")
        state = "dispatched_unconfirmed"
    if finished:
        observed = finished[1]
        if (dispatched is None or observed.get("scope") != data["scope"] or observed.get("attempt_id") != identity or
                observed.get("prepared_hash") != _hash(data) or observed.get("dispatch_hash") != _hash(dispatched[1]) or
                observed.get("status") not in ("returned", "unknown")):
            raise JobError("model_attempt_evidence_corrupt")
        state = observed["status"]
    elif dispatched is None:
        from .model_budgets import is_closed_unstarted

        if await is_closed_unstarted(manager, tx, data):
            state = "not_dispatched"
    return {"attempt_id": identity, "generation": str(data["scope"]["generation"]),
        "round_index": data["round_index"], "retry_index": data["retry_index"], "model": data["model"],
        "purpose": data.get("purpose", "llm"), "max_tokens": data["max_tokens"], "request_fingerprint": data["request_hash"], "status": state,
        "observed_usage": observed.get("usage"), "estimated": observed.get("estimated"),
        "provider_reported_cost": observed.get("provider_cost"), "provider_cost_currency": None,
        "provider_response_id": observed.get("provider_response_id"), "provider_query_supported": False,
        "error_type": observed.get("error_type"), "response_received": state == "returned",
        "prepared_event_id": raw["event_id"], "dispatched_event_id": dispatched[0]["event_id"] if dispatched else None,
        "finished_event_id": finished[0]["event_id"] if finished else None, "execution_authority": False}


async def list_attempts(manager, actor, job_id, *, limit=20, max_chars=8000, cursor=None):
    if (not isinstance(job_id, str) or not 0 < len(job_id) <= 64 or type(limit) is not int or not 1 <= limit <= 100 or
            type(max_chars) is not int or not 1024 <= max_chars <= 32000):
        raise JobError("invalid_model_attempt_query")
    async with _snapshot(manager) as tx:
        job = await manager._authorize(tx, actor, job_id)
        if job["kind"] != "send":
            raise JobError("model_attempts_require_send_job")
        query = _query(manager, actor, job, "attempts")
        page = _decode(cursor, query) if cursor is not None else None
        phases = await _all(manager, tx, job_id)
        rows = sorted((value[0] for value in phases["prepared"].values()), key=_key)
        if len(rows) > _MAX_EVENTS:
            raise JobError("task_event_scope_too_large")
        membership = _hash([row["event_id"] for row in rows])
        if page and page["membership"] != membership:
            raise JobError("model_attempt_snapshot_changed")
        upper = tuple(page["upper"]) if page else max((_key(row) for row in rows), default=(0, "empty"))
        state = {"query": query, "upper": list(upper), "membership": membership}
        selected = [r for r in rows if _key(r) <= upper and (page is None or _key(r) > tuple(page["after"]))]
        result = {"version": 1, "backend": "task_registry", "view": "attempts", "job_id": job_id,
            "billing_status": "unavailable", "items": [], "truncated": bool(selected), "next_cursor": None}
        for index, row in enumerate(selected[:limit]):
            item = await _summary(manager, tx, job, row, phases)
            more = index + 1 < len(selected)
            candidate = {**result, "items": [*result["items"], item], "truncated": more,
                "next_cursor": _cursor(state, _key(row)) if more else None}
            if len(_json(candidate)) > max_chars:
                if not result["items"]:
                    raise JobError("query_result_too_small")
                break
            result = candidate
        return result
