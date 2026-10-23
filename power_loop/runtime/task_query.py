"""Bounded, authorized task directory; queries never dispatch or acknowledge work.

This experimental read model is deliberately independent of model tool handlers.
Historical jobs come from the store, not process-local task/handle dictionaries.
"""
from __future__ import annotations

import base64
import json
from collections import deque

from .jobs import KINDS, JobError, _hash, _json
from .message_ref import integer_id
from .result_revisions import result_id

_SCAN_LIMIT = 5000
_OUTCOMES = frozenset({"succeeded", "failed", "cancelled", "timed_out", "budget_exceeded", "interrupted"})
_STATUSES = _OUTCOMES | frozenset({"created", "queued", "running", "waiting", "settling", "cancelling", "model_closed"})


def _int(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise JobError("invalid_query_" + name)
    return value


def _strings(value, name, allowed):
    if value is None:
        return None
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or x not in allowed for x in value):
        raise JobError("invalid_query_" + name)
    return sorted(set(value))


def _decode(value):
    try:
        if not isinstance(value, str) or len(value) > 2048:
            raise ValueError()
        data = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
        if not isinstance(data, dict) or set(data) != {"query", "upper", "after", "snapshot_at", "membership"}:
            raise ValueError()
        if any(not isinstance(data[k], str) or len(data[k]) != 64 for k in ("query", "membership")):
            raise ValueError()
        for key in ("upper", "after"):
            pair = data[key]
            if (not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not int or pair[0] < 0 or
                    not isinstance(pair[1], str) or not 0 < len(pair[1]) <= 64):
                raise ValueError()
        _int(data["snapshot_at"], "cursor", 0, 2**63 - 1)
        if tuple(data["after"]) > tuple(data["upper"]):
            raise ValueError()
        return data
    except (ValueError, TypeError, UnicodeError) as exc:
        raise JobError("invalid_query_cursor") from exc


def _cursor(query, upper, after, snapshot, membership):
    return base64.urlsafe_b64encode(_json({"query": query, "upper": list(upper), "after": list(after),
        "snapshot_at": snapshot, "membership": membership}).encode()).decode().rstrip("=")


def _key(row):
    return row["created_at"], row["job_id"]


def active_reasons(row):
    reasons = []
    if row["outcome"] is None and row["status"] != "model_closed":
        reasons.append("execution_pending")
    if row["resource_state"] in ("running", "unknown"):
        reasons.append("resource_" + row["resource_state"])
    if row["result_status"] not in ("none", "handled", "not_applicable"):
        reasons.append("result_" + row["result_status"])
    if row["supervision_status"] != "closed":
        reasons.append("supervision_open")
    return reasons


async def _forest(manager, tx, actor, root, max_depth, upper):
    if root is not None:
        seeds = [await manager._authorize(tx, actor, root)]
    else:
        seeds = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? "
            "AND (session_id=? OR created_by_session_id=?) ORDER BY created_at,job_id LIMIT ?",
            (actor.tenant_id, actor.principal_id, actor.session_id, actor.session_id, _SCAN_LIMIT + 1))
    if len(seeds) > _SCAN_LIMIT:
        raise JobError("query_scope_too_large")
    rows, depths, pending = {}, {}, []
    # Seeds may include both a node and one of its sends. Preserve their actual
    # depths relative to the highest authorized seed, rather than flattening them.
    seed_ids = {r["job_id"] for r in seeds}
    for row in seeds:
        await manager._authorize(tx, actor, row["job_id"])
        if row["parent_job_id"] not in seed_ids and (upper is None or _key(row) <= upper):
            pending.append((row, 0))
    if seeds and not pending:
        raise JobError("invalid_job_lineage")
    pending = deque(pending)
    discovered = {r["job_id"] for r, _ in pending}
    truncated = False
    while pending:
        row, depth = pending.popleft()
        jid = row["job_id"]
        await manager._authorize(tx, actor, jid)
        rows[jid], depths[jid] = row, depth
        room = _SCAN_LIMIT - len(rows) - len(pending)
        children = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE parent_job_id=? "
            "ORDER BY created_at,job_id LIMIT ?", (jid, 1 if depth >= max_depth else room + 1))
        children = [c for c in children if upper is None or _key(c) <= upper]
        if children and depth >= max_depth:
            truncated = True
            continue
        if len(children) > room:
            truncated = True
        for child in children[:room]:
            if child["job_id"] in discovered:
                raise JobError("invalid_job_lineage")
            discovered.add(child["job_id"])
            pending.append((child, depth + 1))
    return rows, depths, truncated


def _summary(row, *, context=False):
    result = {
        "job_id": row["job_id"], "parent_job_id": row["parent_job_id"], "kind": row["kind"],
        "status": row["status"], "context_only": context,
    }
    if context:
        return result
    result.update({k: row[k] for k in ("session_id", "created_by_session_id", "generation", "version", "resource_state",
        "supervision_status", "result_status", "outcome", "created_at", "updated_at")})
    result.update(title=row["title"][:160], blocks_completion=bool(row["blocks_completion"]),
        active_reason=active_reasons(row), billing_status="unavailable", cancel_requested=row["cancel_requested_at"] is not None,
        controls={"request_cancel": row["supervision_status"] != "closed", "resource_stopped": row["resource_state"] == "stopped",
            "reason": "Persistent cancellation acceptance is not confirmation of physical stop."})
    return result


async def _index(manager, tx, row):
    creation = await tx.fetchone(f"SELECT event_id,call_ref_json,receipt_ref_json FROM {manager.events} "
        "WHERE job_id=? AND event_type='operation' AND action='create'", (row["job_id"],))
    current = await tx.fetchone(f"SELECT event_id,job_version,payload_json FROM {manager.events} WHERE event_id=? AND event_type='result' AND job_id=?",
        (result_id(row["job_id"], row["generation"]), row["job_id"]))
    refs = {"creation_operation_id": creation["event_id"] if creation else None,
        "call_ref": _reference(manager, row, creation["call_ref_json"], "assistant") if creation else None,
        "receipt_ref": _reference(manager, row, creation["receipt_ref_json"], "tool") if creation else None,
        "result_ref": {"store_id": manager.store.store_id, "event_id": current["event_id"],
            "generation": row["generation"], "revision": current["job_version"]} if current else None}
    if row["result_json"]:
        try:
            saved = json.loads(row["result_json"])
            canonical = json.loads(current["payload_json"]) if current else None
            if not isinstance(saved, dict) or not isinstance(canonical, dict) or not isinstance(canonical.get("result"), dict):
                raise ValueError()
            data = canonical["result"]
            if canonical.get("generation", 0 if canonical.get("unstarted") else 1) != row["generation"]:
                raise ValueError()
            text = data.get("final_text")
            if not isinstance(text, str) and isinstance(data.get("final"), dict):
                text = data["final"].get("text")
            refs["output_excerpt"] = text[:400] if isinstance(text, str) else ""
        except (TypeError, ValueError) as exc:
            raise JobError("query_result_corrupt") from exc
    return refs


def _reference(manager, row, raw, role):
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        expected = {"version", "store_id", "session_id", "id", "seq", "role", "tool_call_id", "send_index", "round_index", "id_status"}
        if (not isinstance(data, dict) or set(data) != expected or type(data["version"]) is not int or data["version"] != 1 or
                data["store_id"] != manager.store.store_id or data["session_id"] != row["created_by_session_id"] or
                data["role"] != role or data["id_status"] != "available" or
                (data["tool_call_id"] is not None and not isinstance(data["tool_call_id"], str))):
            raise ValueError()
        integer_id(data["id"])
        integer_id(data["seq"], "seq")
        if data["send_index"] is not None and (not isinstance(data["send_index"], str) or
                not data["send_index"].isascii() or not data["send_index"].isdecimal()):
            raise ValueError()
        if data["round_index"] is not None and (type(data["round_index"]) is not int or data["round_index"] < 0):
            raise ValueError()
        return data
    except (TypeError, ValueError) as exc:
        raise JobError("query_message_ref_corrupt") from exc


async def list_jobs(manager, actor, *, root_job_id=None, scope="descendants", status="active", kinds=None,
                    billing_status=None, created_after=None, created_before=None, max_depth=63, limit=20,
                    max_chars=8000, cursor=None):
    """Read a bounded task forest and immutable output/message pointers.

    Cursor pages retain the creation upper bound and recheck authorization and
    current status. They are not a frozen snapshot of mutable execution state.
    """
    if root_job_id is not None and (not isinstance(root_job_id, str) or not 0 < len(root_job_id) <= 64):
        raise JobError("invalid_query_root")
    if scope not in ("self", "descendants") or (scope == "self" and root_job_id is None):
        raise JobError("invalid_query_scope")
    if isinstance(status, list):
        status = _strings(status, "status", _STATUSES)
    elif not isinstance(status, str) or status not in ("active", "terminal", "all"):
        raise JobError("invalid_query_status")
    kinds = _strings(kinds, "kinds", KINDS)
    if billing_status is not None:
        raise JobError("billing_adapter_unavailable")
    _int(max_depth, "max_depth", 0, 63)
    _int(limit, "limit", 1, 100)
    _int(max_chars, "max_chars", 1024, 32000)
    for value in (created_after, created_before):
        if value is not None:
            _int(value, "time", 0, 2**63 - 1)
    if created_after is not None and created_before is not None and created_after > created_before:
        raise JobError("invalid_query_time")
    query = _hash([manager.store.store_id, actor.tenant_id, actor.principal_id, actor.session_id,
        root_job_id, scope, status, kinds, created_after, created_before, max_depth])
    page = _decode(cursor) if cursor is not None else None
    if page and page["query"] != query:
        raise JobError("query_cursor_mismatch")
    async with manager._write() as tx:
        if scope == "self":
            row = await manager._authorize(tx, actor, root_job_id)
            rows, depths, tree_truncated = {root_job_id: row}, {root_job_id: 0}, False
        else:
            rows, depths, tree_truncated = await _forest(manager, tx, actor, root_job_id, max_depth,
                tuple(page["upper"]) if page else None)
        membership = _hash(sorted((r["job_id"], r["parent_job_id"]) for r in rows.values()))
        if page and page["membership"] != membership:
            # Random job IDs are not an insertion sequence. A same-millisecond
            # insert below the saved upper key must not silently enter a page.
            raise JobError("query_snapshot_changed")
        upper = tuple(page["upper"]) if page else max((_key(r) for r in rows.values()), default=(0, "empty"))
        snapshot = page["snapshot_at"] if page else manager.now()
        matches = []
        for row in rows.values():
            if root_job_id is not None and (row["job_id"] == root_job_id) != (scope == "self"):
                continue
            if (kinds is not None and row["kind"] not in kinds) or (page and _key(row) <= tuple(page["after"])):
                continue
            if (created_after is not None and row["created_at"] < created_after) or (created_before is not None and row["created_at"] > created_before):
                continue
            if ((status == "active" and not active_reasons(row)) or (status == "terminal" and row["outcome"] is None) or
                    (isinstance(status, list) and row["status"] not in status)):
                continue
            matches.append(row)
        matches.sort(key=_key)
        envelope = {"version": 1, "snapshot_at": snapshot, "observed_at": manager.now(),
            "root": {"job_id": root_job_id, "session_id": actor.session_id}, "nodes": [], "matched_count": 0,
            "tree_truncated": tree_truncated, "truncated": tree_truncated or bool(matches), "next_cursor": None}
        selected = set()
        for index, row in enumerate(matches[:limit]):
            additions = {}
            parent = row["parent_job_id"]
            while parent in rows and parent not in selected:
                additions[parent] = _summary(rows[parent], context=True)
                parent = rows[parent]["parent_job_id"]
            additions[row["job_id"]] = {**_summary(row), **await _index(manager, tx, row)}
            combined = {n["job_id"]: n for n in envelope["nodes"]}
            for jid, node in additions.items():
                if jid not in combined or not node["context_only"]:
                    combined[jid] = node
            candidate = {**envelope, "nodes": sorted(combined.values(), key=lambda n: (depths[n["job_id"]], _key(rows[n["job_id"]]))),
                "matched_count": index + 1, "truncated": tree_truncated or index + 1 < len(matches),
                "next_cursor": _cursor(query, upper, _key(row), snapshot, membership) if index + 1 < len(matches) else None}
            if len(_json(candidate)) > max_chars:
                if not envelope["matched_count"]:
                    raise JobError("query_result_too_small")
                break
            envelope = candidate
            selected.add(row["job_id"])
        return envelope


async def get_job(manager, actor, job_id, *, max_chars=8000):
    """A current saved summary and references, never execution control secrets."""
    _int(max_chars, "max_chars", 1024, 32000)
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, job_id)
        output = {"version": 1, "observed_at": manager.now(), "node": {**_summary(row), **await _index(manager, tx, row)}}
        if len(_json(output)) > max_chars:
            raise JobError("query_result_too_small")
        return output
