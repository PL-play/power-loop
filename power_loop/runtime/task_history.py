"""Task-qualified, bounded historical reads; never grants execution authority."""
from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager

from .jobs import JobError, _hash
from .message_ref import MessageRef
from .recall import _export, _screen, dumps
from .store.task_schema import TASK_PROTOCOL_VERSION
from .task_sessions import _bindings

_EXPORT_MAX_RECORDS = 10_000
_EXPORT_MAX_BYTES = 16 * 1024 * 1024


def _number(value, name, minimum=1, maximum=2**63-1):
    if isinstance(value, str) and len(value) <= 19 and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or not minimum <= value <= maximum:
        raise JobError("invalid_task_history_"+name)
    return value


def _text(value, name, maximum):
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise JobError("invalid_task_history_"+name)
    return value


@asynccontextmanager
async def _snapshot(manager):
    # No control-row FOR UPDATE lock and no transaction across an LLM request.
    async with manager.db.transaction() as tx:
        if manager.db.dialect.name == "postgres":
            await tx.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        await _verify_store(manager, tx)
        yield tx


async def _verify_store(manager, tx):
    row = await tx.fetchone(f"SELECT store_id,protocol_version FROM {manager.prefix}task_store WHERE id=1")
    if row is None or row["store_id"] != manager.store.store_id or row["protocol_version"] != TASK_PROTOCOL_VERSION:
        raise JobError("store_identity_changed")


async def _historical_caller(manager, tx, actor, refs, cache):
    """Older steering saved SID/send refs but not a caller lease in payload.

    Resolve only an unambiguous immutable send descriptor in the actor's domain;
    matching session metadata or a live loop alone is never authority.
    """
    pairs = {(ref["session_id"], ref["send_index"]) for ref in refs}
    if len(pairs) != 1:
        raise JobError("task_history_origin_scope_mismatch")
    sid, send = next(iter(pairs))
    if sid not in cache:
        rows = await tx.fetchall(f"SELECT j.job_id,e.payload_json FROM {manager.jobs} j JOIN {manager.events} e "
            "ON e.job_id=j.job_id AND e.event_type='operation' AND e.action='create' "
            "WHERE j.kind='send' AND j.generation>0 AND j.session_id=? AND j.tenant_id=? AND j.principal_id=? "
            "ORDER BY j.created_at,j.job_id LIMIT ?", (sid, actor.tenant_id, actor.principal_id, 5001))
        if len(rows) > 5000:
            raise JobError("task_history_operation_scope_too_large")
        found = {}
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
                request = payload.get("request") if isinstance(payload, dict) else None
                index = request.get("send_index") if isinstance(request, dict) else None
                if type(index) is int and 1 <= index < 2**63:
                    found.setdefault(str(index), []).append(row["job_id"])
            except (ValueError, TypeError):
                continue  # An unwitnessed unrelated send does not grant a match.
        cache[sid] = found
    matches = cache[sid].get(send, [])
    if len(matches) != 1:
        raise JobError("task_history_operation_origin_unavailable")
    return matches[0]


async def _origin_scope(manager, tx, actor, job, sid, record_ids):
    """Exact operation references never grant a caller's whole transcript."""
    from .execution_membership import _lease
    from .task_query import _reference

    if record_ids is None:
        return None
    events = await tx.fetchall(f"SELECT event_id,action,caller_session_id,tool_call_id,call_ref_json,receipt_ref_json FROM {manager.events} "
        "WHERE job_id=? AND event_type='operation' ORDER BY created_at,event_id LIMIT ?", (job["job_id"], 5001))
    if len(events) > 5000:
        raise JobError("task_history_operation_scope_too_large")
    selected, origins, source_cache = [], set(), {}
    for event in events:
        subject = {**job, "created_by_session_id": event["caller_session_id"]}
        refs = [_reference(manager, subject, event[key], role) for key, role in
            (("call_ref_json", "assistant"), ("receipt_ref_json", "tool")) if event[key]]
        chosen = [ref for ref in refs if int(ref["id"]) in record_ids]
        if not chosen:
            continue
        if event["action"] == "create":
            source_id = job["parent_job_id"]
        else:
            raw = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND job_id=?", (event["event_id"], job["job_id"]))
            try:
                payload = json.loads(raw["payload_json"])
                if not isinstance(payload, dict):
                    raise ValueError()
                request = payload.get("request")
                if isinstance(request, dict) and "caller" in request:
                    proof = request["caller"]
                    _lease(proof)
                    source_id = proof["job_id"]
                else:
                    source_id = await _historical_caller(manager, tx, actor, chosen, source_cache)
            except (ValueError, TypeError, KeyError) as exc:
                raise JobError("task_history_operation_origin_unavailable") from exc
        origins.add(source_id)
        selected.extend((ref, event["event_id"], event["tool_call_id"]) for ref in chosen)
    if {int(ref["id"]) for ref, _, _ in selected} != set(record_ids):
        return None
    if len(origins) != 1:
        raise JobError("task_history_origin_scope_mismatch")
    parent = await manager._authorize(tx, actor, next(iter(origins)))
    refs = [ref for ref, _, _ in selected]
    if (parent["kind"] != "send" or any(ref["session_id"] != parent["session_id"] for ref in refs) or
            sid is not None and sid != parent["session_id"]):
        raise JobError("task_history_origin_scope_mismatch")
    [binding] = await _bindings(manager, tx, parent)
    if any(ref["send_index"] != str(binding["send_index"]) for ref in refs):
        raise JobError("task_history_origin_scope_mismatch")
    for ref, _, call_id in selected:
        row = await tx.fetchone(f"SELECT * FROM {manager.store.t.messages} WHERE session_id=? AND id=?",
            (parent["session_id"], int(ref["id"])))
        if row is not None:
            actual = MessageRef(manager.store.store_id, row["session_id"], row["id"], row["seq"], row["role"],
                row["tool_call_id"], row["send_index"], row["round_index"]).to_dict()
            if _hash(actual) != _hash(ref) or ref["role"] == "tool" and ref["tool_call_id"] != call_id:
                raise JobError("task_history_origin_reference_mismatch")
            if ref["role"] == "assistant":
                try:
                    calls = json.loads(row["tool_calls_json"] or "[]")
                    if not isinstance(calls, list) or not any(isinstance(call, dict) and call.get("id") == call_id for call in calls):
                        raise ValueError()
                except (ValueError, TypeError) as exc:
                    raise JobError("task_history_origin_reference_mismatch") from exc
    return {"job_id": job["job_id"], "job_generation": job["generation"], "kind": job["kind"],
        "session_id": parent["session_id"], "bindings": [{**binding, "history_scope": "exact_origin_messages", "message_refs": refs,
            "source_operation_ids": sorted({operation for _, operation, _ in selected})}],
        "send_index": binding["send_index"], "available": binding["available"], "reason": binding["reason"]}


async def _scope(manager, tx, actor, job_id, sid, record_ids=None):
    job = await manager._authorize(tx, actor, job_id)
    origin = await _origin_scope(manager, tx, actor, job, sid, record_ids)
    if origin is not None:
        return origin
    bindings = await _bindings(manager, tx, job)
    if sid is None:
        sessions = {b["session_id"] for b in bindings}
        if len(sessions) > 1:
            raise JobError("task_history_session_required")
        sid = next(iter(sessions), None)
    selected = [b for b in bindings if b["session_id"] == sid]
    if bindings and not selected or sid is not None and not selected:
        raise JobError("task_session_access_denied")
    forced = {b.get("send_index") for b in selected if b["history_scope"] == "send"}
    if len(forced) > 1:
        raise JobError("task_session_binding_corrupt")
    return {"job_id": job_id, "job_generation": job["generation"], "kind": job["kind"],
        "session_id": sid, "bindings": selected, "send_index": next(iter(forced), None),
        "available": bool(selected) and all(b["available"] for b in selected),
        "reason": None if selected and all(b["available"] for b in selected) else
            "session_deleted" if selected else "not_created" if job["kind"] in ("send", "subagent", "workflow_node") else "logical_task"}


def _selection(source, ids, seq, id, message_id, message_ids, send_index):
    if source not in ("messages", "projection"):
        raise JobError("invalid_task_history_source")
    if sum(v is not None for v in (ids, seq, id, message_id, message_ids)) > 1:
        raise JobError("ambiguous_task_history_selection")
    if (source == "projection" and any(v is not None for v in (seq, message_id, message_ids)) or
            source == "messages" and id is not None):
        raise JobError("task_history_coordinate_mismatch")
    lo, hi, records = 1, None, None
    single = any(v is not None for v in (seq, id, message_id))
    if ids is not None:
        if not isinstance(ids, list) or len(ids) != 2:
            raise JobError("invalid_task_history_ids")
        lo, hi = [_number(v, "ids") for v in ids]
        if lo > hi:
            raise JobError("invalid_task_history_ids")
    if seq is not None or id is not None:
        lo = hi = _number(seq if seq is not None else id, "coordinate")
    if message_id is not None:
        records = [_number(message_id, "message_id")]
    if message_ids is not None:
        if not isinstance(message_ids, list) or not 1 <= len(message_ids) <= 100:
            raise JobError("invalid_task_history_message_ids")
        records = sorted({_number(v, "message_id") for v in message_ids})
    return {"source": source, "lo": lo, "hi": hi, "record_ids": records, "single": single,
        "send_index": _number(send_index, "send_index") if send_index is not None else None}


def _where(sid, selection, *, forced_send=False):
    coord = "seq" if selection["source"] == "messages" else "id"
    where, params = f"session_id=? AND {coord}>=?", [sid, selection["lo"]]
    if selection["hi"] is not None:
        where += f" AND {coord}<=?"
        params.append(selection["hi"])
    if selection["send_index"] is not None:
        where += " AND send_index=?"
        params.append(selection["send_index"])
    if selection["record_ids"] is not None:
        where += " AND id IN ("+",".join("?" for _ in selection["record_ids"])+")"
        params.extend(selection["record_ids"])
    if forced_send and selection["source"] == "projection":
        where += " AND kind IN ('user','project')"
    return where, params, coord


def _record(manager, source, row):
    value = dict(row)
    if source == "messages":
        value["message_ref"] = MessageRef(manager.store.store_id, row["session_id"], row["id"], row["seq"],
            row["role"], row.get("tool_call_id"), row.get("send_index"), row.get("round_index")).to_dict()
    for key in ("id", "seq", "send_index", "source_seq_lo", "source_seq_hi", "compact_from_send", "compact_to_send"):
        if value.get(key) is not None:
            value[key] = str(_number(value[key], "record_identity", minimum=0 if key == "send_index" else 1))
    return value


async def _rows(manager, tx, sid, selection, upper, after, limit, *, forced_send):
    rows = await manager.store.history_records(sid, source=selection["source"], lo=selection["lo"], hi=upper,
        after=after, limit=limit, send_index=selection["send_index"], record_ids=selection["record_ids"],
        exclude_compact=forced_send, _fetchall=tx.fetchall)
    if selection["source"] == "projection" and forced_send:
        for row in rows:
            lo, hi = row["source_seq_lo"], row["source_seq_hi"]
            # A user projection legitimately has no source range. It belongs
            # to the already verified send; multi-send compact rows never do.
            if row["kind"] == "user" and lo is None and hi is None:
                continue
            lo, hi = _number(lo, "projection_source"), _number(hi, "projection_source")
            if lo > hi:
                raise JobError("task_history_projection_scope_mismatch")
            raw = await tx.fetchone(f"SELECT MIN(seq) AS lo,MAX(seq) AS hi,"
                "SUM(CASE WHEN send_index=? THEN 0 ELSE 1 END) AS outside "
                f"FROM {manager.store.t.messages} WHERE session_id=? AND seq>=? AND seq<=?",
                (selection["send_index"], sid, lo, hi))
            if raw["lo"] != lo or raw["hi"] != hi or raw["outside"]:
                raise JobError("task_history_projection_scope_mismatch")
    return rows


def _state(cursor, query, binding):
    try:
        if not isinstance(cursor, str) or not 0 < len(cursor) <= 2048:
            raise ValueError()
        value = json.loads(base64.urlsafe_b64decode(cursor+"="*(-len(cursor)%4)))
        if not isinstance(value, dict) or set(value) != {"query", "binding", "upper", "after", "snapshot_at"} or value["query"] != query:
            raise ValueError()
        if value["binding"] != binding:
            raise JobError("task_history_binding_changed")
        upper, after = _number(value["upper"], "cursor"), _number(value["after"], "cursor", minimum=0)
        if after > upper or type(value["snapshot_at"]) is not int or value["snapshot_at"] < 0:
            raise ValueError()
        return {**value, "upper": upper, "after": after}
    except JobError:
        raise
    except (ValueError, TypeError, UnicodeError) as exc:
        raise JobError("invalid_task_history_cursor") from exc


def _cursor(state, after):
    return base64.urlsafe_b64encode(dumps({**state, "upper": str(state["upper"]), "after": str(after)}).encode()).decode().rstrip("=")


async def read_history(manager, actor, job_id, *, session_id=None, source="messages", ids=None, seq=None, id=None,
                       message_id=None, message_ids=None, send_index=None, limit=20, cursor=None,
                       char_offset=None, max_chars=8000, output_file=None):
    """Read saved originals/projections with durable task authority on every call."""
    _text(job_id, "job", 64)
    if session_id is not None:
        _text(session_id, "session", 255)
    if type(limit) is not int or not 1 <= limit <= 100 or type(max_chars) is not int or not 1024 <= max_chars <= 32000:
        raise JobError("invalid_task_history_budget")
    selection = _selection(source, ids, seq, id, message_id, message_ids, send_index)
    offset = _number(char_offset, "char_offset", minimum=0) if char_offset is not None else 0
    if char_offset is not None and not selection["single"]:
        raise JobError("task_history_offset_requires_single_record")
    if output_file is not None:
        _text(output_file, "output_file", 4096)
        if cursor is not None or char_offset is not None:
            raise JobError("task_history_export_selection_conflict")
        from .env import RuntimeEnvError, get_runtime_env
        try:
            get_runtime_env().require_workspace_dir()
        except RuntimeEnvError as exc:
            raise JobError("task_history_workspace_required") from exc
    async with _snapshot(manager) as tx:
        scope = await _scope(manager, tx, actor, job_id, session_id, selection["record_ids"])
        sid, forced = scope["session_id"], scope["send_index"] is not None
        public_scope = {**scope, "send_index": str(scope["send_index"]) if forced else None,
            "bindings": [{**b, **({"send_index": str(b["send_index"])} if b.get("send_index") is not None else {})} for b in scope["bindings"]]}
        base = {"version": 1, "store_id": manager.store.store_id, **public_scope, "source": source, "execution_authority": False,
            "projection_content_mutable": source == "projection", "snapshot_kind": "coordinate_upper_bound"}
        if not scope["available"]:
            return _bounded({**base, "status": "unavailable"}, max_chars)
        if forced:
            if selection["send_index"] not in (None, scope["send_index"]):
                raise JobError("task_history_send_scope_mismatch")
            selection["send_index"] = scope["send_index"]
        binding = _hash(scope)
        query = _hash({"store": manager.store.store_id, "actor": [actor.tenant_id, actor.principal_id, actor.session_id],
            "job": job_id, "session": sid, "selection": selection})
        where, params, coord = _where(sid, selection, forced_send=forced)
        table = manager.store.t.messages if source == "messages" else manager.store.t.project_messages
        bounds = await tx.fetchone(f"SELECT MIN({coord}) AS lo,MAX({coord}) AS hi FROM {table} WHERE {where}", params)
        if bounds["hi"] is None:
            return _bounded({**base, "status": "unavailable", "reason": "records_not_found"}, max_chars)
        state = _state(cursor, query, binding) if cursor is not None else {
            "query": query, "binding": binding, "upper": bounds["hi"], "after": 0, "snapshot_at": manager.now()}
        if state["upper"] > bounds["hi"]:
            return _bounded({**base, "status": "unavailable", "reason": "records_changed"}, max_chars)
        rows = await _rows(manager, tx, sid, selection, state["upper"], state["after"], limit+1, forced_send=forced)
    if not rows:
        return _bounded({**base, "status": "unavailable", "reason": "records_not_found"}, max_chars)
    if output_file is not None:
        # Refuse an oversized receipt before publishing an otherwise successful
        # file. Retrying a failed response must not collide with a hidden export.
        _bounded({**base, "status": "exported", "path": output_file, "records": _EXPORT_MAX_RECORDS,
            "bytes": _EXPORT_MAX_BYTES, "upper": str(state["upper"]), "snapshot_at": state["snapshot_at"]}, max_chars)
        async def authorize(tx):
            await _verify_store(manager, tx)
            current = await _scope(manager, tx, actor, job_id, sid, selection["record_ids"])
            if not current["available"] or _hash(current) != binding:
                raise JobError("task_history_binding_changed")
        async def records(tx, after, size):
            return await _rows(manager, tx, sid, selection, state["upper"], after, size, forced_send=forced)
        try:
            result = json.loads(await _export(manager.store, sid, source, selection["lo"], state["upper"], selection["send_index"],
                output_file, authorize=authorize, read_records=records, transform=lambda row: _record(manager, source, row),
                max_records=_EXPORT_MAX_RECORDS, max_bytes=_EXPORT_MAX_BYTES, workspace_only=True))
        except (OSError, ValueError) as exc:
            if isinstance(exc, JobError):
                raise
            raise JobError("task_history_export_failed") from exc
        return _bounded({**base, **result, "upper": str(state["upper"]), "snapshot_at": state["snapshot_at"]}, max_chars)
    if selection["single"]:
        text = dumps(_screen(_record(manager, source, rows[0])))
        def envelope(size):
            piece = text[offset:offset+size]
            end = offset+len(piece)
            return {**base, "status": "ok", "format": "record_json_text", "text": piece, "total_chars": len(text),
                "next_char_offset": end if end < len(text) else None, "truncated": end < len(text)}
        left, right = 0, max(0, min(len(text)-offset, max_chars))
        while left < right:
            mid = (left+right+1)//2
            if len(dumps(envelope(mid))) <= max_chars:
                left = mid
            else:
                right = mid-1
        if not left and offset < len(text):
            raise JobError("task_history_budget_too_small")
        return _bounded(envelope(left), max_chars)
    shown = []
    for row in rows[:limit]:
        public = _record(manager, source, row)
        text = dumps(_screen(public))
        item = {key: public[key] for key in ("id", "seq", "send_index", "role", "name", "kind", "message_ref") if key in public}
        shown.append({**item, "preview": text[:400], "total_chars": len(text), "truncated": len(text)>400})
    while shown:
        more = len(shown) < len(rows)
        result = {**base, "status": "ok", "records": shown, "upper": str(state["upper"]), "snapshot_at": state["snapshot_at"],
            "next_cursor": _cursor(state, int(shown[-1][coord])) if more else None}
        if len(dumps(result)) <= max_chars:
            return result
        if len(shown) == 1:
            if shown[0]["preview"]:
                shown[0].update(preview="", truncated=True)
                continue
            raise JobError("task_history_budget_too_small")
        shown.pop()


def _bounded(value, budget):
    if len(dumps(value)) > budget:
        raise JobError("task_history_budget_too_small")
    return value
