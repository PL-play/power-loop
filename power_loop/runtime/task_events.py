"""Bounded reads of operation references and immutable canonical task results.

Only the public operation envelope is exposed. Execution/request payloads can
contain leases, frozen prompts and dispatch tokens and are never a log view.
"""
from __future__ import annotations

import base64
import json

from .jobs import JobError, _hash
from .recall import _export, _screen, dumps
from .result_revisions import result_id
from .task_history import _bounded, _number, _snapshot, _text, _verify_store
from .task_query import _reference

_MAX_EVENTS = 5000
_MAX_BYTES = 16 * 1024 * 1024
_HEADERS = "event_id,job_id,event_type,job_version,action,caller_session_id,tool_call_id,call_ref_json,receipt_ref_json,delivery_status,created_at"


def _key(event):
    return event["created_at"], event["event_id"]


def _generation(job_id, event_id, *, initial=1):
    first = result_id(job_id, 1)
    if event_id == first:
        return initial
    if not event_id.startswith(first+"_g"):
        raise JobError("task_event_result_corrupt")
    value = event_id[len(first)+2:]
    result = _number(value, "result_generation")
    if result <= 1 or result_id(job_id, result) != event_id:
        raise JobError("task_event_result_corrupt")
    return result


def _header(manager, job, event):
    if event["job_id"] != job["job_id"] or event["event_type"] not in ("operation", "result", "native_callback_returned"):
        raise JobError("task_event_scope_mismatch")
    version = _number(event["job_version"], "event_revision")
    created = _number(event["created_at"], "event_time", minimum=0)
    data = {"store_id": manager.store.store_id, "job_id": job["job_id"], "event_id": event["event_id"],
        "event_type": event["event_type"], "revision": str(version), "created_at": str(created),
        "delivery_status": event["delivery_status"], "execution_authority": False}
    if event['event_type'] == 'native_callback_returned':
        return {**data, 'callback_returned': True, 'resource_stop_proven': False, 'billing_settled': False}
    if event["event_type"] == "result":
        return {**data, "generation": str(_generation(job["job_id"], event["event_id"], initial=0 if job["generation"] == 0 else 1))}
    # Later operations may originate from a different send/SID than creation.
    origin = {**job, "created_by_session_id": event["caller_session_id"]}
    call = _reference(manager, origin, event["call_ref_json"], "assistant")
    receipt = _reference(manager, origin, event["receipt_ref_json"], "tool")
    if call is not None and (not isinstance(event["tool_call_id"], str) or not 0 < len(event["tool_call_id"]) <= 255):
        raise JobError("task_event_origin_corrupt")
    if receipt is not None and (call is None or receipt["session_id"] != call["session_id"] or
            receipt["send_index"] != call["send_index"] or receipt["tool_call_id"] != event["tool_call_id"]):
        raise JobError("task_event_origin_corrupt")
    return {**data, "operation_id": event["event_id"], "action": event["action"],
        "origin": "model" if call else "system", "call_ref": call, "receipt_ref": receipt,
        "receipt_status": "available" if receipt else "missing" if call else "not_applicable"}


def _query(manager, actor, job, view):
    return _hash([manager.store.store_id, actor.tenant_id, actor.principal_id, actor.session_id,
        job["job_id"], job["parent_job_id"], job["root_job_id"], job["created_by_session_id"], view])


def _decode(cursor, query):
    try:
        if not isinstance(cursor, str) or not 0 < len(cursor) <= 2048:
            raise ValueError()
        value = json.loads(base64.urlsafe_b64decode(cursor+"="*(-len(cursor)%4)))
        if not isinstance(value, dict) or set(value) != {"query", "upper", "after", "membership"} or value["query"] != query:
            raise ValueError()
        for name in ("upper", "after"):
            pair = value[name]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError()
            pair[0] = _number(pair[0], "event_cursor", minimum=0)
            _text(pair[1], "event_cursor", 64)
        if tuple(value["after"]) > tuple(value["upper"]) or not isinstance(value["membership"], str):
            raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError) as exc:
        raise JobError("invalid_task_event_cursor") from exc


def _cursor(state, after):
    return base64.urlsafe_b64encode(dumps({**state, "after": list(after)}).encode()).decode().rstrip("=")


async def _events(manager, tx, job, kind, upper=None):
    rows = await tx.fetchall(f"SELECT {_HEADERS} FROM {manager.events} WHERE job_id=? AND event_type=? "
        "ORDER BY created_at,event_id LIMIT ?", (job["job_id"], kind, _MAX_EVENTS+1))
    if len(rows) > _MAX_EVENTS:
        raise JobError("task_event_scope_too_large")
    return [row for row in rows if upper is None or _key(row) <= tuple(upper)]


async def _public_header(manager, tx, job, event):
    public = _header(manager, job, event)
    if event["event_type"] == "operation" and event["action"] == "nudge":
        from .input_dispositions import identity, read_tx, summary
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (identity(manager, event["event_id"]),)):
            original = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event["event_id"],))
            proof = await read_tx(manager, tx, original, required=True)
            public["input_disposition"] = summary(proof)
    return public


async def _detail(manager, tx, job, event):
    public = await _public_header(manager, tx, job, event)
    if event['event_type'] == 'native_callback_returned':
        from .managed_effects import _ref
        from .native_returns import _LIMIT, read
        from .pending_calls import _object
        from .tool_call_plans import read as read_plan
        raw = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (event['event_id'],), _LIMIT)
        if raw is None or not isinstance(raw.get('call_ref'), dict):
            raise JobError('native_return_checkpoint_changed')
        ref = _ref(raw['call_ref'])
        plan = await read_plan(manager, tx, job, ref)
        entries = [entry for entry in plan['entries'] if entry['call_id'] == raw.get('call_id')]
        if len(entries) != 1:
            raise JobError('native_return_checkpoint_changed')
        saved, value = await read(manager, tx, job, ref, entries[0], plan)
        if saved['event_id'] != event['event_id']:
            raise JobError('native_return_checkpoint_changed')
        # Expose the exact returned data and message/operation references,
        # never the internal executor/boot lease or frozen execution scope.
        return {**public, 'call_ref': ref.to_dict(), 'tool_call_id': value['call_id'],
                'return_text': value['return_text'], 'operations': value['operations']}
    if event["event_type"] == "operation":
        return public  # Never expose operation request/lease/dispatch payloads.
    raw = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_id=? AND event_type='result'",
        (job["job_id"], event["event_id"]))
    if raw is None:
        raise JobError("task_event_missing")
    try:
        value = json.loads(raw["payload_json"])
        generation = value.get("generation", 0 if value.get("unstarted") else 1)
        if not isinstance(value, dict) or type(generation) is not int or str(generation) != public["generation"] or not isinstance(value.get("result"), dict):
            raise ValueError()
        return {**public, "outcome": value.get("outcome"), "resource_state": value.get("resource_state"), "result": value["result"]}
    except (ValueError, TypeError, AttributeError) as exc:
        raise JobError("task_event_result_corrupt") from exc


async def read_events(manager, actor, job_id, *, view, event_id=None, limit=20, cursor=None,
                      char_offset=None, max_chars=8000, output_file=None):
    """List public references, or read/export selected saved events, without ACKs."""
    _text(job_id, "job", 64)
    if view not in ("operations", "results", "returns"):
        raise JobError("invalid_task_event_view")
    if type(limit) is not int or not 1 <= limit <= 100 or type(max_chars) is not int or not 1024 <= max_chars <= 32000:
        raise JobError("invalid_task_event_budget")
    if event_id is not None:
        _text(event_id, "event", 64)
    if char_offset is not None and event_id is None:
        raise JobError("task_event_offset_requires_event")
    offset = _number(char_offset, "char_offset", minimum=0) if char_offset is not None else 0
    if cursor is not None and event_id is not None:
        raise JobError("task_event_selection_conflict")
    if output_file is not None:
        _text(output_file, "output_file", 4096)
        if cursor is not None or char_offset is not None:
            raise JobError("task_event_export_selection_conflict")
        from .env import RuntimeEnvError, get_runtime_env
        try:
            get_runtime_env().require_workspace_dir()
        except RuntimeEnvError as exc:
            raise JobError("task_event_workspace_required") from exc
    kind = {"operations": "operation", "results": "result", "returns": "native_callback_returned"}[view]
    async with _snapshot(manager) as tx:
        job = await manager._authorize(tx, actor, job_id)
        query = _query(manager, actor, job, view)
        state = _decode(cursor, query) if cursor is not None else None
        if event_id is not None:
            row = await tx.fetchone(f"SELECT {_HEADERS} FROM {manager.events} WHERE job_id=? AND event_id=? AND event_type=?", (job_id, event_id, kind))
            events = [row] if row else []
        else:
            events = await _events(manager, tx, job, kind, state["upper"] if state else None)
        membership = _hash([list(_key(row)) for row in events])
        if state is not None and state["membership"] != membership:
            raise JobError("task_event_snapshot_changed")
        base = {"version": 1, "backend": "task_registry", "store_id": manager.store.store_id, "job_id": job_id, "view": view,
            "execution_authority": False, "read_only": True}
        if not events:
            return _bounded({**base, "status": "unavailable", "reason": "events_not_found"}, max_chars)
        state = state or {"query": query, "upper": list(_key(events[-1])), "after": [0, ""], "membership": membership}
        if output_file is None and event_id is not None:
            value = await _detail(manager, tx, job, events[0])
        elif output_file is None:
            remaining = [event for event in events if _key(event) > tuple(state["after"])]
            shown = remaining[:limit]
            headers = [await _public_header(manager, tx, job, row) for row in shown]
            while True:
                result = {**base, "status": "ok", "events": headers,
                    "matched_count": len(events), "truncated": len(shown) < len(remaining),
                    "next_cursor": _cursor(state, _key(shown[-1])) if shown and len(shown) < len(remaining) else None}
                if len(dumps(result)) <= max_chars:
                    return result
                if len(shown) <= 1:
                    raise JobError("task_event_budget_too_small")
                shown.pop()
                headers.pop()
    if output_file is not None:
        _bounded({**base, "status": "exported", "path": output_file, "records": _MAX_EVENTS, "bytes": _MAX_BYTES}, max_chars)
        async def authorize(tx):
            await _verify_store(manager, tx)
            current = await manager._authorize(tx, actor, job_id)
            if _query(manager, actor, current, view) != query:
                raise JobError("task_event_scope_mismatch")
        async def records(tx, after, size):
            current = await manager._authorize(tx, actor, job_id)
            result = []
            for index, original in enumerate(events[after:after+size], after+1):
                actual = await tx.fetchone(f"SELECT {_HEADERS} FROM {manager.events} WHERE job_id=? AND event_id=? AND event_type=?",
                    (job_id, original["event_id"], kind))
                if actual is None or _key(actual) != _key(original) or actual["job_version"] != original["job_version"]:
                    raise JobError("task_event_snapshot_changed")
                result.append({"id": index, "event": await _detail(manager, tx, current, actual)})
            return result
        try:
            result = json.loads(await _export(manager.store, actor.session_id, "task_events", 0, None, None, output_file,
                authorize=authorize, read_records=records, transform=lambda row: row["event"], max_records=_MAX_EVENTS,
                max_bytes=_MAX_BYTES, workspace_only=True))
        except (ValueError, OSError) as exc:
            if isinstance(exc, JobError):
                raise
            raise JobError("task_event_export_failed") from exc
        return _bounded({**base, **result}, max_chars)
    text = dumps(_screen(value))
    def envelope(size):
        end = offset+size
        return {**base, "event_id": event_id, "status": "ok", "format": "event_json_text", "text": text[offset:end],
            "total_chars": len(text), "next_char_offset": end if end < len(text) else None, "truncated": end < len(text)}
    low, high = 0, min(max_chars, max(0, len(text)-offset))
    while low < high:
        mid = (low+high+1)//2
        if len(dumps(envelope(mid))) <= max_chars:
            low = mid
        else:
            high = mid-1
    if not low and offset < len(text):
        raise JobError("task_event_budget_too_small")
    return _bounded(envelope(low), max_chars)
