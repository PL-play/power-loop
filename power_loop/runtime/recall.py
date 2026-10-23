"""Bounded audit reads and exclusive JSONL exports; never runs recalled tools."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class RecallInputError(ValueError):
    """Library-owned validation, distinct from arbitrary backend ValueError."""


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def authorize(store: Any, parent: str, child: str | None) -> tuple[str, str | None]:
    if child is None:
        return parent, None
    row = await store.get_session(child)
    if row is not None:
        if row.parent_session_id != parent:
            raise ValueError("access_denied: sub_session must be your direct child")
        return child, None
    ownership = await store.get_runtime_state(parent, "subagent:child:" + child, default=None)
    if isinstance(ownership, dict) and ownership.get("deleted"):
        return child, "session_deleted"
    raise ValueError("access_denied: no authorized child session")


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RecallInputError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def _screen(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"data:[^\s\"']*?;base64,[A-Za-z0-9+/=]+", "[inline media omitted]", value)
    if isinstance(value, dict):
        return {k: _screen(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_screen(v) for v in value]
    return value


async def recall(
    store: Any, parent: str, *, sub_session: str | None = None, source: str = "messages",
    ids: list[int] | None = None, seq: int | None = None, id: int | None = None,
    send_index: int | None = None, limit: int = 20, cursor: str | None = None,
    char_offset: int | None = None, char_limit: int | None = None,
    output_file: str | None = None,
    _scope_id: str | None = None, _export_options: dict | None = None, _strict_errors: bool = False,
    _all_records: bool = False,
) -> str:
    """Return valid JSON envelopes; cursor binds the selection and captured upper bound."""
    try:
        sid, unavailable = await authorize(store, parent, sub_session)
        if unavailable:
            return dumps({"status": "unavailable", "reason": unavailable, "session_id": sid})
        if source not in {"messages", "projection"}:
            raise RecallInputError("source must be messages or projection")
        limit = _integer(limit, "limit", 1, 100)
        offset = _integer(0 if char_offset is None else char_offset, "char_offset", 0, 2**63 - 1)
        budget = _integer(8000 if char_limit is None else char_limit, "char_limit", 256, 32000)
        if seq is not None and source != "messages" or id is not None and source != "projection":
            raise RecallInputError("seq selects messages; id selects projection")
        single = seq if source == "messages" else id
        if ids is not None and (single is not None or send_index is not None):
            raise RecallInputError("choose ids OR a single record OR send_index")
        if ids is None and single is None and send_index is None and not _all_records:
            raise RecallInputError("select ids=[lo,hi], seq (messages), id (projection), or send_index")
        if send_index is not None:
            _integer(send_index, "send_index", 1, 2**63 - 1)
        lo, hi = 0, None
        if ids is not None:
            if not isinstance(ids, list) or len(ids) != 2:
                raise RecallInputError("ids must be [inclusive_start,inclusive_end]")
            lo, hi = [_integer(x, "ids", 1, 2**63 - 1) for x in ids]
            if lo > hi:
                raise RecallInputError("ids start exceeds end")
        if single is not None:
            lo = hi = _integer(single, "record coordinate", 1, 2**63 - 1)
        elif offset:
            raise RecallInputError("char_offset requires seq or id")
        if output_file is not None and not str(output_file).strip():
            raise RecallInputError("output_file must be a new workspace path")
        if output_file is not None and (cursor is not None or char_offset is not None or char_limit is not None):
            raise RecallInputError("output_file exports full records; omit cursor/char_offset/char_limit")
        coord = "seq" if source == "messages" else "id"
        selection = {"session_id": sid, "source": source, "ids": ids, "single": single,
                     "send_index": send_index}
        if _scope_id is not None:
            selection["authority"] = _scope_id
        signature = hashlib.sha256(dumps(selection).encode()).hexdigest()
        after = 0
        if cursor:
            try:
                state = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if state["selection"] != signature:
                    raise ValueError
                after = _integer(state["after"], "cursor", 0, 2**63 - 1)
                hi = _integer(state["hi"], "cursor upper bound", 1, 2**63 - 1)
                if ids is not None and hi > ids[1] or after < lo - 1:
                    raise ValueError
            except (ValueError, TypeError, KeyError, UnicodeError):
                raise RecallInputError("invalid cursor for this session/selection") from None
        else:
            bounds = await store.history_bounds(sid, source=source)
            if bounds is None:
                return dumps({"status": "unavailable", "reason": "records_not_found"})
            hi = min(hi, bounds[1]) if hi is not None else bounds[1]
        rows = await store.history_records(sid, source=source, lo=lo, hi=hi, after=after,
                                           limit=limit + 1, send_index=send_index)
        if not rows:
            return dumps({"status": "unavailable", "reason": "records_not_found"})
        if output_file:
            return await _export(store, sid, source, lo, hi, send_index, output_file, **(_export_options or {}))
        if single is not None:
            text = dumps(_screen(rows[0]))
            def envelope(size):
                piece = text[offset:offset + size]
                end = offset + len(piece)
                return dumps({"status": "ok", "session_id": sid, "source": source, coord: single,
                              "format": "record_json_text", "text": piece, "total_chars": len(text),
                              "next_char_offset": end if end < len(text) else None,
                              "truncated": end < len(text)})
            left, right = 0, min(budget, max(0, len(text) - offset))
            while left < right:
                middle = (left + right + 1) // 2
                if len(envelope(middle)) <= budget:
                    left = middle
                else:
                    right = middle - 1
            if not left and offset < len(text):
                raise RecallInputError("char_limit is too small for record metadata; increase it")
            return envelope(left)
        shown = []
        # Reserve room for envelope/cursor. Each preview includes ALL large fields in its budget.
        used = 0
        for row in rows[:limit]:
            item = {coord: row[coord], "send_index": row.get("send_index")}
            for key in ("role", "kind", "source_seq_lo", "source_seq_hi", "projector_version"):
                if key in row:
                    item[key] = row[key]
            text = dumps(_screen(row))
            room = max(0, budget - used - len(dumps(item)) - 80)
            if room < 80 and shown:
                break
            item.update(preview=text[:min(room, 500)], total_chars=len(text),
                        truncated=len(text) > min(room, 500))
            shown.append(item)
            used += len(dumps(item))
        while shown:
            last = shown[-1][coord]
            more = len(shown) < len(rows)
            next_cursor = base64.urlsafe_b64encode(dumps(
                {"selection": signature, "after": last, "hi": hi}).encode()).decode() if more else None
            answer = dumps({"status": "ok", "session_id": sid, "source": source,
                            "records": shown, "next_cursor": next_cursor})
            if len(answer) <= budget:
                return answer
            if len(shown) == 1:
                if shown[0]["preview"]:
                    shown[0]["preview"] = ""
                    shown[0]["truncated"] = True
                    continue
                raise RecallInputError("char_limit is too small for list metadata; increase it")
            shown.pop()
    except (ValueError, OSError) as exc:
        if _strict_errors:
            raise
        return dumps({"status": "error", "error": str(exc)})


async def _export(store, sid, source, lo, hi, send_index, output_file, *,
                  authorize=None, read_records=None, transform=None,
                  max_records=None, max_bytes=None, workspace_only=False,
                  transaction=None, configure_snapshot=True):
    from power_loop.runtime.env import safe_path

    directory, filename = os.path.split(output_file)
    if not filename or filename in {".", ".."}:
        raise RecallInputError("output_file must name a new file")
    # Resolve/check the parent but preserve the final component: a dangling symlink is an
    # existing target too, not permission to write through it to a different path.
    target = safe_path(directory or ".", "w") / filename
    checked = safe_path(str(target), "w")
    if workspace_only:
        from power_loop.runtime.env import get_runtime_env
        if not checked.is_relative_to(get_runtime_env().require_workspace_dir().resolve()):
            raise RecallInputError("output_file must stay inside the workspace")
    if target.exists() or target.is_symlink():
        raise RecallInputError("output_file already exists; choose a new path")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".recall-", dir=target.parent)
    stream = os.fdopen(fd, "wb")
    count = size = 0
    try:
        # One read snapshot, no transaction spans an LLM/network request. Stream batches to disk.
        async with (transaction if transaction is not None else store._db.transaction()) as tx:
            if configure_snapshot and store._db.dialect.name == "postgres":
                await tx.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            if authorize is not None:
                await authorize(tx)
            with stream:
                after = 0
                while True:
                    batch = await read_records(tx, after, 100) if read_records is not None else await store.history_records(
                        sid, source=source, lo=lo, hi=hi, after=after, limit=100,
                        send_index=send_index, _fetchall=tx.fetchall)
                    if not batch:
                        break
                    for row in batch:
                        value = transform(row) if transform is not None else row
                        encoded = (dumps({"source": source, **value}) + "\n").encode("utf-8")
                        if max_records is not None and count >= max_records or max_bytes is not None and size+len(encoded) > max_bytes:
                            raise RecallInputError("export_limit_exceeded")
                        stream.write(encoded)
                        size += len(encoded)
                        count += 1
                    after = batch[-1]["seq" if source == "messages" else "id"]
                stream.flush()
                os.fsync(stream.fileno())
        if not count:
            raise RecallInputError("records_not_found")
        # link is atomic and fails if another writer created the target; never replace it.
        os.link(temporary, target)
    finally:
        stream.close()
        Path(temporary).unlink(missing_ok=True)
    return dumps({"status": "exported", "path": output_file, "records": count, "bytes": size})
