"""Inline session-row fold statistics (Design 217 M0 Decision 2 / M1b).

All fold aggregates and bounded cross-send deduplication keys (12-hex SHA256 -> send_index,
capped at 512 entries, < 10 KB) live inline in ``sessions.fold_stats_json``.
Zero reads or writes to ``session_runtime_state``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.fold import fold_due
from power_loop.runtime.store.types import ProjectMessageRow

logger = logging.getLogger(__name__)
# Retained constants for export filtering of legacy runtime_state rows; NEVER written by M1b.
STATE_KEY = "__pl_projection_fold"
OWNER_PREFIX = STATE_KEY + "/"
PAGE_SIZE = 256
FOLD_DEDUP_MAX_KEYS = 512


class _ProjectionSnapshotConflict(Exception):
    """Only an explicit zero-row conditional statistics write permits the cold retry."""


@dataclass(frozen=True)
class ProjectionFoldSnapshot:
    prior: ProjectMessageRow | None
    rows: list[ProjectMessageRow]
    revision: str


def fold_dedup_hash(key: Any) -> str:
    """12-hex SHA256 short hash for inline ``sessions.fold_stats_json`` dedup map."""
    if isinstance(key, str):
        raw = key.encode("utf-8")
    else:
        raw = json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _encode_state(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def parse_state(value_json: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse ``sessions.fold_stats_json``; malformed statistics are rebuilt, never trusted."""
    if isinstance(value_json, dict):
        return value_json
    try:
        value = json.loads(value_json) if value_json is not None else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def update_fold_stats(
    existing: dict[str, Any] | None,
    *,
    send_index: int,
    dedup_keys: Sequence[Any] = (),
    token_delta: int = 0,
    row_delta: int = 2,
    compact_reset: bool = False,
    max_keys: int = FOLD_DEDUP_MAX_KEYS,
) -> dict[str, Any]:
    """Pure update helper for inline ``sessions.fold_stats_json`` (v3, 512-key cap, 0 KV SQL).

    - Stores ``dedup`` as ``{12-hex-sha256: latest_send_index}`` capped at ``max_keys`` (512).
    - Evicts oldest ``send_index`` entries on overflow (deterministic tie-break by hash).
    - Clears ``dedup`` completely when ``compact_reset=True``.
    """
    base = dict(existing) if isinstance(existing, dict) else {}
    if compact_reset:
        return {
            "v": 3,
            "sends": 0,
            "rows": 0,
            "tokens": max(0, int(token_delta)),
            "maximum": int(send_index),
            "compact_to_send": int(send_index),
            "dedup": {},
            "revision": secrets.token_hex(8),
        }
    dedup: dict[str, int] = {}
    raw_dedup = base.get("dedup")
    if isinstance(raw_dedup, dict):
        for k, v in raw_dedup.items():
            dedup[str(k)] = int(v)
    for key in dedup_keys:
        if key:
            dedup[fold_dedup_hash(key)] = int(send_index)
    if len(dedup) > max_keys:
        keep = sorted(dedup.items(), key=lambda kv: (kv[1], kv[0]))[-max_keys:]
        dedup = dict(keep)
    out: dict[str, Any] = {
        "v": 3,
        "sends": int(base.get("sends") or 0) + 1,
        "rows": int(base.get("rows") or 0) + int(row_delta),
        "tokens": max(0, int(base.get("tokens") or 0) + int(token_delta)),
        "maximum": max(int(base.get("maximum") or 0), int(send_index)),
        "compact_to_send": base.get("compact_to_send"),
        "dedup": dedup,
        "revision": secrets.token_hex(8),
    }
    if "policy" in base:
        out["policy"] = base["policy"]
    if "store_id" in base:
        out["store_id"] = base["store_id"]
    return out


async def read_state(store, reader, sid: str) -> dict[str, Any] | None:
    row = await reader.fetchone(
        f"SELECT fold_stats_json FROM {store.t.sessions} WHERE session_id=?",
        (sid,),
    )
    return parse_state(row["fold_stats_json"] if row else None)


async def _put_state(store, writer, sid: str, value: dict[str, Any]) -> None:
    await writer.execute(
        f"UPDATE {store.t.sessions} SET fold_stats_json=? WHERE session_id=?",
        (_encode_state(value), sid),
    )


def _invalid(state: dict[str, Any] | None, *, compact_to_send: int | None = None, clear_dedup: bool = False) -> dict[str, Any]:
    prev_compact = compact_to_send if compact_to_send is not None else (
        state.get("compact_to_send") if isinstance(state, dict) else None
    )
    prev_dedup = {} if clear_dedup else (
        dict(state.get("dedup") or {}) if isinstance(state, dict) and isinstance(state.get("dedup"), dict) else {}
    )
    return {
        "v": 3,
        "invalid": True,
        "revision": secrets.token_hex(8),
        "compact_to_send": prev_compact,
        "dedup": prev_dedup,
    }


async def invalidate(store, tx, sid: str, state: dict[str, Any] | None = None, *,
                     compact_to_send: int | None = None, clear_dedup: bool = False) -> None:
    """Caller holds the session lock; all projection mutations invalidate the same revision."""
    if state is None:
        state = await read_state(store, tx, sid)
    await _put_state(store, tx, sid, _invalid(state, compact_to_send=compact_to_send, clear_dedup=clear_dedup))


def _keys(rows: Sequence[ProjectMessageRow], policy) -> set[Any]:
    if policy is None or not policy.deduplicate:
        return set()
    return {
        t["k"]
        for r in rows
        if r.kind == "project" and isinstance(r.content, dict)
        for t in (r.content.get("tools") or [])
        if isinstance(t, dict) and t.get("k")
    }


def _policy_tag(policy) -> str:
    return policy.fingerprint[:16]


def _cost(projector, row: ProjectMessageRow, maximum: int, owners: dict[Any, int], policy) -> int:
    # Rendering annotations never mutate the caller's persisted/projected rows.
    row_copy = copy.copy(row)
    row_copy.recency = max(0, maximum - row_copy.send_index)
    row_copy.render_ctx = {"latest_key": owners, "hot_window": policy.hot_window}
    return estimate_tokens(projector.render([row_copy]))


def _base_state(store, sid: str, policy, cutoff: int | None) -> dict[str, Any]:
    return {
        "v": 3,
        "store_id": store.store_id,
        "policy": _policy_tag(policy),
        "compact_to_send": cutoff,
        "tokens": 0,
        "rows": 0,
        "sends": 0,
        "maximum": 0,
        "dedup": {},
        "revision": secrets.token_hex(8),
    }


def _state_ok(store, sid: str, state: dict[str, Any] | None, policy) -> bool:
    """Statistics written for this store/policy by v3 inline code path."""
    if not isinstance(state, dict) or state.get("v") != 3 or state.get("invalid"):
        return False
    if state.get("store_id") != store.store_id or state.get("policy") != _policy_tag(policy):
        return False
    if not isinstance(state.get("dedup"), dict):
        return False
    return all(
        type(state.get(k)) is int and state[k] >= 0
        for k in ("tokens", "rows", "sends", "maximum")
    )


def _matches(store, sid: str, state: dict[str, Any] | None, policy, cutoff: int | None) -> bool:
    return _state_ok(store, sid, state, policy) and state.get("compact_to_send") == cutoff


def _owners_from_dedup(dedup: dict[str, int], keys: set[Any]) -> dict[Any, int]:
    owners: dict[Any, int] = {}
    for key in keys:
        h = fold_dedup_hash(key)
        if h in dedup:
            owners[key] = int(dedup[h])
    return owners


def _cap_dedup(dedup: dict[str, int], max_keys: int = FOLD_DEDUP_MAX_KEYS) -> dict[str, int]:
    if len(dedup) <= max_keys:
        return dedup
    keep = sorted(dedup.items(), key=lambda kv: (kv[1], kv[0]))[-max_keys:]
    return dict(keep)


def compute_inline_fold_stats(
    store,
    sid: str,
    *,
    existing_raw: str | dict[str, Any] | None,
    projector,
    policy,
    new_rows: list[ProjectMessageRow],
    send_index: int,
    window=None,
) -> tuple[dict[str, Any] | None, bool]:
    """Pure in-memory inline ``fold_stats_json`` update (0 SQL!).

    Returns ``(updated_state, needs_db_fallback)``. On the healthy fast path (cold start,
    non-dedup/zero-hot-window renderer, or when all affected rows live inside ``window``),
    computes exact v3 statistics in memory so ``Final`` writes ``sessions.fold_stats_json``
    in the SAME ``UPDATE sessions`` statement with 0 extra transactions and 0 KV SQL.
    """
    if policy is None or not new_rows or not all(r.kind in ("user", "project") for r in new_rows):
        return None, True

    cutoff = window.stats.compact_to_send if (window is not None and window.stats is not None) else None
    existing = parse_state(existing_raw)
    if existing is None:
        non_seed = [r for r in (window.rows if window is not None else ()) if r.kind in ("user", "project")]
        if not non_seed:
            state = _base_state(store, sid, policy, cutoff)
        else:
            return None, True
    elif _matches(store, sid, existing, policy, cutoff) and send_index > existing["maximum"]:
        state = existing
    else:
        return None, True

    dedup = {str(k): int(v) for k, v in (state.get("dedup") or {}).items()}
    new_keys = _keys(new_rows, policy)
    owners = _owners_from_dedup(dedup, new_keys)

    cutoff_i = cutoff if cutoff is not None else -1
    lower = max(cutoff_i, state["maximum"] - policy.hot_window)
    upper = min(state["maximum"], send_index - policy.hot_window)
    owner_sends = {send for send in owners.values() if cutoff_i < send <= state["maximum"]}
    window_lo = window.window_lo if window is not None else send_index

    # Verify all affected rows are already covered by window in memory; otherwise fall back.
    if policy.hot_window and lower < min(upper, window_lo - 1):
        return None, True
    if any(s < window_lo for s in owner_sends):
        return None, True

    changed: list[ProjectMessageRow] = []
    if window is not None and (owner_sends or (policy.hot_window and upper >= window_lo and lower < upper)):
        found: dict[tuple[int, str], ProjectMessageRow] = {}
        for row in window.rows:
            if window_lo <= row.send_index < send_index and (
                (policy.hot_window and lower < row.send_index <= upper and row.kind in ("user", "project"))
                or (row.kind == "project" and row.send_index in owner_sends)
            ):
                found[(row.send_index, row.kind)] = row
        changed = list(found.values())

    old_keys = _keys(changed, policy)
    owners.update(_owners_from_dedup(dedup, old_keys - owners.keys()))
    # Evicted keys (>512 cap) conservatively default to the row's own send_index (never error).
    for row in changed:
        for key in _keys([row], policy):
            owners.setdefault(key, row.send_index)

    before = sum(_cost(projector, row, state["maximum"], owners, policy) for row in changed)
    owners.update(dict.fromkeys(new_keys, send_index))
    after = sum(_cost(projector, row, send_index, owners, policy) for row in [*changed, *new_rows])

    for key in new_keys:
        dedup[fold_dedup_hash(key)] = send_index
    dedup = _cap_dedup(dedup)

    new_tokens = max(0, state["tokens"] + after - before)
    updated = {
        **state,
        "tokens": new_tokens,
        "rows": state["rows"] + len(new_rows),
        "sends": state["sends"] + int(any(r.kind in ("user", "project") for r in new_rows)),
        "maximum": send_index,
        "dedup": dedup,
        "revision": secrets.token_hex(8),
    }
    return updated, False


async def _changed_rows(store, tx, sid: str, state: dict[str, Any], policy, maximum: int,
                        owners: dict[Any, int], window=None) -> list[ProjectMessageRow]:
    """Only rows becoming cold, or the previous visible owner of a new dedup key, can change."""
    from .store import _row_to_project_message

    found: dict[tuple[int, str], ProjectMessageRow] = {}
    cutoff = state["compact_to_send"] if state.get("compact_to_send") is not None else -1
    lower = max(cutoff, state["maximum"] - policy.hot_window)
    upper = min(state["maximum"], maximum - policy.hot_window)
    indices = {send for send in owners.values() if cutoff < send <= state["maximum"]}
    window_lo = window.window_lo if window is not None else maximum
    covered_owners = {send for send in indices if window_lo <= send < maximum}
    if window is not None and (covered_owners or (policy.hot_window and upper >= window_lo and lower < upper)):
        for row in window.rows:
            if window_lo <= row.send_index < maximum and (
                (policy.hot_window and lower < row.send_index <= upper and row.kind in ("user", "project"))
                or (row.kind == "project" and row.send_index in covered_owners)
            ):
                found[(row.send_index, row.kind)] = row
    outside_upper = min(upper, window_lo - 1)
    if policy.hot_window and lower < outside_upper:
        rows = await tx.fetchall(
            f"SELECT * FROM {store.t.project_messages} WHERE session_id=? AND send_index>? "
            "AND send_index<=? AND kind IN ('user','project')",
            (sid, lower, outside_upper),
        )
        found.update({(r["send_index"], r["kind"]): _row_to_project_message(r) for r in rows})
    indices_list = sorted(indices - covered_owners)
    for offset in range(0, len(indices_list), PAGE_SIZE):
        page = indices_list[offset:offset + PAGE_SIZE]
        rows = await tx.fetchall(
            f"SELECT * FROM {store.t.project_messages} WHERE session_id=? AND kind='project' "
            f"AND send_index IN ({','.join('?' for _ in page)})",
            (sid, *page),
        )
        found.update({(r["send_index"], r["kind"]): _row_to_project_message(r) for r in rows})
    return list(found.values())


async def _advance(store, tx, sid: str, state: dict[str, Any], projector, policy,
                   new_rows: list[ProjectMessageRow], send_index: int, window=None) -> dict[str, Any]:
    dedup = {str(k): int(v) for k, v in (state.get("dedup") or {}).items()}
    new_keys = _keys(new_rows, policy)
    owners = _owners_from_dedup(dedup, new_keys)
    changed = await _changed_rows(store, tx, sid, state, policy, send_index, owners, window=window)
    old_keys = _keys(changed, policy)
    owners.update(_owners_from_dedup(dedup, old_keys - owners.keys()))
    for row in changed:
        for key in _keys([row], policy):
            owners.setdefault(key, row.send_index)
    before = sum(_cost(projector, row, state["maximum"], owners, policy) for row in changed)
    owners.update(dict.fromkeys(new_keys, send_index))
    after = sum(_cost(projector, row, send_index, owners, policy) for row in [*changed, *new_rows])
    for key in new_keys:
        dedup[fold_dedup_hash(key)] = send_index
    dedup = _cap_dedup(dedup)
    new_tokens = max(0, state["tokens"] + after - before)
    return {
        **state,
        "tokens": new_tokens,
        "rows": state["rows"] + len(new_rows),
        "sends": state["sends"] + int(any(r.kind in ("user", "project") for r in new_rows)),
        "maximum": send_index,
        "dedup": dedup,
    }


async def persist(store, sid: str, *, send_index: int, rows, source_seq_lo, source_seq_hi,
                  projector_version: int, projector=None, policy=None, ownership=None, window=None):
    """Atomically append projections and advance/invalidate ``sessions.fold_stats_json``.

    Zero ``session_runtime_state`` reads or writes. Used by standalone/repair/fold callers;
    normal sends write projections + ``fold_stats_json`` inline inside ``RuntimeInputStore.final``.
    """
    import time

    from .store import _dumps, _nul_safe

    new_rows = [
        ProjectMessageRow(
            session_id=sid, send_index=send_index, kind=kind,
            content=json.loads(_dumps(content if content is not None else {})),
            rendered_text=_nul_safe(text), source_seq_lo=source_seq_lo, source_seq_hi=source_seq_hi,
            compact_from_send=None, compact_to_send=None, projector_version=projector_version,
            token_estimate=None, created_at=0, completed_through_send_index=send_index,
        )
        for kind, content, text in rows
    ]
    now = time.time_ns() // 1_000_000
    header = window.stats if window is not None else None
    snapshot = parse_state(header.value_json) if header is not None else None
    incremental = (
        policy is not None
        and bool(new_rows)
        and all(row.kind in ("user", "project") for row in new_rows)
    )
    healthy = (
        incremental
        and header is not None
        and header.row_exists
        and _matches(store, sid, snapshot, policy, header.compact_to_send)
        and send_index > snapshot["maximum"]
    )
    first = incremental and header is not None and not header.row_exists and send_index == 1

    async def save(use_header: bool):
        async with store._db.transaction() as tx:
            locked = await store._db.dialect.lock_state(tx, store.t.sessions, sid)
            store.fence_owner_row(locked, sid, ownership)
            locked_raw = locked["fold_stats_json"]
            if use_header:
                if healthy and locked_raw != header.value_json:
                    raise _ProjectionSnapshotConflict()
                if not healthy:
                    if locked_raw is not None:
                        raise _ProjectionSnapshotConflict()
                    existing_any = await tx.fetchone(
                        f"SELECT 1 AS present FROM {store.t.project_messages} WHERE session_id=? LIMIT 1",
                        (sid,),
                    )
                    if existing_any is not None:
                        raise _ProjectionSnapshotConflict()
                state = snapshot if healthy else _base_state(store, sid, policy, None)
                cutoff = header.compact_to_send
            else:
                state = parse_state(locked_raw)
                if policy is not None and _state_ok(store, sid, state, policy):
                    cutoff = state.get("compact_to_send")
                else:
                    compact_row = await store._query_latest_project_compact(tx.fetchone, sid)
                    cutoff = compact_row.compact_to_send if compact_row is not None else None
                    if policy is not None and state is None:
                        any_row = await tx.fetchone(
                            f"SELECT send_index FROM {store.t.project_messages} WHERE session_id=? LIMIT 1",
                            (sid,),
                        )
                        if any_row is None:
                            state = _base_state(store, sid, policy, cutoff)
            updated = None
            if (
                policy is not None
                and new_rows
                and all(r.kind in ("user", "project") for r in new_rows)
                and _matches(store, sid, state, policy, cutoff)
                and send_index > state["maximum"]
            ):
                try:
                    updated = await _advance(
                        store, tx, sid, state, projector, policy, new_rows, send_index,
                        window=window if use_header and healthy else None,
                    )
                except (KeyError, TypeError, ValueError):
                    logger.warning("projection fold statistics require rebuilding for %s", sid, exc_info=True)
            if updated is None:
                updated = _invalid(state, compact_to_send=cutoff)
            else:
                updated["revision"] = secrets.token_hex(8)
            if new_rows:
                await store._upsert_project_messages_tx(tx, [
                    (
                        sid, int(send_index), row.kind,
                        _dumps(row.content if row.content is not None else {}),
                        _nul_safe(row.rendered_text),
                        source_seq_lo, source_seq_hi, None, None,
                        int(projector_version), None, int(send_index), now,
                    )
                    for row in new_rows
                ])
            await _put_state(store, tx, sid, updated)
        return updated

    if healthy or first:
        try:
            return await save(True)
        except _ProjectionSnapshotConflict:
            pass
    return await save(False)


async def read_snapshot(store, sid: str, *, revision: str | None = None) -> ProjectionFoldSnapshot | None:
    """Keyset-page bodies without a session row lock, fenced by ``fold_stats_json.revision``."""
    from .store import _PROJECT_KIND_ORDER, _row_to_project_message

    state = await read_state(store, store._db, sid)
    if not state or (revision is not None and state.get("revision") != revision):
        return None
    prior = await store._query_latest_project_compact(store._db.fetchone, sid)
    cutoff = prior.compact_to_send if prior is not None else None
    rows: list[ProjectMessageRow] = []
    last = None
    while True:
        where, params = "session_id=?", [sid]
        if cutoff is not None:
            where += " AND send_index>?"
            params.append(cutoff)
        if last is not None:
            where += " AND (send_index>? OR (send_index=? AND kind>?))"
            params.extend((last[0], last[0], last[1]))
        page = await store._db.fetchall(
            f"SELECT * FROM {store.t.project_messages} WHERE {where} ORDER BY send_index,kind LIMIT ?",
            (*params, PAGE_SIZE),
        )
        rows.extend(_row_to_project_message(r) for r in page)
        if len(page) < PAGE_SIZE:
            break
        last = page[-1]["send_index"], page[-1]["kind"]
    current = await read_state(store, store._db, sid)
    if not current or current.get("revision") != state.get("revision"):
        return None
    rows.sort(key=lambda r: (r.send_index, _PROJECT_KIND_ORDER.get(r.kind, 9)))
    return ProjectionFoldSnapshot(prior, rows, str(state["revision"]))


async def _rebuild(store, sid: str, projector, policy, snapshot: ProjectionFoldSnapshot, _obsolete=None):
    """Cold/version-change repair: render outside locks; install inline into ``sessions.fold_stats_json``."""
    prior, rows = snapshot.prior, snapshot.rows
    cutoff = prior.compact_to_send if prior is not None else None
    state = _base_state(store, sid, policy, cutoff)
    sends = {r.send_index for r in rows if r.kind in ("user", "project")}
    maximum = max(sends, default=0)
    owners: dict[Any, int] = {}
    dedup: dict[str, int] = {}
    for row in rows:
        for key in _keys([row], policy):
            latest = max(row.send_index, owners.get(key, row.send_index))
            owners[key] = latest
            dedup[fold_dedup_hash(key)] = latest
    dedup = _cap_dedup(dedup)
    state.update(
        tokens=sum(_cost(projector, row, maximum, owners, policy) for row in rows)
        + (estimate_tokens(projector.render([prior])) if prior is not None else 0),
        rows=sum(r.kind in ("user", "project") for r in rows),
        sends=len(sends),
        maximum=maximum,
        dedup=dedup,
        revision=snapshot.revision,
    )
    async with store._db.transaction() as tx:
        await store._db.dialect.lock_state(tx, store.t.sessions, sid)
        current = await read_state(store, tx, sid)
        provider = getattr(projector, "fold_statistics_policy", None)
        if current and current.get("revision") == snapshot.revision and provider() == policy:
            await _put_state(store, tx, sid, state)
            return state
    return None


async def prepare(store, sid: str, *, projector, keep_last_sends: int, threshold: int,
                  row_threshold=None, ownership=None, window=None, **projection):
    """Persist this send; read historical bodies only after a positive gate or during repair."""
    provider = getattr(projector, "fold_statistics_policy", None)
    policy = provider() if provider is not None else None
    state = await persist(
        store, sid, projector=projector, policy=policy, ownership=ownership,
        window=window, **projection,
    )
    if provider is not None and provider() != policy:
        async with store._db.transaction() as tx:
            await store._db.dialect.lock_state(tx, store.t.sessions, sid)
            current = await read_state(store, tx, sid)
            if current and current.get("revision") == state.get("revision"):
                await invalidate(store, tx, sid)
        return None
    if keep_last_sends <= 0:
        return None
    cutoff = state.get("compact_to_send")
    if policy is not None and _matches(store, sid, state, policy, cutoff):
        if state["sends"] <= keep_last_sends or not fold_due(state["tokens"], threshold, state["rows"], row_threshold):
            return None
    snapshot = await read_snapshot(store, sid, revision=state.get("revision"))
    if snapshot is None:
        return None
    if policy is not None and not _matches(store, sid, state, policy, cutoff):
        state = await _rebuild(store, sid, projector, policy, snapshot)
        if state is None or state["sends"] <= keep_last_sends or not fold_due(state["tokens"], threshold, state["rows"], row_threshold):
            return None
    if policy is None:
        sends = {r.send_index for r in snapshot.rows if r.kind in ("user", "project")}
        if len(sends) <= keep_last_sends:
            return None
        if hasattr(projector, "stamp_render_context"):
            projector.stamp_render_context(snapshot.rows, max(sends) + 1)
        if not fold_due(
            estimate_tokens(projector.render(([snapshot.prior] if snapshot.prior else []) + snapshot.rows)),
            threshold, len(snapshot.rows), row_threshold,
        ):
            return None
    return snapshot
