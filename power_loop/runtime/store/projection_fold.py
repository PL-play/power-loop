"""Incremental, renderer-versioned fold statistics; projection bodies are read outside locks.

The small aggregate and (only for deduplicating renderers) latest-key owners live in
reserved runtime-state keys. They are derived data, not task or accounting state.
No schema migration or second authoritative message representation is required.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass

from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.fold import fold_due
from power_loop.runtime.store.types import ProjectMessageRow

logger = logging.getLogger(__name__)
STATE_KEY = "__pl_projection_fold"
OWNER_PREFIX = STATE_KEY + "/"
PAGE_SIZE = 256


class _ProjectionSnapshotConflict(Exception):
    """Only an explicit zero-row conditional statistics write permits the cold retry."""


@dataclass(frozen=True)
class ProjectionFoldSnapshot:
    prior: ProjectMessageRow | None
    rows: list[ProjectMessageRow]
    revision: str


def parse_state(value_json):
    """Parse a runtime-state JSON value; malformed statistics are rebuilt, never trusted."""
    try:
        value = json.loads(value_json) if value_json is not None else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


async def read_state(store, reader, sid):
    row = await reader.fetchone(
        f"SELECT value_json FROM {store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
        (sid, STATE_KEY),
    )
    return parse_state(row["value_json"] if row else None)


async def _put(store, writer, sid, key, value):
    # This is private DERIVED state (fold statistics and their key owners). No runtime
    # projector polls these keys, so bumping session_state.runtime_version here would wake
    # every runtime-state reader (e.g. Todo) for a value that did not change. Readers of the
    # derived state query it directly under the session lock.
    sql = store._db.dialect.upsert(
        store.t.session_runtime_state, ("session_id", "state_key"), ("value_json", "updated_at"),
    )
    await writer.execute(sql, (sid, key, json.dumps(value, ensure_ascii=False), time.time_ns() // 1_000_000))


def _invalid(state):
    return {"revision": secrets.token_hex(16), "obsolete": (state or {}).get("generation")
            or (state or {}).get("obsolete")}


async def invalidate(store, tx, sid, state=None):
    """Caller holds the session lock; all projection mutations invalidate the same revision.

    ``state`` lets a caller that already read the statistics in its lock statement reuse it.
    """
    if state is None:
        state = await read_state(store, tx, sid)
    await _put(store, tx, sid, STATE_KEY, _invalid(state))


def _keys(rows, policy):
    if not policy.deduplicate:
        return set()
    return {t["k"] for r in rows if r.kind == "project" and isinstance(r.content, dict)
            for t in r.content.get("tools") or [] if isinstance(t, dict) and t.get("k")}


def _owner_key(generation, key):
    digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return OWNER_PREFIX + generation + "/" + digest


async def _owners(store, reader, sid, generation, keys):
    by_name = {_owner_key(generation, key): key for key in keys}
    names = list(by_name)
    result = {}
    for offset in range(0, len(names), PAGE_SIZE):
        page = names[offset:offset + PAGE_SIZE]
        rows = await reader.fetchall(
            f"SELECT state_key,value_json FROM {store.t.session_runtime_state} WHERE session_id=? "
            f"AND state_key IN ({','.join('?' for _ in page)})", (sid, *page),
        )
        for row in rows:
            result[by_name[row["state_key"]]] = int(json.loads(row["value_json"]))
    return result


def _cost(projector, row, maximum, owners, policy):
    # Rendering annotations never mutate the caller's persisted/projected rows.
    row = copy.copy(row)
    row.recency = max(0, maximum - row.send_index)
    row.render_ctx = {"latest_key": owners, "hot_window": policy.hot_window}
    return estimate_tokens(projector.render([row]))


def _base_state(store, sid, policy, cutoff):
    return {"v": 2, "store_id": store.store_id, "session_id": sid, "policy": policy.fingerprint,
            "cutoff": cutoff, "generation": secrets.token_hex(16), "tokens": 0, "rows": 0,
            "sends": 0, "maximum": 0}


def _state_ok(store, sid, state, policy):
    """Statistics written for this store/session/policy by a current code path."""
    return (isinstance(state, dict) and state.get("v") == 2
            and state.get("store_id") == store.store_id and state.get("session_id") == sid
            and state.get("policy") == policy.fingerprint
            and isinstance(state.get("generation"), str)
            and all(type(state.get(k)) is int and state[k] >= 0 for k in ("tokens", "sends", "maximum")))


def _matches(store, sid, state, policy, cutoff):
    return _state_ok(store, sid, state, policy) and state.get("cutoff") == cutoff


async def _changed_rows(store, tx, sid, state, policy, maximum, owners, window=None):
    """Only rows becoming cold, or the previous visible owner of a new dedup key, can change."""
    from .store import _row_to_project_message

    found = {}
    cutoff = state["cutoff"] if state["cutoff"] is not None else -1
    lower = max(cutoff, state["maximum"] - policy.hot_window)
    upper = min(state["maximum"], maximum - policy.hot_window)
    indices = {send for send in owners.values() if cutoff < send <= state["maximum"]}
    window_lo = window.window_lo if window is not None else maximum
    covered_owners = {send for send in indices if window_lo <= send < maximum}
    if window is not None and (covered_owners or policy.hot_window and upper >= window_lo and lower < upper):
        # One pass, retaining references only to affected rows. The CAS that follows
        # validates this entire statement snapshot; rendering annotations are reset by _cost.
        for row in window.rows:
            if window_lo <= row.send_index < maximum and (
                    policy.hot_window and lower < row.send_index <= upper and row.kind in ("user", "project")
                    or row.kind == "project" and row.send_index in covered_owners):
                found[row.send_index, row.kind] = row
    outside_upper = min(upper, window_lo - 1)
    if policy.hot_window and lower < outside_upper:
        rows = await tx.fetchall(
            f"SELECT * FROM {store.t.project_messages} WHERE session_id=? AND send_index>? "
            "AND send_index<=? AND kind IN ('user','project')", (sid, lower, outside_upper),
        )
        found.update({(r["send_index"], r["kind"]): _row_to_project_message(r) for r in rows})
    indices = sorted(indices - covered_owners)
    for offset in range(0, len(indices), PAGE_SIZE):
        page = indices[offset:offset + PAGE_SIZE]
        rows = await tx.fetchall(
            f"SELECT * FROM {store.t.project_messages} WHERE session_id=? AND kind='project' "
            f"AND send_index IN ({','.join('?' for _ in page)})", (sid, *page),
        )
        found.update({(r["send_index"], r["kind"]): _row_to_project_message(r) for r in rows})
    return list(found.values())


async def _advance(store, tx, sid, state, projector, policy, new_rows, send_index, window=None):
    new_keys = _keys(new_rows, policy)
    owners = await _owners(store, tx, sid, state["generation"], new_keys)
    changed = await _changed_rows(store, tx, sid, state, policy, send_index, owners, window=window)
    old_keys = _keys(changed, policy)
    owners.update(await _owners(store, tx, sid, state["generation"], old_keys - owners.keys()))
    if any(key not in owners or owners[key] < row.send_index
           for row in changed for key in _keys([row], policy)):
        raise ValueError("missing projection fold key owner")
    before = sum(_cost(projector, row, state["maximum"], owners, policy) for row in changed)
    owners.update(dict.fromkeys(new_keys, send_index))
    after = sum(_cost(projector, row, send_index, owners, policy) for row in [*changed, *new_rows])
    updated = {**state, "tokens": state["tokens"] + after - before,
               "rows": state["rows"] + len(new_rows),
               "sends": state["sends"] + int(any(r.kind in ("user", "project") for r in new_rows)),
               "maximum": send_index}
    if updated["tokens"] < 0:
        raise ValueError("invalid projection fold token total")
    for key in new_keys:
        await _put(store, tx, sid, _owner_key(state["generation"], key), send_index)
    return updated


async def persist(store, sid, *, send_index, rows, source_seq_lo, source_seq_hi, projector_version,
                  projector=None, policy=None, ownership=None, window=None):
    """Atomically append projections and advance/invalidate statistics under a short session lock.

    ``ownership`` (design 215 §8.2): when the writer holds a session owner fence, the
    derived rows are only written after the same transaction verifies epoch/token, so a
    stale owner's late projection cannot clobber the new owner's rebuild."""
    from .store import _dumps, _nul_safe

    new_rows = [ProjectMessageRow(
        session_id=sid, send_index=send_index, kind=kind, content=json.loads(_dumps(content if content is not None else {})),
        rendered_text=_nul_safe(text), source_seq_lo=source_seq_lo, source_seq_hi=source_seq_hi,
        compact_from_send=None, compact_to_send=None, projector_version=projector_version,
        token_estimate=None, created_at=0, completed_through_send_index=send_index,
    ) for kind, content, text in rows]
    now = time.time_ns() // 1_000_000
    header = window.stats if window is not None else None
    snapshot = parse_state(header.value_json) if header is not None else None
    incremental = (policy is not None and new_rows
                   and all(row.kind in ("user", "project") for row in new_rows))
    healthy = (incremental and header is not None and header.row_exists
               and _matches(store, sid, snapshot, policy, header.compact_to_send)
               and send_index > snapshot["maximum"])
    first = incremental and header is not None and not header.row_exists and send_index == 1

    async def save(use_header):
        async with store._db.transaction() as tx:
            # Ownership always comes from the real locked row, never the run snapshot.
            locked = await store._db.dialect.lock_state(tx, store.t.session_state, sid)
            store.fence_owner_row(locked, sid, ownership)
            if use_header:
                state = snapshot if healthy else _base_state(store, sid, policy, None)
                cutoff = header.compact_to_send
            else:
                # After the lock, one statement returns the statistics and the cold-path facts. This is
                # fresh (no pre-wait join snapshot): the compact subquery uses the (session_id,kind)
                # coverage index, and any_send stops at the first row of the session index. Hot saves
                # pay these bounded probes but issue no extra statement.
                compact_cte = ("WITH compact AS (" if store._db.dialect.name == "mysql"
                               else "WITH compact AS MATERIALIZED (")
                facts = await tx.fetchone(
                    f"{compact_cte}SELECT * FROM {store.t.project_messages} "
                    "WHERE session_id=? AND kind='compact') "
                    f"SELECT (SELECT value_json FROM {store.t.session_runtime_state} "
                    "WHERE session_id=? AND state_key=?) AS stats_json, "
                    "(SELECT compact_to_send FROM compact ORDER BY send_index DESC LIMIT 1) "
                    "AS compact_to_send, "
                    f"(SELECT send_index FROM {store.t.project_messages} WHERE session_id=? LIMIT 1) AS any_send",
                    (sid, sid, STATE_KEY, sid),
                )
                state = parse_state(facts["stats_json"] if facts is not None else None)
                # A valid statistics row is written under this same session lock and any fold/rewrite
                # invalidates it, so its cutoff is current; the compact/any_send probes are ignored on
                # this healthy path and only retained for the cold/rebuild decision below.
                if policy is not None and _state_ok(store, sid, state, policy):
                    cutoff = state.get("cutoff")
                else:
                    cutoff = facts["compact_to_send"] if facts is not None else None
                    if policy is not None and state is None and facts is not None and facts["any_send"] is None:
                        state = _base_state(store, sid, policy, cutoff)
            updated = None
            if (policy is not None and new_rows and all(r.kind in ("user", "project") for r in new_rows)
                    and _matches(store, sid, state, policy, cutoff)
                    and send_index > state["maximum"]):
                try:
                    updated = await _advance(store, tx, sid, state, projector, policy, new_rows, send_index,
                                             window=window if use_header and healthy else None)
                except (KeyError, TypeError, ValueError):
                    logger.warning("projection fold statistics require rebuilding for %s", sid, exc_info=True)
            if updated is None:
                updated = _invalid(state)
            else:
                updated["revision"] = secrets.token_hex(16)
            if use_header:
                # Keep the DB's original string as the expected value. In particular,
                # MySQL's text collation must not erase trailing spaces or Unicode differences.
                value = json.dumps(updated, ensure_ascii=False)
                if healthy:
                    equality = ("CAST(value_json AS BINARY) = CAST(? AS BINARY)" if store._db.dialect.name == "mysql"
                                else "value_json=?")
                    affected = await tx.execute(
                        f"UPDATE {store.t.session_runtime_state} SET value_json=?,updated_at=? "
                        f"WHERE session_id=? AND state_key=? AND {equality}",
                        (value, now, sid, STATE_KEY, header.value_json),
                    )
                else:
                    # First send is not evidence of an empty projection table. Both
                    # absence predicates are evaluated BEFORE inserting the new projections.
                    affected = await tx.execute(
                        f"INSERT INTO {store.t.session_runtime_state} "
                        "(session_id,state_key,value_json,updated_at) SELECT ?,?,?,? "
                        f"WHERE NOT EXISTS (SELECT 1 FROM {store.t.session_runtime_state} "
                        "WHERE session_id=? AND state_key=?) "
                        f"AND NOT EXISTS (SELECT 1 FROM {store.t.project_messages} WHERE session_id=?)",
                        (sid, STATE_KEY, value, now, sid, STATE_KEY, sid),
                    )
                if affected == 0:
                    raise _ProjectionSnapshotConflict()
            if new_rows:
                await store._upsert_project_messages_tx(tx, [(
                    sid, int(send_index), row.kind,
                    _dumps(row.content if row.content is not None else {}),
                    _nul_safe(row.rendered_text),
                    source_seq_lo, source_seq_hi, None, None,
                    int(projector_version), None, int(send_index), now,
                ) for row in new_rows])
            if not use_header:
                await _put(store, tx, sid, STATE_KEY, updated)
        return updated

    if healthy or first:
        try:
            return await save(True)
        except _ProjectionSnapshotConflict:
            # The transaction manager has rolled back key owners and statistics.
            # Legal mutators share the state lock: the existing locked facts path is
            # sufficient now. Never retry an unknown commit or replay external work.
            pass
    return await save(False)


async def read_snapshot(store, sid, *, revision=None):
    """Keyset-page bodies without a session row lock, fenced by the projection-write revision."""
    from .store import _PROJECT_KIND_ORDER, _row_to_project_message

    state = await read_state(store, store._db, sid)
    if not state or (revision is not None and state["revision"] != revision):
        return None
    prior = await store._query_latest_project_compact(store._db.fetchone, sid)
    cutoff = prior.compact_to_send if prior is not None else None
    rows = []
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
    if not current or current["revision"] != state["revision"]:
        return None
    rows.sort(key=lambda r: (r.send_index, _PROJECT_KIND_ORDER.get(r.kind, 9)))
    return ProjectionFoldSnapshot(prior, rows, state["revision"])


async def _discard_owners(store, sid, generation):
    if not generation:
        return
    prefix = OWNER_PREFIX + generation + "/"
    pattern = prefix.replace("!", "!!").replace("_", "!_").replace("%", "!%") + "%"
    after = None
    while True:
        async with store._db.transaction() as tx:
            where = "session_id=? AND state_key LIKE ? ESCAPE '!'"
            params = [sid, pattern]
            if after is not None:
                where += " AND state_key>?"
                params.append(after)
            rows = await tx.fetchall(
                f"SELECT state_key FROM {store.t.session_runtime_state} WHERE {where} "
                "ORDER BY state_key LIMIT ?", (*params, PAGE_SIZE),
            )
            if not rows:
                return
            await store.signal_changes(tx, sid, "runtime")
            await tx.execute(
                f"DELETE FROM {store.t.session_runtime_state} WHERE session_id=? "
                f"AND state_key IN ({','.join('?' for _ in rows)})", (sid, *(r["state_key"] for r in rows)),
            )
            after = rows[-1]["state_key"]


async def _rebuild(store, sid, projector, policy, snapshot, obsolete):
    """Cold/version-change repair: render outside locks; install only if no projection changed.

    Key owners are staged in bounded transactions under a fresh generation. A concurrent
    projection writer prevents publication, so unfinished derived rows can never authorize
    a fold or overwrite a newer aggregate.
    """
    prior, rows = snapshot.prior, snapshot.rows
    cutoff = prior.compact_to_send if prior is not None else None
    state = _base_state(store, sid, policy, cutoff)
    sends = {r.send_index for r in rows if r.kind in ("user", "project")}
    maximum = max(sends, default=0)
    owners = {}
    for row in rows:
        for key in _keys([row], policy):
            owners[key] = max(row.send_index, owners.get(key, row.send_index))
    state.update(tokens=sum(_cost(projector, row, maximum, owners, policy) for row in rows)
                 + (estimate_tokens(projector.render([prior])) if prior is not None else 0),
                 rows=sum(r.kind in ("user", "project") for r in rows),
                 sends=len(sends), maximum=maximum, revision=snapshot.revision)
    installed = False
    replaced = obsolete
    try:
        entries = list(owners.items())
        for offset in range(0, len(entries), PAGE_SIZE):
            async with store._db.transaction() as tx:
                for key, send in entries[offset:offset + PAGE_SIZE]:
                    await _put(store, tx, sid, _owner_key(state["generation"], key), send)
        async with store._db.transaction() as tx:
            await store._db.dialect.lock_state(tx, store.t.session_state, sid)
            current = await read_state(store, tx, sid)
            provider = getattr(projector, "fold_statistics_policy", None)
            if current and current["revision"] == snapshot.revision and provider() == policy:
                replaced = current.get("generation") or current.get("obsolete")
                await _put(store, tx, sid, STATE_KEY, state)
                installed = True
    finally:
        if not installed:
            await _discard_owners(store, sid, state["generation"])
    if installed:
        await _discard_owners(store, sid, replaced)
        return state
    return None


async def prepare(store, sid, *, projector, keep_last_sends, threshold, row_threshold=None,
                  ownership=None, window=None, **projection):
    """Persist this send; read historical bodies only after a positive gate or during repair."""
    provider = getattr(projector, "fold_statistics_policy", None)
    policy = provider() if provider is not None else None
    state = await persist(store, sid, projector=projector, policy=policy, ownership=ownership,
                          window=window, **projection)
    if provider is not None and provider() != policy:
        async with store._db.transaction() as tx:
            await store._db.dialect.lock_state(tx, store.t.session_state, sid)
            current = await read_state(store, tx, sid)
            if current and current["revision"] == state["revision"]:
                await invalidate(store, tx, sid)
        return None
    if keep_last_sends <= 0:
        return None
    cutoff = state.get("cutoff")
    if policy is not None and _matches(store, sid, state, policy, cutoff):
        if state["sends"] <= keep_last_sends or not fold_due(state["tokens"], threshold, state["rows"], row_threshold):
            return None
    snapshot = await read_snapshot(store, sid, revision=state["revision"])
    if snapshot is None:
        return None  # a concurrent writer won; its send will re-evaluate the new revision
    if policy is not None and not _matches(store, sid, state, policy, cutoff):
        state = await _rebuild(store, sid, projector, policy, snapshot, state.get("obsolete"))
        if state is None or state["sends"] <= keep_last_sends or not fold_due(state["tokens"], threshold, state["rows"], row_threshold):
            return None
    if policy is None:
        sends = {r.send_index for r in snapshot.rows if r.kind in ("user", "project")}
        if len(sends) <= keep_last_sends:
            return None
        if hasattr(projector, "stamp_render_context"):
            projector.stamp_render_context(snapshot.rows, max(sends) + 1)
        if not fold_due(estimate_tokens(projector.render(([snapshot.prior] if snapshot.prior else []) + snapshot.rows)),
                        threshold, len(snapshot.rows), row_threshold):
            return None
    return snapshot
