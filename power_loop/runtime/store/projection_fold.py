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
from power_loop.runtime.store.types import ProjectMessageRow

logger = logging.getLogger(__name__)
STATE_KEY = "__pl_projection_fold"
OWNER_PREFIX = STATE_KEY + "/"
PAGE_SIZE = 256


@dataclass(frozen=True)
class ProjectionFoldSnapshot:
    prior: ProjectMessageRow | None
    rows: list[ProjectMessageRow]
    revision: str


async def read_state(store, reader, sid):
    row = await reader.fetchone(
        f"SELECT value_json FROM {store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
        (sid, STATE_KEY),
    )
    try:
        value = json.loads(row["value_json"]) if row else None
    except (TypeError, ValueError):
        return None  # malformed derived statistics must be rebuilt, never trusted
    return value if isinstance(value, dict) else None


async def _put(store, writer, sid, key, value):
    sql = store._db.dialect.upsert(
        store.t.session_runtime_state, ("session_id", "state_key"), ("value_json", "updated_at"),
    )
    await writer.execute(sql, (sid, key, json.dumps(value, ensure_ascii=False), time.time_ns() // 1_000_000))


def _invalid(state):
    return {"revision": secrets.token_hex(16), "obsolete": (state or {}).get("generation")
            or (state or {}).get("obsolete")}


async def invalidate(store, tx, sid):
    """Caller holds the session lock; all projection mutations invalidate the same revision."""
    state = _invalid(await read_state(store, tx, sid))
    await _put(store, tx, sid, STATE_KEY, state)


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
    return {"v": 1, "store_id": store.store_id, "session_id": sid, "policy": policy.fingerprint,
            "cutoff": cutoff, "generation": secrets.token_hex(16), "tokens": 0, "sends": 0, "maximum": 0}


def _matches(store, sid, state, policy, cutoff):
    return (isinstance(state, dict) and state.get("v") == 1
            and state.get("store_id") == store.store_id and state.get("session_id") == sid
            and state.get("policy") == policy.fingerprint and state.get("cutoff") == cutoff
            and isinstance(state.get("generation"), str)
            and all(type(state.get(k)) is int and state[k] >= 0 for k in ("tokens", "sends", "maximum")))


async def _changed_rows(store, tx, sid, state, policy, maximum, owners):
    """Only rows becoming cold, or the previous visible owner of a new dedup key, can change."""
    from .store import _row_to_project_message

    found = {}
    cutoff = state["cutoff"] if state["cutoff"] is not None else -1
    lower = max(cutoff, state["maximum"] - policy.hot_window)
    upper = min(state["maximum"], maximum - policy.hot_window)
    if policy.hot_window and lower < upper:
        rows = await tx.fetchall(
            f"SELECT * FROM {store.t.project_messages} WHERE session_id=? AND send_index>? "
            "AND send_index<=? AND kind IN ('user','project')", (sid, lower, upper),
        )
        found.update({(r["send_index"], r["kind"]): _row_to_project_message(r) for r in rows})
    indices = sorted({send for send in owners.values() if cutoff < send <= state["maximum"]})
    for offset in range(0, len(indices), PAGE_SIZE):
        page = indices[offset:offset + PAGE_SIZE]
        rows = await tx.fetchall(
            f"SELECT * FROM {store.t.project_messages} WHERE session_id=? AND kind='project' "
            f"AND send_index IN ({','.join('?' for _ in page)})", (sid, *page),
        )
        found.update({(r["send_index"], r["kind"]): _row_to_project_message(r) for r in rows})
    return list(found.values())


async def _advance(store, tx, sid, state, projector, policy, new_rows, send_index):
    new_keys = _keys(new_rows, policy)
    owners = await _owners(store, tx, sid, state["generation"], new_keys)
    changed = await _changed_rows(store, tx, sid, state, policy, send_index, owners)
    old_keys = _keys(changed, policy)
    owners.update(await _owners(store, tx, sid, state["generation"], old_keys - owners.keys()))
    if any(key not in owners or owners[key] < row.send_index
           for row in changed for key in _keys([row], policy)):
        raise ValueError("missing projection fold key owner")
    before = sum(_cost(projector, row, state["maximum"], owners, policy) for row in changed)
    owners.update(dict.fromkeys(new_keys, send_index))
    after = sum(_cost(projector, row, send_index, owners, policy) for row in [*changed, *new_rows])
    updated = {**state, "tokens": state["tokens"] + after - before,
               "sends": state["sends"] + int(any(r.kind in ("user", "project") for r in new_rows)),
               "maximum": send_index}
    if updated["tokens"] < 0:
        raise ValueError("invalid projection fold token total")
    for key in new_keys:
        await _put(store, tx, sid, _owner_key(state["generation"], key), send_index)
    return updated


async def persist(store, sid, *, send_index, rows, source_seq_lo, source_seq_hi, projector_version,
                  projector=None, policy=None):
    """Atomically append projections and advance/invalidate statistics under a short session lock."""
    from .store import _dumps, _nul_safe

    new_rows = [ProjectMessageRow(
        session_id=sid, send_index=send_index, kind=kind, content=json.loads(_dumps(content if content is not None else {})),
        rendered_text=_nul_safe(text), source_seq_lo=source_seq_lo, source_seq_hi=source_seq_hi,
        compact_from_send=None, compact_to_send=None, projector_version=projector_version,
        token_estimate=None, created_at=0,
    ) for kind, content, text in rows]
    async with store._db.transaction() as tx:
        await store._db.dialect.lock_state(tx, store.t.session_state, sid)
        state = await read_state(store, tx, sid)
        compact = await tx.fetchone(
            f"SELECT compact_to_send FROM {store.t.project_messages} WHERE session_id=? AND kind='compact' "
            "ORDER BY send_index DESC LIMIT 1", (sid,),
        )
        cutoff = compact["compact_to_send"] if compact is not None else None
        if policy is not None and state is None and compact is None:
            exists = await tx.fetchone(
                f"SELECT send_index FROM {store.t.project_messages} WHERE session_id=? LIMIT 1", (sid,),
            )
            if exists is None:
                state = _base_state(store, sid, policy, cutoff)
        updated = None
        if (policy is not None and new_rows and all(r.kind in ("user", "project") for r in new_rows)
                and _matches(store, sid, state, policy, cutoff)
                and send_index > state["maximum"]):
            try:
                updated = await _advance(store, tx, sid, state, projector, policy, new_rows, send_index)
            except (KeyError, TypeError, ValueError):
                logger.warning("projection fold statistics require rebuilding for %s", sid, exc_info=True)
        for row in new_rows:
            await store._upsert_project_message_tx(
                tx, sid, send_index=send_index, kind=row.kind, content=row.content,
                rendered_text=row.rendered_text, source_seq_lo=source_seq_lo, source_seq_hi=source_seq_hi,
                projector_version=projector_version, now=time.time_ns() // 1_000_000,
            )
        if updated is None:
            updated = _invalid(state)
        else:
            updated["revision"] = secrets.token_hex(16)
        await _put(store, tx, sid, STATE_KEY, updated)
    return updated


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


async def prepare(store, sid, *, projector, keep_last_sends, threshold, **projection):
    """Persist this send; read historical bodies only after a positive gate or during repair."""
    provider = getattr(projector, "fold_statistics_policy", None)
    policy = provider() if provider is not None else None
    state = await persist(store, sid, projector=projector, policy=policy, **projection)
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
        if state["sends"] <= keep_last_sends or state["tokens"] < threshold:
            return None
    snapshot = await read_snapshot(store, sid, revision=state["revision"])
    if snapshot is None:
        return None  # a concurrent writer won; its send will re-evaluate the new revision
    if policy is not None and not _matches(store, sid, state, policy, cutoff):
        state = await _rebuild(store, sid, projector, policy, snapshot, state.get("obsolete"))
        if state is None or state["sends"] <= keep_last_sends or state["tokens"] < threshold:
            return None
    if policy is None:
        sends = {r.send_index for r in snapshot.rows if r.kind in ("user", "project")}
        if len(sends) <= keep_last_sends:
            return None
        if hasattr(projector, "stamp_render_context"):
            projector.stamp_render_context(snapshot.rows, max(sends) + 1)
        if estimate_tokens(projector.render(([snapshot.prior] if snapshot.prior else []) + snapshot.rows)) < threshold:
            return None
    return snapshot
