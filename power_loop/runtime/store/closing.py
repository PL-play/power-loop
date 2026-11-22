"""Session closure and bounded deletion, serialized with admission on sessions."""
from __future__ import annotations

import time

from power_loop.contracts.errors import SessionBusy

from .retention import batch_limit, delete_batch


async def busy(store, tx, session_id, state=None):
    """Check whether a session has active/queued work based on ``sessions`` state.

    Readiness for closing depends on ``sessions`` state (``execution_state='idle' AND
    wake_pending_count=0``, no unresolved ``pending_tool_seq`` or active lease) and active
    ``waiting_continuation`` rows.
    """
    if state is None:
        state = await tx.fetchone(
            f'SELECT * FROM {store.t.sessions} WHERE session_id=?', (session_id,))
    if state is not None and (
        state['execution_state'] != 'idle'
        or state['pending_tool_seq'] is not None
        or int(state['wake_pending_count']) > 0
        or (state['owner_node_id'] is not None
            and state['owner_lease_until_ms'] is not None
            and state['owner_lease_until_ms'] > time.time_ns() // 1_000_000)
    ):
        return True
    return bool(await tx.fetchone(
        f'SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? '
        "AND status='waiting_continuation' LIMIT 1",
        (session_id,),
    ))


class _CloseTreeChanged(Exception):
    """Internal retry signal when linked descendants mutate before deep-first lock completes."""


async def _discover_linked_tree(store, tx, seed_ids, cascade):
    """Unlocked BFS discovery of target session(s) and their linked descendants."""
    discovered: dict[str, dict] = {}
    pending = list(sorted(set(seed_ids)))
    while pending:
        sid = pending.pop()
        if sid in discovered:
            continue
        state = await tx.fetchone(f'SELECT * FROM {store.t.sessions} WHERE session_id=?', (sid,))
        if state is None:
            continue
        discovered[sid] = dict(state)
        if cascade:
            children = await tx.fetchall(
                f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? '
                "AND cancel_propagation='linked' ORDER BY session_id",
                (sid,),
            )
            pending.extend(r['session_id'] for r in reversed(children))
    return discovered


async def lock_sessions(store, tx, session_id, cascade, *, stop_on_busy=False):
    """Lock linked tree strictly deep-first (depth DESC, session_id ASC) to match Final/cancel/set_lifecycle.

    When ``stop_on_busy=True``, returns ``None`` immediately if any unlocked discovered row is
    already busy (preventing unnecessary lock waits when closing an in-flight Driver or leaf).
    """
    seed_ids = [session_id] if isinstance(session_id, str) else list(session_id)
    discovered = await _discover_linked_tree(store, tx, seed_ids, cascade)
    if not discovered:
        return []
    if stop_on_busy:
        for sid, row in discovered.items():
            if await busy(store, tx, sid, row):
                return None

    by_depth: dict[int, list[str]] = {}
    for sid, row in discovered.items():
        by_depth.setdefault(int(row.get('depth') or 0), []).append(sid)

    suffix = '' if store._db.dialect.name == 'sqlite' else ' FOR UPDATE'
    locked: dict[str, dict] = {}
    for d_val in sorted(by_depth, reverse=True):
        lvl_ids = sorted(by_depth[d_val])
        ph = ','.join('?' for _ in lvl_ids)
        lvl_rows = await tx.fetchall(
            f'SELECT * FROM {store.t.sessions} WHERE session_id IN ({ph}) '
            f'ORDER BY session_id ASC{suffix}',
            tuple(lvl_ids),
        )
        for st in lvl_rows:
            locked[str(st['session_id'])] = {**dict(st), 'state': st}

    if set(locked.keys()) != set(discovered.keys()):
        raise _CloseTreeChanged()
    if cascade and locked:
        # Verify under locks that no concurrent create_many/rebind added a new linked child or moved depth.
        for sid, row in locked.items():
            if int(row.get('depth') or 0) != int(discovered[sid].get('depth') or 0):
                raise _CloseTreeChanged()
        l_ids = sorted(locked.keys())
        ph = ','.join('?' for _ in l_ids)
        recheck = await tx.fetchall(
            f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id IN ({ph}) '
            "AND cancel_propagation='linked'",
            tuple(l_ids),
        )
        for r in recheck:
            if str(r['session_id']) not in locked:
                raise _CloseTreeChanged()

    # Return parent-first (depth ASC, session_id ASC) so delete_sessions' reversed(rows) deletes children first.
    return sorted(locked.values(), key=lambda r: (int(r.get('depth') or 0), str(r['session_id'])))


async def mark_closing(store, tx, row):
    now = time.time_ns() // 1_000_000
    changed = await tx.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='closing',"
        "pending_count=0,pending_bytes=0,updated_at=? "
        "WHERE session_id=? AND lifecycle IN ('open','suspended')",
        (now, row['session_id']),
    )
    if changed and int(row.get('pending_count') or 0) > 0:
        # Void any residual wake=0 pending notices so retention can purge the closed session.
        await tx.execute(
            f"UPDATE {store.t.follow_up_queue} SET status='void',terminal_status='cancelled',"
            "cancel_by='removed',error_code='session_closed',completed_at=? "
            "WHERE session_id=? AND status='pending'",
            (now, row['session_id']),
        )
    return changed


async def close_if_idle(store, session_ids, *, cascade=True, limit=1000, resource_guard=None):
    remaining = batch_limit(limit)
    seeds = sorted(set(session_ids))
    for _attempt in range(8):
        try:
            async with store._db.transaction() as tx:
                rows = await lock_sessions(store, tx, seeds, cascade, stop_on_busy=True)
                if rows is None:
                    return 'busy'
                for row in rows:
                    if await busy(store, tx, row['session_id'], row['state']):
                        return 'busy'
                if resource_guard is not None and not resource_guard([row['session_id'] for row in rows]):
                    return 'busy'
                for row in rows:
                    if row['lifecycle'] not in ('open', 'suspended'):
                        continue
                    if not remaining:
                        return 'cleanup_pending'
                    remaining -= await mark_closing(store, tx, row)
            return 'closed'
        except _CloseTreeChanged:
            continue
    return 'busy'


SESSION_TABLE_KEYS = (
    ('follow_up_queue', ('id',)), ('project_messages', ('id',)),
    ('session_runtime_state', ('state_key',)), ('messages', ('id',)),
)


async def delete_sessions(store, session_id, *, cascade=True, limit=1000, resource_guard=None):
    """One row allowance includes closure, dependents, reparenting and session rows."""
    remaining, deleted = batch_limit(limit), []
    if remaining < 1:
        raise ValueError('session deletion requires a limit of at least 1')
    for _attempt in range(8):
        try:
            async with store._db.transaction() as tx:
                rows = await lock_sessions(store, tx, session_id, cascade, stop_on_busy=True)
                if rows is None:
                    raise SessionBusy('session or descendant has unfinished work')
                for row in rows:
                    if await busy(store, tx, row['session_id'], row['state']):
                        raise SessionBusy('session or descendant has unfinished work')
                if resource_guard is not None and not resource_guard([row['session_id'] for row in rows]):
                    raise SessionBusy('session resources still in use')
                for row in rows:
                    if row['lifecycle'] in ('open', 'suspended'):
                        if not remaining:
                            return deleted
                        remaining -= await mark_closing(store, tx, row)
                for row in reversed(rows):
                    sid = row['session_id']
                    if not remaining:
                        break
                    if cascade and await tx.fetchone(
                        f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? '
                        "AND cancel_propagation='linked' LIMIT 1", (sid,),
                    ):
                        continue
                    for name, keys in SESSION_TABLE_KEYS:
                        remaining -= await delete_batch(
                            tx, store.table_prefix + name, keys, 'session_id=?', (sid,), remaining)
                        if not remaining:
                            return deleted
                    children = await tx.fetchall(
                        f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? LIMIT ?',
                        (sid, remaining),
                    )
                    for child in children:
                        remaining -= await tx.execute(
                            f'UPDATE {store.t.sessions} SET parent_session_id=NULL WHERE session_id=?',
                            (child['session_id'],),
                        )
                    if remaining < 1:
                        return deleted
                    count = await tx.execute(f'DELETE FROM {store.t.sessions} WHERE session_id=?', (sid,))
                    remaining -= count
                    if count:
                        deleted.append(sid)
                        store._forget_appended(sid)
            return deleted
        except _CloseTreeChanged:
            continue
    raise SessionBusy('session tree changed concurrently')
