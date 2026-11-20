"""Session closure and bounded deletion, serialized with admission on session_state."""
from __future__ import annotations

import time

from power_loop.contracts.errors import SessionBusy

from .retention import batch_limit, delete_batch


async def busy(store, tx, session_id, state=None):
    if state is None:
        state = await tx.fetchone(
            f'SELECT * FROM {store.t.session_state} WHERE session_id=?', (session_id,))
    if state is not None and (
        state['execution_state'] != 'idle' or state['pending_json']
        or (state['owner_token'] is not None
            and state['owner_lease_until_ms'] > time.time_ns() // 1_000_000)
    ):
        return True
    return bool(await tx.fetchone(
        f'SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? '
        "AND terminal_status IS NULL AND status IN ('pending','waiting_continuation','delivered') LIMIT 1",
        (session_id,),
    ))


async def lock_sessions(store, tx, session_id, cascade):
    """Lock the parent before discovering linked children; independent children survive."""
    rows, pending = [], [session_id]
    while pending:
        sid = pending.pop()
        suffix = '' if store._db.dialect.name == 'sqlite' else ' FOR UPDATE'
        state = await tx.fetchone(f'SELECT * FROM {store.t.session_state} WHERE session_id=?' + suffix, (sid,))
        if state is None:
            continue
        row = await tx.fetchone(
            f'SELECT session_id,lifecycle FROM {store.t.sessions} WHERE session_id=?', (sid,))
        if row is None:
            continue
        rows.append({**dict(row), 'state': state})
        if cascade:
            children = await tx.fetchall(
                f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? '
                "AND cancel_propagation='linked' ORDER BY session_id", (sid,))
            pending.extend(r['session_id'] for r in reversed(children))
    return rows


async def mark_closing(store, tx, row):
    changed = await tx.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='closing',updated_at=? "
        "WHERE session_id=? AND lifecycle='open'",
        (time.time_ns() // 1_000_000, row['session_id']),
    )
    return changed


async def close_if_idle(store, session_ids, *, cascade=True, limit=1000, resource_guard=None):
    remaining = batch_limit(limit)
    async with store._db.transaction() as tx:
        found = {}
        for sid in sorted(set(session_ids)):
            for row in await lock_sessions(store, tx, sid, cascade):
                found[row['session_id']] = row
        rows = list(found.values())
        for row in rows:
            if await busy(store, tx, row['session_id'], row['state']):
                return 'busy'
        if resource_guard is not None and not resource_guard([row['session_id'] for row in rows]):
            return 'busy'
        for row in rows:
            if row['lifecycle'] != 'open':
                continue
            if not remaining:
                return 'cleanup_pending'
            remaining -= await mark_closing(store, tx, row)
    return 'closed'


SESSION_TABLE_KEYS = (
    ('follow_up_queue', ('id',)), ('project_messages', ('id',)),
    ('session_runtime_state', ('state_key',)), ('messages', ('id',)),
)


async def delete_sessions(store, session_id, *, cascade=True, limit=1000, resource_guard=None):
    """One row allowance includes closure, dependents, reparenting and session rows."""
    remaining, deleted = batch_limit(limit), []
    if remaining < 2:
        raise ValueError('session deletion requires a limit of at least 2')
    async with store._db.transaction() as tx:
        rows = await lock_sessions(store, tx, session_id, cascade)
        for row in rows:
            if await busy(store, tx, row['session_id'], row['state']):
                raise SessionBusy('session or descendant has unfinished work')
        if resource_guard is not None and not resource_guard([row['session_id'] for row in rows]):
            raise SessionBusy('session resources still in use')
        for row in rows:
            if row['lifecycle'] == 'open':
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
            if remaining < 2:
                return deleted
            remaining -= await tx.execute(f'DELETE FROM {store.t.session_state} WHERE session_id=?', (sid,))
            count = await tx.execute(f'DELETE FROM {store.t.sessions} WHERE session_id=?', (sid,))
            remaining -= count
            if count:
                deleted.append(sid)
                store._forget_appended(sid)
    return deleted
