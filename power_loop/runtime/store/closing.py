"""Explicit lifecycle closure, serialized with input admission on session_state."""
from __future__ import annotations

from power_loop.contracts.errors import SessionBusy

from .retention import batch_limit, delete_batch, delete_tree, lock_root, tree_runtime


async def busy(store, tx, session_id):
    p = store.table_prefix
    if await tx.fetchone(f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? "
                        "AND status IN ('pending','claimed') LIMIT 1", (session_id,)):
        return True
    if not store.has_message_ids:
        return False
    # Only unfinished jobs and obligations; no event/history reconstruction.
    for condition in ("supervision_status<>'closed'", "resource_state<>'stopped'"):
        if await tx.fetchone(f'SELECT job_id FROM {p}jobs WHERE session_id=? AND {condition} LIMIT 1', (session_id,)):
            return True
    if await tx.fetchone(f'SELECT source_id FROM {p}session_obligations WHERE session_id=? LIMIT 1', (session_id,)):
        return True
    if await tx.fetchone(f'SELECT b.attempt_id FROM {p}jobs j JOIN {p}task_budget_attempts b ON b.job_id=j.job_id '
                        "WHERE j.session_id=? AND (b.pending=1 OR b.phase NOT IN ('released','settling')) LIMIT 1", (session_id,)):
        return True
    return False


async def lock_sessions(store, tx, session_id, cascade):
    """Parent before child, also used by child creation; detached sessions survive."""
    rows, pending = [], [session_id]
    suffix = '' if store._db.dialect.name == 'sqlite' else ' FOR UPDATE'
    while pending:
        sid = pending.pop()
        row = await tx.fetchone(f'SELECT session_id,lifecycle,inbox_version FROM {store.t.session_state} '
                               'WHERE session_id=?'+suffix, (sid,))
        if row is None:
            continue
        rows.append(row)
        if cascade:
            children = await tx.fetchall(f'SELECT session_id FROM {store.t.sessions} '
                "WHERE parent_session_id=? AND lifecycle<>'detached' ORDER BY session_id", (sid,))
            pending.extend(r['session_id'] for r in reversed(children))
    return rows


async def close_if_idle(store, session_ids, *, cascade=True, limit=1000, resource_guard=None):
    remaining = batch_limit(limit)
    async with store._db.transaction() as tx:
        found = {}
        for sid in sorted(set(session_ids)):
            for row in await lock_sessions(store, tx, sid, cascade):
                found[row['session_id']] = row
        rows = list(found.values())
        for row in rows:
            if await busy(store, tx, row['session_id']):
                return 'busy'
        if resource_guard is not None and not resource_guard([row['session_id'] for row in rows]):
            return 'busy'
        for row in rows:
            if row['lifecycle'] != 'open':
                continue
            if not remaining:
                return 'cleanup_pending'
            remaining -= await tx.execute(f"UPDATE {store.t.session_state} SET lifecycle='closing', "
                "inbox_version=inbox_version+1 WHERE session_id=? AND inbox_version=? AND lifecycle='open'",
                (row['session_id'], row['inbox_version']))
    return 'closed'


SESSION_TABLE_KEYS = (
    ('messages', ('seq',)), ('compactions', ('compact_seq',)),
    ('usage_observations', ('send_index','round_index')), ('timers', ('timer_id',)),
    ('session_runtime_state', ('state_key',)), ('background_tasks', ('task_id',)),
    ('notes', ('note_id',)), ('project_messages', ('id',)),
    ('hook_events', ('event_id',)), ('follow_up_queue', ('id',)),
    ('session_stats', ('session_id',)),
)


async def delete_sessions(store, session_id, *, cascade=True, limit=1000, resource_guard=None):
    """One allowance includes lifecycle writes, task data, reparenting and sessions."""
    remaining, deleted = batch_limit(limit), []
    if remaining < 2:
        raise ValueError("session deletion requires a limit of at least 2")
    async with store._db.transaction() as tx:
        rows = await lock_sessions(store, tx, session_id, cascade)
        for row in rows:
            if await busy(store, tx, row['session_id']):
                raise SessionBusy('session or descendant has unfinished work')
        if resource_guard is not None and not resource_guard([row['session_id'] for row in rows]):
            raise SessionBusy('session resources still in use')
        for row in rows:
            if row['lifecycle'] == 'open':
                if not remaining:
                    return deleted
                remaining -= await tx.execute(f"UPDATE {store.t.session_state} SET lifecycle='closing', "
                    "inbox_version=inbox_version+1 WHERE session_id=? AND inbox_version=? AND lifecycle='open'",
                    (row['session_id'], row['inbox_version']))
        for row in rows:
            sid = row['session_id']
            if not remaining:
                break
            if store.has_message_ids:
                roots = await tx.fetchall(f'SELECT job_id FROM {store.table_prefix}jobs '
                    'WHERE session_id=? AND parent_job_id IS NULL ORDER BY job_id LIMIT ?', (sid, remaining))
                for root in roots:
                    await lock_root(store, tx, root['job_id'])
                    count, retained = await tree_runtime(store, tx, root['job_id'], 0, remaining, force=True)
                    remaining -= count
                    if retained or not remaining:
                        return deleted
                    counts = {}
                    done = await delete_tree(store, tx, root['job_id'], remaining, counts)
                    remaining -= sum(v for k, v in counts.items() if k != 'task_trees')
                    if not done:
                        return deleted
                if roots and not remaining:
                    return deleted
        for row in reversed(rows):
            sid = row['session_id']
            if not remaining:
                break
            if store.has_message_ids and await tx.fetchone(
                    f'SELECT job_id FROM {store.table_prefix}jobs WHERE session_id=? LIMIT 1', (sid,)):
                continue
            if cascade and await tx.fetchone(
                    f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? "
                    "AND lifecycle<>'detached' LIMIT 1", (sid,)):
                continue
            specs = list(SESSION_TABLE_KEYS)
            if store.has_message_ids:
                specs += [('task_context_events', ('event_id',)), ('session_controls', ('session_id',))]
            for name, keys in specs:
                count = await delete_batch(tx, store.table_prefix+name, keys, 'session_id=?', (sid,), remaining)
                remaining -= count
                if not remaining:
                    return deleted
            # Detached children (or all children with cascade=False) are retained.
            children = await tx.fetchall(f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? LIMIT ?',
                                         (sid, remaining))
            for child in children:
                remaining -= await tx.execute(f'UPDATE {store.t.sessions} SET parent_session_id=NULL WHERE session_id=?',
                                              (child['session_id'],))
            if remaining < 2:
                return deleted
            remaining -= await tx.execute(f'DELETE FROM {store.t.session_state} WHERE session_id=?', (sid,))
            count = await tx.execute(f'DELETE FROM {store.t.sessions} WHERE session_id=?', (sid,))
            remaining -= count
            if count:
                deleted.append(sid)
                store._forget_appended(sid)
    return deleted
