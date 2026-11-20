"""Bounded retention over schema 17 sessions, terminal inputs and derived history."""
from __future__ import annotations

from dataclasses import dataclass

DAY_MS = 86400000
DEFAULT_DAYS = {'sessions': 30, 'messages': 30, 'inbox': 7, 'runtime': 30, 'projections': 30}


@dataclass(frozen=True)
class RetentionPage:
    deleted: dict[str, int]
    scanned: int
    fetched: int
    next_after: str | None


def batch_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('retention limit must be between 1 and 1000')
    return limit


async def delete_batch(tx, table, keys, where, params, limit, *, order=None):
    if limit <= 0:
        return 0
    rows = await tx.fetchall(f"SELECT {','.join(keys)} FROM {table} WHERE {where} "
        f"ORDER BY {order or ','.join(keys)} LIMIT ?", (*params, limit))
    if not rows:
        return 0
    marks = ','.join('('+','.join('?' for _ in keys)+')' for _ in rows)
    values = tuple(row[key] for row in rows for key in keys)
    return await tx.execute(f"DELETE FROM {table} WHERE ({','.join(keys)}) IN ({marks}) AND ({where})",
                            (*values, *params))


async def purge_expired(store, now, *, limit, retention_days, session_id=None, categories=None,
                        candidate_limit=None, after_session_id=None):
    from .closing import busy

    remaining = batch_limit(limit)
    days = {**DEFAULT_DAYS, **(retention_days or {})}
    if days.keys() != DEFAULT_DAYS.keys() or any(type(v) not in (int, float) or v < 0 for v in days.values()):
        raise ValueError('invalid retention days')
    wanted = set(DEFAULT_DAYS if categories is None else categories)
    if not wanted <= DEFAULT_DAYS.keys():
        raise ValueError('unknown retention category')
    cutoff = {key: int(now-value*DAY_MS) for key, value in days.items()}
    eligibility_cutoff = max(cutoff.values())
    if candidate_limit is None:
        # Preserve directed and count-only callers' candidate order.
        scope, params = (' AND s.session_id=?', (session_id,)) if session_id else ('', ())
        candidates = await store._db.fetchall(
            f'SELECT s.session_id FROM {store.t.sessions} s JOIN {store.t.session_state} st '
            "ON st.session_id=s.session_id WHERE st.execution_state='idle' AND s.updated_at<?"
            + scope + ' ORDER BY s.updated_at,s.session_id LIMIT ?',
            (eligibility_cutoff, *params, remaining),
        )
    else:
        scope, params = (' WHERE session_id>?', (after_session_id,)) if after_session_id is not None else ('', ())
        candidates = await store._db.fetchall(
            f'SELECT session_id,updated_at FROM {store.t.sessions}{scope} ORDER BY session_id LIMIT ?',
            (*params, candidate_limit),
        )
    counts = {}
    scanned, last = 0, None
    for candidate in candidates:
        if not remaining:
            break
        sid = last = candidate['session_id']
        scanned += 1
        # Skip recent rows from the page snapshot. Deletion eligibility is
        # always checked again under the state lock.
        if candidate_limit is not None and candidate['updated_at'] >= eligibility_cutoff:
            continue
        async with store._db.transaction() as tx:
            try:
                state = await store._db.dialect.lock_state(tx, store.t.session_state, sid)
            except ValueError:
                # A peer can delete this candidate before we lock it.
                continue
            row = await tx.fetchone(f'SELECT lifecycle,updated_at FROM {store.t.sessions} WHERE session_id=?', (sid,))
            if row is None or row['updated_at'] >= eligibility_cutoff or await busy(store, tx, sid, state):
                continue
            if await tx.fetchone(
                f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? LIMIT 1', (sid,),
            ):
                continue
            if 'inbox' in wanted:
                count = await delete_batch(tx, store.t.follow_up_queue, ('id',),
                    'session_id=? AND terminal_status IS NOT NULL AND completed_at<?',
                    (sid, cutoff['inbox']), remaining)
                counts['inbox'] = counts.get('inbox', 0) + count
                remaining -= count
            # Open sessions retain their context/journal. Closed sessions can be
            # drained in bounded batches, but results remain while inputs reference them.
            if row['lifecycle'] == 'open' or await tx.fetchone(
                f'SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? LIMIT 1', (sid,),
            ):
                continue
            for category, table, keys, age_column in (
                ('projections', store.t.project_messages, ('id',), 'created_at'),
                ('runtime', store.t.session_runtime_state, ('state_key',), 'updated_at'),
                ('messages', store.t.messages, ('id',), 'created_at'),
            ):
                if category in wanted:
                    count = await delete_batch(tx, table, keys,
                        f'session_id=? AND {age_column}<?', (sid, cutoff[category]), remaining)
                    counts[category] = counts.get(category, 0) + count
                    remaining -= count
            if 'sessions' not in wanted or remaining < 2 or row['updated_at'] >= cutoff['sessions']:
                continue
            retained = False
            for table in (store.t.project_messages, store.t.session_runtime_state, store.t.messages):
                if await tx.fetchone(f'SELECT session_id FROM {table} WHERE session_id=? LIMIT 1', (sid,)):
                    retained = True
                    break
            if retained:
                continue
            count = await tx.execute(f'DELETE FROM {store.t.session_state} WHERE session_id=?', (sid,))
            count += await tx.execute(f'DELETE FROM {store.t.sessions} WHERE session_id=?', (sid,))
            counts['sessions'] = counts.get('sessions', 0) + count
            remaining -= count
            store._forget_appended(sid)
    fetched = len(candidates)
    # A full page needs one more seek to confirm the tail. Resume at the last
    # checked candidate when deletion used the allowance before the page ended.
    next_after = last if fetched == candidate_limit or scanned < fetched else None
    return RetentionPage(counts, scanned, fetched, next_after)
