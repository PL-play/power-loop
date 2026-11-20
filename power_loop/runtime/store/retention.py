"""Bounded retention over schema 18 sessions, settled inputs and derived history."""
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
        scope, params = (' AND session_id=?', (session_id,)) if session_id else ('', ())
        candidates = await store._db.fetchall(
            f"SELECT session_id FROM {store.t.sessions} "
            "WHERE execution_state='idle' AND updated_at<?"
            + scope + ' ORDER BY updated_at,session_id LIMIT ?',
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
        if candidate_limit is not None and candidate['updated_at'] >= eligibility_cutoff:
            continue
        async with store._db.transaction() as tx:
            try:
                state = await store._db.dialect.lock_state(tx, store.t.sessions, sid)
            except ValueError:
                continue
            if state['updated_at'] >= eligibility_cutoff or await busy(store, tx, sid, state):
                continue
            if await tx.fetchone(
                f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? LIMIT 1', (sid,),
            ):
                continue
            if 'inbox' in wanted:
                cur_si = int(state['send_index'])
                count = await delete_batch(
                    tx, store.t.follow_up_queue, ('id',),
                    "session_id=? AND ("
                    "(terminal_status IS NOT NULL AND completed_at<?) "
                    "OR (status='delivered' AND terminal_status IS NULL "
                    "AND delivered_at<? AND delivered_send_index<=?))",
                    (sid, cutoff['inbox'], cutoff['inbox'], cur_si), remaining,
                )
                counts['inbox'] = counts.get('inbox', 0) + count
                remaining -= count
            if state['lifecycle'] in ('open', 'suspended') or await tx.fetchone(
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
            if 'sessions' not in wanted or remaining < 1 or state['updated_at'] >= cutoff['sessions']:
                continue
            retained = False
            for table in (store.t.project_messages, store.t.session_runtime_state, store.t.messages):
                if await tx.fetchone(f'SELECT session_id FROM {table} WHERE session_id=? LIMIT 1', (sid,)):
                    retained = True
                    break
            if retained:
                continue
            count = await tx.execute(f'DELETE FROM {store.t.sessions} WHERE session_id=?', (sid,))
            counts['sessions'] = counts.get('sessions', 0) + count
            remaining -= count
            store._forget_appended(sid)
    fetched = len(candidates)
    next_after = last if fetched == candidate_limit or scanned < fetched else None
    return RetentionPage(counts, scanned, fetched, next_after)
