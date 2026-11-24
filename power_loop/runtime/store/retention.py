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


_LIFECYCLE_ORDER = ('closed', 'closing', 'open', 'suspended')
_CURSOR_PREFIX = 'lc:'
_INCLUSIVE_CURSOR_PREFIX = 'lci:'
_RAW_INCLUSIVE_PREFIX = 'sidi:'


def _encode_cursor(lifecycle: str, updated_at: int, session_id: str, *, inclusive: bool = False) -> str:
    prefix = _INCLUSIVE_CURSOR_PREFIX if inclusive else _CURSOR_PREFIX
    return f'{prefix}{lifecycle}:{int(updated_at)}:{session_id}'


def _decode_cursor(token: str) -> tuple[str, int, str, bool] | None:
    if token.startswith(_INCLUSIVE_CURSOR_PREFIX):
        inclusive = True
        body = token[len(_INCLUSIVE_CURSOR_PREFIX):]
    elif token.startswith(_CURSOR_PREFIX):
        inclusive = False
        body = token[len(_CURSOR_PREFIX):]
    else:
        return None
    parts = body.split(':', 2)
    if len(parts) != 3 or parts[0] not in _LIFECYCLE_ORDER:
        return None
    try:
        return parts[0], int(parts[1]), parts[2], inclusive
    except ValueError:
        return None


async def _fetch_lifecycle_candidates(
    store, *, eligibility_cutoff: int, limit: int,
    lifecycles: tuple[str, ...], after_session_id: str | None = None,
    session_id: str | None = None,
) -> tuple[list[dict], bool]:
    """Scan idle expired candidate sessions via ``idx_sessions_lifecycle``.

    Index columns: ``(lifecycle, execution_state, updated_at, session_id)``.
    Supports both structured lifecycle cursors (``lc:<lifecycle>:<updated_at>:<session_id>``
    and inclusive resume cursors ``lci:...``) produced by ``purge_expired_page`` and raw
    ``session_id`` cursors for backward compatibility.
    """
    if session_id is not None:
        row = await store._db.fetchone(
            f"SELECT session_id,lifecycle,updated_at FROM {store.t.sessions} "
            "WHERE session_id=? AND execution_state='idle' AND updated_at<?",
            (session_id, eligibility_cutoff),
        )
        return ([dict(row)] if row is not None else []), False

    if after_session_id is not None:
        parsed = _decode_cursor(after_session_id)
        if parsed is None:
            # Backward-compatible raw session_id cursor (or inclusive resume cursor for raw mode).
            inclusive_raw = after_session_id.startswith(_RAW_INCLUSIVE_PREFIX)
            raw_sid = (
                after_session_id[len(_RAW_INCLUSIVE_PREFIX):]
                if inclusive_raw else after_session_id
            )
            cmp_op = '>=' if inclusive_raw else '>'
            ph = ','.join('?' for _ in lifecycles)
            rows = await store._db.fetchall(
                f"SELECT session_id,lifecycle,updated_at FROM {store.t.sessions} "
                f"WHERE lifecycle IN ({ph}) AND execution_state='idle' "
                f"AND updated_at<? AND session_id{cmp_op}? ORDER BY session_id ASC LIMIT ?",
                (*lifecycles, eligibility_cutoff, raw_sid, limit),
            )
            return [dict(r) for r in rows], False
        start_lc, start_ts, start_sid, start_inclusive = parsed
    else:
        start_lc, start_ts, start_sid, start_inclusive = None, None, None, False

    collected: list[dict] = []
    started = start_lc is None
    for lc in lifecycles:
        if len(collected) >= limit:
            break
        if not started:
            if lc != start_lc:
                continue
            started = True
            need = limit - len(collected)
            cmp_op = '>=' if start_inclusive else '>'
            rows = await store._db.fetchall(
                f"SELECT session_id,lifecycle,updated_at FROM {store.t.sessions} "
                "WHERE lifecycle=? AND execution_state='idle' AND updated_at<? "
                f"AND (updated_at>? OR (updated_at=? AND session_id{cmp_op}?)) "
                "ORDER BY updated_at ASC,session_id ASC LIMIT ?",
                (lc, eligibility_cutoff, start_ts, start_ts, start_sid, need),
            )
        else:
            need = limit - len(collected)
            rows = await store._db.fetchall(
                f"SELECT session_id,lifecycle,updated_at FROM {store.t.sessions} "
                "WHERE lifecycle=? AND execution_state='idle' AND updated_at<? "
                "ORDER BY updated_at ASC,session_id ASC LIMIT ?",
                (lc, eligibility_cutoff, need),
            )
        collected.extend(dict(r) for r in rows)
    return collected, True


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
    lifecycles = _LIFECYCLE_ORDER if 'inbox' in wanted else ('closed', 'closing')
    scan_limit = remaining if candidate_limit is None else candidate_limit
    candidates, use_lifecycle_cursor = await _fetch_lifecycle_candidates(
        store,
        eligibility_cutoff=eligibility_cutoff,
        limit=scan_limit,
        lifecycles=lifecycles,
        after_session_id=after_session_id if candidate_limit is not None else None,
        session_id=session_id if candidate_limit is None else None,
    )
    counts = {}
    scanned, last = 0, None
    truncated_cursor: str | None = None
    for candidate in candidates:
        if not remaining:
            break
        sid = candidate['session_id']
        scanned += 1
        session_deleted = False
        if candidate['updated_at'] >= eligibility_cutoff:
            last = (
                _encode_cursor(candidate['lifecycle'], candidate['updated_at'], sid)
                if use_lifecycle_cursor else sid
            )
            continue
        async with store._db.transaction() as tx:
            try:
                state = await store._db.dialect.lock_state(tx, store.t.sessions, sid)
            except ValueError:
                state = None
            if (
                state is not None
                and state['updated_at'] < eligibility_cutoff
                and not await busy(store, tx, sid, state)
                and not await tx.fetchone(
                    f'SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? LIMIT 1', (sid,),
                )
            ):
                if 'inbox' in wanted:
                    cur_si = int(state['send_index'])
                    execution_state = str(state['execution_state'])
                    driver_clause = ''
                    driver_params: tuple = ()
                    if (
                        state['session_kind'] == 'workflow_driver'
                        and state['start_input_message_id'] is not None
                    ):
                        driver_clause = ' AND item_id<>?'
                        driver_params = (state['start_input_message_id'],)
                    count = await delete_batch(
                        tx, store.t.follow_up_queue, ('id',),
                        "session_id=? AND ("
                        "(terminal_status IS NOT NULL AND completed_at<?) "
                        "OR (status='delivered' AND terminal_status IS NULL AND delivered_at<? "
                        "AND (delivered_send_index<? OR (delivered_send_index=? AND ?<>'running'))"
                        f'{driver_clause}))',
                        (sid, cutoff['inbox'], cutoff['inbox'], cur_si, cur_si, execution_state, *driver_params),
                        remaining,
                    )
                    counts['inbox'] = counts.get('inbox', 0) + count
                    remaining -= count
                if (
                    remaining > 0
                    and state['lifecycle'] not in ('open', 'suspended')
                    and not await tx.fetchone(
                        f'SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? LIMIT 1', (sid,),
                    )
                ):
                    for category, table, keys, age_column in (
                        ('projections', store.t.project_messages, ('id',), 'created_at'),
                        ('runtime', store.t.session_runtime_state, ('state_key',), 'updated_at'),
                        ('messages', store.t.messages, ('id',), 'created_at'),
                    ):
                        if category in wanted and remaining > 0:
                            count = await delete_batch(tx, table, keys,
                                f'session_id=? AND {age_column}<?', (sid, cutoff[category]), remaining)
                            counts[category] = counts.get(category, 0) + count
                            remaining -= count
                    if 'sessions' in wanted and remaining >= 1 and state['updated_at'] < cutoff['sessions']:
                        retained = False
                        for table in (store.t.project_messages, store.t.session_runtime_state, store.t.messages):
                            if await tx.fetchone(f'SELECT session_id FROM {table} WHERE session_id=? LIMIT 1', (sid,)):
                                retained = True
                                break
                        if not retained:
                            count = await tx.execute(f'DELETE FROM {store.t.sessions} WHERE session_id=?', (sid,))
                            counts['sessions'] = counts.get('sessions', 0) + count
                            remaining -= count
                            if count:
                                session_deleted = True
                                store._forget_appended(sid)
        if remaining == 0 and not session_deleted:
            # Current session was truncated by the row allowance: do NOT advance `last` past it;
            # return an inclusive cursor so the next page re-enters this session cleanly.
            truncated_cursor = (
                _encode_cursor(candidate['lifecycle'], candidate['updated_at'], sid, inclusive=True)
                if use_lifecycle_cursor else f'{_RAW_INCLUSIVE_PREFIX}{sid}'
            )
            break
        last = (
            _encode_cursor(candidate['lifecycle'], candidate['updated_at'], sid)
            if use_lifecycle_cursor else sid
        )
    fetched = len(candidates)
    if truncated_cursor is not None:
        next_after = truncated_cursor
    elif fetched == candidate_limit or scanned < fetched:
        next_after = last
    else:
        next_after = None
    return RetentionPage(counts, scanned, fetched, next_after)
