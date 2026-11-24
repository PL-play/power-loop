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
# Leading ASCII Unit Separator (\x1f) is non-printable and therefore disallowed in all legal
# session IDs (runtime_types.identifier), guaranteeing zero collision between caller-supplied
# raw session IDs and system continuation tokens.
_CURSOR_PREFIX = '\x1flc:'
_INCLUSIVE_CURSOR_PREFIX = '\x1flci:'
_RAW_EXCLUSIVE_PREFIX = '\x1fsid:gt:'
_RAW_INCLUSIVE_PREFIX = '\x1fsid:ge:'
_SUPPLEMENTARY_CURSOR_PREFIX = '\x1fsupp:'


def _encode_supplementary_cursor(step: int) -> str:
    return f'{_SUPPLEMENTARY_CURSOR_PREFIX}{int(step)}'


def _decode_supplementary_cursor(token: str) -> int | None:
    if not token.startswith(_SUPPLEMENTARY_CURSOR_PREFIX):
        return None
    raw = token[len(_SUPPLEMENTARY_CURSOR_PREFIX):]
    if not raw.isdigit() or int(raw) < 1:
        return None
    return int(raw)


async def _delete_active_terminal_inbox(store, tx, *, cutoff_ms: int, limit: int, session_id: str | None) -> int:
    """Clean expired terminal follow_up_queue rows even for active open/suspended sessions.

    Safety invariants:
    - Cleans only settled rows: ``(terminal_status IS NOT NULL AND completed_at < cutoff)``
      OR completed fast-path rows ``(status='delivered' AND terminal_status IS NULL AND
      delivered_at < cutoff AND (delivered_send_index < sessions.send_index OR
      (delivered_send_index = sessions.send_index AND sessions.execution_state <> 'running')))``.
    - Never deletes ``pending``, ``waiting_continuation``, or currently running inputs.
    - Protects ``workflow_driver.start_input_message_id``.
    - Never touches raw ``messages``.
    """
    if limit <= 0:
        return 0
    where = (
        "((f.terminal_status IS NOT NULL AND f.completed_at<?) "
        "OR (f.status='delivered' AND f.terminal_status IS NULL AND f.delivered_at<? "
        "AND (s.session_id IS NULL OR f.delivered_send_index<s.send_index "
        "OR (f.delivered_send_index=s.send_index AND s.execution_state<>'running')))) "
        "AND (s.session_id IS NULL OR NOT (s.session_kind='workflow_driver' "
        "AND s.lifecycle IN ('open','suspended') AND s.start_input_message_id=f.item_id))"
    )
    params: list = [cutoff_ms, cutoff_ms]
    if session_id is not None:
        where += " AND f.session_id=?"
        params.append(session_id)
    rows = await tx.fetchall(
        f"SELECT f.id FROM {store.t.follow_up_queue} f "
        f"LEFT JOIN {store.t.sessions} s ON s.session_id=f.session_id "
        f"WHERE {where} ORDER BY f.id ASC LIMIT ?",
        (*params, limit),
    )
    if not rows:
        return 0
    ids = tuple(row['id'] for row in rows)
    marks = ','.join('?' for _ in ids)
    return await tx.execute(
        f"DELETE FROM {store.t.follow_up_queue} "
        f"WHERE id IN ({marks}) AND ("
        "(terminal_status IS NOT NULL AND completed_at<?) "
        "OR (status='delivered' AND terminal_status IS NULL AND delivered_at<?)) "
        f"AND NOT EXISTS (SELECT 1 FROM {store.t.sessions} s "
        f"WHERE s.session_id={store.t.follow_up_queue}.session_id "
        "AND s.session_kind='workflow_driver' AND s.lifecycle IN ('open','suspended') "
        f"AND s.start_input_message_id={store.t.follow_up_queue}.item_id)",
        (*ids, cutoff_ms, cutoff_ms),
    )


async def _delete_superseded_projections(store, tx, *, cutoff_ms: int, limit: int, session_id: str | None) -> int:
    """Clean expired project_messages superseded by the reader-selected latest compact row.

    Mirrors ``SessionStore.latest_project_compact`` (``ORDER BY send_index DESC LIMIT 1``):
    keeps the highest-``send_index`` compact row and only deletes non-compact rows covered
    by THAT row's ``compact_to_send`` plus strictly older compact rows (``send_index < latest``).

    The conditional ``DELETE`` re-evaluates the current committed latest compact row at delete
    time so concurrent ``upsert_project_message`` commits under ``READ COMMITTED`` (PostgreSQL /
    MySQL) cannot cause rows needed by a newer or rewritten compact summary to be deleted.
    """
    if limit <= 0:
        return 0
    sub_where = "kind='compact'"
    sub_params: list = []
    if session_id is not None:
        sub_where += " AND session_id=?"
        sub_params.append(session_id)
    tbl = store.t.project_messages
    rows = await tx.fetchall(
        f"SELECT p.id, p.session_id FROM {tbl} p "
        f"JOIN ("
        f"SELECT c1.session_id, c1.send_index AS latest_compact_send, "
        f"c1.compact_to_send AS latest_compact_to "
        f"FROM {tbl} c1 "
        f"JOIN (SELECT session_id, MAX(send_index) AS max_send FROM {tbl} "
        f"WHERE {sub_where} GROUP BY session_id) m "
        "ON m.session_id=c1.session_id AND m.max_send=c1.send_index "
        "WHERE c1.kind='compact' AND c1.compact_to_send IS NOT NULL"
        ") c ON c.session_id=p.session_id "
        "WHERE p.created_at<? AND ("
        "(p.kind<>'compact' AND p.send_index<=c.latest_compact_to) "
        "OR (p.kind='compact' AND p.send_index<c.latest_compact_send)) "
        "ORDER BY p.id ASC LIMIT ?",
        (*sub_params, cutoff_ms, limit),
    )
    if not rows:
        return 0
    ids = tuple(row['id'] for row in rows)
    marks = ','.join('?' for _ in ids)
    if store._db.dialect.name == 'mysql':
        sids = tuple(dict.fromkeys(row['session_id'] for row in rows))
        sid_marks = ','.join('?' for _ in sids)
        return await tx.execute(
            f"DELETE p FROM {tbl} p "
            f"JOIN ("
            f"SELECT c1.session_id, c1.send_index AS latest_compact_send, "
            f"c1.compact_to_send AS latest_compact_to "
            f"FROM {tbl} c1 "
            f"JOIN (SELECT session_id, MAX(send_index) AS max_send FROM {tbl} "
            f"WHERE kind='compact' AND session_id IN ({sid_marks}) GROUP BY session_id) m "
            "ON m.session_id=c1.session_id AND m.max_send=c1.send_index "
            "WHERE c1.kind='compact' AND c1.compact_to_send IS NOT NULL"
            ") c ON c.session_id=p.session_id "
            f"WHERE p.id IN ({marks}) AND p.created_at<? AND ("
            "(p.kind<>'compact' AND p.send_index<=c.latest_compact_to) "
            "OR (p.kind='compact' AND p.send_index<c.latest_compact_send))",
            (*sids, *ids, cutoff_ms),
        )
    return await tx.execute(
        f"DELETE FROM {tbl} WHERE id IN ({marks}) AND created_at<? "
        f"AND EXISTS (SELECT 1 FROM {tbl} c1 "
        f"WHERE c1.session_id={tbl}.session_id "
        "AND c1.kind='compact' AND c1.compact_to_send IS NOT NULL "
        f"AND c1.send_index=(SELECT MAX(c2.send_index) FROM {tbl} c2 "
        f"WHERE c2.session_id={tbl}.session_id AND c2.kind='compact') "
        f"AND (({tbl}.kind<>'compact' AND {tbl}.send_index<=c1.compact_to_send) "
        f"OR ({tbl}.kind='compact' AND {tbl}.send_index<c1.send_index)))",
        (*ids, cutoff_ms),
    )


async def _delete_orphan_shared_state(store, tx, *, cutoff_ms: int, limit: int, session_id: str | None) -> int:
    """Clean expired session-scoped ``shared:*`` rows with no owning session row.

    Host-owned blackboards (``session_id LIKE 'blackboard:%'``) never have a
    matching ``sessions`` row and are managed by the host via ``StoreBlackboard.delete()``,
    so they are explicitly preserved.
    """
    if limit <= 0:
        return 0
    where = (
        "r.state_key LIKE 'shared:%' "
        "AND r.session_id NOT LIKE 'blackboard:%' "
        "AND r.updated_at<? AND s.session_id IS NULL"
    )
    params: list = [cutoff_ms]
    if session_id is not None:
        where += " AND r.session_id=?"
        params.append(session_id)
    rows = await tx.fetchall(
        f"SELECT r.session_id, r.state_key FROM {store.t.session_runtime_state} r "
        f"LEFT JOIN {store.t.sessions} s ON s.session_id=r.session_id "
        f"WHERE {where} ORDER BY r.session_id ASC, r.state_key ASC LIMIT ?",
        (*params, limit),
    )
    if not rows:
        return 0
    marks = ','.join('(?,?)' for _ in rows)
    values = tuple(val for row in rows for val in (row['session_id'], row['state_key']))
    return await tx.execute(
        f"DELETE FROM {store.t.session_runtime_state} "
        f"WHERE (session_id,state_key) IN ({marks}) "
        "AND state_key LIKE 'shared:%' AND session_id NOT LIKE 'blackboard:%' AND updated_at<? "
        f"AND NOT EXISTS (SELECT 1 FROM {store.t.sessions} s WHERE s.session_id={store.t.session_runtime_state}.session_id)",
        (*values, cutoff_ms),
    )


def _encode_cursor(
    lifecycle: str, updated_at: int, session_id: str, *, inclusive: bool = False, step: int = 1,
) -> str:
    if inclusive:
        return f'{_INCLUSIVE_CURSOR_PREFIX}{int(step)}:{lifecycle}:{int(updated_at)}:{session_id}'
    return f'{_CURSOR_PREFIX}{lifecycle}:{int(updated_at)}:{session_id}'


def _encode_raw_cursor(session_id: str, *, inclusive: bool = False, step: int = 1) -> str:
    if inclusive:
        return f'{_RAW_INCLUSIVE_PREFIX}{int(step)}:{session_id}'
    return f'{_RAW_EXCLUSIVE_PREFIX}{session_id}'


def _decode_cursor(token: str) -> tuple[str, int, str, bool, int] | None:
    if token.startswith(_INCLUSIVE_CURSOR_PREFIX):
        body = token[len(_INCLUSIVE_CURSOR_PREFIX):]
        parts = body.split(':', 3)
        if len(parts) != 4 or parts[1] not in _LIFECYCLE_ORDER:
            return None
        try:
            step = int(parts[0])
            ts = int(parts[2])
        except ValueError:
            return None
        if step < 1:
            return None
        return parts[1], ts, parts[3], True, step
    if token.startswith(_CURSOR_PREFIX):
        body = token[len(_CURSOR_PREFIX):]
        parts = body.split(':', 2)
        if len(parts) != 3 or parts[0] not in _LIFECYCLE_ORDER:
            return None
        try:
            return parts[0], int(parts[1]), parts[2], False, 0
        except ValueError:
            return None
    return None


async def _fetch_lifecycle_candidates(
    store, *, eligibility_cutoff: int, limit: int,
    lifecycles: tuple[str, ...], after_session_id: str | None = None,
    session_id: str | None = None,
) -> tuple[list[dict], bool, int]:
    """Scan idle expired candidate sessions via ``idx_sessions_lifecycle``.

    Index columns: ``(lifecycle, execution_state, updated_at, session_id)``.
    Supports structured lifecycle cursors (exclusive and monotonic-step inclusive resume cursors)
    produced by ``purge_expired_page`` as well as raw ``session_id`` cursors.
    """
    if session_id is not None:
        row = await store._db.fetchone(
            f"SELECT session_id,lifecycle,updated_at FROM {store.t.sessions} "
            "WHERE session_id=? AND execution_state='idle' AND updated_at<?",
            (session_id, eligibility_cutoff),
        )
        return ([dict(row)] if row is not None else []), False, 0

    if after_session_id is not None:
        parsed = _decode_cursor(after_session_id)
        if parsed is None:
            # Backward-compatible raw session_id cursor, plus explicit \x1fsid:gt:/\x1fsid:ge: tokens
            # emitted by purge_expired_page so arbitrary printable session_ids never collide.
            raw_step = 0
            if after_session_id.startswith(_RAW_INCLUSIVE_PREFIX):
                body = after_session_id[len(_RAW_INCLUSIVE_PREFIX):]
                step_s, sep, rem_sid = body.partition(':')
                if sep and step_s.isdigit() and int(step_s) >= 1 and rem_sid:
                    raw_step = int(step_s)
                    raw_sid = rem_sid
                    cmp_op = '>='
                else:
                    raw_sid = after_session_id
                    cmp_op = '>'
            elif after_session_id.startswith(_RAW_EXCLUSIVE_PREFIX):
                raw_sid = after_session_id[len(_RAW_EXCLUSIVE_PREFIX):]
                cmp_op = '>'
            else:
                raw_sid = after_session_id
                cmp_op = '>'
            ph = ','.join('?' for _ in lifecycles)
            rows = await store._db.fetchall(
                f"SELECT session_id,lifecycle,updated_at FROM {store.t.sessions} "
                f"WHERE lifecycle IN ({ph}) AND execution_state='idle' "
                f"AND updated_at<? AND session_id{cmp_op}? ORDER BY session_id ASC LIMIT ?",
                (*lifecycles, eligibility_cutoff, raw_sid, limit),
            )
            return [dict(r) for r in rows], False, raw_step
        start_lc, start_ts, start_sid, start_inclusive, cursor_step = parsed
    else:
        start_lc, start_ts, start_sid, start_inclusive, cursor_step = None, None, None, False, 0

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
    return collected, True, cursor_step


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
    supp_step = (
        _decode_supplementary_cursor(after_session_id)
        if candidate_limit is not None and after_session_id is not None
        else None
    )
    if supp_step is not None:
        candidates, use_lifecycle_cursor, cursor_step = [], True, 0
    else:
        candidates, use_lifecycle_cursor, cursor_step = await _fetch_lifecycle_candidates(
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
                if use_lifecycle_cursor else _encode_raw_cursor(sid)
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
                        and state['lifecycle'] in ('open', 'suspended')
                        and state['start_input_message_id'] is not None
                    ):
                        driver_clause = ' AND item_id<>?'
                        driver_params = (state['start_input_message_id'],)
                    count = await delete_batch(
                        tx, store.t.follow_up_queue, ('id',),
                        "session_id=? AND ("
                        "(terminal_status IS NOT NULL AND completed_at<?) "
                        "OR (status='delivered' AND terminal_status IS NULL AND delivered_at<? "
                        "AND (delivered_send_index<? OR (delivered_send_index=? AND ?<>'running')))"
                        f"){driver_clause}",
                        (sid, cutoff['inbox'], cutoff['inbox'], cur_si, cur_si, execution_state, *driver_params),
                        remaining,
                    )
                    if count:
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
                            if count:
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
                            if count:
                                counts['sessions'] = counts.get('sessions', 0) + count
                            remaining -= count
                            if count:
                                session_deleted = True
                                store._forget_appended(sid)
        if remaining == 0 and not session_deleted:
            # Current session was truncated by the row allowance: do NOT advance `last` past it;
            # return a strictly advancing inclusive cursor (monotonic step) so callers such as
            # scheduler progress checks confirm forward progress while re-entering this session.
            next_step = cursor_step + 1
            truncated_cursor = (
                _encode_cursor(
                    candidate['lifecycle'], candidate['updated_at'], sid,
                    inclusive=True, step=next_step,
                )
                if use_lifecycle_cursor
                else _encode_raw_cursor(sid, inclusive=True, step=next_step)
            )
            break
        last = (
            _encode_cursor(candidate['lifecycle'], candidate['updated_at'], sid)
            if use_lifecycle_cursor else _encode_raw_cursor(sid)
        )
    fetched = len(candidates)
    if truncated_cursor is not None:
        next_after = truncated_cursor
    elif fetched == candidate_limit or scanned < fetched:
        next_after = last
    else:
        next_after = None

    # Supplementary bounded cleanup after lifecycle candidates complete:
    # 1) Active open/suspended sessions' expired terminal follow_up_queue rows
    # 2) Expired project_messages superseded by compact_to_send
    # 3) Orphaned shared:* runtime_state rows without an owning session
    # Never touches active open sessions' raw messages.
    if next_after is None and remaining > 0:
        supp_sid = session_id if candidate_limit is None else None
        supp_deleted = 0
        async with store._db.transaction() as tx:
            if 'inbox' in wanted and remaining > 0:
                count = await _delete_active_terminal_inbox(
                    store, tx, cutoff_ms=cutoff['inbox'], limit=remaining, session_id=supp_sid,
                )
                if count:
                    counts['inbox'] = counts.get('inbox', 0) + count
                    supp_deleted += count
                    remaining -= count
            if 'projections' in wanted and remaining > 0:
                count = await _delete_superseded_projections(
                    store, tx, cutoff_ms=cutoff['projections'], limit=remaining, session_id=supp_sid,
                )
                if count:
                    counts['projections'] = counts.get('projections', 0) + count
                    supp_deleted += count
                    remaining -= count
            if 'runtime' in wanted and remaining > 0:
                count = await _delete_orphan_shared_state(
                    store, tx, cutoff_ms=cutoff['runtime'], limit=remaining, session_id=supp_sid,
                )
                if count:
                    counts['runtime'] = counts.get('runtime', 0) + count
                    supp_deleted += count
                    remaining -= count
        if remaining == 0 and supp_deleted > 0 and candidate_limit is not None:
            next_after = _encode_supplementary_cursor((supp_step or 0) + 1)

    return RetentionPage(counts, scanned, fetched, next_after)
