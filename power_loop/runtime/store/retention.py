"""Host-invoked retention; one shared row allowance per call, including every transaction."""
from __future__ import annotations

from . import message_usage

DAY_MS = 86400000
DEFAULT_DAYS = {'task_trees': 30, 'executors': 1, 'follow_ups': 7, 'timers': 7,
                'hooks': 30, 'usage': 90, 'runtime': 30}


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


async def prune_timers(store, session_id, *, statuses, cutoff, limit):
    limit = batch_limit(limit)
    if not statuses:
        return 0
    if not set(statuses) <= {'fired', 'cancelled'}:
        raise ValueError('only terminal timers can be pruned')
    where, params = 'status IN ('+','.join('?' for _ in statuses)+')', list(statuses)
    if session_id is not None:
        where += ' AND session_id=?'
        params.append(session_id)
    if cutoff is not None:
        where += ' AND updated_at<?'
        params.append(cutoff)
    async with store._db.transaction() as tx:
        return await delete_batch(tx, store.t.timers, ('session_id', 'timer_id'), where, params,
                                  limit, order='updated_at,session_id,timer_id')


async def prune_usage(store, session_id, *, cutoff, limit):
    """Both observation sources, without reading the historical reporting view."""
    limit = batch_limit(limit)
    where, params = 'created_at<?', [cutoff]
    if session_id is not None:
        where += ' AND session_id=?'
        params.append(session_id)
    backend = store._db.dialect.name
    present = ("CAST(meta_json AS jsonb)->'_round_usage'" if backend == 'postgres'
               else "JSON_EXTRACT(meta_json,'$._round_usage')")
    columns = 'session_id,send_index,round_index,created_at'
    async with store._db.transaction() as tx:
        observations = await tx.fetchall(f'SELECT {columns} FROM {store.t.usage_observations} '
            f'WHERE {where} ORDER BY created_at LIMIT ?', (*params, limit))
        messages = await tx.fetchall(f'SELECT seq,{columns} FROM {store.t.messages} '
            f'WHERE {where} AND {present} IS NOT NULL ORDER BY created_at LIMIT ?', (*params, limit))
        changed = 0
        for row in sorted([*observations, *messages], key=lambda row: row['created_at']):
            key = (row['session_id'], row['send_index'] or 0, row['round_index'])
            if changed == limit:
                break
            if 'seq' in row:
                affected = await tx.execute(f'UPDATE {store.t.messages} SET meta_json={message_usage.remove_sql(backend)} '
                    f'WHERE session_id=? AND seq=? AND created_at<? AND {present} IS NOT NULL',
                    (row['session_id'], row['seq'], cutoff))
            else:
                affected = await tx.execute(f'DELETE FROM {store.t.usage_observations} '
                    'WHERE session_id=? AND send_index=? AND round_index=? AND created_at<?', (*key, cutoff))
            changed += affected
        return changed


def ready_condition(prefix):
    # All subqueries use a root/job/session key index. A pending alert remains
    # actionable even if its root has already published a closure.
    return ("root.supervision_status='closed' AND NOT EXISTS ("
        f'SELECT 1 FROM {prefix}jobs j WHERE j.root_job_id=root.job_id AND ('
        "j.supervision_status<>'closed' OR j.resource_state<>'stopped' OR j.intake<>'closed')) "
        f'AND NOT EXISTS (SELECT 1 FROM {prefix}jobs j JOIN {prefix}session_obligations o ON o.job_id=j.job_id '
        'WHERE j.root_job_id=root.job_id) '
        f'AND NOT EXISTS (SELECT 1 FROM {prefix}jobs j JOIN {prefix}task_budget_attempts b ON b.job_id=j.job_id '
        "WHERE j.root_job_id=root.job_id AND (b.pending=1 OR b.phase NOT IN ('released','settling'))) "
        f'AND NOT EXISTS (SELECT 1 FROM {prefix}jobs j JOIN {prefix}job_events e ON e.job_id=j.job_id '
        f'JOIN {prefix}follow_up_queue q ON q.session_id=root.session_id AND q.item_id=e.event_id '
        "WHERE j.root_job_id=root.job_id AND e.event_type='task_alert' AND q.status IN ('pending','claimed'))")


async def ready_tree(tx, prefix, root_id):
    return await tx.fetchone(f'SELECT job_id FROM {prefix}jobs root WHERE root.job_id=? AND '
                            +ready_condition(prefix), (root_id,))


async def lock_root(store, tx, root_id):
    suffix = '' if store._db.dialect.name == 'sqlite' else ' FOR UPDATE'
    return await tx.fetchone(f'SELECT * FROM {store.table_prefix}jobs WHERE job_id=?'+suffix, (root_id,))


def runtime_tree_condition(store):
    p, r = store.table_prefix, store.t.session_runtime_state
    concat = (lambda *parts: 'CONCAT('+','.join(parts)+')') if store._db.dialect.name == 'mysql' else (lambda *parts: ' || '.join(parts))
    child = concat("'subagent:child:'", 'j.session_id')
    run = concat("'subagent:run:'", 'SUBSTR(e.event_id,4)')
    workflow = concat("'workflow:run:'", 'SUBSTR(e.event_id,4)')
    node = concat("'workflow:node:'", 'e.operation_key')
    # Restrict the outer runtime table by its (session_id,state_key) primary key.
    # Historical descriptors belonging to other trees are not cleanup candidates.
    sessions = (f'{r}.session_id IN (SELECT j.session_id FROM {p}jobs j WHERE j.root_job_id=? '
        f'UNION SELECT j.created_by_session_id FROM {p}jobs j WHERE j.root_job_id=? '
        f'UNION SELECT e.caller_session_id FROM {p}jobs j JOIN {p}job_events e ON e.job_id=j.job_id '
        f"WHERE j.root_job_id=? AND e.event_type='operation') AND ")
    return sessions + (f"EXISTS (SELECT 1 FROM {p}jobs j WHERE j.root_job_id=? AND ("
        f"(j.created_by_session_id={r}.session_id AND {r}.state_key={child}) OR "
        f"EXISTS (SELECT 1 FROM {p}job_events e WHERE e.job_id=j.job_id AND e.event_type='operation' AND ("
        f"({r}.session_id=COALESCE(e.caller_session_id,j.created_by_session_id) AND {r}.state_key={run}) OR "
        f"({r}.session_id=j.session_id AND {r}.state_key={workflow}) OR "
        f"({r}.session_id=j.created_by_session_id AND {r}.state_key={node})))))")


async def tree_runtime(store, tx, root_id, cutoff, limit, *, force=False):
    where = runtime_tree_condition(store)
    params = (root_id,)*4 + (() if force else (cutoff,))
    count = await delete_batch(tx, store.t.session_runtime_state, ('session_id', 'state_key'),
        where + ('' if force else ' AND updated_at<?'), params, limit)
    retained = await tx.fetchone(f'SELECT state_key FROM {store.t.session_runtime_state} WHERE {where} LIMIT 1', (root_id,)*4)
    return count, bool(retained)


async def delete_tree(store, tx, root_id, limit, counts):
    """Remove dependents before jobs and keep the root until the final batch."""
    p = store.table_prefix
    remaining = limit
    jobs = f'SELECT job_id FROM {p}jobs WHERE root_job_id=?'
    # Follow-up references must disappear before their event IDs do.
    where = (f'delivered_job_id IN ({jobs}) OR target_job_id IN ({jobs})')
    specs = [(store.t.follow_up_queue, 'follow_up_queue', ('id',), where, (root_id, root_id))]
    for table, keys in (
        ('job_events', ('event_id',)), ('execution_checkpoints', ('job_id','generation')),
        ('execution_allocations', ('allocation_id',)), ('task_budget_attempts', ('attempt_id',)),
        ('session_obligations', ('session_id','kind','source_id'))):
        specs.append((p+table, table, keys, f'job_id IN ({jobs})', (root_id,)))
    for table, category, keys, where, params in specs:
        if not remaining:
            return False
        deleted = await delete_batch(tx, table, keys, where, params, remaining)
        counts[category] = counts.get(category, 0)+deleted
        remaining -= deleted
        if not remaining:
            return False
    # Bound the child deletion as well. The root is not among these candidates.
    deleted = await delete_batch(tx, p+'jobs', ('job_id',), 'root_job_id=? AND job_id<>?',
                                 (root_id, root_id), remaining)
    counts['jobs'] = counts.get('jobs', 0)+deleted
    remaining -= deleted
    if not remaining:
        return False
    # A partial child batch spends the whole allowance, so reaching here means
    # every dependent and child was removed in this or previous batches.
    deleted = await tx.execute(f'DELETE FROM {p}jobs WHERE job_id=?', (root_id,))
    counts['jobs'] = counts.get('jobs', 0)+deleted
    counts['task_trees'] = counts.get('task_trees', 0)+deleted
    return True


async def purge_expired(store, now, *, limit, retention_days):
    remaining = batch_limit(limit)
    days = {**DEFAULT_DAYS, **(retention_days or {})}
    if days.keys() != DEFAULT_DAYS.keys() or any(type(v) not in (int, float) or v < 0 for v in days.values()):
        raise ValueError('invalid retention days')
    cutoff = {key: int(now-value*DAY_MS) for key, value in days.items()}
    counts = {}
    async with store._db.transaction() as tx:
        ordinary = (f' AND NOT EXISTS (SELECT 1 FROM {store.table_prefix}jobs j '
                    f'WHERE (j.job_id={store.t.follow_up_queue}.delivered_job_id '
                    f'OR j.job_id={store.t.follow_up_queue}.target_job_id) '
                    "AND (j.supervision_status<>'closed' OR j.resource_state<>'stopped'))") if store.has_message_ids else ''
        for category, table, keys, where, params, order in (
            ('follow_up_queue', store.t.follow_up_queue, ('id',),
             "status IN ('delivered','void') AND created_at<? AND (delivered_at IS NULL OR delivered_at<?)"+ordinary,
             (cutoff['follow_ups'], cutoff['follow_ups']), 'created_at,id'),
            ('hook_events', store.t.hook_events, ('session_id','event_id'), 'created_at<?',
             (cutoff['hooks'],), 'created_at,session_id,event_id')):
            counts[category] = await delete_batch(tx, table, keys, where, params, remaining, order=order)
            remaining -= counts[category]
    for category, prune, age in (('timers', store.prune_timers, 'timers'),
                                 ('usage_rounds', store.prune_usage_rounds, 'usage')):
        if not remaining:
            return counts
        counts[category] = await prune(None, older_than_ms=int(now-cutoff[age]), now=now, limit=remaining)
        remaining -= counts[category]
    if not store.has_message_ids or not remaining:
        return counts
    p = store.table_prefix
    runtime_ref = runtime_tree_condition(store).replace('j.root_job_id=?', 'j.root_job_id=root.job_id')
    roots = await store._db.fetchall(f'SELECT job_id FROM {p}jobs root WHERE '
        +ready_condition(p)+f' AND parent_job_id IS NULL AND updated_at<? AND NOT EXISTS ('
        f'SELECT 1 FROM {store.t.session_runtime_state} WHERE updated_at>=? AND {runtime_ref}) '
        'ORDER BY updated_at,job_id LIMIT ?',
        (min(cutoff['task_trees'], cutoff['runtime']), cutoff['runtime'], remaining))
    for candidate in roots:
        if not remaining:
            return counts
        async with store._db.transaction() as tx:
            root = await lock_root(store, tx, candidate['job_id'])
            if root is None or root['updated_at'] >= cutoff['task_trees'] or not await ready_tree(tx, p, root['job_id']):
                continue
            deleted, retained = await tree_runtime(store, tx, root['job_id'], cutoff['runtime'], remaining)
            counts['session_runtime_state'] = counts.get('session_runtime_state', 0)+deleted
            remaining -= deleted
            if not retained and remaining:
                before = sum(v for k, v in counts.items() if k != 'task_trees')
                await delete_tree(store, tx, root['job_id'], remaining, counts)
                remaining -= sum(v for k, v in counts.items() if k != 'task_trees')-before
    if not remaining:
        return counts
    unreferenced = f'NOT EXISTS (SELECT 1 FROM {p}jobs j WHERE j.executor_key={p}task_executors.executor_key)'
    candidates = await store._db.fetchall(f'SELECT executor_key FROM {p}task_executors '
        f"WHERE state='draining' AND updated_at<? AND {unreferenced} ORDER BY updated_at LIMIT ?",
        (cutoff['executors'], remaining))
    candidates += await store._db.fetchall(f'SELECT executor_key FROM {p}task_executors '
        f'WHERE lease_until_ms<? AND {unreferenced} ORDER BY lease_until_ms LIMIT ?', (cutoff['executors'], remaining))
    for key in sorted({row['executor_key'] for row in candidates}):
        if not remaining:
            break
        async with store._db.transaction() as tx:
            suffix = '' if store._db.dialect.name == 'sqlite' else ' FOR UPDATE'
            row = await tx.fetchone(f'SELECT * FROM {p}task_executors WHERE executor_key=?'+suffix, (key,))
            if row is None or not (row['lease_until_ms'] < cutoff['executors'] or
                                  row['state'] == 'draining' and row['updated_at'] < cutoff['executors']):
                continue
            if await tx.fetchone(f'SELECT job_id FROM {p}jobs WHERE executor_key=? LIMIT 1', (key,)):
                continue
            deleted = await delete_batch(tx, p+'task_executor_events', ('event_id',), 'executor_key=?', (key,), remaining)
            counts['task_executor_events'] = counts.get('task_executor_events', 0)+deleted
            remaining -= deleted
            if remaining:
                deleted = await tx.execute(f'DELETE FROM {p}task_executors WHERE executor_key=?', (key,))
                counts['task_executors'] = counts.get('task_executors', 0)+deleted
                remaining -= deleted
    if remaining >= 2:
        table = store.t.sessions
        present = ("CAST(metadata_json AS jsonb)->'runtime_descriptor'" if store._db.dialect.name == 'postgres'
                   else "JSON_EXTRACT(metadata_json,'$.runtime_descriptor')")
        where = (f"kind='subagent' AND updated_at<? AND {present} IS NOT NULL "
            f'AND NOT EXISTS (SELECT 1 FROM {p}jobs j WHERE j.session_id={table}.session_id) '
            f'AND NOT EXISTS (SELECT 1 FROM {store.t.follow_up_queue} q WHERE q.session_id={table}.session_id) '
            f'AND NOT EXISTS (SELECT 1 FROM {p}session_obligations o WHERE o.session_id={table}.session_id)')
        rows = await store._db.fetchall(f'SELECT session_id FROM {table} WHERE {where} ORDER BY updated_at,session_id LIMIT ?',
                                       (cutoff['runtime'], remaining//2))
        for row in rows:
            async with store._db.transaction() as tx:
                sid = row['session_id']
                await store._db.dialect.lock_state(tx, store.t.session_state, sid)
                if not await tx.fetchone(f'SELECT session_id FROM {table} WHERE session_id=? AND {where}',
                                        (sid, cutoff['runtime'])):
                    continue
                removed = ("CAST(CAST(metadata_json AS jsonb)-'runtime_descriptor' AS text)"
                           if store._db.dialect.name == 'postgres' else "JSON_REMOVE(metadata_json,'$.runtime_descriptor')")
                count = await tx.execute(f"UPDATE {store.t.session_state} SET lifecycle='closed',inbox_version=inbox_version+1 "
                                         'WHERE session_id=?', (sid,))
                count += await tx.execute(f'UPDATE {table} SET metadata_json={removed} WHERE session_id=?', (sid,))
                counts['descriptor_rows'] = counts.get('descriptor_rows', 0)+count
                remaining -= count
    return counts
