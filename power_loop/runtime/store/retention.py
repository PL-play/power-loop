"""Host-invoked retention, with indexed candidates and bounded delete statements."""
from __future__ import annotations

import json

from . import message_usage

DAY_MS = 86400000
DEFAULT_DAYS = {'task_trees': 30, 'executors': 1, 'follow_ups': 7, 'timers': 7,
                'hooks': 30, 'usage': 90, 'runtime': 30}


def batch_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('retention limit must be between 1 and 1000')
    return limit


async def delete_batch(tx, table, keys, where, params, limit, *, order=None):
    rows = await tx.fetchall(f"SELECT {','.join(keys)} FROM {table} WHERE {where} "
        f"ORDER BY {order or ','.join(keys)} LIMIT ?", (*params, limit))
    if not rows:
        return 0
    marks = ','.join('('+','.join('?' for _ in keys)+')' for _ in rows)
    values = tuple(row[key] for row in rows for key in keys)
    return await tx.execute(f"DELETE FROM {table} WHERE ({','.join(keys)}) IN ({marks}) AND {where}",
                            (*values, *params))


def inbox_event_key(store):
    item = store.t.follow_up_queue+'.item_id'
    operation = (f"CONCAT('op_',SUBSTR({item},10))" if store._db.dialect.name == 'mysql'
                 else f"'op_' || SUBSTR({item},10)")
    return (f"CASE WHEN {item} LIKE 'nudge:%' THEN SUBSTR({item},7) "
            f"WHEN {item} LIKE 'subagent:%' THEN {operation} ELSE {item} END")


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
        chosen, changed = set(), set()
        for row in sorted([*observations, *messages], key=lambda row: row['created_at']):
            key = (row['session_id'], row['send_index'] or 0, row['round_index'])
            if key not in chosen and len(chosen) == limit:
                continue
            chosen.add(key)
            if 'seq' in row:
                affected = await tx.execute(f'UPDATE {store.t.messages} SET meta_json={message_usage.remove_sql(backend)} '
                    f'WHERE session_id=? AND seq=? AND created_at<? AND {present} IS NOT NULL',
                    (row['session_id'], row['seq'], cutoff))
            else:
                affected = await tx.execute(f'DELETE FROM {store.t.usage_observations} '
                    'WHERE session_id=? AND send_index=? AND round_index=? AND created_at<?', (*key, cutoff))
            if affected:
                changed.add(key)
        return len(changed)


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


async def drop_runtime_key(store, tx, row):
    sid, key = row['session_id'], row['state_key']
    if key.startswith('workflow:run:'):
        # The existing workflow index uses the session-state lock for mutations.
        await store._db.dialect.lock_state(tx, store.t.session_state, sid)
        index = await tx.fetchone(f'SELECT value_json FROM {store.t.session_runtime_state} '
                                  "WHERE session_id=? AND state_key='workflow:index'", (sid,))
        if index is not None:
            from power_loop.workflow.journal import run_key
            remaining = [run for run in json.loads(index['value_json']) if run_key(run) != key]
            await tx.execute(f'UPDATE {store.t.session_runtime_state} SET value_json=? '
                "WHERE session_id=? AND state_key='workflow:index'", (json.dumps(remaining), sid))
    return await tx.execute(f'DELETE FROM {store.t.session_runtime_state} '
        'WHERE session_id=? AND state_key=? AND updated_at=?', (sid, key, row['updated_at']))


async def runtime_root(store, tx, row):
    p, key = store.table_prefix, row['state_key']
    if key.startswith('subagent:child:'):
        return await tx.fetchone(f'SELECT root_job_id FROM {p}jobs WHERE session_id=? '
            "AND kind IN ('subagent','workflow_node') ORDER BY updated_at DESC LIMIT 1", (key[15:],))
    for prefix in ('subagent:run:', 'workflow:run:', 'workflow:history:'):
        if key.startswith(prefix):
            return await tx.fetchone(f'SELECT j.root_job_id FROM {p}job_events e '
                f'JOIN {p}jobs j ON j.job_id=e.job_id WHERE e.event_id=?', ('op_'+key[len(prefix):],))
    return None


async def purge_runtime(store, cutoff, limit):
    table, count = store.t.session_runtime_state, 0
    rows = await store._db.fetchall(f'SELECT * FROM {table} WHERE updated_at<? '
        "AND (state_key LIKE 'subagent:run:%' OR state_key LIKE 'subagent:child:%' "
        "OR state_key LIKE 'workflow:run:%' OR state_key LIKE 'workflow:history:%') "
        'ORDER BY updated_at LIMIT ?', (cutoff, limit))
    for row in rows:
        owner = await runtime_root(store, store._db, row)
        if owner is None:
            continue
        async with store._db.transaction() as tx:
            root = await lock_root(store, tx, owner['root_job_id'])
            if root is not None and root['updated_at'] < cutoff and await ready_tree(tx, store.table_prefix, root['job_id']):
                count += await drop_runtime_key(store, tx, row)
    return count


async def tree_runtime(store, tx, root_id, cutoff, limit, *, force=False):
    """Remove keys while their operation IDs still identify the owning tree.

    A recent runtime record keeps its task tree until the runtime retention
    window also ends. No registry of retention dependencies is persisted.
    """
    p, cursor, count, retained = store.table_prefix, '', 0, False
    while True:
        jobs = await tx.fetchall(f'SELECT * FROM {p}jobs WHERE root_job_id=? AND job_id>? '
                                'ORDER BY job_id LIMIT ?', (root_id, cursor, limit))
        if not jobs:
            return count, retained
        for job in jobs:
            keys = []
            if job['session_id'] and job['kind'] in ('subagent', 'workflow_node'):
                keys.append((job['created_by_session_id'], 'subagent:child:'+job['session_id']))
            event_cursor = ''
            while True:
                events = await tx.fetchall(f'SELECT event_id,caller_session_id FROM {p}job_events '
                    "WHERE job_id=? AND event_type='operation' AND event_id>? ORDER BY event_id LIMIT ?",
                    (job['job_id'], event_cursor, limit))
                for event in events:
                    sid = event['caller_session_id'] or job['created_by_session_id']
                    run_id = event['event_id'][3:]
                    for namespace in ('subagent:run:', 'workflow:run:', 'workflow:history:'):
                        keys.append((sid, namespace+run_id))
                for sid, key in keys:
                    row = await tx.fetchone(f'SELECT * FROM {store.t.session_runtime_state} '
                                            'WHERE session_id=? AND state_key=?', (sid, key))
                    if row is None:
                        continue
                    if not force and row['updated_at'] >= cutoff:
                        retained = True
                    else:
                        count += await drop_runtime_key(store, tx, row)
                keys.clear()
                if not events:
                    break
                event_cursor = events[-1]['event_id']
        cursor = jobs[-1]['job_id']


async def delete_tree(store, tx, root_id, limit, counts):
    p = store.table_prefix
    # Delete bounded job pages, leaving the locked root row until the very end.
    while True:
        rows = await tx.fetchall(f'SELECT job_id FROM {p}jobs WHERE root_job_id=? AND job_id<>? '
                                'ORDER BY job_id LIMIT ?', (root_id, root_id, limit))
        ids = [row['job_id'] for row in rows] or [root_id]
        marks = ','.join('?' for _ in ids)
        # Notification readers need their inbox row as long as the task exists.
        # Exact event IDs, nudge operation IDs and subagent run IDs are the
        # existing inbox identities; no new association is persisted.
        cursor = ''
        while events := await tx.fetchall(f'SELECT event_id FROM {p}job_events '
                f'WHERE job_id IN ({marks}) AND event_id>? ORDER BY event_id LIMIT ?', (*ids, cursor, limit)):
            items = []
            for event in events:
                key = event['event_id']
                items.append(key)
                if key.startswith('op_'):
                    items.extend(('nudge:'+key, 'subagent:'+key[3:]))
            where = 'item_id IN ('+','.join('?' for _ in items)+')'
            while deleted := await delete_batch(tx, store.t.follow_up_queue, ('id',), where, items, limit):
                counts['follow_up_queue'] = counts.get('follow_up_queue', 0)+deleted
            cursor = events[-1]['event_id']
        for table, keys in (
            ('job_events', ('event_id',)), ('execution_checkpoints', ('job_id','generation')),
            ('execution_allocations', ('allocation_id',)), ('task_budget_attempts', ('attempt_id',)),
            ('session_obligations', ('session_id','kind','source_id'))):
            while deleted := await delete_batch(tx, p+table, keys, f'job_id IN ({marks})', ids, limit):
                counts[table] = counts.get(table, 0)+deleted
        deleted = await tx.execute(f'DELETE FROM {p}jobs WHERE job_id IN ({marks})', ids)
        counts['jobs'] = counts.get('jobs', 0)+deleted
        if not rows:
            counts['task_trees'] = counts.get('task_trees', 0)+int(deleted > 0)
            return


async def delete_session_tasks(store, tx, session_id):
    if not store.has_message_ids:
        return
    p = store.table_prefix
    while roots := await tx.fetchall(f'SELECT job_id FROM {p}jobs WHERE session_id=? '
                                    'AND parent_job_id IS NULL ORDER BY job_id LIMIT 1000', (session_id,)):
        for root in roots:
            if await lock_root(store, tx, root['job_id']) is not None:
                await tree_runtime(store, tx, root['job_id'], 0, 1000, force=True)
                await delete_tree(store, tx, root['job_id'], 1000, {})
    await tx.execute(f'DELETE FROM {p}task_context_events WHERE session_id=?', (session_id,))


async def purge_expired(store, now, *, limit, retention_days):
    limit = batch_limit(limit)
    days = {**DEFAULT_DAYS, **(retention_days or {})}
    if days.keys() != DEFAULT_DAYS.keys() or any(type(v) not in (int, float) or v < 0 for v in days.values()):
        raise ValueError('invalid retention days')
    cutoff = {key: int(now-value*DAY_MS) for key, value in days.items()}
    counts = {}
    async with store._db.transaction() as tx:
        ordinary = (f' AND NOT EXISTS (SELECT 1 FROM {store.table_prefix}job_events e '
                    f'WHERE e.event_id={inbox_event_key(store)})') if store.has_message_ids else ''
        counts['follow_up_queue'] = await delete_batch(tx, store.t.follow_up_queue, ('id',),
            "status IN ('delivered','void','transferred') AND created_at<? AND (delivered_at IS NULL OR delivered_at<?)"+ordinary,
            (cutoff['follow_ups'], cutoff['follow_ups']), limit, order='created_at,id')
        counts['hook_events'] = await delete_batch(tx, store.t.hook_events, ('session_id','event_id'),
            'created_at<?', (cutoff['hooks'],), limit, order='created_at,session_id,event_id')
    counts['timers'] = await store.prune_timers(None, older_than_ms=int(now-cutoff['timers']), now=now, limit=limit)
    counts['usage_rounds'] = await store.prune_usage_rounds(None, older_than_ms=int(now-cutoff['usage']), now=now, limit=limit)
    if not store.has_message_ids:
        return counts
    p = store.table_prefix
    counts['session_runtime_state'] = await purge_runtime(store, cutoff['runtime'], limit)
    roots = await store._db.fetchall(f'SELECT job_id FROM {p}jobs root WHERE '
        +ready_condition(p)+' AND parent_job_id IS NULL AND updated_at<? ORDER BY updated_at,job_id LIMIT ?',
        (cutoff['task_trees'], limit))
    for candidate in roots:
        async with store._db.transaction() as tx:
            root = await lock_root(store, tx, candidate['job_id'])
            if root is None or root['updated_at'] >= cutoff['task_trees'] or not await ready_tree(tx, p, root['job_id']):
                continue
            runtime_cutoff = cutoff['runtime'] if root['updated_at'] < cutoff['runtime'] else -1
            deleted, retained = await tree_runtime(store, tx, root['job_id'], runtime_cutoff, limit)
            counts['session_runtime_state'] += deleted
            if not retained:
                await delete_tree(store, tx, root['job_id'], limit, counts)
    # Filter references before LIMIT so retained instances cannot starve later
    # candidates. Recheck under the existing row lock before deleting.
    unreferenced = f'NOT EXISTS (SELECT 1 FROM {p}jobs j WHERE j.executor_key={p}task_executors.executor_key)'
    candidates = await store._db.fetchall(f'SELECT executor_key FROM {p}task_executors '
        f"WHERE state='draining' AND updated_at<? AND {unreferenced} ORDER BY updated_at LIMIT ?",
        (cutoff['executors'], limit))
    candidates += await store._db.fetchall(f'SELECT executor_key FROM {p}task_executors '
        f'WHERE lease_until_ms<? AND {unreferenced} ORDER BY lease_until_ms LIMIT ?', (cutoff['executors'], limit))
    for key in sorted({row['executor_key'] for row in candidates})[:limit]:
        async with store._db.transaction() as tx:
            suffix = '' if store._db.dialect.name == 'sqlite' else ' FOR UPDATE'
            row = await tx.fetchone(f'SELECT * FROM {p}task_executors WHERE executor_key=?'+suffix, (key,))
            if row is None or not (row['lease_until_ms'] < cutoff['executors'] or
                                  row['state'] == 'draining' and row['updated_at'] < cutoff['executors']):
                continue
            if await tx.fetchone(f'SELECT job_id FROM {p}jobs WHERE executor_key=? LIMIT 1', (key,)):
                continue
            while deleted := await delete_batch(tx, p+'task_executor_events', ('event_id',), 'executor_key=?', (key,), limit):
                counts['task_executor_events'] = counts.get('task_executor_events', 0)+deleted
            counts['task_executors'] = counts.get('task_executors', 0)+await tx.execute(
                f'DELETE FROM {p}task_executors WHERE executor_key=?', (key,))
    return counts
