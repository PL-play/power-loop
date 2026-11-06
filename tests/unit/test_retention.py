"""Retention preserves unfinished work and accounting authority on three DBs."""
import asyncio
import json

import pytest

from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.retention import DAY_MS

from .test_job_registry_192 import registry_store
from .test_send_sql_profile import _measure

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
NOW = 200 * DAY_MS
OLD = 1


async def insert(db, table, **row):
    await db.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))


async def root(store, name, *, ended=True, at=OLD):
    manager = JobManager(store, now_ms=lambda: at)
    sid = await store.create_session()
    registration = await manager.register_job(JobActor('retention', 'host', sid), kind='send',
                                               session_id=sid, system_key=name)
    job_id = registration.job.job_id
    if ended:
        await store._db.execute(f"UPDATE {manager.jobs} SET status='completed',supervision_status='closed',"
            "resource_state='stopped',intake='closed',outcome='succeeded' WHERE job_id=?", (job_id,))
    return job_id, sid, registration.operation.operation_id[3:]


@pytest.mark.parametrize('source', ['stopped', 'expired'])
@pytest.mark.parametrize('limit', [1, 1000])
async def test_executor_retention_progresses_past_references(registry_store, monkeypatch, source, limit):
    store, connect = registry_store
    db, p = store._db, store.table_prefix
    job, _, _ = await root(store, 'active-reference', ended=False)
    parent = await db.fetchone(f'SELECT * FROM {p}jobs WHERE job_id=?', (job,))
    async with db.transaction() as tx:
        for i in range(limit+3):
            key = f'executor-{i}'
            await insert(tx, p+'task_executors', executor_key=key, tenant_id='retention', principal_id='host',
                         executor_id=key, boot_id='boot', contract_hash='a'*64, version=1,
                         state='draining' if source == 'stopped' else 'accepting',
                         lease_until_ms=NOW if source == 'stopped' else OLD+i, created_at=OLD, updated_at=OLD+i)
            await insert(tx, p+'task_executor_events', event_id=key, executor_key=key, action='register',
                         operation_key=key, request_hash='a'*64, payload_json='{}', created_at=OLD)
            if i < limit:
                await insert(tx, p+'jobs', **{**parent, 'job_id': f'child-{i}', 'parent_job_id': job,
                                             'executor_key': key, 'kind': 'subagent'})
        # More than one full batch of events for one deletable executor.
        for i in range(1000 if limit == 1000 else 0):
            await insert(tx, p+'task_executor_events', event_id=f'event-{i}', executor_key=f'executor-{limit}',
                         action='heartbeat', operation_key=f'event-{i}', request_hash='a'*64,
                         payload_json='{}', created_at=OLD)
    peer = await connect()
    async with db.transaction() as tx:
        transaction_type = type(tx)
    execute, deletions = transaction_type.execute, []

    async def count_delete(tx, sql, params=()):
        count = await execute(tx, sql, params)
        if sql.startswith('DELETE FROM ' + p):
            assert 0 <= count <= limit
            deletions.append(count)
        return count

    monkeypatch.setattr(transaction_type, 'execute', count_delete)
    first = await store.purge_expired(NOW, limit=limit)
    assert first['task_executors'] == min(limit, 3)
    # Separate connections compete for the same next candidate; only its
    # deleting transaction may count it. Repeated calls must keep progressing.
    reports = [first]
    for _ in range(2):
        reports.extend(await asyncio.gather(store.purge_expired(NOW, limit=1), peer.purge_expired(NOW, limit=1)))
    assert sum(r.get('task_executors', 0) for r in reports) == 3
    assert sum(r.get('task_executor_events', 0) for r in reports) == (1003 if limit == 1000 else 3)
    assert sum(deletions) == (1006 if limit == 1000 else 6)
    if limit == 1000:
        assert max(deletions) == 1000
    retained = {f'executor-{i}' for i in range(limit)}
    for table in ('task_executors', 'task_executor_events'):
        assert {r['executor_key'] for r in await db.fetchall(f'SELECT executor_key FROM {p}{table}')} == retained
    assert not any((await store.purge_expired(NOW, limit=1)).values())


@pytest.mark.parametrize('change', ['reference', 'updated_at', 'state', 'lease_until_ms'])
async def test_executor_retention_rechecks_selected_candidates(registry_store, monkeypatch, change):
    store, _ = registry_store
    db, p = store._db, store.table_prefix
    job, _, _ = await root(store, 'late-reference', ended=False)
    await insert(db, p+'task_executors', executor_key='candidate', tenant_id='retention', principal_id='host',
                 executor_id='candidate', boot_id='boot', contract_hash='a'*64, version=1,
                 state='accepting' if change == 'lease_until_ms' else 'draining',
                 lease_until_ms=OLD if change == 'lease_until_ms' else NOW, created_at=OLD, updated_at=OLD)
    await insert(db, p+'task_executor_events', event_id='registered', executor_key='candidate', action='register',
                 operation_key='registered', request_hash='a'*64, payload_json='{}', created_at=OLD)
    fetchall, changed = db.fetchall, []

    async def select_then_change(sql, params=()):
        rows = await fetchall(sql, params)
        if not changed and rows and sql.startswith(f'SELECT executor_key FROM {p}task_executors '):
            changed.append(True)
            if change == 'reference':
                await db.execute(f"UPDATE {p}jobs SET executor_key='candidate' WHERE job_id=?", (job,))
            else:
                await db.execute(f"UPDATE {p}task_executors SET {change}=? WHERE executor_key='candidate'",
                                 ('accepting' if change == 'state' else NOW,))
        return rows

    monkeypatch.setattr(db, 'fetchall', select_then_change)
    report = await store.purge_expired(NOW, limit=1)
    assert changed and report.get('task_executors', 0) == report.get('task_executor_events', 0) == 0
    assert await db.fetchone(f"SELECT executor_key FROM {p}task_executors WHERE executor_key='candidate'")
    assert await db.fetchone(f"SELECT event_id FROM {p}task_executor_events WHERE event_id='registered'")


async def test_categories_expire_without_removing_recent_or_active_work(registry_store):
    store, connect = registry_store
    db, p = store._db, store.table_prefix
    roots = [await root(store, state, ended=state != 'active', at=NOW if state == 'recent' else OLD)
             for state in ('expired', 'recent', 'active')]
    for i, (job, sid, run) in enumerate(roots):
        at = NOW if i == 1 else OLD
        child_sid = await store.create_session(parent_session_id=sid)
        parent = await db.fetchone(f'SELECT * FROM {p}jobs WHERE job_id=?', (job,))
        await insert(db, p+'jobs', **{**parent, 'job_id': f'child-{i}', 'parent_job_id': job,
                                    'session_id': child_sid, 'kind': 'subagent'})
        await insert(db, store.t.session_runtime_state, session_id=sid, state_key='subagent:child:'+child_sid,
                     value_json='{"deleted":false}', updated_at=at)
        await insert(db, store.t.follow_up_queue, session_id=sid, item_id=str(i), kind='user', mode='queue',
                     status='pending' if i == 2 else 'delivered', content='keep message', created_at=at, delivered_at=at)
        for item, status in (('op_'+run, 'delivered'), ('nudge:op_'+run, 'void'), ('subagent:'+run, 'transferred')):
            await insert(db, store.t.follow_up_queue, session_id=sid, item_id=item, kind='task_done', mode='queue',
                         status=status, content='task history', created_at=OLD, delivered_at=OLD)
        await insert(db, store.t.timers, session_id=sid, timer_id=1, due_at=at, note='timer',
                     status='armed' if i == 2 else 'fired', created_at=at, updated_at=at)
        await insert(db, store.t.hook_events, session_id=sid, event_id=1, hook_point='after', kind='metadata', created_at=at)
        await insert(db, store.t.usage_observations, session_id=sid, send_index=1, round_index=1,
                     total_tokens=5, created_at=at)
        await store.append_message(sid, role='assistant', content='retained body', send_index=2, round_index=1,
                                   meta={'_round_usage': {'total_tokens': 5}})
        await db.execute(f'UPDATE {store.t.messages} SET created_at=? WHERE session_id=?', (at, sid))
        for namespace in ('subagent:run:', 'workflow:run:', 'workflow:history:'):
            await insert(db, store.t.session_runtime_state, session_id=sid, state_key=namespace+run,
                         value_json=json.dumps({'run_id': run}), updated_at=at)
        await insert(db, store.t.session_runtime_state, session_id=sid, state_key='workflow:index',
                     value_json=json.dumps([run]), updated_at=at)
        await insert(db, p+'task_executors', executor_key=f'executor-{i}', tenant_id='retention', principal_id='host',
                     executor_id=f'executor-{i}', boot_id='boot', contract_hash='a'*64, state='draining' if i < 2 else 'accepting',
                     version=1, lease_until_ms=at, created_at=at, updated_at=at)
        await insert(db, p+'task_executor_events', event_id=f'boot-{i}', executor_key=f'executor-{i}', action='register',
                     operation_key=f'boot-{i}', request_hash='a'*64, payload_json='{}', created_at=at)
        if i == 2:
            await db.execute(f'UPDATE {p}jobs SET executor_key=? WHERE job_id=?', (f'executor-{i}', job))
    peer = await connect()
    reports = await asyncio.gather(store.purge_expired(NOW, limit=1000), peer.purge_expired(NOW, limit=1000))
    assert sum(r.get('task_trees', 0) for r in reports) == 1
    assert sum(r.get('task_executors', 0) for r in reports) == 1
    remaining = {r['job_id'] for r in await db.fetchall(f'SELECT job_id FROM {p}jobs')}
    assert remaining == {roots[1][0], roots[2][0], 'child-1', 'child-2'}
    assert (await store.get_runtime_state(roots[0][1], 'workflow:index')) == []
    for i, (_, sid, run) in enumerate(roots):
        assert bool(await store.get_runtime_state(sid, 'workflow:run:'+run)) == (i != 0)
        child_keys = await db.fetchall(f"SELECT state_key FROM {store.t.session_runtime_state} "
                                      "WHERE session_id=? AND state_key LIKE 'subagent:child:%'", (sid,))
        assert bool(child_keys) == (i != 0)
        assert bool(await store.get_timer(sid, 1)) == (i != 0)
        assert bool(await db.fetchone(f'SELECT id FROM {store.t.follow_up_queue} WHERE session_id=?', (sid,))) == (i != 0)
        for item in ('op_'+run, 'nudge:op_'+run, 'subagent:'+run):
            assert bool(await db.fetchone(f'SELECT id FROM {store.t.follow_up_queue} '
                                         'WHERE session_id=? AND item_id=?', (sid, item))) == (i != 0)
        # Hooks and usage expire by age independently of task status.
        assert bool(await db.fetchone(f'SELECT event_id FROM {store.t.hook_events} WHERE session_id=?', (sid,))) == (i == 1)
        assert bool(await store.list_usage_rounds(sid)) == (i == 1)
        assert (await store.load_all_messages(sid))[0].content == 'retained body'
    assert not any((await store.purge_expired(NOW)).values())


async def test_unsettled_fees_obligations_and_alerts_keep_their_tree(registry_store):
    store, _ = registry_store
    db, p = store._db, store.table_prefix
    jobs = [await root(store, reason) for reason in ('fee', 'obligation', 'alert')]
    await insert(db, p+'task_budget_attempts', attempt_id='unacknowledged', job_id=jobs[0][0], policy_key='policy',
                 version=1, phase='settling', pending=1, state_hash='unused', state_json='{}', created_at=OLD, updated_at=OLD)
    await insert(db, p+'session_obligations', session_id=jobs[1][1], kind='result', source_id='pending',
                 job_id=jobs[1][0], opened_version=1)
    await insert(db, p+'job_events', event_id='alert', job_id=jobs[2][0], event_type='task_alert', job_version=1,
                 payload_json='{}', delivery_status='pending', created_at=OLD)
    await insert(db, store.t.follow_up_queue, session_id=jobs[2][1], item_id='alert', kind='task_alert', mode='queue',
                 status='claimed', content='still actionable', created_at=OLD)
    try:
        assert (await store.purge_expired(NOW)).get('task_trees', 0) == 0
        assert len(await db.fetchall(f'SELECT job_id FROM {p}jobs')) == 3
    finally:
        # Complete the three fixture obligations, then retention can delete them.
        await db.execute(f'UPDATE {p}task_budget_attempts SET pending=0')
        await db.execute(f'DELETE FROM {p}session_obligations')
        await db.execute(f"UPDATE {store.t.follow_up_queue} SET status='delivered',delivered_at=?", (OLD,))
    assert (await store.purge_expired(NOW))['task_trees'] == 3


async def test_batch_bound_and_session_delete_preserve_parent_tree(registry_store):
    store, _ = registry_store
    db, p = store._db, store.table_prefix
    parent, sid, _ = await root(store, 'parent', ended=False)
    child_sid = await store.create_session(parent_session_id=sid)
    job = await db.fetchone(f'SELECT * FROM {p}jobs WHERE job_id=?', (parent,))
    await insert(db, p+'jobs', **{**job, 'job_id': 'child', 'parent_job_id': parent, 'session_id': child_sid, 'kind': 'subagent'})
    for i in range(5):
        await insert(db, store.t.follow_up_queue, session_id=sid, item_id=f'void-{i}', content='old',
                     kind='user', mode='queue', status='void', created_at=OLD)
    assert (await store.purge_expired(NOW, limit=2))['follow_up_queue'] == 2
    assert (await store.purge_expired(NOW, limit=2))['follow_up_queue'] == 2
    await store.close_session(child_sid)
    assert len(await db.fetchall(f'SELECT job_id FROM {p}jobs')) == 2
    await store.close_session(sid)
    assert not await db.fetchall(f'SELECT job_id FROM {p}jobs')
    assert not await db.fetchall(f'SELECT event_id FROM {p}job_events')
    assert await store.close_session(sid) == 0
    with pytest.raises(ValueError, match='between 1 and 1000'):
        await store.purge_expired(NOW, limit=1001)


async def test_purged_task_trees_do_not_reset_budget_policy_or_balances(registry_store):
    store, connect = registry_store
    await _measure(store, 1, True)
    p = store.table_prefix
    original = {table: await store._db.fetchall(f'SELECT * FROM {p}{table}')
                for table in ('task_budget_policies', 'task_budget_scopes', 'session_stats')}
    messages = await store._db.fetchall(f'SELECT session_id,seq,content FROM {store.t.messages} ORDER BY session_id,seq')
    newest = (await store._db.fetchone(f'SELECT MAX(updated_at) AS at FROM {p}jobs'))['at']
    report = await store.purge_expired(newest+100*DAY_MS)
    assert report['task_trees'] == 2 and report['jobs'] == 2
    for table, rows in original.items():
        assert await store._db.fetchall(f'SELECT * FROM {p}{table}') == rows
    assert await store._db.fetchall(f'SELECT session_id,seq,content FROM {store.t.messages} ORDER BY session_id,seq') == messages
    assert not await store._db.fetchall(f"SELECT event_id FROM {p}job_events WHERE event_type IN ('model_budget_policy','model_budget_scope')")
    await _measure(await connect(), 1, True)
    sid = (await store._db.fetchone(f'SELECT session_id FROM {p}jobs LIMIT 1'))['session_id']
    await store.close_session(sid)
    assert not await store._db.fetchall(f'SELECT job_id FROM {p}jobs')
    assert await store._db.fetchone(f'SELECT policy_key FROM {p}task_budget_policies')
