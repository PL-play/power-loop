"""Host discovery and boot admission cost depend on unfinished work only."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from power_loop.runtime import executor_registry as registry
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_inspection import list_open_roots

from .test_executor_lifecycle_192 import host
from .test_executor_registry_192 import boot
from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled

__all__ = ['registry_store', 'boot', 'host']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def insert_rows(manager, table, rows):
    columns = tuple(rows[0])
    unit = '(' + ','.join('?' for _ in columns) + ')'
    async with manager.db.transaction() as tx:
        for offset in range(0, len(rows), 100):
            batch = rows[offset:offset+100]
            await tx.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES " + ','.join(unit for _ in batch),
                             tuple(row[key] for row in batch for key in columns))


async def measured(c, action):
    original = c.store._db
    c.store._db = profile = Profiled(original)
    manager = JobManager(c.store, now_ms=lambda: c.now[0])
    started = time.perf_counter()
    try:
        value = await action(manager)
    finally:
        elapsed = time.perf_counter() - started
        c.store._db = original
    return value, {'sql': len(profile.records), 'returned_rows': sum(r['rows'] or 0 for r in profile.records),
                   'seconds': elapsed}


async def test_discovery_and_activation_ignore_ten_thousand_finished_records(boot):
    c = boot
    root = await c.manager.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='open-root')
    job = await c.manager.db.fetchone(f'SELECT * FROM {c.manager.jobs} WHERE job_id=?', (root.job.job_id,))
    attempts = c.manager.prefix + 'task_budget_attempts'
    fee = dict(attempt_id='fee-0', job_id=job['job_id'], policy_key='policy', version=1,
               phase='settling', pending=1, state_hash='unused', state_json='{}', created_at=1, updated_at=1)
    await insert_rows(c.manager, attempts, [fee, {**fee, 'attempt_id': 'fee-1', 'phase': 'dispatch_authorized', 'pending': 0}])
    reports = []
    for history in (0, 10000):
        identity = replace(c.identity, boot_id=f'boot-{history}')
        await registry.register(c.manager, identity, contract_hash='a'*64)
        roots, roots_cost = await measured(c, list_open_roots)
        fees, fees_cost = await measured(c, budgets.list_unsettled)
        _, activation_cost = await measured(c, lambda manager: registry.activate(
            manager, identity, expected_version=1, system_key='start'))
        assert [r['job_id'] for r in roots['roots']] == [job['job_id']]
        assert [r['attempt_id'] for r in fees['attempts']] == ['fee-0', 'fee-1']
        reports.append({'history': history, 'roots': roots_cost, 'fees': fees_cost, 'activation': activation_cost})
        await registry.drain(c.manager, identity, system_key='stop', reason='finished')
        if history == 0:
            closed = {**job, 'status': 'completed', 'supervision_status': 'closed', 'resource_state': 'stopped',
                      'intake': 'closed', 'outcome': 'succeeded', 'executor_id': identity.executor_id,
                      'boot_id': identity.boot_id}
            await insert_rows(c.manager, c.manager.jobs, [
                {**closed, 'job_id': f'old-{i:05}', 'root_job_id': f'old-{i:05}', 'completion_scope_job_id': f'old-{i:05}'}
                for i in range(10000)])
            await insert_rows(c.manager, attempts, [{**fee, 'attempt_id': f'old-{i:05}', 'pending': 0,
                'phase': 'released' if i % 2 else 'settling'} for i in range(10000)])
            executor = await c.manager.db.fetchone(f'SELECT * FROM {c.manager.prefix}task_executors WHERE executor_key=?',
                (registry._key(c.manager, identity),))
            await insert_rows(c.manager, c.manager.prefix+'task_executors', [
                {**executor, 'executor_key': f'old-{i:05}', 'boot_id': f'old-{i:05}'} for i in range(10000)])
    for name in ('roots', 'fees', 'activation'):
        first, last = reports[0][name], reports[1][name]
        assert last['sql'] == first['sql'] and last['returned_rows'] == first['returned_rows']
        assert last['seconds'] <= max(.1, first['seconds'] * 10), reports
    target = os.environ.get('POWER_LOOP_SQL_PROFILE')
    if target:
        path = Path(target).parent / 'history.jsonl'
        with path.open('a') as output:
            output.write(json.dumps({'db': c.manager.db.dialect.name, 'measurements': reports})+'\n')


async def test_unsettled_pages_include_dispatch_and_pending_but_not_confirmed_fees(boot):
    c = boot
    root = await c.manager.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='root')
    phases = ['local_reserved', 'host_reserving', 'host_reserved', 'dispatch_authorized', 'dispatch_closed', 'settling']
    rows = [dict(attempt_id=f'fee-{i}', job_id=root.job.job_id, policy_key='policy', version=1, phase=phase,
                 pending=int(phase in ('host_reserving', 'dispatch_closed', 'settling')),
                 state_hash='unused', state_json='{}', created_at=1, updated_at=1) for i, phase in enumerate(phases)]
    await insert_rows(c.manager, c.manager.prefix+'task_budget_attempts', rows + [
        {**rows[-1], 'attempt_id': 'settled', 'pending': 0}, {**rows[-1], 'attempt_id': 'released', 'pending': 0, 'phase': 'released'}])
    found, cursor = [], ''
    while True:
        page = await budgets.list_unsettled(c.manager, after_attempt_id=cursor, limit=2)
        found.extend(r['attempt_id'] for r in page['attempts'])
        if page['next_cursor'] is None:
            break
        cursor = page['next_cursor']
    assert found == [r['attempt_id'] for r in rows]


async def test_one_minute_of_live_heartbeats_adds_no_events(host):
    _, manager, runtime = host
    await runtime.start()
    first = await registry.get(manager, runtime._lifecycle.identity)
    before = await manager.db.fetchall(f'SELECT * FROM {manager.prefix}task_executor_events ORDER BY event_id')
    # Exercise the real scheduled heartbeat over a full minute, not a simulated clock.
    await asyncio.sleep(60)
    current = await registry.get(manager, runtime._lifecycle.identity)
    assert current['admission_open'] and current['lease_until_ms'] >= first['lease_until_ms'] + 57000
    assert current['version'] == first['version']
    assert await manager.db.fetchall(f'SELECT * FROM {manager.prefix}task_executor_events ORDER BY event_id') == before
