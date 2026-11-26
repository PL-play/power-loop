"""Measure the new host ports independently of model execution and ordinary send."""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop import (
    AgentHooks,
    HookPoint,
    RuntimeClient,
    StoreBlackboard,
    StoreTimerBackend,
    TimerContext,
    TimerRunner,
)
from power_loop.agent.runtime_transport import MemoryMessageTransport
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from power_loop.runtime.store.process_guard import ProcessStoreGuard
from tests.row_profile import _snapshot
from tests.write_meter import WriteMeter

from .session_native_fixture import fixture
from .test_managed_effects_sessions_215 import _exercise_managed_notice
from .test_send_sql_profile import _CATEGORIES, Profiled, _category, _counts
from .test_workflow_sessions_215 import _exercise_halt_late_result


class NativeConnection:
    def __init__(self, original, profile):
        self.original, self.profile = original, profile

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def fetchval(self, sql, *args):
        result = await self.original.fetchval(sql, *args)
        self.profile.note(sql, int(result is not None))
        return result

    async def ping(self, **kwargs):
        # COM_PING is a middleware round trip, not SQL.
        self.profile.pings += 1
        return await self.original.ping(**kwargs)

    @asynccontextmanager
    async def cursor(self):
        async with self.original.cursor() as cursor:
            profile = self.profile
            class Cursor:
                async def execute(self, sql, args):
                    result = await cursor.execute(sql, args)
                    profile.note(sql, result)
                    return result
                async def fetchone(self):
                    return await cursor.fetchone()
            yield Cursor()


class PortsProfile(Profiled):
    pings = 0
    async def open_lock_connection(self):
        return NativeConnection(await self.db.open_lock_connection(), self)


@pytest.mark.parametrize('operation', ['timer_create', 'timer_cancel', 'timer_scan_initial',
    'timer_scan_empty', 'timer_fire_acceptance', 'timer_fire_unknown',
    'board_first', 'board_update', 'board_delete', 'startup_lock'])
async def test_host_port_cost(registry_store, operation):
    store, _ = registry_store
    _, target, actor = await fixture(store)
    original = store._db
    transport = MemoryMessageTransport()
    # No worker: the firing measurement includes full acceptance, not model execution.
    client = RuntimeClient(store=store, transport=transport)
    backend = StoreTimerBackend(store)
    source = TimerContext(store.store_id, target.session_id, actor)
    board = StoreBlackboard(store)
    if operation in {'board_update', 'board_delete'}:
        await board.post('cost-board', text='first')
    timer = None
    if operation in {'timer_cancel', 'timer_fire_acceptance', 'timer_fire_unknown'}:
        timer = await backend.create(source, delay_s=3600 if operation == 'timer_cancel' else -1, note='cost timer')
    meter = await WriteMeter.install(original, store.table_prefix)
    profile = PortsProfile(original, meter)
    store._db = profile
    # Recreate client store ports after attaching the profiler.
    client = RuntimeClient(store=store, transport=transport)
    hooks = AgentHooks()
    if operation == 'timer_fire_unknown':
        def fail_before_append(ctx):
            raise RuntimeError('timer cost hook failure')
        hooks.register(HookPoint.TIMER_FIRE, fail_before_append)
    driver = TimerRunner(store=store, client=client, hooks=hooks)
    if operation != 'timer_scan_initial':
        driver._last_recovery = time.monotonic()
    guard = ProcessStoreGuard(store, MemorySessionOwnershipCoordinator(), transport)
    try:
        before = await _snapshot(original, store.table_prefix)
        await meter.reset()
        started = time.monotonic()
        if operation == 'timer_create':
            await backend.create(source, delay_s=3600, note='cost timer')
        elif operation == 'timer_cancel':
            assert await backend.cancel(source, timer.timer_id)
        elif operation.startswith('timer_scan'):
            assert await driver.scan_once() == 0
        elif operation == 'timer_fire_acceptance':
            assert await driver.scan_once() == 1
        elif operation == 'timer_fire_unknown':
            assert await driver.scan_once() == 0
            assert 'timer cost hook failure' in driver.status().last_error
        elif operation == 'board_first':
            await board.post('cost-board', text='first')
        elif operation == 'board_update':
            await board.update('cost-board', 1, text='updated')
        elif operation == 'board_delete':
            await board.delete('cost-board')
        else:
            await guard.acquire()
            await guard.check()
            await guard.release()
        elapsed = time.monotonic() - started
        records = list(profile.records)
        tx_sites = list(profile.tx_sites)
        if operation == 'startup_lock':
            expected = {
                'sqlite': ([], 0),
                'postgres': (['SELECT pg_try_advisory_lock($1)', 'SELECT 1'], 0),
                'mysql': (['SELECT GET_LOCK(%s,0)', 'SELECT RELEASE_LOCK(%s)'], 1),
            }
            statements, pings = expected[original.dialect.name]
            assert [record['sql'] for record in records] == statements
            assert profile.pings == pings and profile.transactions == 0
            assert not tx_sites and not profile.tx_categories
            assert all(_category(record['chain'], record['sql']) == 'session_control' for record in records)
        writes = await meter.report()
        after = await _snapshot(original, store.table_prefix)
        delta = {name: [after[name][i]-before[name][i] for i in (0, 1)]
                 for name in after if after[name] != before[name]}
        data = dict(db=original.dialect.name, operation=operation, samples=1,
            sql={**_counts(records, tx_sites), 'total':len(records)}, transactions=len(tx_sites),
            native_pings=profile.pings, records=records, tx_sites=tx_sites, writes=writes,
            net_tables=delta, instrumented_elapsed_s=elapsed,
            timing_note='Write triggers enabled; not a latency benchmark. Native connection setup is not SQL-profiled.')
        if output := os.environ.get('P7_PORTS_OUT'):
            with Path(output).open('a') as file:
                file.write(json.dumps(data)+'\n')
    finally:
        await guard.release()
        await client.aclose()
        await transport.aclose()
        store._db = original
        await meter.close()


async def test_late_child_result_retention_cost(registry_store, monkeypatch):
    store, _ = registry_store
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix)
    profile = PortsProfile(original, meter)
    store._db = profile  # Attach before the workflow runtime and ports are built.
    data = {}

    @asynccontextmanager
    async def measure():
        await meter.reset()
        profile.records.clear()
        profile.tx_sites.clear()
        profile.tx_categories.clear()
        profile.transactions = profile.pings = 0
        started = time.monotonic()
        yield
        elapsed = time.monotonic() - started
        records = [{**r, 'category': _category(r['chain'], r['sql'])} for r in profile.records]
        tx_sites, tx_categories = list(profile.tx_sites), list(profile.tx_categories)
        transactions = profile.transactions
        counts = _counts(records, tx_sites, tx_categories)
        assert records and len(tx_sites) == len(tx_categories) == transactions
        assert counts.get('unfinished_transactions', 0) == 0
        assert sum(counts[name] for name in _CATEGORIES) == len(records)
        assert sum(counts[name + '_tx'] for name in _CATEGORIES) == transactions
        assert any('agent.runtime_store:_river_deliver_tx' in r['chain'] and 'follow_up_queue' in r['sql']
                   for r in records)
        assert any('agent.runtime_store:_river_deliver_tx' in r['chain'] and 'sessions' in r['sql']
                   for r in records)
        assert not any('workflow.runner:' in r['chain'] for r in records)
        writes = await meter.report()
        data.update(db=original.dialect.name, operation='workflow_halt_late_result', samples=1,
            sql={**counts, 'total': len(records)}, transactions=transactions, records=records,
            tx_sites=tx_sites, tx_categories=tx_categories, native_pings=profile.pings,
            writes=writes, net_tables={name: [delta['rows'], delta['bytes']]
                for name, delta in writes['net_residue'].items() if delta['rows'] or delta['bytes']},
            instrumented_elapsed_s=elapsed,
            timing_note='B release through driver commit and worker drain, measured with WriteMeter enabled.')

    try:
        data['verification'] = await _exercise_halt_late_result(store, original, monkeypatch, measure=measure)
        if output := os.environ.get('P7_PORTS_OUT'):
            with Path(output).open('a') as file:
                file.write(json.dumps(data)+'\n')
    finally:
        store._db = original
        await meter.close()


@pytest.mark.parametrize('operation', ['managed_notice_known', 'managed_notice_resolved'])
async def test_notice_cost(registry_store, operation):
    store, connect = registry_store
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix)
    profile = PortsProfile(original, meter)
    store._db = profile

    data = {}

    @asynccontextmanager
    async def measure():
        await meter.reset()
        profile.records.clear()
        profile.tx_sites.clear()
        profile.tx_categories.clear()
        profile.transactions = profile.pings = 0
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            records = [{**r, 'category': _category(r['chain'], r['sql'])} for r in profile.records]
            tx_sites, tx_categories = list(profile.tx_sites), list(profile.tx_categories)
            counts = _counts(records, tx_sites, tx_categories)
            writes = await meter.report()
            data.update(db=original.dialect.name, operation=operation, samples=1,
                sql={**counts, 'total': len(records)}, transactions=profile.transactions,
                tx_sites=tx_sites, tx_categories=tx_categories, records=records, writes=writes,
                net_tables={name: [delta['rows'], delta['bytes']]
                    for name, delta in writes['net_residue'].items() if delta['rows'] or delta['bytes']},
                instrumented_elapsed_s=elapsed, native_pings=profile.pings,
                timing_note='Release known delivery or read-only recovery through parent input final and worker drain. '
                            'Preparation, initial unknown notice, meter SQL and later test readbacks are excluded.')

    try:
        data['verification'] = await _exercise_managed_notice(store, original,
            pending=operation == 'managed_notice_resolved', measure=measure, connect=connect)
        counts = data['sql']
        assert counts['total'] > 0 and counts.get('unfinished_transactions', 0) == 0
        assert sum(counts[name] for name in _CATEGORIES) == counts['total']
        assert sum(counts[name + '_tx'] for name in _CATEGORIES) == data['transactions']
    finally:
        if data and (output := os.environ.get('P7_PORTS_OUT')):
            with Path(output).open('a') as file:
                file.write(json.dumps(data)+'\n')
        store._db = original
        await meter.close()
