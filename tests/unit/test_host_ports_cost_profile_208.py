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
from tests.write_meter import WriteMeter

from .test_job_registry_192 import registry_store
from .test_runtime_messaging_203 import fixture
from .test_send_row_profile import _snapshot
from .test_send_sql_profile import Profiled, _counts

__all__ = ['registry_store']


class NativeConnection:
    def __init__(self, original, profile):
        self.original, self.profile = original, profile

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def fetchval(self, sql, *args):
        self.profile.note(sql)
        return await self.original.fetchval(sql, *args)

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
                    profile.note(sql)
                    return await cursor.execute(sql, args)
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


async def test_late_child_result_retention_cost(registry_store):
    from .test_late_node_results_208 import (
        test_late_child_is_retained_and_consumed_only_by_explicit_node_resume as exercise,
    )

    store, _ = registry_store
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix)
    profile = PortsProfile(original, meter)
    store._db = profile  # Every client, worker and job manager is built afterward.

    @asynccontextmanager
    async def measure(store):
        before = await _snapshot(original, store.table_prefix)
        await meter.reset()
        profile.records.clear()
        profile.tx_sites.clear()
        started = time.monotonic()
        yield
        elapsed = time.monotonic() - started
        records, tx_sites = list(profile.records), list(profile.tx_sites)
        after = await _snapshot(original, store.table_prefix)
        data = dict(db=original.dialect.name, operation='late_child_result_retention', samples=1,
            sql={**_counts(records, tx_sites), 'total':len(records)},
            transactions=len(tx_sites), records=records, tx_sites=tx_sites,
            writes=await meter.report(), net_tables={name:[after[name][i]-before[name][i] for i in (0,1)]
                for name in after if after[name]!=before[name]}, instrumented_elapsed_s=elapsed,
            timing_note='Child release through worker drain; write triggers enabled, not a latency benchmark.')
        if output := os.environ.get('P7_PORTS_OUT'):
            with Path(output).open('a') as file:
                file.write(json.dumps(data)+'\n')
    try:
        await exercise(registry_store, late_measure=measure)
    finally:
        store._db = original
        await meter.close()


@pytest.mark.parametrize('operation', ['wait_continue', 'wait_defer', 'managed_notice_known', 'managed_notice_resolved'])
async def test_wait_and_notice_cost(registry_store, operation):
    store, _ = registry_store
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix)
    profile = PortsProfile(original, meter)
    store._db = profile

    @asynccontextmanager
    async def measure(store):
        before = await _snapshot(original, store.table_prefix)
        await meter.reset()
        profile.records.clear()
        profile.tx_sites.clear()
        started = time.monotonic()
        yield
        elapsed = time.monotonic() - started
        # Test assertions read via the instrumented store too; exclude those
        # calls without an SDK frame, and retain every actual SDK SQL call.
        records = [row for row in profile.records if row['chain']]
        tx_sites = [chain for chain in profile.tx_sites if chain]
        after = await _snapshot(original, store.table_prefix)
        data = dict(db=original.dialect.name, operation=operation, samples=1,
            sql={**_counts(records, tx_sites), 'total':len(records)},
            transactions=len(tx_sites), records=records, tx_sites=tx_sites,
            writes=await meter.report(), net_tables={name:[after[name][i]-before[name][i] for i in (0,1)]
                for name in after if after[name]!=before[name]}, instrumented_elapsed_s=elapsed,
            timing_note='Whole finite wait through original child completion, or accepted notice through original main completion; includes worker drain/close. Assertion-only reads excluded. Write triggers enabled; not a latency benchmark.')
        if output := os.environ.get('P7_PORTS_OUT'):
            with Path(output).open('a') as file:
                file.write(json.dumps(data)+'\n')
    try:
        if operation.startswith('wait_'):
            from .test_wait_review_208 import test_finite_wait_preserves_original_child_and_capacity_one
            await test_finite_wait_preserves_original_child_and_capacity_one(registry_store,
                operation.removeprefix('wait_'), measure=measure)
        else:
            from .test_managed_delivery_192 import (
                test_main_actually_consumes_the_known_outcome_and_original_error_is_unchanged,
            )
            await test_main_actually_consumes_the_known_outcome_and_original_error_is_unchanged(registry_store,
                operation == 'managed_notice_resolved', measure=measure)
    finally:
        store._db = original
        await meter.close()
