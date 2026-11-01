"""A real SQL cancellation during negative publication retains the result."""
from __future__ import annotations

import asyncio
import os

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime import delegation
from power_loop.runtime.store.factory import _make_database
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_executor_lifecycle_192 import host
from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import AdapterModel

__all__ = ['host', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_shutdown_cancelling_a_native_negative_result_write_keeps_one_original_result(host, monkeypatch):
    store, manager, runtime = host
    tools, model = ToolRegistry(), AdapterModel()
    register_spawn_agent(tools)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=tools,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=3))
    sid = await loop.new_session()
    model.root = sid
    start, semaphore = delegation.start, asyncio.Semaphore(0)

    async def queued(*args, **kwargs):
        return await start(*args, **kwargs, admission=semaphore)

    monkeypatch.setattr(delegation, 'start', queued)
    await loop.send('delegate', sid)
    [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='subagent'")
    run = delegation._RUNNING[id(store), child['session_id']]
    assert model.child_calls == 0 and child['outcome'] is None
    if manager.db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if manager.db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    blocker_db, observer = await _make_database(dsn), await _make_database(dsn)
    locked, release, native_wait = asyncio.Event(), asyncio.Event(), asyncio.Event()
    record, signal = runtime.record_subagent, runtime._lifecycle._signal
    held_task, monitor_task = None, None
    writes, forced = [], []

    async def hold():
        async with blocker_db.transaction() as tx:
            assert await tx.execute(f'UPDATE {manager.jobs} SET version=version WHERE job_id=?', (child['job_id'],)) == 1
            locked.set()
            await release.wait()

    async def observe_native():
        async with asyncio.timeout(3):
            while True:
                if manager.db.dialect.name == 'sqlite':
                    waiting = manager.db._lock.locked() and not manager.db._conn.in_transaction
                elif manager.db.dialect.name == 'postgres':
                    waiting = await observer.fetchone('SELECT pid FROM pg_stat_activity WHERE datname=current_database() '
                        "AND wait_event_type='Lock' AND query LIKE ?", ('%'+manager.prefix+'jobs%',))
                else:
                    waiting = await observer.fetchone('SELECT w.REQUESTING_ENGINE_TRANSACTION_ID '
                        'FROM performance_schema.data_lock_waits w JOIN performance_schema.data_locks l '
                        'ON l.ENGINE_LOCK_ID=w.REQUESTING_ENGINE_LOCK_ID '
                        'WHERE l.OBJECT_SCHEMA=DATABASE() AND l.OBJECT_NAME=?', (manager.jobs,))
                if waiting:
                    native_wait.set()
                    return
                await asyncio.sleep(.01)

    async def recorded(binding, result):
        nonlocal held_task, monitor_task
        writes.append(result['status'])
        if len(writes) == 1:
            held_task = asyncio.create_task(hold())
            await locked.wait()
            monitor_task = asyncio.create_task(observe_native())
        return await record(binding, result)

    def signalled(entries, **kwargs):
        hit = kwargs.get('force') and any(entry['task'] is run.task and not run.task.done() for entry in entries)
        if hit:
            assert native_wait.is_set(), 'shutdown must cancel an actual native database lock wait'
            forced.append(True)
        value = signal(entries, **kwargs)
        if hit:
            release.set()
        return value

    monkeypatch.setattr(runtime, 'record_subagent', recorded)
    monkeypatch.setattr(runtime._lifecycle, '_signal', signalled)
    try:
        result = await runtime.shutdown(timeout_s=3, grace_s=1)
        assert forced and native_wait.is_set() and writes
        assert result['durable_drain_confirmed'] and not result['pending_handles']
        assert model.child_calls == 0
        row = await manager.get_job(runtime.actor(sid), child['job_id'])
        assert row.outcome == 'cancelled' and row.cancel_requested_at is not None
        assert row.resource_state == 'stopped' and row.result['status'] == 'cancelled'
        events = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='result'", (child['job_id'],))
        assert len(events) == 1
        assert not runtime._local_executions and not runtime._lifecycle._owned_tasks
    finally:
        release.set()
        if held_task is not None:
            await held_task
        if monitor_task is not None:
            await monitor_task
        await blocker_db.close()
        await observer.close()
        await loop.aclose(drain_timeout_s=.1)
