"""The root serializes its descendants while unrelated server roots proceed."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import control_runtime, model_attempts
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction
from power_loop.runtime.store.factory import _make_database
from power_loop.runtime.task_runtime import TaskRuntime

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def wait_for_admission_lock(manager, observer, action):
    engine = manager.db.dialect.name
    async with asyncio.timeout(5):
        while not action.done():
            if engine == 'sqlite':
                waiting = manager.db._lock.locked() and not manager.db._conn.in_transaction
            elif engine == 'postgres':
                waiting = await observer.fetchone('SELECT pid FROM pg_stat_activity '
                    "WHERE datname=current_database() AND wait_event_type='Lock' AND query LIKE ?",
                    ('%'+manager.prefix+'%',))
            else:
                waiting = await observer.fetchone('SELECT w.REQUESTING_ENGINE_TRANSACTION_ID '
                    'FROM performance_schema.data_lock_waits w JOIN performance_schema.data_locks l '
                    'ON l.ENGINE_LOCK_ID=w.REQUESTING_ENGINE_LOCK_ID '
                    'WHERE l.OBJECT_SCHEMA=DATABASE() AND l.OBJECT_NAME LIKE ?', (manager.prefix+'%',))
            if waiting:
                break
            await asyncio.sleep(.01)
    assert not action.done(), 'writer passed the conflicting root lock'


@pytest.mark.parametrize('target', ['root', 'child', 'other-root', 'checkpoint'])
async def test_actual_root_writer_blocks_its_tree_and_releases_unrelated_roots(registry_store, tmp_path, target):
    store, _ = registry_store
    original = store._db
    store._db = profiled = Profiled(original)
    manager = JobManager(store, now_ms=lambda: 1000000)
    actors = [JobActor('root-lock', 'host', await store.create_session()) for _ in range(2)]
    roots = [await manager.register_job(actor, kind='send', session_id=actor.session_id,
        system_key='root', request={'model_attempt_protocol':1}) for actor in actors]
    runtime = TaskRuntime(manager, tenant_id='root-lock', principal_id='host', executor_id='e', boot_id='b')
    binding = await runtime.claim(actors[0], roots[0])
    lease = binding.lease
    child = await manager.register_job(actors[0], kind='subagent', parent=lease, system_key='child', session_id=actors[0].session_id)
    if target == 'checkpoint':
        await model_attempts.enable(binding)
        control_runtime.current(binding).record('lifecycle', 'root-lock-proof', {'observed':True})
    engine = manager.db.dialect.name
    dsn = next(row['file'] for row in await manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main') \
        if engine == 'sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    observer = await _make_database(dsn)
    ready, release = tmp_path/'root.ready', tmp_path/'root.release'
    config = {'actor':asdict(actors[0]),'job_id':roots[0].job.job_id,'now':1000000}
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._root_writer_worker',
        dsn, store.table_prefix, json.dumps(config), str(ready), str(release),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    action = None

    async def mutate():
        if target == 'root':
            await manager.close_model_execution(actors[0], lease)
            return lease
        if target == 'checkpoint':
            await control_runtime.flush_binding(binding)
            return lease
        actor, registration = (actors[0], child) if target == 'child' else (actors[1], roots[1])
        return await manager.claim_job(actor, registration.job.job_id, executor_id='e', boot_id='b', expected_version=1)

    try:
        await wait_ready(process, ready)
        assert json.loads(ready.read_text()) == {'pid':process.pid,'root':roots[0].job.job_id}
        profiled.records.clear()
        action = asyncio.create_task(mutate())
        blocked = engine == 'sqlite' or target != 'other-root'
        if blocked:
            await wait_for_admission_lock(manager, observer, action)
            assert not action.done()
        else:
            result = await asyncio.wait_for(asyncio.shield(action), 5)
            assert result.job_id == roots[1].job.job_id and result.generation == 1
            assert process.returncode is None and not release.exists()
        release.touch()
        result = await asyncio.wait_for(action, 10)
        assert result.generation == 1
        assert result.job_id == (lease.job_id if target in ('root', 'checkpoint') else
            child.job.job_id if target == 'child' else roots[1].job.job_id)
        if target == 'checkpoint':
            assert control_runtime.current(binding).cursor >= 1
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stdout.decode()
        assert not [r for r in profiled.records if store.table_prefix+'task_store' in r['sql']]
        assert [r for r in profiled.records if 'SELECT root.*' in r['sql'] and store.table_prefix+'jobs' in r['sql']]
    finally:
        release.touch()
        if action is not None and not action.done():
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
        await stop_worker(process)
        manager.db = original
        store._db = original
        await observer.close()


@pytest.mark.parametrize('conflict', [False, True])
async def test_first_root_creation_is_atomic_and_replays_only_after_the_root_lock(registry_store, monkeypatch, conflict):
    store, connect = registry_store
    managers = [JobManager(store), JobManager(await connect())]
    sid = await store.create_session()
    actor = JobActor('root-birth', 'host', sid)
    insert = ControlTransaction.insert_job
    waiting, ready = [], asyncio.Event()

    async def together(tx, job_id, root_job_id, sql, params=()):
        waiting.append(job_id)
        if tx.dialect.name != 'sqlite':
            if len(waiting) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 5)
        return await insert(tx, job_id, root_job_id, sql, params)

    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, 'insert_job', together)
        result = await asyncio.wait_for(asyncio.gather(*(manager.register_job(actor, kind='send',
            session_id=sid, system_key='same-root', request={'value':int(conflict and index == 1)})
            for index, manager in enumerate(managers)), return_exceptions=True), 10)
    if conflict:
        errors = [value for value in result if isinstance(value, JobError)]
        assert len(errors) == 1 and errors[0].code == 'operation_conflict', result
        saved = [value for value in result if not isinstance(value, BaseException)]
        assert len(saved) == 1 and not saved[0].replayed
    else:
        assert all(not isinstance(value, BaseException) for value in result), result
        assert result[0].job.job_id == result[1].job.job_id
        assert sorted(value.replayed for value in result) == [False, True]
    assert len(set(waiting)) == 1
    manager = managers[0]
    assert (await manager.db.fetchone(f'SELECT COUNT(*) AS n FROM {manager.jobs}'))['n'] == 1
    original = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE action='create'")
    assert len(original) == 1
    row = await manager.db.fetchone(f'SELECT generation,version FROM {manager.jobs}')
    assert row == {'generation':0,'version':1}


async def test_undeclared_conditional_job_write_is_rejected_before_its_cte_runs(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    sid = await store.create_session()
    actor = JobActor('root-intent', 'host', sid)
    await manager.register_job(actor, kind='send', session_id=sid, system_key='root')
    before = await manager.db.fetchall(f'SELECT * FROM {manager.jobs}')
    with pytest.raises(AssertionError, match='must declare an explicit intent'):
        async with manager._write() as tx:
            await tx.execute_flag(f'WITH probe AS (SELECT 1 AS n) UPDATE {manager.jobs} SET version=version+1')
    assert await manager.db.fetchall(f'SELECT * FROM {manager.jobs}') == before
