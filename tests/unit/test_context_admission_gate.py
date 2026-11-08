"""Exercise the context admission gate independently of the old global mutex.

The worker uses the actual SDK admission read in a native transaction. Public
whole-send/process races use test_shared_session_roots and test_runtime_messaging_203.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import root_context, root_control
from power_loop.runtime.jobs import JobError
from power_loop.runtime.session_controls import ControlTransaction
from power_loop.runtime.store.factory import _make_database

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store
from .test_root_context_192 import NAMESPACE, root_case
from .test_root_control_192 import PAYLOAD, send, stop
from .test_send_sql_profile import Profiled, _counts

__all__ = ['registry_store', 'root_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('mode', ['hold-admission', 'hold-stop'])
async def test_context_admission_and_stop_serialize_without_the_global_store_lock(root_case, tmp_path, mode):
    c = root_case
    controls = c.manager.prefix+'session_controls'
    assert (await c.manager.db.fetchone(f'SELECT version FROM {controls} WHERE session_id=?', (c.sid,)))['version'] == 0
    document = root_context._document(NAMESPACE, PAYLOAD)
    if c.manager.db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await c.manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if c.manager.db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    reader_db, observer = await _make_database(dsn), await _make_database(dsn)
    ready, release, output = (tmp_path/name for name in ('context.ready', 'context.release', 'context.done'))
    config = {'actor': asdict(c.actor), 'document': document}
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._context_admission_gate_worker',
        dsn, c.manager.prefix, json.dumps(config), mode, str(ready), str(release), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    action = None

    async def admit():
        async with reader_db.transaction() as raw:
            tx = ControlTransaction(raw, c.manager)
            try:
                await root_control._admit(c.manager, tx, c.actor, document)
            except JobError as exc:
                return exc.code
            return 'admitted'

    try:
        await wait_ready(process, ready)
        assert json.loads(ready.read_text()) == {'mode': mode, 'pid': process.pid}
        action = asyncio.create_task(stop(c) if mode == 'hold-admission' else admit())
        waiting_db = c.manager.db if mode == 'hold-admission' else reader_db
        async with asyncio.timeout(5):
            while not action.done():
                if c.manager.db.dialect.name == 'sqlite':
                    waiting = waiting_db._lock.locked() and not waiting_db._conn.in_transaction
                elif c.manager.db.dialect.name == 'postgres':
                    waiting = await observer.fetchone('SELECT pid FROM pg_stat_activity WHERE datname=current_database() '
                        "AND wait_event_type='Lock' AND query LIKE ?", ('%'+controls+'%',))
                else:
                    waiting = await observer.fetchone('SELECT w.REQUESTING_ENGINE_TRANSACTION_ID '
                        'FROM performance_schema.data_lock_waits w JOIN performance_schema.data_locks l '
                        'ON l.ENGINE_LOCK_ID=w.REQUESTING_ENGINE_LOCK_ID '
                        'WHERE l.OBJECT_SCHEMA=DATABASE() AND l.OBJECT_NAME=?', (controls,))
                if waiting:
                    break
                await asyncio.sleep(.01)
        assert not action.done(), 'stop/admission crossed an uncommitted context gate'
        release.touch()
        value = await asyncio.wait_for(action, 10)
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stdout.decode()
        if mode == 'hold-admission':
            assert value['state'] == 'accepted' and value['roots'] == []
            assert json.loads(output.read_text()) == {'state': 'admitted'}
        else:
            assert value == 'root_context_cancelled'
            assert json.loads(output.read_text())['state'] == 'accepted'
        with pytest.raises(JobError, match='root_context_cancelled'):
            await c.manager.register_job(c.actor, kind='send', session_id=c.sid, system_key='after-gate',
                request={'host_context': document})
        assert not c.model.calls
        assert not await c.manager.db.fetchall(f'SELECT job_id FROM {c.manager.jobs}')
    finally:
        release.touch()
        await stop_worker(process)
        if action is not None:
            await asyncio.gather(action, return_exceptions=True)
        await reader_db.close()
        await observer.close()


@pytest.mark.parametrize('stage', ['first', 'same-context', 'new-context'])
async def test_context_admission_reads_current_control_state_without_rewriting_jobs(root_case, stage):
    c = root_case
    manager, payload = c.manager, PAYLOAD
    receipt = None
    if stage in ('same-context', 'new-context'):
        await send(c)
    if stage == 'new-context':
        payload = {'origin_run_id': 202}
    assert await manager.db.fetchone(f'SELECT version FROM {manager.prefix}session_controls WHERE session_id=?', (c.sid,)) is not None
    before = await manager.db.fetchall(f'SELECT * FROM {manager.prefix}task_context_events ORDER BY event_id')
    jobs = await manager.db.fetchall(f'SELECT * FROM {manager.jobs} ORDER BY job_id')
    original = manager.db
    profiled = Profiled(original)
    manager.db = manager.store._db = profiled
    try:
        async with manager._write() as tx:
            await root_control._admit(manager, tx, c.actor, root_context._document(NAMESPACE, payload))
    finally:
        manager.db = manager.store._db = original
    queries = [record for record in profiled.records if any(site.startswith('runtime.root_control:') for site in record['chain'])]
    # Current ControlTransaction checks the session control row, then reads its
    # context admission state. Both calls already exist at the P7A source start.
    assert len(queries) == 2
    assert _counts(profiled.records, profiled.tx_sites)['control_tx'] == 1
    assert await manager.db.fetchall(f'SELECT * FROM {manager.prefix}task_context_events ORDER BY event_id') == before
    assert await manager.db.fetchall(f'SELECT * FROM {manager.jobs} ORDER BY job_id') == jobs
    if receipt is not None:
        assert await stop(c, manager=manager) == receipt
