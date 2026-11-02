"""A context gate must notice a statement snapshot older than its row lock.

Database primitive only; full public cancellation/admission races are separate.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop.runtime.store.factory import _make_database

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_one_locked_gate_read_detects_a_precommit_original_snapshot(registry_store, tmp_path):
    store, _ = registry_store
    db, table = store._db, store.table_prefix+'context_gate_probe'
    await db.execute(f'CREATE TABLE {table} (kind VARCHAR(16) PRIMARY KEY,value INTEGER NOT NULL)')
    for kind in ('gate', 'barrier'):
        await db.execute(f'INSERT INTO {table} (kind,value) VALUES (?,?)', (kind, 0))
    if db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    observer = await _make_database(dsn)
    ready, release = tmp_path/'context-gate.ready', tmp_path/'context-gate.release'
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._context_gate_snapshot_worker',
        dsn, table, str(ready), str(release), cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    entered = asyncio.Event()
    action = None

    async def read():
        async with db.transaction() as tx:
            entered.set()
            suffix = '' if db.dialect.name == 'sqlite' else ' FOR UPDATE'
            row = await tx.fetchone(f'SELECT g.value AS current_version,'
                f"(SELECT value FROM {table} WHERE kind='gate') AS observed_version,"
                f"(SELECT value FROM {table} WHERE kind='barrier') AS blocked "
                f"FROM {table} g WHERE g.kind='gate'"+suffix)
            safe = row['current_version'] == row['observed_version'] and row['blocked'] == 0
            if row['current_version'] != row['observed_version']:
                fresh = await tx.fetchone(f"SELECT value FROM {table} WHERE kind='barrier'")
                assert fresh['value'] == 1
            return row, safe

    try:
        await wait_ready(process, ready)
        assert json.loads(ready.read_text())['pid'] == process.pid
        action = asyncio.create_task(read())
        if db.dialect.name == 'sqlite':
            await asyncio.sleep(.05)
            assert db._lock.locked() and not entered.is_set()
        else:
            await entered.wait()
            async with asyncio.timeout(5):
                while True:
                    if db.dialect.name == 'postgres':
                        waiting = await observer.fetchone('SELECT pid FROM pg_stat_activity WHERE datname=current_database() '
                            "AND wait_event_type='Lock' AND query LIKE ?", ('%'+table+'%',))
                    else:
                        waiting = await observer.fetchone('SELECT w.REQUESTING_ENGINE_TRANSACTION_ID '
                            'FROM performance_schema.data_lock_waits w JOIN performance_schema.data_locks l '
                            'ON l.ENGINE_LOCK_ID=w.REQUESTING_ENGINE_LOCK_ID '
                            'WHERE l.OBJECT_SCHEMA=DATABASE() AND l.OBJECT_NAME=?', (table,))
                    if waiting:
                        break
                    await asyncio.sleep(.01)
        assert not action.done()
        release.touch()
        row, safe = await asyncio.wait_for(action, 10)
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stdout.decode()
        assert row['current_version'] == 1 and not safe
        if db.dialect.name == 'postgres':
            # Looking only at the barrier subquery would incorrectly admit.
            assert row['observed_version'] == row['blocked'] == 0
        else:
            assert row['blocked'] == 1 or row['current_version'] != row['observed_version']
    finally:
        release.touch()
        await stop_worker(process)
        if action is not None:
            await asyncio.gather(action, return_exceptions=True)
        await observer.close()
        await db.execute(f'DROP TABLE {table}')
