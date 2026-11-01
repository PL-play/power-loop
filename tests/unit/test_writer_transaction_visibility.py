"""A writer's earlier lookup must not hide changes committed before its root lock.

This tests database visibility,not the final SDK lease-takeover acceptance gate.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('snapshot', [False, True])
async def test_lookup_then_root_lock_sees_committed_parent_change_while_snapshot_stays_coherent(
        registry_store, tmp_path, snapshot):
    store, _ = registry_store
    db, table = store._db, store.table_prefix+'visibility_probe'
    await db.execute(f'CREATE TABLE {table} (node_id VARCHAR(16) PRIMARY KEY,generation INTEGER NOT NULL)')
    for node in ('root', 'parent', 'leaf'):
        await db.execute(f'INSERT INTO {table} (node_id,generation) VALUES (?,?)', (node, 1))
    if db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    ready, release, attempted, output = (tmp_path/name for name in ('visibility.ready', 'visibility.release', 'visibility.attempt', 'visibility.done'))
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._writer_visibility_worker',
        dsn, table, str(ready), str(release), str(attempted), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        await wait_ready(process, ready)
        assert json.loads(ready.read_text())['pid'] > 0
        scope = db.read_snapshot if snapshot else db.transaction
        async with scope() as tx:
            assert (await tx.fetchone(f'SELECT generation FROM {table} WHERE node_id=?', ('leaf',)))['generation'] == 1
            release.touch()
            await wait_ready(process, attempted)
            serialized = db.dialect.name == 'sqlite' and not snapshot
            if not serialized:
                await wait_ready(process, output)
            if not snapshot:
                lock = '' if db.dialect.name == 'sqlite' else ' FOR UPDATE'
                assert (await tx.fetchone(f'SELECT generation FROM {table} WHERE node_id=?'+lock, ('root',)))['generation'] == 1
            parent = await tx.fetchone(f'SELECT generation FROM {table} WHERE node_id=?', ('parent',))
            assert parent['generation'] == (1 if snapshot or serialized else 2)
            assert (await tx.fetchone(f'SELECT generation FROM {table} WHERE node_id=?', ('leaf',)))['generation'] == 1
        await wait_ready(process, output)
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stdout.decode()
        assert json.loads(output.read_text()) == {'changed': 1}
        assert (await db.fetchone(f'SELECT generation FROM {table} WHERE node_id=?', ('parent',)))['generation'] == 2
    finally:
        release.touch()
        await stop_worker(process)
        await db.execute(f'DROP TABLE {table}')
