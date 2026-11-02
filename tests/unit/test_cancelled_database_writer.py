"""A cancelled native lock wait must retain its original cancellation cause."""
from __future__ import annotations

import asyncio
import os
import threading

import pytest

from power_loop.runtime.store.factory import _make_database

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_cancelled_lock_wait_does_not_replace_cancellation_with_rollback_disconnect(registry_store):
    store, _ = registry_store
    db, table = store._db, store.table_prefix+'session_controls'
    sid = await store.create_session()
    assert await db.execute(f'UPDATE {table} SET version=? WHERE session_id=?', (1, sid)) == 1
    if db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    writer_db, observer = await _make_database(dsn), await _make_database(dsn)
    attempted, entered = asyncio.Event(), asyncio.Event()

    async def writer():
        attempted.set()
        async with writer_db.transaction() as tx:
            entered.set()
            await tx.execute(f'UPDATE {table} SET version=version+1 WHERE session_id=?', (sid,))

    action = None
    try:
        async with db.transaction() as tx:
            await tx.execute(f'UPDATE {table} SET version=version WHERE session_id=?', (sid,))
            action = asyncio.create_task(writer())
            await attempted.wait()
            if db.dialect.name == 'sqlite':
                # BEGIN IMMEDIATE itself waits in the actual SQLite worker.
                await asyncio.sleep(.05)
                assert writer_db._lock.locked() and not entered.is_set()
            else:
                await entered.wait()
                async with asyncio.timeout(5):
                    while True:
                        if db.dialect.name == 'postgres':
                            waiting = await observer.fetchone('SELECT pid FROM pg_stat_activity WHERE datname=current_database() '
                                "AND wait_event_type='Lock' AND query LIKE ?", ('%UPDATE '+table+'%',))
                        else:
                            waiting = await observer.fetchone('SELECT w.REQUESTING_ENGINE_TRANSACTION_ID '
                                'FROM performance_schema.data_lock_waits w JOIN performance_schema.data_locks l '
                                'ON l.ENGINE_LOCK_ID=w.REQUESTING_ENGINE_LOCK_ID '
                                'WHERE l.OBJECT_SCHEMA=DATABASE() AND l.OBJECT_NAME=?', (table,))
                        if waiting:
                            break
                        await asyncio.sleep(.01)
            assert not action.done()
            action.cancel('cancel actual database wait')
            await asyncio.sleep(.01)
        with pytest.raises(asyncio.CancelledError, match='cancel actual database wait'):
            await action
        assert (await db.fetchone(f'SELECT version FROM {table} WHERE session_id=?', (sid,)))['version'] == 1
        # The pool must provide a usable connection after that cancellation.
        async with writer_db.transaction() as tx:
            assert await tx.execute(f'UPDATE {table} SET version=version+1 WHERE session_id=?', (sid,)) == 1
        assert (await db.fetchone(f'SELECT version FROM {table} WHERE session_id=?', (sid,)))['version'] == 2
    finally:
        if action is not None and not action.done():
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
        await writer_db.close()
        await observer.close()


async def test_cancelled_native_snapshot_query_keeps_cancellation_and_releases_connection(registry_store):
    store, _ = registry_store
    db, owner = store._db, 'snapshot_wait '+store.table_prefix
    observer = None
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    if db.dialect.name == 'sqlite':
        def blocked():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10)
            return 1
        db._conn.create_function('snapshot_cancel_wait', 0, blocked)
        query = 'SELECT snapshot_cancel_wait() AS value'
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
        observer = await _make_database(dsn)
        query = ('SELECT pg_sleep(10)' if db.dialect.name == 'postgres' else 'SELECT SLEEP(10)')+' /* '+owner+' */'

    async def inspect():
        async with db.read_snapshot() as tx:
            await tx.fetchone(query)

    action = asyncio.create_task(inspect())
    try:
        if db.dialect.name == 'sqlite':
            await asyncio.wait_for(entered.wait(), 3)
        else:
            async with asyncio.timeout(5):
                while True:
                    if db.dialect.name == 'postgres':
                        native = await observer.fetchone('SELECT pid FROM pg_stat_activity WHERE datname=current_database() '
                            "AND state='active' AND query LIKE ?", ('%'+owner+'%',))
                    else:
                        native = await observer.fetchone('SELECT id FROM performance_schema.processlist '
                            "WHERE id<>CONNECTION_ID() AND COMMAND='Query' AND INFO LIKE ?", ('%'+owner+'%',))
                    if native:
                        break
                    await asyncio.sleep(.01)
        assert not action.done()
        action.cancel('cancel actual snapshot query')
        await asyncio.sleep(.01)
        release.set()
        with pytest.raises(asyncio.CancelledError, match='cancel actual snapshot query'):
            await action
        async with db.read_snapshot() as tx:
            assert (await tx.fetchone('SELECT 1 AS value'))['value'] == 1
    finally:
        release.set()
        if not action.done():
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
        if observer is not None:
            await observer.close()
