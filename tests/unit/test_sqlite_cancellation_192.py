"""Real worker barriers: cancellation must not release a busy connection."""
import asyncio
import sqlite3
import threading

import pytest

from power_loop import SessionStore
from power_loop.runtime.store.backends.sqlite import SqliteDatabase
from power_loop.runtime.store.factory import open_store

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def entered(event):
    async with asyncio.timeout(5):
        while not event.is_set():
            await asyncio.sleep(.001)


async def cancelled_twice(task):
    task.cancel()
    await asyncio.sleep(.01)
    task.cancel()
    await asyncio.sleep(.01)
    assert not task.done(), 'thread must drain before cancellation releases ownership'


@pytest.mark.parametrize('factory', ['session', 'dsn'])
@pytest.mark.parametrize('cancel', [False, True])
async def test_async_open_does_not_block_and_closes_late_connection(tmp_path, monkeypatch, factory, cancel):
    original = SqliteDatabase.open
    gate, release = threading.Event(), threading.Event()
    opened = []
    def slow_open(path):
        db = original(path)
        opened.append(db)
        gate.set()
        assert release.wait(5)
        return db
    monkeypatch.setattr(SqliteDatabase, 'open', staticmethod(slow_open))
    opening = SessionStore.open if factory == 'session' else open_store
    task = asyncio.create_task(opening(str(tmp_path/'open.db')))
    try:
        await entered(gate)
        if cancel:
            await cancelled_twice(task)
        assert not task.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert opened[0]._closed
        else:
            store = await task
            assert await store.create_session()
            await store.close()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for db in opened:
            await db.close()


@pytest.mark.parametrize('stage', ['begin', 'statement', 'commit', 'rollback'])
async def test_transaction_cancellation_drains_worker_then_restores_connection(stage):
    db = await SqliteDatabase.connect()
    await db.execute('CREATE TABLE t (id INTEGER PRIMARY KEY)')
    original = db._b_exec
    gate, release = threading.Event(), threading.Event()
    sql_match = {'begin': 'BEGIN IMMEDIATE', 'statement': 'INSERT INTO t VALUES (1)', 'commit': 'COMMIT', 'rollback': 'ROLLBACK'}[stage]
    def blocked(sql, params):
        result = original(sql, params)
        if sql == sql_match and not gate.is_set():
            gate.set()
            assert release.wait(5)
        return result
    db._b_exec = blocked
    async def transaction():
        async with db.transaction() as tx:
            await tx.execute('INSERT INTO t VALUES (1)')
            if stage == 'rollback':
                raise ValueError('original body failed')
    task = asyncio.create_task(transaction())
    contender = None
    try:
        await entered(gate)
        await cancelled_twice(task)
        contender = asyncio.create_task(db.execute('INSERT INTO t VALUES (2)'))
        await asyncio.sleep(.02)
        assert not contender.done() and db._lock.locked()
        release.set()
        with pytest.raises(ValueError if stage == 'rollback' else asyncio.CancelledError):
            await task
        await contender
        assert not db._conn.in_transaction
        assert await db.fetchall('SELECT id FROM t ORDER BY id') == ([{'id': 1}, {'id': 2}] if stage == 'commit' else [{'id': 2}])
    finally:
        release.set()
        await asyncio.gather(task, *([contender] if contender else []), return_exceptions=True)
        await db.close()


@pytest.mark.parametrize('method', ['execute', 'fetchone', 'fetchall'])
async def test_autocommit_cancellation_keeps_close_waiting_for_actual_worker(method):
    db = await SqliteDatabase.connect()
    gate, release = threading.Event(), threading.Event()
    def waiting():
        gate.set()
        assert release.wait(5)
        return 42
    db._conn.create_function('worker_gate', 0, waiting)
    task = asyncio.create_task(getattr(db, method)('SELECT worker_gate()'))
    closing = None
    try:
        await entered(gate)
        await cancelled_twice(task)
        closing = asyncio.create_task(db.close())
        await asyncio.sleep(.02)
        assert not closing.done() and not db._closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closing
        assert db._closed
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        await db.close()


async def test_failed_rollback_quarantines_connection_and_preserves_original_error(monkeypatch):
    db = await SqliteDatabase.connect()
    original = db._b_exec
    def broken(sql, params):
        if sql == 'ROLLBACK':
            raise sqlite3.OperationalError('rollback unavailable')
        return original(sql, params)
    monkeypatch.setattr(db, '_b_exec', broken)
    try:
        with pytest.raises(ValueError, match='original'):
            async with db.transaction():
                raise ValueError('original')
        assert db._unusable and db._closed
        with pytest.raises(RuntimeError, match='unusable'):
            await db.execute('SELECT 1')
    finally:
        await db.close()


async def test_failed_pragma_closes_partially_initialized_connection(tmp_path, monkeypatch):
    original = sqlite3.connect
    connections = []
    class Connection(sqlite3.Connection):
        closed = False
        def execute(self, sql, *args):
            if sql == 'PRAGMA journal_mode=WAL':
                raise sqlite3.OperationalError('cannot initialize WAL')
            return super().execute(sql, *args)
        def close(self):
            super().close()
            self.closed = True
    def connect(*args, **kwargs):
        connection = original(*args, **kwargs, factory=Connection)
        connections.append(connection)
        return connection
    monkeypatch.setattr(sqlite3, 'connect', connect)
    with pytest.raises(sqlite3.OperationalError, match='initialize WAL'):
        await SqliteDatabase.connect(str(tmp_path/'broken.db'))
    assert len(connections) == 1 and connections[0].closed


@pytest.mark.parametrize('operation', ['checkpoint', 'vacuum', 'backup', 'close'])
async def test_maintenance_and_close_drain_their_real_worker(tmp_path, monkeypatch, operation):
    from power_loop.runtime.store.backends import sqlite as backend
    db = await SqliteDatabase.connect(str(tmp_path/'main.db'))
    gate, release = threading.Event(), threading.Event()
    offload = backend._offload
    def blocked(function, args):
        value = function(*args)
        gate.set()
        assert release.wait(5)
        return value
    async def run(function, *args):
        return await offload(blocked, function, args)
    monkeypatch.setattr(backend, '_offload', run)
    coro = db.backup(str(tmp_path/'copy.db')) if operation == 'backup' else getattr(db, operation)()
    task = asyncio.create_task(coro)
    try:
        await entered(gate)
        # Simulate the host cancelling every new asyncio Task. Executor work
        # must not depend on a separate wrapper Task surviving that operation.
        owned_tasks = asyncio.all_tasks() - {asyncio.current_task()}
        for child in owned_tasks:
            if child is task or child.get_coro().__qualname__ == 'to_thread':
                child.cancel()
        await cancelled_twice(task)
        assert db._lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not db._lock.locked()
        if operation == 'close':
            assert db._closed
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await db.close()
