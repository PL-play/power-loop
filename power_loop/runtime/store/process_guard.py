"""Startup-only exclusion for process-local coordination domains.

Native locks belong to a dedicated connection/file, never a task lease table.
The bounded registry holds only live runtimes and drops the last reference.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from power_loop.agent.runtime_types import RuntimeRequestError

_entries = {}


class ProcessStoreGuard:
    def __init__(self, store, coordinator, transport):
        self.store, self.db = store, store._db
        self.domain = (coordinator, transport)
        self.kind = self.db.dialect.name
        self.key = (self.kind, store.store_id)
        if self.kind == 'sqlite':
            path = self.db.path
            self.key = ('sqlite', id(self.db)) if path == ':memory:' else ('sqlite', str(Path(path).resolve()))
        self.entry = None

    async def acquire(self):
        existing = _entries.get(self.key)
        if existing is not None:
            if existing.domain != self.domain:
                raise RuntimeRequestError('invalid_options', 'store already has a different process-local coordination domain')
            await asyncio.shield(existing.ready)
            existing.users += 1
            self.entry = existing
            return
        self.ready = asyncio.get_running_loop().create_future()
        self.users, self.connection, self.file = 1, None, None
        _entries[self.key] = self
        try:
            if self.kind == 'sqlite':
                if self.db.path != ':memory:':
                    import fcntl
                    self.file = open(self.db.path, 'rb')
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif self.kind == 'postgres':
                self.connection = await self.db.open_lock_connection()
                self.lock_key = int.from_bytes(hashlib.sha256(self.store.table_prefix.encode()).digest()[:8], 'big', signed=True)
                if not await self.connection.fetchval('SELECT pg_try_advisory_lock($1)', self.lock_key):
                    raise RuntimeRequestError('invalid_options', 'store already owned by another default runtime')
            elif self.kind == 'mysql':
                self.connection = await self.db.open_lock_connection()
                self.lock_key = 'power-loop-' + hashlib.sha256((self.connection.db.decode() if isinstance(self.connection.db, bytes) else self.connection.db).encode() + self.store.table_prefix.encode()).hexdigest()[:48]
                async with self.connection.cursor() as cursor:
                    await cursor.execute('SELECT GET_LOCK(%s,0)', (self.lock_key,))
                    if (await cursor.fetchone())[0] != 1:
                        raise RuntimeRequestError('invalid_options', 'store already owned by another default runtime')
            else:
                raise RuntimeRequestError('invalid_options', 'store cannot hold a native process lock')
        except BaseException as exc:
            _entries.pop(self.key, None)
            await self._close()
            self.ready.set_exception(exc)
            self.ready.exception()
            if isinstance(exc, BlockingIOError):
                raise RuntimeRequestError('invalid_options', 'store already owned by another default runtime') from exc
            raise
        self.ready.set_result(None)
        self.entry = self

    async def check(self):
        entry = self.entry
        if entry.kind == 'postgres':
            await entry.connection.fetchval('SELECT 1')
        elif entry.kind == 'mysql':
            await entry.connection.ping(reconnect=False)
        elif entry.file is not None:
            os.fstat(entry.file.fileno())

    async def release(self):
        entry, self.entry = self.entry, None
        if entry is None:
            return
        entry.users -= 1
        if entry.users == 0:
            _entries.pop(self.key, None)
            await entry._close()

    async def _close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.connection is not None:
            connection, self.connection = self.connection, None
            # Closing the native connection releases its lock even on failure.
            if self.kind == 'postgres':
                await connection.close()
            else:
                connection.close()
