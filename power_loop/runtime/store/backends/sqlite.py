"""SQLite backend (default, zero-dependency).

Wraps a single stdlib ``sqlite3`` connection (``check_same_thread=False``, autocommit
so transaction boundaries are explicit) and runs every statement in a worker thread via
``asyncio.to_thread`` so the async store never blocks the event loop. An ``asyncio.Lock``
serializes writers — preserving SQLite's one-writer model — so a transaction's
``SELECT next_seq → … → UPDATE`` is atomic without DB row locks. WAL keeps readers
non-blocking. (A dedicated read pool can be layered later; for now reads share the write
connection under the lock.)
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from power_loop.runtime.store.db import Params, Row
from power_loop.runtime.store.dialect import Dialect, SqliteDialect


class _SqliteTransaction:
    """Lock-free statement runner; the owning Database holds its write lock for the
    whole ``transaction()`` block, so these never interleave with another transaction."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    async def execute(self, sql: str, params: Params = ()) -> int:
        return await self._db._exec(self._db.dialect.translate(sql), params)

    async def fetchone(self, sql: str, params: Params = ()) -> Row | None:
        return await self._db._fetchone(self._db.dialect.translate(sql), params)

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]:
        return await self._db._fetchall(self._db.dialect.translate(sql), params)


class SqliteDatabase:
    def __init__(self, conn: sqlite3.Connection, *, path: str) -> None:
        self._conn = conn
        self.path = path
        self.dialect: Dialect = SqliteDialect()
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    def open(cls, path: str = ":memory:") -> SqliteDatabase:
        if path != ":memory:":
            from pathlib import Path

            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            path = str(p)
        # isolation_level=None → autocommit; we issue explicit BEGIN/COMMIT/ROLLBACK so
        # the transaction boundary is uniform with the server backends.
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        if path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return cls(conn, path=path)

    # ── blocking primitives (run in a worker thread) ───────────────────────────
    def _b_exec(self, sql: str, params: Params) -> int:
        return self._conn.execute(sql, tuple(params)).rowcount

    def _b_fetchone(self, sql: str, params: Params) -> Row | None:
        r = self._conn.execute(sql, tuple(params)).fetchone()
        return dict(r) if r is not None else None

    def _b_fetchall(self, sql: str, params: Params) -> list[Row]:
        return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    async def _exec(self, sql: str, params: Params = ()) -> int:
        return await asyncio.to_thread(self._b_exec, sql, params)

    async def _fetchone(self, sql: str, params: Params = ()) -> Row | None:
        return await asyncio.to_thread(self._b_fetchone, sql, params)

    async def _fetchall(self, sql: str, params: Params = ()) -> list[Row]:
        return await asyncio.to_thread(self._b_fetchall, sql, params)

    # ── transaction ────────────────────────────────────────────────────────────
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_SqliteTransaction]:
        async with self._lock:
            await self._exec("BEGIN")
            try:
                yield _SqliteTransaction(self)
            except BaseException:
                await self._exec("ROLLBACK")
                raise
            else:
                await self._exec("COMMIT")

    # ── autocommit reads / single writes (lock-guarded) ────────────────────────
    async def fetchone(self, sql: str, params: Params = ()) -> Row | None:
        async with self._lock:
            return await self._fetchone(self.dialect.translate(sql), params)

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]:
        async with self._lock:
            return await self._fetchall(self.dialect.translate(sql), params)

    async def execute(self, sql: str, params: Params = ()) -> int:
        async with self._lock:
            return await self._exec(self.dialect.translate(sql), params)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            await asyncio.to_thread(self._conn.close)

    # ── Maintenance capability (SQLite-only; see store/capabilities.py) ─────────
    async def checkpoint(self, *, mode: str = "TRUNCATE") -> None:
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            raise ValueError(f"invalid checkpoint mode: {mode!r}")
        async with self._lock:
            await asyncio.to_thread(self._conn.execute, f"PRAGMA wal_checkpoint({mode})")

    async def vacuum(self, *, incremental: bool = True) -> None:
        async with self._lock:
            # VACUUM cannot run inside a transaction; this conn is in autocommit mode.
            sql = "PRAGMA incremental_vacuum" if incremental else "VACUUM"
            await asyncio.to_thread(self._conn.execute, sql)

    async def backup(self, dest_path: str) -> None:
        def _backup() -> None:
            dest = sqlite3.connect(dest_path)
            try:
                self._conn.backup(dest)
            finally:
                dest.close()

        async with self._lock:
            await asyncio.to_thread(_backup)


__all__ = ["SqliteDatabase"]
