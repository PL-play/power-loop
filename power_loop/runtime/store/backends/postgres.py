"""PostgreSQL backend (``power-loop[postgres]`` extra).

A thin async wrapper over an ``asyncpg`` connection pool. Unlike SQLite, this is a real
multi-writer client/server backend: per-session ``seq`` allocation stays correct across
processes because :meth:`PostgresDialect.lock_state` takes a ``SELECT … FOR UPDATE`` row
lock inside the transaction (the ``(session_id, seq)`` PK is the backstop). ``asyncpg`` is
imported lazily in :meth:`connect`, so the core never depends on it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from power_loop.runtime.store.db import Params, Row
from power_loop.runtime.store.dialect import Dialect, PostgresDialect


def _affected(tag: str) -> int:
    """Parse asyncpg's command tag into an affected-row count.

    Tags look like ``"UPDATE 3"`` / ``"DELETE 1"`` / ``"INSERT 0 1"`` / ``"CREATE TABLE"``.
    The trailing integer is the row count (0 for DDL / no-count statements)."""
    parts = tag.split()
    return int(parts[-1]) if parts and parts[-1].isdigit() else 0


class _PgTransaction:
    def __init__(self, db: PostgresDatabase, conn: Any) -> None:
        self._db = db
        self._conn = conn

    async def execute(self, sql: str, params: Params = ()) -> int:
        return _affected(await self._conn.execute(self._db.dialect.translate(sql), *params))

    async def fetchone(self, sql: str, params: Params = ()) -> Row | None:
        r = await self._conn.fetchrow(self._db.dialect.translate(sql), *params)
        return dict(r) if r is not None else None

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]:
        rows = await self._conn.fetch(self._db.dialect.translate(sql), *params)
        return [dict(r) for r in rows]


class PostgresDatabase:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.dialect: Dialect = PostgresDialect()
        self._closed = False

    @classmethod
    async def connect(cls, dsn: str, *, min_size: int = 1, max_size: int = 10) -> PostgresDatabase:
        import asyncpg  # optional extra: power-loop[postgres]

        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        return cls(pool)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_PgTransaction]:
        async with self._pool.acquire() as conn, conn.transaction():
            yield _PgTransaction(self, conn)

    async def fetchone(self, sql: str, params: Params = ()) -> Row | None:
        async with self._pool.acquire() as conn:
            r = await conn.fetchrow(self.dialect.translate(sql), *params)
            return dict(r) if r is not None else None

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(self.dialect.translate(sql), *params)
            return [dict(r) for r in rows]

    async def execute(self, sql: str, params: Params = ()) -> int:
        async with self._pool.acquire() as conn:
            return _affected(await conn.execute(self.dialect.translate(sql), *params))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._pool.close()


__all__ = ["PostgresDatabase"]
