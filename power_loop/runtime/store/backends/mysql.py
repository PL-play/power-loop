"""MySQL backend (``power-loop[mysql]`` extra).

A thin async wrapper over an ``aiomysql`` connection pool (pure-Python driver, no build
step). Like Postgres this is a real multi-writer client/server backend: per-session
``seq`` allocation stays correct across processes because :meth:`MySQLDialect.lock_state`
takes a ``SELECT … FOR UPDATE`` row lock inside the transaction (the ``(session_id, seq)``
PK is the backstop). ``aiomysql`` is imported lazily in :meth:`connect`, so the core
never depends on it.

Two MySQL-specific choices made here:

* **``CLIENT.FOUND_ROWS``** — without it MySQL's ``rowcount`` after an UPDATE counts
  *changed* rows, not *matched* rows, so a no-op UPDATE (e.g. a timer heartbeat that
  re-stamps the same value, or a note update to identical content) would report 0 and
  break the store's CAS/affected-rows checks. With the flag, ``rowcount`` = matched rows,
  matching SQLite/Postgres semantics.
* **``autocommit=True``** on the pool, so a bare ``execute``/``fetch`` outside an explicit
  transaction commits immediately (mirrors asyncpg/SQLite); ``transaction()`` wraps a
  ``begin()``/``commit()``/``rollback()`` block that suspends autocommit for its duration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import unquote, urlparse

from power_loop.runtime.store.db import Params, Row
from power_loop.runtime.store.dialect import Dialect, MySQLDialect


def _args(params: Params) -> tuple[Any, ...] | None:
    # Pass None for a parameterless statement so the driver skips its ``query % args``
    # %-formatting pass entirely (our DDL has no ``%`` literals, and dialect.translate
    # already doubled any that a parameterized statement might carry).
    return tuple(params) if params else None


class _MyTransaction:
    def __init__(self, db: MySQLDatabase, conn: Any) -> None:
        self._db = db
        self._conn = conn

    async def execute(self, sql: str, params: Params = ()) -> int:
        async with self._conn.cursor() as cur:
            await cur.execute(self._db.dialect.translate(sql), _args(params))
            return cur.rowcount

    async def fetchone(self, sql: str, params: Params = ()) -> Row | None:
        import aiomysql  # optional extra: power-loop[mysql]

        async with self._conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(self._db.dialect.translate(sql), _args(params))
            r = await cur.fetchone()
            return dict(r) if r is not None else None

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]:
        import aiomysql

        async with self._conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(self._db.dialect.translate(sql), _args(params))
            return [dict(r) for r in await cur.fetchall()]


class MySQLDatabase:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.dialect: Dialect = MySQLDialect()
        self._closed = False

    @classmethod
    async def connect(cls, dsn: str, *, minsize: int = 1, maxsize: int = 10) -> MySQLDatabase:
        import aiomysql  # optional extra: power-loop[mysql]
        from pymysql.constants import CLIENT  # aiomysql depends on PyMySQL

        u = urlparse(dsn)
        pool = await aiomysql.create_pool(
            host=u.hostname or "localhost",
            port=u.port or 3306,
            user=unquote(u.username) if u.username else None,
            password=unquote(u.password) if u.password else "",
            db=(u.path or "/").lstrip("/") or None,
            autocommit=True,
            client_flag=CLIENT.FOUND_ROWS,  # rowcount = matched rows (CAS parity)
            # Suppress benign Note-level diagnostics (e.g. "table already exists" from
            # CREATE TABLE IF NOT EXISTS); real Warnings/Errors still surface.
            init_command="SET sql_notes=0",
            minsize=minsize,
            maxsize=maxsize,
        )
        return cls(pool)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_MyTransaction]:
        async with self._pool.acquire() as conn:
            await conn.begin()
            tx = _MyTransaction(self, conn)
            try:
                yield tx
            except BaseException:
                await conn.rollback()
                raise
            else:
                await conn.commit()

    async def fetchone(self, sql: str, params: Params = ()) -> Row | None:
        import aiomysql

        async with self._pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(self.dialect.translate(sql), _args(params))
            r = await cur.fetchone()
            return dict(r) if r is not None else None

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]:
        import aiomysql

        async with self._pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(self.dialect.translate(sql), _args(params))
            return [dict(r) for r in await cur.fetchall()]

    async def execute(self, sql: str, params: Params = ()) -> int:
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(self.dialect.translate(sql), _args(params))
            return cur.rowcount

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.close()
        await self._pool.wait_closed()


__all__ = ["MySQLDatabase"]
