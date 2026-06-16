"""``open_store(dsn)`` — pick a backend by DSN scheme and return an open async store.

    sqlite:///path/to.db | sqlite://:memory: | :memory: | ./relative.db   → SQLite
    postgresql://user:pw@host:port/db | postgres://…                       → PostgreSQL
    mysql://user:pw@host:port/db                                           → MySQL

The SQLite backend is the zero-dependency default; postgres/mysql pull their driver from
the matching extra only when their scheme is used.
"""

from __future__ import annotations

from power_loop.runtime.store.db import Database
from power_loop.runtime.store.schema import SchemaPolicy, ensure_schema
from power_loop.runtime.store.store import (
    DEFAULT_MAX_SPAWN_DEPTH,
    DEFAULT_TABLE_PREFIX,
    SessionStore,
)


async def open_store(
    dsn: str = ":memory:",
    *,
    max_spawn_depth: int = DEFAULT_MAX_SPAWN_DEPTH,
    table_prefix: str = DEFAULT_TABLE_PREFIX,
    schema: SchemaPolicy | str | None = None,
    create_schema: bool | None = None,
) -> SessionStore:
    """Open an async store for ``dsn`` (scheme → backend), provisioned per ``schema``
    (:class:`SchemaPolicy`, default AUTO_CREATE). ``create_schema`` (bool) is a deprecated
    alias (True→AUTO_CREATE, False→VERIFY)."""
    db = await _make_database(dsn)
    await ensure_schema(db, table_prefix, policy=schema, create_schema=create_schema)
    return SessionStore(db, max_spawn_depth=max_spawn_depth, table_prefix=table_prefix)


async def _make_database(dsn: str) -> Database:
    if dsn.startswith(("postgresql://", "postgres://")):
        from power_loop.runtime.store.backends.postgres import PostgresDatabase

        return await PostgresDatabase.connect(dsn)
    if dsn.startswith("mysql://"):
        from power_loop.runtime.store.backends.mysql import MySQLDatabase

        return await MySQLDatabase.connect(dsn)
    # SQLite (default): ":memory:", a "sqlite://…" URL, or a bare filesystem path.
    from power_loop.runtime.store.backends.sqlite import SqliteDatabase

    path = dsn
    if dsn.startswith("sqlite://"):
        rest = dsn[len("sqlite://") :]
        # sqlite:///abs → /abs ; sqlite://:memory: → :memory: ; sqlite://rel → rel
        path = rest[1:] if rest.startswith("/") and not rest.startswith("//") else rest
        path = path or ":memory:"
    return SqliteDatabase.open(path)


__all__ = ["open_store"]
