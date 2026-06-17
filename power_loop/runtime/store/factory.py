"""``open_store(dsn)`` — pick a backend by DSN scheme and return an open async store.

    ./relative.db | :memory: | sqlite:///rel.db | sqlite:////abs.db        → SQLite
    postgresql://user:pw@host:port/db | postgres://…                       → PostgreSQL
    mysql://user:pw@host:port/db                                           → MySQL

SQLite URLs follow the SQLAlchemy convention: ``sqlite:///rel.db`` is RELATIVE,
``sqlite:////abs.db`` is ABSOLUTE (four slashes), ``sqlite://`` is in-memory. A bare
filesystem path is taken verbatim. A SQLAlchemy-style ``+driver`` suffix
(``postgresql+psycopg://``) is accepted and ignored. An unrecognized scheme raises rather
than silently creating a SQLite file named after the DSN.

The SQLite backend is the zero-dependency default; postgres/mysql pull their driver from
the matching extra only when their scheme is used.
"""

from __future__ import annotations

import re

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


_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9.+-]*)://")


def _split_scheme(dsn: str) -> tuple[str | None, str]:
    """Return ``(base_scheme, normalized_dsn)``.

    ``base_scheme`` is the URL scheme lowercased with any SQLAlchemy-style ``+driver``
    suffix stripped (``postgresql+psycopg`` → ``postgresql``); ``normalized_dsn`` rewrites
    the scheme to that base so the underlying driver (asyncpg / aiomysql) — which does not
    understand the ``+driver`` form — accepts it. A bare filesystem path or ``:memory:``
    (no ``scheme://``) yields ``(None, dsn)``."""
    m = _SCHEME_RE.match(dsn)
    if m is None:
        return None, dsn
    base = m.group(1).lower().split("+", 1)[0]
    return base, base + dsn[m.end(1):]


def _sqlite_path(dsn: str) -> str:
    """Resolve a SQLite DSN to a filesystem path / ``:memory:``.

    Follows the SQLAlchemy convention so an absolute path stays absolute:
    ``sqlite:///rel.db`` → ``rel.db`` (relative), ``sqlite:////abs.db`` → ``/abs.db``
    (absolute), ``sqlite://`` / ``sqlite://:memory:`` → in-memory, and a bare path (no
    scheme) is returned verbatim."""
    scheme, _ = _split_scheme(dsn)
    if scheme is None:
        return dsn  # bare filesystem path or ':memory:'
    rest = dsn.split("://", 1)[1] if "://" in dsn else ""
    path = rest[1:] if rest.startswith("/") else rest
    return path or ":memory:"


async def _make_database(dsn: str) -> Database:
    scheme, normalized = _split_scheme(dsn)
    if scheme in ("postgresql", "postgres"):
        from power_loop.runtime.store.backends.postgres import PostgresDatabase

        return await PostgresDatabase.connect(normalized)
    if scheme == "mysql":
        from power_loop.runtime.store.backends.mysql import MySQLDatabase

        return await MySQLDatabase.connect(normalized)
    if scheme in (None, "sqlite", "sqlite3"):
        # ":memory:", a "sqlite://…" URL, or a bare filesystem path.
        from power_loop.runtime.store.backends.sqlite import SqliteDatabase

        return SqliteDatabase.open(_sqlite_path(dsn))
    # Unknown scheme: do NOT silently fall through to a SQLite file named after the DSN —
    # that masks a typo/misconfig and writes to the wrong place.
    raise ValueError(
        f"unsupported store DSN scheme {scheme!r} in {dsn!r}; expected sqlite:// (or a "
        "bare filesystem path / ':memory:'), postgresql://, or mysql:// — a SQLAlchemy-style "
        "'+driver' suffix (e.g. postgresql+psycopg://) is accepted and ignored."
    )


__all__ = ["open_store"]
