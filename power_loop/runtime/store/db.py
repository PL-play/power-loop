"""The driver port: async ``Database`` / ``Transaction`` every backend implements.

The ``SessionStore`` facade is written ONCE against these two protocols plus a
:class:`~power_loop.runtime.store.dialect.Dialect`; each backend (SQLite/PostgreSQL/
MySQL) supplies a concrete ``Database``. SQL is authored once with ``?`` placeholders and
unqualified-but-prefixed table names; the backend translates the paramstyle (``?`` →
``%s`` / ``$1``) via its dialect. Reads outside a transaction are autocommit; every
multi-statement write runs inside ``async with db.transaction() as tx:`` so it is atomic
on any engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from power_loop.runtime.store.dialect import Dialect

Row = Mapping[str, Any]
Params = Sequence[Any]


@runtime_checkable
class Transaction(Protocol):
    """A single in-flight transaction on one connection. All statements run on the
    same connection, serialized; the context manager commits on success, rolls back
    on exception."""

    async def execute(self, sql: str, params: Params = ()) -> int:
        """Run a write/DDL statement; return affected-row count (DB-defined)."""
        ...

    async def fetchone(self, sql: str, params: Params = ()) -> Row | None: ...

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]: ...


@runtime_checkable
class Database(Protocol):
    """A backend connection/pool. Owns its dialect, hands out transactions, and serves
    autocommit reads. Construction (DSN/pool/PRAGMAs) is backend-specific."""

    dialect: Dialect

    def transaction(self) -> AbstractAsyncContextManager[Transaction]:
        """Begin a transaction; commit on clean exit, roll back on exception."""
        ...

    async def fetchone(self, sql: str, params: Params = ()) -> Row | None:
        """Autocommit read (may run on a pooled read connection)."""
        ...

    async def fetchall(self, sql: str, params: Params = ()) -> list[Row]: ...

    async def execute(self, sql: str, params: Params = ()) -> int:
        """Autocommit single-statement write (used for schema bootstrap)."""
        ...

    async def close(self) -> None: ...


__all__ = ["Database", "Transaction", "Row", "Params"]
