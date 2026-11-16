"""SQL dialect differences for the schema-17 session store."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .session_native_ddl import session_native_ddl

if TYPE_CHECKING:
    from power_loop.runtime.store.db import Row, Transaction


@runtime_checkable
class Dialect(Protocol):
    name: str

    def translate(self, sql: str) -> str: ...
    def ddl(self, prefix: str) -> list[str]: ...
    def insert_ignore(self, table: str, cols: Sequence[str]) -> str: ...
    def upsert(
        self, table: str, key_cols: Sequence[str], val_cols: Sequence[str], *,
        add_cols: Sequence[str] = (), insert_only_cols: Sequence[str] = (),
    ) -> str: ...
    def upsert_many(
        self, table: str, key_cols: Sequence[str], val_cols: Sequence[str], *, rows: int,
        add_cols: Sequence[str] = (), insert_only_cols: Sequence[str] = (),
    ) -> str: ...
    async def lock_state(self, tx: Transaction, state_table: str, session_id: str) -> Row: ...


class SqliteDialect:
    name = "sqlite"

    def translate(self, sql):
        return sql

    def ddl(self, prefix):
        return session_native_ddl(self.name, prefix)

    def insert_ignore(self, table, cols):
        return _insert_conflict(table, cols)

    def upsert(self, table, key_cols, val_cols, *, add_cols=(), insert_only_cols=()):
        return _onconflict_upsert(table, key_cols, val_cols, add_cols, insert_only_cols)

    def upsert_many(self, table, key_cols, val_cols, *, rows, add_cols=(), insert_only_cols=()):
        return _onconflict_upsert_many(
            table, key_cols, val_cols, rows, add_cols, insert_only_cols)

    async def lock_state(self, tx, state_table, session_id):
        return await _locked_state(tx, state_table, session_id, "")


class PostgresDialect:
    name = "postgres"

    def translate(self, sql):
        out, n = [], 0
        for char in sql:
            if char == "?":
                n += 1
                out.append(f"${n}")
            else:
                out.append(char)
        return "".join(out)

    def ddl(self, prefix):
        return session_native_ddl(self.name, prefix)

    def insert_ignore(self, table, cols):
        return _insert_conflict(table, cols)

    def upsert(self, table, key_cols, val_cols, *, add_cols=(), insert_only_cols=()):
        return _onconflict_upsert(table, key_cols, val_cols, add_cols, insert_only_cols)

    def upsert_many(self, table, key_cols, val_cols, *, rows, add_cols=(), insert_only_cols=()):
        return _onconflict_upsert_many(
            table, key_cols, val_cols, rows, add_cols, insert_only_cols)

    async def lock_state(self, tx, state_table, session_id):
        return await _locked_state(tx, state_table, session_id, " FOR UPDATE")


class MySQLDialect:
    name = "mysql"

    def translate(self, sql):
        return sql.replace("%", "%%").replace("?", "%s")

    def ddl(self, prefix):
        return session_native_ddl(self.name, prefix)

    def insert_ignore(self, table, cols):
        return (
            f"INSERT IGNORE INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})"
        )

    def upsert(self, table, key_cols, val_cols, *, add_cols=(), insert_only_cols=()):
        return self.upsert_many(
            table, key_cols, val_cols, rows=1, add_cols=add_cols,
            insert_only_cols=insert_only_cols)

    def upsert_many(self, table, key_cols, val_cols, *, rows, add_cols=(), insert_only_cols=()):
        cols = [*key_cols, *val_cols, *add_cols, *insert_only_cols]
        placeholders = ",".join("(" + ",".join("?" * len(cols)) + ")" for _ in range(rows))
        sets = [f"{c}=new_row.{c}" for c in val_cols]
        sets += [f"{c}={table}.{c}+new_row.{c}" for c in add_cols]
        return (
            f"INSERT INTO {table} ({','.join(cols)}) VALUES {placeholders} AS new_row "
            f"ON DUPLICATE KEY UPDATE {','.join(sets)}"
        )

    async def lock_state(self, tx, state_table, session_id):
        return await _locked_state(tx, state_table, session_id, " FOR UPDATE")


async def _locked_state(tx, state_table, session_id, suffix):
    row = await tx.fetchone(
        f"SELECT * FROM {state_table} WHERE session_id=?" + suffix, (session_id,)
    )
    if row is None:
        raise ValueError(f"unknown session: {session_id}")
    return row


def _insert_conflict(table, cols):
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
        "ON CONFLICT DO NOTHING"
    )


def _onconflict_upsert(table, key_cols, val_cols, add_cols, insert_only_cols):
    return _onconflict_upsert_many(table, key_cols, val_cols, 1, add_cols, insert_only_cols)


def _onconflict_upsert_many(table, key_cols, val_cols, rows, add_cols, insert_only_cols):
    cols = [*key_cols, *val_cols, *add_cols, *insert_only_cols]
    placeholders = ",".join("(" + ",".join("?" * len(cols)) + ")" for _ in range(rows))
    sets = [f"{c}=excluded.{c}" for c in val_cols]
    sets += [f"{c}={table}.{c}+excluded.{c}" for c in add_cols]
    return (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES {placeholders} "
        f"ON CONFLICT ({','.join(key_cols)}) DO UPDATE SET {','.join(sets)}"
    )


__all__ = ["Dialect", "SqliteDialect", "PostgresDialect", "MySQLDialect"]
