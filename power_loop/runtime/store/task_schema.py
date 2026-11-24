"""Schema 18 / protocol 43 verification. Older stores are never migrated."""
from __future__ import annotations

import re

from .db import Database
from .schema import (
    _STORE_TABLES,
    StoreSchemaError,
    _acquire_provision_lock,
    _release_provision_lock,
    _table_exists_tx,
    provisioning_ddl,
    validate_table_prefix,
)
from .session_native_ddl import INDEXES

TASK_SCHEMA_VERSION = 18
TASK_PROTOCOL_VERSION = 43
# Retained as an empty tuple for host test compatibility until agent/tests drops top-level imports.
TASK_TABLES: tuple[str, ...] = ()


async def verify_task_schema(db: Database, prefix: str) -> None:
    async with db.transaction() as tx:
        missing = [
            table for table in ("schema_migrations", *_STORE_TABLES)
            if not await _table_exists_tx(tx, db.dialect.name, prefix + table)
        ]
        if missing:
            raise StoreSchemaError(f"session schema incomplete: {missing}")
        row = await tx.fetchone(
            f"SELECT version,store_id,protocol_version FROM {prefix}schema_migrations WHERE id=1"
        )
        if row is None or int(row["version"]) != TASK_SCHEMA_VERSION:
            raise StoreSchemaError("existing store has an unsupported schema version")
        if not row["store_id"] or int(row["protocol_version"]) != TASK_PROTOCOL_VERSION:
            raise StoreSchemaError("session store identity/protocol missing or unsupported")
        await _verify_indexes(tx, db, prefix)


async def _verify_indexes(tx, db, prefix):
    expected = {prefix + name: (table, columns) for name, (table, columns) in INDEXES.items()}
    actual = {}
    if db.dialect.name == "sqlite":
        for name, (table, _columns) in expected.items():
            if not await tx.fetchone(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=? AND tbl_name=?",
                (name, prefix + table),
            ):
                continue
            rows = await tx.fetchall(f"PRAGMA index_info({name})")
            actual[name] = tuple(row["name"] for row in rows)
    elif db.dialect.name == "postgres":
        names = tuple(expected)
        rows = await tx.fetchall(
            "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname=current_schema() "
            f"AND indexname IN ({','.join('?' for _ in names)})", names,
        )
        for row in rows:
            name = row["indexname"]
            if row["tablename"] != prefix + expected[name][0]:
                continue
            matched = re.search(r"\(([^()]*)\)", row["indexdef"])
            actual[name] = (
                tuple(value.strip().strip('"') for value in matched[1].split(","))
                if matched else ()
            )
    else:
        names = tuple(expected)
        rows = await tx.fetchall(
            "SELECT table_name AS table_name,index_name AS index_name,column_name AS column_name "
            "FROM information_schema.statistics "
            "WHERE table_schema=DATABASE() "
            f"AND index_name IN ({','.join('?' for _ in names)}) "
            "ORDER BY table_name,index_name,seq_in_index", names,
        )
        for row in rows:
            name = row["index_name"]
            if row["table_name"] != prefix + expected[name][0]:
                continue
            actual[name] = (*actual.get(name, ()), row["column_name"])
    for name, (_table, columns) in expected.items():
        if actual.get(name) != columns:
            raise StoreSchemaError(f"session index missing or incompatible: {name}")


async def ensure_task_schema(db: Database, prefix: str, *, create: bool) -> int:
    validate_table_prefix(prefix)
    vtable = prefix + "schema_migrations"
    async with db.transaction() as tx:
        await _acquire_provision_lock(tx, db, prefix)
        try:
            exists = await _table_exists_tx(tx, db.dialect.name, vtable)
            row = await tx.fetchone(f"SELECT version FROM {vtable} WHERE id=1") if exists else None
            if row is not None:
                if int(row["version"]) != TASK_SCHEMA_VERSION:
                    raise StoreSchemaError(
                        f"existing store is incompatible; create a fresh schema v{TASK_SCHEMA_VERSION} store"
                    )
            else:
                if not create:
                    raise StoreSchemaError("session schema not initialized", ddl=provisioning_ddl(db, prefix))
                if await _table_exists_tx(tx, db.dialect.name, prefix + "sessions"):
                    raise StoreSchemaError("unstamped store requires inspection")
                statements = provisioning_ddl(db, prefix)
                try:
                    for statement in statements:
                        await tx.execute(statement)
                except Exception as exc:
                    raise StoreSchemaError(
                        "schema provisioning failed; check DDL permissions and use an empty store",
                        ddl=statements,
                    ) from exc
        finally:
            await _release_provision_lock(tx, db, prefix)
    await verify_task_schema(db, prefix)
    return TASK_SCHEMA_VERSION
