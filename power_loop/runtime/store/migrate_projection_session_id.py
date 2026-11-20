"""Offline schema-17 migration for the projection session/id index only."""
from __future__ import annotations

import argparse
import asyncio
import json
import re

from .db import Database, Transaction
from .schema import StoreSchemaError, validate_table_prefix
from .task_schema import verify_task_schema

INDEX = "idx_projection_session_id"
COLUMNS = ("session_id", "id")


async def _index_columns(tx: Transaction, db: Database, prefix: str):
    name = prefix + INDEX
    if db.dialect.name == "sqlite":
        row = await tx.fetchone(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=? AND tbl_name=?",
            (name, prefix + "project_messages"),
        )
        if row is None:
            return None
        rows = await tx.fetchall(f"PRAGMA index_info({name})")
        return tuple(row["name"] for row in rows)
    if db.dialect.name == "postgres":
        row = await tx.fetchone(
            "SELECT tablename,indexdef FROM pg_indexes WHERE schemaname=current_schema() "
            "AND indexname=?", (name,),
        )
        if row is None:
            return None
        if row["tablename"] != prefix + "project_messages":
            return ()
        matched = re.search(r"\(([^()]*)\)", row["indexdef"])
        return tuple(value.strip().strip('"') for value in matched[1].split(",")) if matched else ()
    rows = await tx.fetchall(
        "SELECT table_name AS table_name,column_name AS column_name FROM information_schema.statistics "
        "WHERE table_schema=DATABASE() AND index_name=? ORDER BY seq_in_index", (name,),
    )
    if not rows:
        return None
    if any(row["table_name"] != prefix + "project_messages" for row in rows):
        return ()
    return tuple(row["column_name"] for row in rows)


async def add_projection_session_id_index(db: Database, prefix: str) -> dict:
    validate_table_prefix(prefix)
    name = prefix + INDEX
    async with db.transaction() as tx:
        columns = await _index_columns(tx, db, prefix)
        if columns == COLUMNS:
            return {"index": name, "changed": False}
        if columns is not None:
            raise StoreSchemaError(f"session index missing or incompatible: {name}")
        await tx.execute(
            f"CREATE INDEX {name} ON {prefix}project_messages({','.join(COLUMNS)})"
        )
    await verify_task_schema(db, prefix)
    return {"index": name, "changed": True}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    from .factory import _make_database

    db = await _make_database(args.dsn)
    try:
        result = await add_projection_session_id_index(db, args.prefix)
        print(json.dumps(result))
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
