"""Schema-policy bootstrap for fresh session-native stores."""
from __future__ import annotations

import hashlib
import re
from enum import Enum

from power_loop.runtime.store.db import Database, Transaction
from power_loop.runtime.store.session_native_ddl import CONTRACT_TABLES

CURRENT_SCHEMA_VERSION = 18
_STORE_TABLES = CONTRACT_TABLES
_STORE_VIEWS: tuple[str, ...] = ()
_TABLE_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_TABLE_PREFIX_LEN = 24


def validate_table_prefix(prefix: str) -> str:
    if prefix == "":
        return prefix
    if not isinstance(prefix, str) or not _TABLE_PREFIX_RE.match(prefix):
        raise ValueError(f"invalid table_prefix {prefix!r}")
    if len(prefix) > _MAX_TABLE_PREFIX_LEN:
        raise ValueError(f"table_prefix {prefix!r} is too long")
    return prefix


class SchemaPolicy(str, Enum):
    AUTO_CREATE = "auto_create"
    VERIFY = "verify"
    TASKS = "tasks"


class StoreSchemaError(RuntimeError):
    def __init__(self, message: str, *, ddl: list[str] | None = None) -> None:
        self.ddl = ddl or []
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if not self.ddl:
            return base
        return base + "\n\n--- provisioning script ---\n" + ";\n".join(self.ddl) + ";"


def provisioning_ddl(db: Database, prefix: str) -> list[str]:
    import secrets

    from .task_schema import TASK_PROTOCOL_VERSION

    vtable = prefix + "schema_migrations"
    sid = "VARCHAR(64)" if db.dialect.name == "mysql" else "TEXT"
    opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if db.dialect.name == "mysql" else ""
    insert = "INSERT IGNORE" if db.dialect.name == "mysql" else "INSERT"
    conflict = "" if db.dialect.name == "mysql" else " ON CONFLICT(id) DO NOTHING"
    store_id = "store_" + secrets.token_hex(16)
    return [
        f"CREATE TABLE IF NOT EXISTS {vtable} (id INTEGER PRIMARY KEY CHECK (id=1), "
        f"version INTEGER NOT NULL, store_id {sid} NOT NULL, "
        f"protocol_version INTEGER NOT NULL){opts}",
        *db.dialect.ddl(prefix),
        f"{insert} INTO {vtable} (id,version,store_id,protocol_version) "
        f"VALUES (1,{CURRENT_SCHEMA_VERSION},'{store_id}',{TASK_PROTOCOL_VERSION}){conflict}",
    ]


async def _table_exists_tx(tx: Transaction, dialect_name: str, table: str) -> bool:
    if dialect_name == "sqlite":
        row = await tx.fetchone(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?", (table,)
        )
    elif dialect_name == "postgres":
        row = await tx.fetchone("SELECT to_regclass(?) AS reg", (table,))
        return row is not None and row["reg"] is not None
    else:
        row = await tx.fetchone(
            "SELECT 1 AS present FROM information_schema.tables "
            "WHERE table_schema=DATABASE() AND table_name=?", (table,)
        )
    return row is not None


def _provision_lock_key(prefix: str) -> int:
    digest = hashlib.blake2b(f"power_loop:provision:{prefix}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


async def _acquire_provision_lock(tx: Transaction, db: Database, prefix: str) -> None:
    if db.dialect.name == "postgres":
        await tx.execute("SELECT pg_advisory_xact_lock(?)", (_provision_lock_key(prefix),))
    elif db.dialect.name == "mysql":
        await tx.execute("SELECT GET_LOCK(?, 30)", (_mysql_lock_name(prefix),))


async def _release_provision_lock(tx: Transaction, db: Database, prefix: str) -> None:
    if db.dialect.name == "mysql":
        await tx.execute("SELECT RELEASE_LOCK(?)", (_mysql_lock_name(prefix),))


def _mysql_lock_name(prefix: str) -> str:
    return f"plprov_{_provision_lock_key(prefix) & 0xFFFFFFFF:08x}"


async def ensure_schema(db: Database, prefix: str, *, policy: SchemaPolicy | str | None = None) -> int:
    from .task_schema import ensure_task_schema

    eff = SchemaPolicy(policy) if policy is not None else SchemaPolicy.AUTO_CREATE
    return await ensure_task_schema(
        db, prefix, create=eff in {SchemaPolicy.AUTO_CREATE, SchemaPolicy.TASKS}
    )


__all__ = [
    "CURRENT_SCHEMA_VERSION", "SchemaPolicy", "StoreSchemaError", "ensure_schema",
    "provisioning_ddl",
]
