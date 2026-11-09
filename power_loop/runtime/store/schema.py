"""Portable current-schema bootstrap (replaces SQLite's PRAGMA user_version).

Every backend stores its version in a single-row ``{prefix}schema_migrations`` table, so
fresh-stamp / verify / refuse-newer works identically on SQLite/PostgreSQL/MySQL. DDL is
rendered per-dialect (``Dialect.ddl(prefix)``).

Provisioning is governed by a :class:`SchemaPolicy`:

* ``AUTO_CREATE`` (default, zero-infra DX): probe the version table; if absent, create every
  table + index and stamp the version. If the DDL fails (e.g. the DB role lacks CREATE
  rights) raise :class:`StoreSchemaError` carrying the **full** provisioning script so an
  operator can run it as a privileged user.
* ``VERIFY``: probe only; raise :class:`StoreSchemaError` (with the DDL) if the schema is
  missing or its version differs. For roles with no DDL rights — provision out-of-band, then
  open with VERIFY.

Concurrency note: AUTO_CREATE is idempotent (``CREATE TABLE IF NOT EXISTS`` + an
``ON CONFLICT/IGNORE`` version stamp) and serializes concurrent *first-boot* across processes
with a cross-process provisioning lock (``pg_advisory_xact_lock`` on PostgreSQL, a named
``GET_LOCK`` on MySQL, a no-op on single-writer SQLite) so N app instances racing against a
fresh server DB don't surface a raw duplicate-object error. It is still NOT a single atomic
transaction on MySQL (DDL auto-commits there), but the lock makes the race converge.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Awaitable, Callable
from enum import Enum

from power_loop.runtime.store.db import Database, Row, Transaction

logger = logging.getLogger(__name__)

#: A table_prefix is concatenated into raw SQL identifiers (CREATE/SELECT/INSERT/…) with no
#: quoting, so it MUST be a safe bare identifier — empty (the 1.x layout) or a leading
#: letter/underscore then word chars. Validated at the entry points to keep the prefix from
#: ever reaching the SQL builders unchecked (injection / malformed-DDL on a tenant-derived value).
_TABLE_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


#: Cap on table_prefix length (store-dialect-3). The longest derived identifier is
#: ``{prefix}idx_background_tasks_session_status`` (~35 chars after the prefix); MySQL caps
#: identifiers at 64, so a long prefix would overflow index/table names there. 24 leaves margin.
_MAX_TABLE_PREFIX_LEN = 24


def validate_table_prefix(prefix: str) -> str:
    """Return ``prefix`` if it is a safe table-name prefix (empty or ``[A-Za-z_]\\w*``, ≤24 chars);
    raise :class:`ValueError` otherwise."""
    if prefix == "":
        return prefix
    if not isinstance(prefix, str) or not _TABLE_PREFIX_RE.match(prefix):
        raise ValueError(
            f"invalid table_prefix {prefix!r}: must be empty or match [A-Za-z_][A-Za-z0-9_]* "
            "(it is interpolated into SQL identifiers without quoting)"
        )
    if len(prefix) > _MAX_TABLE_PREFIX_LEN:
        raise ValueError(
            f"table_prefix {prefix!r} is too long ({len(prefix)} > {_MAX_TABLE_PREFIX_LEN}): the "
            "longest derived index name must stay within MySQL's 64-char identifier limit"
        )
    return prefix

# Only the current task structure can be opened. No runtime migration.
CURRENT_SCHEMA_VERSION = 16

#: The store's data tables (besides ``{prefix}schema_migrations``) — used by VERIFY to
#: confirm the FULL schema is present, not just the version row. Keep in sync with
#: ``store._Tables`` / ``Dialect.ddl``.
_STORE_TABLES: tuple[str, ...] = (
    "sessions", "messages", "compactions", "usage_observations", "session_state",
    "session_runtime_state", "shared_state", "background_tasks", "session_stats",
    "timers", "notes", "project_messages", "hook_events",
    "follow_up_queue", "context_cache", "context_attempts",
)
_STORE_VIEWS = ('usage_rounds',)








async def _table_exists_tx(tx: Transaction, dialect_name: str, table: str) -> bool:
    """:func:`_table_exists` on the OPEN migration transaction (a fresh ``db`` query would
    deadlock on SQLite's single connection)."""
    if dialect_name == "sqlite":
        row = await tx.fetchone(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?", (table,)
        )
        return row is not None
    if dialect_name == "postgres":
        row = await tx.fetchone("SELECT to_regclass(?) AS reg", (table,))
        return row is not None and row["reg"] is not None
    row = await tx.fetchone(
        "SELECT 1 AS present FROM information_schema.tables "
        "WHERE table_schema=DATABASE() AND table_name=?",
        (table,),
    )
    return row is not None






class SchemaPolicy(str, Enum):
    """How :func:`ensure_schema` provisions the store at open time."""

    AUTO_CREATE = "auto_create"  # create tables/indexes if missing, then stamp
    VERIFY = "verify"            # verify only; raise (with DDL) if missing/stale
    TASKS = "tasks"              # current task schema, fresh or already current only


def provisioning_ddl(db: Database, prefix: str) -> list[str]:
    """Complete current-schema script for an empty database, including identity."""
    import secrets

    from .context_cache import context_cache_ddl
    from .task_schema import TASK_PROTOCOL_VERSION, message_identity_ddl, task_tables_ddl
    vtable = prefix + "schema_migrations"
    statements = [f"CREATE TABLE IF NOT EXISTS {vtable} "
                  "(id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL)",
                  *db.dialect.ddl(prefix), *context_cache_ddl(db.dialect.name, prefix),
                  *task_tables_ddl(db.dialect.name, prefix)]
    output, indexes = [], set()
    for statement in statements:
        if statement.startswith(f"CREATE TABLE IF NOT EXISTS {prefix}messages ("):
            statement = message_identity_ddl(statement, db.dialect.name)
        match = re.match(r"CREATE (?:UNIQUE )?INDEX (?:IF NOT EXISTS )?(\w+)", statement)
        if match:
            if match[1] in indexes:
                continue
            indexes.add(match[1])
        if statement.startswith("CREATE TABLE"):
            indexes.update(re.findall(r"\b(?:INDEX|KEY)\s+(\w+)\s*\(", statement))
        output.append(statement)
    store_id = "store_" + secrets.token_hex(16)
    insert = "INSERT IGNORE" if db.dialect.name == "mysql" else "INSERT"
    conflict = "" if db.dialect.name == "mysql" else " ON CONFLICT(id) DO NOTHING"
    output.append(f"{insert} INTO {prefix}task_store (id,store_id,protocol_version) "
                  f"VALUES (1,'{store_id}',{TASK_PROTOCOL_VERSION}){conflict}")
    output.append(f"{insert} INTO {vtable} (id,version) VALUES (1,{CURRENT_SCHEMA_VERSION}){conflict}")
    return output


class StoreSchemaError(RuntimeError):
    """The store schema is missing, stale (and auto-create is off / failed), or newer than this
    build. Carries :attr:`ddl` — the full provisioning script — and prints it so an operator
    can create the schema by hand."""

    def __init__(self, message: str, *, ddl: list[str] | None = None) -> None:
        self.ddl = ddl or []
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if not self.ddl:
            return base
        script = ";\n".join(self.ddl) + ";"
        return (
            f"{base}\n\n"
            f"--- run this provisioning script as a user with DDL rights, then reopen "
            f"with schema=SchemaPolicy.VERIFY ---\n{script}"
        )


_FetchOne = Callable[..., Awaitable["Row | None"]]






def _version_stamp_sql(db: Database, vtable: str) -> str:
    """An idempotent 'insert the version row only if absent' (best-effort guard against a
    concurrent first-boot double-stamp; the full cross-process lock is a follow-up)."""
    if db.dialect.name == "mysql":
        return f"INSERT IGNORE INTO {vtable} (id, version) VALUES (1, ?)"
    return f"INSERT INTO {vtable} (id, version) VALUES (1, ?) ON CONFLICT(id) DO NOTHING"


async def _table_exists(db: Database, table: str) -> bool:
    """Catalog probe — True iff ``table`` exists. Unlike ``SELECT … FROM {table}`` this never
    raises 'no such table', so a real connection/permission failure surfaces as itself."""
    name = db.dialect.name
    if name == "sqlite":
        row = await db.fetchone(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?", (table,)
        )
        return row is not None
    if name == "postgres":
        row = await db.fetchone("SELECT to_regclass(?) AS reg", (table,))
        return row is not None and row["reg"] is not None
    if name == "mysql":
        row = await db.fetchone(
            "SELECT 1 AS present FROM information_schema.tables "
            "WHERE table_schema=DATABASE() AND table_name=?",
            (table,),
        )
        return row is not None
    return False  # pragma: no cover - unknown dialect


async def _column_exists(tx: Transaction, dialect_name: str, table: str, column: str) -> bool:
    """Catalog probe — True iff ``table`` has ``column``. Runs on the OPEN transaction ``tx``
    (NOT a fresh ``db`` query, which would deadlock waiting on the single SQLite connection the
    migration's transaction already holds). Makes ``ALTER … ADD COLUMN`` migration steps
    idempotent (no portable ``ADD COLUMN IF NOT EXISTS`` across SQLite/MySQL)."""
    if dialect_name == "sqlite":
        # table is the store's own prefixed name (validated prefix); PRAGMA can't be parameterized.
        rows = await tx.fetchall(f"PRAGMA table_info({table})")
        return any(r["name"] == column for r in rows)
    if dialect_name in ("postgres", "mysql"):
        # Scope to the CURRENT schema — otherwise a same-named table in ANOTHER schema (PG
        # search_path / multi-schema deployments) makes the probe return True for a column the
        # current-schema table lacks, so the ALTER … ADD COLUMN is skipped but the version is still
        # stamped → every subsequent append referencing that column crashes. Mirrors _table_exists
        # (PG to_regclass honors search_path; MySQL DATABASE()).
        scope = (
            "AND table_schema=current_schema() "
            if dialect_name == "postgres"
            else "AND table_schema=DATABASE() "
        )
        row = await tx.fetchone(
            "SELECT 1 AS present FROM information_schema.columns "
            f"WHERE table_name=? {scope}AND column_name=?",
            (table, column),
        )
        return row is not None
    return False  # pragma: no cover - unknown dialect


def _provision_lock_key(prefix: str) -> int:
    """Deterministic signed 63/64-bit int key for an advisory lock (stable across processes,
    unlike the salted builtin ``hash``)."""
    digest = hashlib.blake2b(f"power_loop:provision:{prefix}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


async def _acquire_provision_lock(tx: Transaction, db: Database, prefix: str) -> None:
    """Serialize concurrent first-boot AUTO_CREATE across processes so a race on
    ``CREATE TABLE`` doesn't surface as a raw duplicate-object error (PG) or a misleading
    'permission' hint. PG: transaction-scoped advisory lock (auto-released on commit/rollback).
    MySQL: a named ``GET_LOCK`` released in :func:`_release_provision_lock`. SQLite: a no-op —
    its single-writer model already serializes provisioning."""
    name = db.dialect.name
    if name == "postgres":
        await tx.execute("SELECT pg_advisory_xact_lock(?)", (_provision_lock_key(prefix),))
    elif name == "mysql":
        await tx.execute("SELECT GET_LOCK(?, 30)", (_mysql_lock_name(prefix),))


async def _release_provision_lock(tx: Transaction, db: Database, prefix: str) -> None:
    """Release the MySQL named lock (a session/connection-scoped lock that would otherwise
    persist on the pooled connection). PG/SQLite need no explicit release."""
    if db.dialect.name == "mysql":
        try:
            await tx.execute("SELECT RELEASE_LOCK(?)", (_mysql_lock_name(prefix),))
        except Exception:  # pragma: no cover - release must never mask provisioning result
            logger.warning("provision lock release failed", exc_info=True)


def _mysql_lock_name(prefix: str) -> str:
    # MySQL lock names cap at 64 chars; derive a short bounded name from the prefix hash.
    return f"plprov_{_provision_lock_key(prefix) & 0xFFFFFFFF:08x}"


async def ensure_schema(
    db: Database,
    prefix: str,
    *,
    policy: SchemaPolicy | str | None = None,
) -> int:
    """Create the current task structure or verify it; never open an older mode."""
    from .task_schema import ensure_task_schema
    eff = SchemaPolicy(policy) if policy is not None else SchemaPolicy.AUTO_CREATE
    return await ensure_task_schema(db, prefix, create=eff in {SchemaPolicy.AUTO_CREATE, SchemaPolicy.TASKS})


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SchemaPolicy",
    "StoreSchemaError",
    "ensure_schema",
    "provisioning_ddl",
]
