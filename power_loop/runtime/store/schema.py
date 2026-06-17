"""Portable schema bootstrap + version ladder (replaces SQLite's PRAGMA user_version).

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
from collections.abc import Awaitable, Callable
from enum import Enum

from power_loop.runtime.store.db import Database, Row, Transaction

logger = logging.getLogger(__name__)

#: Bump + append a migration step for ANY schema change.
CURRENT_SCHEMA_VERSION = 1

#: The store's data tables (besides ``{prefix}schema_migrations``) — used by VERIFY to
#: confirm the FULL schema is present, not just the version row. Keep in sync with
#: ``store._Tables`` / ``Dialect.ddl``.
_STORE_TABLES: tuple[str, ...] = (
    "sessions", "messages", "compactions", "usage_rounds", "session_state",
    "session_runtime_state", "shared_state", "background_tasks", "session_stats",
    "timers", "notes",
)


class SchemaPolicy(str, Enum):
    """How :func:`ensure_schema` provisions the store at open time."""

    AUTO_CREATE = "auto_create"  # create tables/indexes if missing, then stamp
    VERIFY = "verify"            # verify only; raise (with DDL) if missing/stale


def _coerce_policy(policy: SchemaPolicy | str | None, create_schema: bool | None) -> SchemaPolicy:
    """Resolve the effective policy. ``create_schema`` (bool) is the deprecated alias:
    True → AUTO_CREATE, False → VERIFY. An explicit ``policy`` wins over it."""
    if policy is not None:
        return SchemaPolicy(policy)
    if create_schema is not None:
        return SchemaPolicy.AUTO_CREATE if create_schema else SchemaPolicy.VERIFY
    return SchemaPolicy.AUTO_CREATE


def provisioning_ddl(db: Database, prefix: str) -> list[str]:
    """The COMPLETE provisioning script: the version table, every store table/index, and the
    version-row stamp — i.e. exactly what a privileged user must run so that a later
    ``VERIFY`` open succeeds. (``Dialect.ddl`` alone omits the migrations table + stamp.)

    The whole script is idempotent (``CREATE TABLE IF NOT EXISTS`` + an ``ON CONFLICT/IGNORE``
    version stamp) so infra automation (Terraform/Ansible/k8s init) can re-apply it safely."""
    vtable = f"{prefix}schema_migrations"
    if db.dialect.name == "mysql":
        stamp = f"INSERT IGNORE INTO {vtable} (id, version) VALUES (1, {CURRENT_SCHEMA_VERSION})"
    else:
        stamp = (
            f"INSERT INTO {vtable} (id, version) VALUES (1, {CURRENT_SCHEMA_VERSION}) "
            "ON CONFLICT(id) DO NOTHING"
        )
    return [
        f"CREATE TABLE IF NOT EXISTS {vtable} "
        f"(id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL)",
        *db.dialect.ddl(prefix),
        stamp,
    ]


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


async def _legacy_unprefixed_present(fetch: _FetchOne, dialect_name: str, prefix: str) -> bool:
    """True iff this looks like a PRE-2.0 power-loop SQLite database: legacy UNPREFIXED tables
    exist but the prefixed 2.0 schema does not. 1.x was SQLite-only and used no table prefix,
    so on upgrade the 2.0 ``pl_`` schema would open as an empty store while the old sessions
    sit (intact) under the unprefixed names. Only meaningful for SQLite with a non-empty prefix."""
    if not prefix or dialect_name != "sqlite":
        return False
    try:
        legacy = await fetch("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        prefixed = await fetch(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (f"{prefix}sessions",)
        )
    except Exception:  # pragma: no cover - detection is best-effort, never fatal
        return False
    return legacy is not None and prefixed is None


def _legacy_message(prefix: str) -> str:
    return (
        f"found a pre-2.0 power-loop database (legacy UNPREFIXED tables) but no '{prefix}' "
        f"schema at this path. power-loop 2.0 uses the '{prefix}' table prefix, so prior "
        "sessions are NOT visible under it — they are intact, just under the old unprefixed "
        "tables. To read the legacy data open the store with table_prefix='' (the 1.x layout); "
        f"otherwise a fresh '{prefix}' schema is used and the old sessions are ignored."
    )


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
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
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
    create_schema: bool | None = None,
) -> int:
    """Bring the store to :data:`CURRENT_SCHEMA_VERSION` per ``policy``; return the live version.

    ``policy`` is a :class:`SchemaPolicy`. ``create_schema`` (bool) is a deprecated alias
    (True→AUTO_CREATE, False→VERIFY); an explicit ``policy`` takes precedence.
    """
    eff = _coerce_policy(policy, create_schema)
    vtable = f"{prefix}schema_migrations"

    if eff is SchemaPolicy.VERIFY:
        # Probe the version table via the CATALOG (never via SELECT … FROM, which would
        # raise on a missing table and force us to swallow ALL exceptions — masking a
        # transient connection/permission failure as 'schema not initialized').
        if not await _table_exists(db, vtable):
            if await _legacy_unprefixed_present(db.fetchone, db.dialect.name, prefix):
                raise StoreSchemaError(_legacy_message(prefix), ddl=provisioning_ddl(db, prefix))
            raise StoreSchemaError(
                f"store schema not initialized ({vtable} missing). Open with "
                "schema=SchemaPolicy.AUTO_CREATE or provision the schema first.",
                ddl=provisioning_ddl(db, prefix),
            )
        # The version table exists: a failure reading it now is a REAL fault (connection /
        # permission), so let it propagate as itself rather than reporting 'not initialized'.
        row = await db.fetchone(f"SELECT version FROM {vtable} WHERE id=1")
        if row is None:
            raise StoreSchemaError(
                f"store schema not initialized ({vtable} has no version row). Open with "
                "schema=SchemaPolicy.AUTO_CREATE or provision the schema first.",
                ddl=provisioning_ddl(db, prefix),
            )
        version = int(row["version"])
        if version != CURRENT_SCHEMA_VERSION:
            raise StoreSchemaError(
                f"store schema version {version} != code {CURRENT_SCHEMA_VERSION}; "
                "run an upgrade (auto-create is disabled).",
                ddl=provisioning_ddl(db, prefix),
            )
        # The version row can be present + current while a DATA table was dropped (partial
        # restore, a manual/DBA drop, a half-applied provisioning script). Probe each so
        # VERIFY is a real pre-flight instead of passing then crashing on the first write.
        missing = [
            f"{prefix}{name}" for name in _STORE_TABLES if not await _table_exists(db, f"{prefix}{name}")
        ]
        if missing:
            raise StoreSchemaError(
                f"store schema incomplete: version stamped v{version} but data table(s) "
                f"missing: {', '.join(missing)}.",
                ddl=provisioning_ddl(db, prefix),
            )
        return version

    # AUTO_CREATE
    async with db.transaction() as tx:
        # Serialize concurrent first-boot across processes so a race on CREATE TABLE doesn't
        # surface as a raw duplicate-object error / misleading 'permission' hint (PG xact
        # advisory lock auto-releases; MySQL named lock released in finally; SQLite no-op).
        await _acquire_provision_lock(tx, db, prefix)
        try:
            await tx.execute(
                f"CREATE TABLE IF NOT EXISTS {vtable} "
                f"(id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL)"
            )
            row = await tx.fetchone(f"SELECT version FROM {vtable} WHERE id=1")
            if row is None:
                # Pre-2.0 DB at this path? Warn loudly (the prefix change is accepted-breaking,
                # but silently fronting an empty store would strand the user's old sessions).
                # Then proceed to create the fresh prefixed schema.
                if await _legacy_unprefixed_present(tx.fetchone, db.dialect.name, prefix):
                    logger.warning("power-loop store: %s", _legacy_message(prefix))
                # Fresh store: create every table (only on first init — a no-op CREATE IF NOT
                # EXISTS otherwise, but it warns per-table on MySQL), then stamp. Wrap the DDL
                # so a permission failure surfaces the full provisioning script instead of a
                # raw driver error.
                try:
                    for stmt in db.dialect.ddl(prefix):
                        await tx.execute(stmt)
                except Exception as exc:
                    raise StoreSchemaError(
                        "failed to auto-create the store schema. If this is a permission "
                        "error, run the DDL below as a user with CREATE rights and reopen "
                        f"with schema=SchemaPolicy.VERIFY. Underlying error: {exc!r}",
                        ddl=provisioning_ddl(db, prefix),
                    ) from exc
                await tx.execute(_version_stamp_sql(db, vtable), (CURRENT_SCHEMA_VERSION,))
                return CURRENT_SCHEMA_VERSION
            version = int(row["version"])
            if version > CURRENT_SCHEMA_VERSION:
                raise StoreSchemaError(
                    f"store schema version {version} is newer than this power_loop build "
                    f"supports (max {CURRENT_SCHEMA_VERSION})"
                )
            # (No migration steps exist at v1; future steps run here, per dialect, then stamp.)
            return version
        finally:
            await _release_provision_lock(tx, db, prefix)


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SchemaPolicy",
    "StoreSchemaError",
    "ensure_schema",
    "provisioning_ddl",
]
