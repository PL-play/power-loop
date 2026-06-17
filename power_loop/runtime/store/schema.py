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

Concurrency note (MVP): AUTO_CREATE is idempotent (``CREATE TABLE IF NOT EXISTS`` + an
``ON CONFLICT/IGNORE`` version stamp) and self-heals across retries, but it is NOT atomic on
MySQL (DDL auto-commits there) and does not take a cross-process lock. Concurrent *first-boot*
of N app instances against a fresh server DB should provision out-of-band and open VERIFY; a
``pg_advisory_xact_lock`` / ``GET_LOCK`` guard for true concurrent first-boot is a documented
follow-up.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from enum import Enum

from power_loop.runtime.store.db import Database, Row

logger = logging.getLogger(__name__)

#: Bump + append a migration step for ANY schema change.
CURRENT_SCHEMA_VERSION = 1


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
    ``VERIFY`` open succeeds. (``Dialect.ddl`` alone omits the migrations table + stamp.)"""
    vtable = f"{prefix}schema_migrations"
    return [
        f"CREATE TABLE IF NOT EXISTS {vtable} "
        f"(id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL)",
        *db.dialect.ddl(prefix),
        f"INSERT INTO {vtable} (id, version) VALUES (1, {CURRENT_SCHEMA_VERSION})",
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
        try:
            row = await db.fetchone(f"SELECT version FROM {vtable} WHERE id=1")
        except Exception:
            row = None
        if row is None:
            if await _legacy_unprefixed_present(db.fetchone, db.dialect.name, prefix):
                raise StoreSchemaError(_legacy_message(prefix), ddl=provisioning_ddl(db, prefix))
            raise StoreSchemaError(
                f"store schema not initialized ({vtable} missing). Open with "
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
        return version

    # AUTO_CREATE
    async with db.transaction() as tx:
        await tx.execute(
            f"CREATE TABLE IF NOT EXISTS {vtable} "
            f"(id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL)"
        )
        row = await tx.fetchone(f"SELECT version FROM {vtable} WHERE id=1")
        if row is None:
            # Pre-2.0 DB at this path? Warn loudly (the prefix change is accepted-breaking, but
            # silently fronting an empty store would strand the user's old sessions). Then
            # proceed to create the fresh prefixed schema.
            if await _legacy_unprefixed_present(tx.fetchone, db.dialect.name, prefix):
                logger.warning("power-loop store: %s", _legacy_message(prefix))
            # Fresh store: create every table (only on first init — a no-op CREATE IF NOT
            # EXISTS otherwise, but it warns per-table on MySQL), then stamp. Wrap the DDL so
            # a permission failure surfaces the full provisioning script instead of a raw
            # driver error.
            try:
                for stmt in db.dialect.ddl(prefix):
                    await tx.execute(stmt)
            except Exception as exc:
                raise StoreSchemaError(
                    "failed to auto-create the store schema. If this is a permission error, "
                    "run the DDL below as a user with CREATE rights and reopen with "
                    f"schema=SchemaPolicy.VERIFY. Underlying error: {exc!r}",
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


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SchemaPolicy",
    "StoreSchemaError",
    "ensure_schema",
    "provisioning_ddl",
]
