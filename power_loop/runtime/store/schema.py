"""Portable schema bootstrap + version ladder (replaces SQLite's PRAGMA user_version).

Every backend stores its version in a single-row ``{prefix}schema_migrations`` table, so
fresh-stamp / upgrade / refuse-newer works identically on SQLite/PostgreSQL/MySQL. DDL is
rendered per-dialect (``Dialect.ddl(prefix)``). Auto-create is the default (power-loop's
zero-infra DX); ``create_schema=False`` verifies the version and raises ``StoreSchemaError``
if the schema is missing or stale instead of touching DDL.
"""

from __future__ import annotations

from power_loop.runtime.store.db import Database

#: Bump + append a migration step for ANY schema change.
CURRENT_SCHEMA_VERSION = 1


class StoreSchemaError(RuntimeError):
    """The on-disk/server schema is missing, older than this build (and auto-create is
    off), or newer than this build supports."""


async def ensure_schema(db: Database, prefix: str, *, create_schema: bool = True) -> int:
    """Bring the store to :data:`CURRENT_SCHEMA_VERSION`; return the live version.

    ``create_schema`` True (default): create tables IF NOT EXISTS, stamp/upgrade the
    version. False: verify only — raise :class:`StoreSchemaError` if uninitialized or
    stale. A newer-than-code version is always refused.
    """
    vtable = f"{prefix}schema_migrations"

    if not create_schema:
        try:
            row = await db.fetchone(f"SELECT version FROM {vtable} WHERE id=1")
        except Exception:
            row = None
        if row is None:
            raise StoreSchemaError(
                f"store schema not initialized ({vtable} missing). Open with "
                "create_schema=True or run schema provisioning first."
            )
        version = int(row["version"])
        if version != CURRENT_SCHEMA_VERSION:
            raise StoreSchemaError(
                f"store schema version {version} != code {CURRENT_SCHEMA_VERSION}; "
                "run an upgrade (auto-create is disabled)."
            )
        return version

    async with db.transaction() as tx:
        # NOTE: on a multi-writer server backend this transaction must be guarded by a
        # DB advisory lock (pg_advisory_xact_lock / GET_LOCK) so two processes can't race
        # the version bump; SQLite's single writer makes it safe as-is.
        await tx.execute(
            f"CREATE TABLE IF NOT EXISTS {vtable} "
            f"(id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL)"
        )
        for stmt in db.dialect.ddl(prefix):
            await tx.execute(stmt)
        row = await tx.fetchone(f"SELECT version FROM {vtable} WHERE id=1")
        if row is None:
            await tx.execute(
                f"INSERT INTO {vtable} (id, version) VALUES (1, ?)", (CURRENT_SCHEMA_VERSION,)
            )
            return CURRENT_SCHEMA_VERSION
        version = int(row["version"])
        if version > CURRENT_SCHEMA_VERSION:
            raise StoreSchemaError(
                f"store schema version {version} is newer than this power_loop build "
                f"supports (max {CURRENT_SCHEMA_VERSION})"
            )
        # (No migration steps exist at v1; future steps run here, per dialect, then stamp.)
        return version


__all__ = ["CURRENT_SCHEMA_VERSION", "StoreSchemaError", "ensure_schema"]
