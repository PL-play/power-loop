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

#: Bump + append a migration step for ANY schema change.
#: v2 (2026-06): adds the ``{prefix}project_messages`` table (send-context projection).
#: v3 (2026-06): adds the ``{prefix}hook_events`` table (ephemeral hook-augmentation audit log).
#: v4 (2026-06): widen MySQL free-text/JSON columns TEXT→LONGTEXT.
#: v5 (2026-06): adds ``cached_tokens`` (prompt cache-read tokens) to usage_rounds + session_stats.
#: v6 (2026-07): usage_rounds PK (session_id, round_index) → (session_id, send_index, round_index).
#:   round_index RESETS per send, so the old key silently overwrote earlier sends' per-round rows —
#:   a session's accounting kept only the LAST send's detail. Existing rows backfill send_index=0
#:   (their true send is unrecoverable); new rows carry the real send index.
#: v7 (2026-07): adds ``{prefix}session_leases`` + ``{prefix}follow_up_queue`` — cross-PROCESS
#:   session mutual exclusion. The in-process lock only serializes one interpreter; with several
#:   agent processes on one database, two of them could drive a session concurrently. The lease row
#:   is the shared arbiter, and the queue lets a process that loses the race hand its steering to
#:   the holder instead of starting a competing run.
CURRENT_SCHEMA_VERSION = 7

#: The store's data tables (besides ``{prefix}schema_migrations``) — used by VERIFY to
#: confirm the FULL schema is present, not just the version row. Keep in sync with
#: ``store._Tables`` / ``Dialect.ddl``.
_STORE_TABLES: tuple[str, ...] = (
    "sessions", "messages", "compactions", "usage_rounds", "session_state",
    "session_runtime_state", "shared_state", "background_tasks", "session_stats",
    "timers", "notes", "project_messages", "hook_events",
    "session_leases", "follow_up_queue",
)


def _send_index_column_type(dialect_name: str) -> str:
    return "INTEGER" if dialect_name == "sqlite" else "BIGINT"


async def _migration_steps(
    tx: Transaction, db: Database, prefix: str, *, from_version: int
) -> list[str]:
    """DDL to migrate an existing store from ``from_version`` to :data:`CURRENT_SCHEMA_VERSION`.
    CREATE statements are ``IF NOT EXISTS`` (idempotent); the ``ALTER … ADD COLUMN`` (no portable
    ``IF NOT EXISTS`` across SQLite/MySQL) is guarded by a catalog probe (on the open ``tx``) so a
    half-applied or re-run migration is safe.

    On SQLite/PostgreSQL this whole ladder runs inside one transaction (atomic — a failure rolls
    back). On MySQL each DDL auto-commits, so a mid-ladder failure leaves a half-applied schema
    with the version NOT bumped; because every step is idempotent, simply reopening with
    AUTO_CREATE once the cause (permissions/disk) is fixed completes it (see the migration-failure
    error in :func:`ensure_schema`, which surfaces these exact steps)."""
    steps: list[str] = []
    if from_version < 2:
        # v1 → v2: add the project_messages (send-context projection) table + index, and the
        # authoritative send_index column on messages (NULL on pre-existing rows).
        steps += db.dialect.project_messages_ddl(prefix)
        if not await _column_exists(tx, db.dialect.name, f"{prefix}messages", "send_index"):
            steps.append(
                f"ALTER TABLE {prefix}messages ADD COLUMN send_index "
                f"{_send_index_column_type(db.dialect.name)}"
            )
    if from_version < 3:
        # v2 → v3: add the hook_events (ephemeral hook-augmentation audit) table + index. A new
        # CREATE TABLE IF NOT EXISTS — no ALTER on the hot messages table, so no _column_exists
        # probe needed (the CREATE is itself idempotent).
        steps += db.dialect.hook_events_ddl(prefix)
    if from_version < 4:
        # v3 → v4: widen the MySQL free-text/JSON columns TEXT(64 KiB)→LONGTEXT so large LLM
        # content/tool output/system prompts no longer fail the write (H2). MySQL-only — SQLite/
        # Postgres TEXT is already unbounded, so this is a pure version bump there. MODIFY is
        # idempotent. (Fresh stores already provision LONGTEXT via Dialect.ddl.)
        if db.dialect.name == "mysql":
            steps += db.dialect.widen_text_columns_ddl(prefix)
    if from_version < 5:
        # v4 → v5: add cached_tokens (prompt cache-read tokens) to usage_rounds + session_stats.
        # ALTER … ADD COLUMN has no portable IF NOT EXISTS, so probe the catalog on the open tx.
        int_t = "INTEGER" if db.dialect.name == "sqlite" else "BIGINT"
        if not await _column_exists(tx, db.dialect.name, f"{prefix}usage_rounds", "cached_tokens"):
            steps.append(f"ALTER TABLE {prefix}usage_rounds ADD COLUMN cached_tokens {int_t}")
        if not await _column_exists(tx, db.dialect.name, f"{prefix}session_stats", "cached_tokens"):
            steps.append(
                f"ALTER TABLE {prefix}session_stats ADD COLUMN cached_tokens {int_t} NOT NULL DEFAULT 0"
            )
    if from_version < 6:
        # v5 → v6: usage_rounds PK gains send_index (round_index resets per send; the old
        # 2-column key overwrote prior sends' rows). SQLite can't alter a PK → rebuild the
        # table; PG/MySQL alter in place. The _column_exists probe makes a re-run safe on
        # MySQL's auto-committing DDL (a half-applied ladder resumes past the done steps).
        if not await _column_exists(tx, db.dialect.name, f"{prefix}usage_rounds", "send_index"):
            steps += _usage_rounds_v6_ddl(db.dialect.name, prefix)
    if from_version < 7:
        # v6 → v7: add the session_leases + follow_up_queue tables (cross-process session mutual
        # exclusion). Both are new CREATE TABLE IF NOT EXISTS — idempotent, no ALTER, so no
        # catalog probe is needed.
        steps += db.dialect.leases_ddl(prefix)
    return steps


def _usage_rounds_v6_ddl(dialect_name: str, prefix: str) -> list[str]:
    p = prefix
    if dialect_name == "sqlite":
        return [
            f"""CREATE TABLE {p}usage_rounds_v6 (
                session_id TEXT NOT NULL, send_index INTEGER NOT NULL DEFAULT 0,
                round_index INTEGER NOT NULL, prompt_tokens INTEGER,
                completion_tokens INTEGER, total_tokens INTEGER, cached_tokens INTEGER, model TEXT,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (session_id, send_index, round_index))""",
            f"INSERT INTO {p}usage_rounds_v6 (session_id, send_index, round_index, prompt_tokens,"
            f" completion_tokens, total_tokens, cached_tokens, model, created_at)"
            f" SELECT session_id, 0, round_index, prompt_tokens, completion_tokens, total_tokens,"
            f" cached_tokens, model, created_at FROM {p}usage_rounds",
            f"DROP TABLE {p}usage_rounds",
            f"ALTER TABLE {p}usage_rounds_v6 RENAME TO {p}usage_rounds",
        ]
    if dialect_name == "postgres":
        return [
            f"ALTER TABLE {p}usage_rounds ADD COLUMN send_index BIGINT NOT NULL DEFAULT 0",
            f"ALTER TABLE {p}usage_rounds DROP CONSTRAINT {p}usage_rounds_pkey",
            f"ALTER TABLE {p}usage_rounds ADD PRIMARY KEY (session_id, send_index, round_index)",
        ]
    # mysql: one statement so the auto-committing DDL can't leave the PK half-swapped.
    return [
        f"ALTER TABLE {p}usage_rounds ADD COLUMN send_index BIGINT NOT NULL DEFAULT 0, "
        f"DROP PRIMARY KEY, ADD PRIMARY KEY (session_id, send_index, round_index)"
    ]


def migration_ddl_for_display(db: Database, prefix: str, *, from_version: int) -> list[str]:
    """The migration DDL as a human-runnable script for an error message — NO catalog probe (the
    ``ALTER`` is shown unconditionally so an operator running it by hand sees every step). Unlike
    :func:`provisioning_ddl` (which only CREATEs fresh tables and would silently SKIP the
    ``ALTER … ADD COLUMN`` on an existing table), this is what actually completes the migration."""
    steps: list[str] = []
    if from_version < 2:
        steps += db.dialect.project_messages_ddl(prefix)
        steps.append(
            f"ALTER TABLE {prefix}messages ADD COLUMN send_index "
            f"{_send_index_column_type(db.dialect.name)}"
        )
    if from_version < 3:
        steps += db.dialect.hook_events_ddl(prefix)
    if from_version < 4 and db.dialect.name == "mysql":
        steps += db.dialect.widen_text_columns_ddl(prefix)
    if from_version < 5:
        int_t = "INTEGER" if db.dialect.name == "sqlite" else "BIGINT"
        steps.append(f"ALTER TABLE {prefix}usage_rounds ADD COLUMN cached_tokens {int_t}")
        steps.append(
            f"ALTER TABLE {prefix}session_stats ADD COLUMN cached_tokens {int_t} NOT NULL DEFAULT 0"
        )
    if from_version < 6:
        steps += _usage_rounds_v6_ddl(db.dialect.name, prefix)
    return steps


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
    create_schema: bool | None = None,
) -> int:
    """Bring the store to :data:`CURRENT_SCHEMA_VERSION` per ``policy``; return the live version.

    ``policy`` is a :class:`SchemaPolicy`. ``create_schema`` (bool) is a deprecated alias
    (True→AUTO_CREATE, False→VERIFY); an explicit ``policy`` takes precedence.
    """
    eff = _coerce_policy(policy, create_schema)
    validate_table_prefix(prefix)
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
            if version < CURRENT_SCHEMA_VERSION:
                # Run the idempotent migration ladder under the provision lock, then bump the
                # stamp. Wrap so a permission failure surfaces the full provisioning script.
                try:
                    for stmt in await _migration_steps(tx, db, prefix, from_version=version):
                        await tx.execute(stmt)
                except Exception as exc:
                    # On MySQL DDL auto-commits, so the failed migration is half-applied and the
                    # version stays at the old value; the steps are idempotent, so reopening with
                    # AUTO_CREATE once the cause is fixed finishes it. Surface the MIGRATION steps
                    # (incl. the ALTER) — NOT provisioning_ddl, which would skip the ADD COLUMN on
                    # the existing messages table and so never actually complete the migration.
                    mysql_note = (
                        " (on MySQL the migration is non-atomic — DDL auto-commits — so this may "
                        "be half-applied; the steps are idempotent, so once the cause is fixed "
                        "simply reopen with AUTO_CREATE to finish.)"
                        if db.dialect.name == "mysql" else ""
                    )
                    raise StoreSchemaError(
                        f"failed to migrate the store schema from v{version} to "
                        f"v{CURRENT_SCHEMA_VERSION}.{mysql_note} If this is a permission error, run "
                        "the migration step(s) below as a user with the needed rights, then reopen "
                        "with schema=SchemaPolicy.AUTO_CREATE (it re-runs the idempotent steps and "
                        f"stamps the version). Underlying error: {exc!r}",
                        ddl=migration_ddl_for_display(db, prefix, from_version=version),
                    ) from exc
                await tx.execute(f"UPDATE {vtable} SET version=? WHERE id=1", (CURRENT_SCHEMA_VERSION,))
                return CURRENT_SCHEMA_VERSION
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
