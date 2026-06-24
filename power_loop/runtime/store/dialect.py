"""SQL dialect seam: the small set of things that genuinely differ per backend.

Everything else (the ~50 store methods' logic) is written once against the
:class:`~power_loop.runtime.store.db.Database`. A ``Dialect`` supplies: placeholder
translation (``?`` → backend paramstyle), per-dialect DDL (prefixed table names), an
``upsert`` renderer, and the atomic per-session ``allocate_seq`` primitive — the one
operation whose CORRECTNESS depends on the engine's concurrency model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from power_loop.runtime.store.db import Row, Transaction


@runtime_checkable
class Dialect(Protocol):
    name: str

    def translate(self, sql: str) -> str:
        """Rewrite ``?`` placeholders to the backend's paramstyle."""
        ...

    def ddl(self, prefix: str) -> list[str]:
        """The CREATE TABLE / CREATE INDEX statements for this backend, table and
        index names carrying ``prefix`` (e.g. ``pl_``). One statement per item."""
        ...

    def project_messages_ddl(self, prefix: str) -> list[str]:
        """DDL for the ``project_messages`` table + index, split out so the v1→v2
        schema migration can add just this table to an existing store. Included in
        :meth:`ddl` for fresh provisioning. Idempotent (CREATE … IF NOT EXISTS)."""
        ...

    def hook_events_ddl(self, prefix: str) -> list[str]:
        """DDL for the ``hook_events`` table + index (audit log of ephemeral hook
        augmentations, e.g. LLM_BEFORE memory injection), split out so the v2→v3
        migration can add just this table. Included in :meth:`ddl` for fresh
        provisioning. Idempotent (CREATE … IF NOT EXISTS)."""
        ...

    def upsert(
        self,
        table: str,
        key_cols: Sequence[str],
        val_cols: Sequence[str],
        *,
        add_cols: Sequence[str] = (),
        insert_only_cols: Sequence[str] = (),
    ) -> str:
        """Render an idempotent INSERT-or-UPDATE. ``val_cols`` are overwritten;
        ``add_cols`` are accumulated (``col = col + new``); ``insert_only_cols`` are set
        on INSERT but PRESERVED on conflict (e.g. ``first_send_at``). ``?`` placeholders
        for ``key_cols + val_cols + add_cols + insert_only_cols`` in that order."""
        ...

    async def lock_state(self, tx: Transaction, state_table: str, session_id: str) -> Row:
        """Lock the ``session_state`` row for ``session_id`` for the rest of the
        transaction and return it, so the caller can atomically read+bump its counters
        (``next_seq`` / ``last_compact_seq``). MUST serialize concurrent writers on a
        server DB (``SELECT … FOR UPDATE``); on SQLite the backend already serializes
        writers, so a plain SELECT suffices. Raises ``ValueError`` for an unknown
        session."""
        ...


# ── SQLite ────────────────────────────────────────────────────────────────────


class SqliteDialect:
    name = "sqlite"

    def translate(self, sql: str) -> str:
        return sql  # qmark is native

    def ddl(self, prefix: str) -> list[str]:
        p = prefix
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}sessions (
                session_id TEXT PRIMARY KEY, created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL, system_prompt TEXT, model TEXT,
                config_json TEXT, status TEXT NOT NULL DEFAULT 'active',
                kind TEXT NOT NULL DEFAULT 'root', parent_session_id TEXT,
                spawn_tool_call_id TEXT, spawn_depth INTEGER NOT NULL DEFAULT 0,
                lifecycle TEXT NOT NULL DEFAULT 'ephemeral', metadata_json TEXT)""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_sessions_parent ON {p}sessions(parent_session_id)",
            f"""CREATE TABLE IF NOT EXISTS {p}messages (
                session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL,
                name TEXT, content TEXT, tool_calls_json TEXT, tool_call_id TEXT,
                round_index INTEGER, state TEXT NOT NULL DEFAULT 'active', meta_json TEXT,
                send_index INTEGER, created_at INTEGER NOT NULL, PRIMARY KEY (session_id, seq))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_messages_session_state "
            f"ON {p}messages(session_id, state, seq)",
            f"""CREATE TABLE IF NOT EXISTS {p}compactions (
                session_id TEXT NOT NULL, compact_seq INTEGER NOT NULL, note_seq INTEGER NOT NULL,
                from_seq INTEGER NOT NULL, to_seq INTEGER NOT NULL, before_tokens INTEGER,
                after_tokens INTEGER, round_index INTEGER, created_at INTEGER NOT NULL,
                PRIMARY KEY (session_id, compact_seq))""",
            f"""CREATE TABLE IF NOT EXISTS {p}usage_rounds (
                session_id TEXT NOT NULL, round_index INTEGER NOT NULL, prompt_tokens INTEGER,
                completion_tokens INTEGER, total_tokens INTEGER, model TEXT,
                created_at INTEGER NOT NULL, PRIMARY KEY (session_id, round_index))""",
            f"""CREATE TABLE IF NOT EXISTS {p}session_state (
                session_id TEXT PRIMARY KEY, next_seq INTEGER NOT NULL DEFAULT 1,
                round_index INTEGER NOT NULL DEFAULT 0, last_compact_seq INTEGER NOT NULL DEFAULT 0,
                pending_json TEXT)""",
            f"""CREATE TABLE IF NOT EXISTS {p}session_runtime_state (
                session_id TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT,
                updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, state_key))""",
            f"""CREATE TABLE IF NOT EXISTS {p}shared_state (
                owner TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT,
                updated_at INTEGER NOT NULL, PRIMARY KEY (owner, state_key))""",
            f"""CREATE TABLE IF NOT EXISTS {p}background_tasks (
                session_id TEXT NOT NULL, task_id TEXT NOT NULL, command TEXT NOT NULL,
                status TEXT NOT NULL, return_code INTEGER, output_tail TEXT, output_path TEXT,
                last_seen_at INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, task_id))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_background_tasks_session_status "
            f"ON {p}background_tasks(session_id, status, updated_at)",
            f"""CREATE TABLE IF NOT EXISTS {p}session_stats (
                session_id TEXT PRIMARY KEY, sends INTEGER NOT NULL DEFAULT 0,
                rounds INTEGER NOT NULL DEFAULT 0, llm_calls INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,
                first_send_at INTEGER, last_send_at INTEGER, updated_at INTEGER NOT NULL)""",
            f"""CREATE TABLE IF NOT EXISTS {p}timers (
                session_id TEXT NOT NULL, timer_id INTEGER NOT NULL, due_at INTEGER NOT NULL,
                note TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'armed', interval_s INTEGER,
                fire_count INTEGER NOT NULL DEFAULT 0, last_fired_at INTEGER,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (session_id, timer_id))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_timers_due ON {p}timers(status, due_at)",
            f"""CREATE TABLE IF NOT EXISTS {p}notes (
                session_id TEXT NOT NULL, note_id INTEGER NOT NULL, content TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, note_id))""",
            *self.project_messages_ddl(p),
            *self.hook_events_ddl(p),
        ]

    def project_messages_ddl(self, prefix: str) -> list[str]:
        p = prefix
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}project_messages (
                session_id TEXT NOT NULL, send_index INTEGER NOT NULL, kind TEXT NOT NULL,
                content_json TEXT NOT NULL, rendered_text TEXT,
                source_seq_lo INTEGER, source_seq_hi INTEGER,
                compact_from_send INTEGER, compact_to_send INTEGER,
                projector_version INTEGER NOT NULL DEFAULT 0, token_estimate INTEGER,
                created_at INTEGER NOT NULL, PRIMARY KEY (session_id, send_index, kind))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_project_messages_session_kind "
            f"ON {p}project_messages(session_id, kind, send_index)",
        ]

    def hook_events_ddl(self, prefix: str) -> list[str]:
        p = prefix
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}hook_events (
                session_id TEXT NOT NULL, event_id INTEGER NOT NULL, message_seq INTEGER,
                round_index INTEGER, send_index INTEGER, hook_point TEXT NOT NULL, hook TEXT,
                position TEXT, kind TEXT NOT NULL, payload_json TEXT, created_at INTEGER NOT NULL,
                PRIMARY KEY (session_id, event_id))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_hook_events_session_msg "
            f"ON {p}hook_events(session_id, message_seq)",
        ]

    def upsert(self, table, key_cols, val_cols, *, add_cols=(), insert_only_cols=()):
        return _onconflict_upsert(table, key_cols, val_cols, add_cols, insert_only_cols)

    async def lock_state(self, tx, state_table, session_id):
        # The SQLite backend serializes writers (one connection under a lock), so a
        # plain SELECT is atomic enough; PG/MySQL add `FOR UPDATE`.
        row = await tx.fetchone(f"SELECT * FROM {state_table} WHERE session_id=?", (session_id,))
        if row is None:
            raise ValueError(f"unknown session: {session_id}")
        return row


def _onconflict_upsert(table, key_cols, val_cols, add_cols, insert_only_cols) -> str:
    """ON CONFLICT … DO UPDATE renderer shared by SQLite and Postgres (identical syntax,
    including the ``excluded`` pseudo-table and the ``col = table.col + excluded.col``
    accumulate). MySQL overrides ``upsert`` with ``ON DUPLICATE KEY UPDATE … AS new_row``."""
    cols = [*key_cols, *val_cols, *add_cols, *insert_only_cols]
    placeholders = ",".join("?" * len(cols))
    sets = [f"{c}=excluded.{c}" for c in val_cols]
    sets += [f"{c}={table}.{c}+excluded.{c}" for c in add_cols]
    conflict = ",".join(key_cols)
    return (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {','.join(sets)}"
    )


# ── PostgreSQL ──────────────────────────────────────────────────────────────
# Pure SQL rendering only (no asyncpg import) so this stays in the zero-dep core; the
# asyncpg-backed Database lives in backends/postgres.py.


class PostgresDialect:
    name = "postgres"

    def translate(self, sql: str) -> str:
        # qmark (?) → numeric ($1, $2, …) for asyncpg. Our SQL never contains a literal
        # '?', so a positional substitution is safe.
        out: list[str] = []
        n = 0
        for ch in sql:
            if ch == "?":
                n += 1
                out.append(f"${n}")
            else:
                out.append(ch)
        return "".join(out)

    def ddl(self, prefix: str) -> list[str]:
        p = prefix
        # Map SQLite types → Postgres: INTEGER→BIGINT (epoch-ms time + counters),
        # TEXT→TEXT, pinned stays a small int (the store passes 0/1 uniformly; the row
        # mapper coerces to bool). JSON columns stay TEXT (the store (de)serializes).
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}sessions (
                session_id TEXT PRIMARY KEY, created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL, system_prompt TEXT, model TEXT,
                config_json TEXT, status TEXT NOT NULL DEFAULT 'active',
                kind TEXT NOT NULL DEFAULT 'root', parent_session_id TEXT,
                spawn_tool_call_id TEXT, spawn_depth BIGINT NOT NULL DEFAULT 0,
                lifecycle TEXT NOT NULL DEFAULT 'ephemeral', metadata_json TEXT)""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_sessions_parent ON {p}sessions(parent_session_id)",
            f"""CREATE TABLE IF NOT EXISTS {p}messages (
                session_id TEXT NOT NULL, seq BIGINT NOT NULL, role TEXT NOT NULL,
                name TEXT, content TEXT, tool_calls_json TEXT, tool_call_id TEXT,
                round_index BIGINT, state TEXT NOT NULL DEFAULT 'active', meta_json TEXT,
                send_index BIGINT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, seq))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_messages_session_state "
            f"ON {p}messages(session_id, state, seq)",
            f"""CREATE TABLE IF NOT EXISTS {p}compactions (
                session_id TEXT NOT NULL, compact_seq BIGINT NOT NULL, note_seq BIGINT NOT NULL,
                from_seq BIGINT NOT NULL, to_seq BIGINT NOT NULL, before_tokens BIGINT,
                after_tokens BIGINT, round_index BIGINT, created_at BIGINT NOT NULL,
                PRIMARY KEY (session_id, compact_seq))""",
            f"""CREATE TABLE IF NOT EXISTS {p}usage_rounds (
                session_id TEXT NOT NULL, round_index BIGINT NOT NULL, prompt_tokens BIGINT,
                completion_tokens BIGINT, total_tokens BIGINT, model TEXT,
                created_at BIGINT NOT NULL, PRIMARY KEY (session_id, round_index))""",
            f"""CREATE TABLE IF NOT EXISTS {p}session_state (
                session_id TEXT PRIMARY KEY, next_seq BIGINT NOT NULL DEFAULT 1,
                round_index BIGINT NOT NULL DEFAULT 0, last_compact_seq BIGINT NOT NULL DEFAULT 0,
                pending_json TEXT)""",
            f"""CREATE TABLE IF NOT EXISTS {p}session_runtime_state (
                session_id TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT,
                updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, state_key))""",
            f"""CREATE TABLE IF NOT EXISTS {p}shared_state (
                owner TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT,
                updated_at BIGINT NOT NULL, PRIMARY KEY (owner, state_key))""",
            f"""CREATE TABLE IF NOT EXISTS {p}background_tasks (
                session_id TEXT NOT NULL, task_id TEXT NOT NULL, command TEXT NOT NULL,
                status TEXT NOT NULL, return_code BIGINT, output_tail TEXT, output_path TEXT,
                last_seen_at BIGINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, task_id))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_background_tasks_session_status "
            f"ON {p}background_tasks(session_id, status, updated_at)",
            f"""CREATE TABLE IF NOT EXISTS {p}session_stats (
                session_id TEXT PRIMARY KEY, sends BIGINT NOT NULL DEFAULT 0,
                rounds BIGINT NOT NULL DEFAULT 0, llm_calls BIGINT NOT NULL DEFAULT 0,
                tool_calls BIGINT NOT NULL DEFAULT 0, prompt_tokens BIGINT NOT NULL DEFAULT 0,
                completion_tokens BIGINT NOT NULL DEFAULT 0, total_tokens BIGINT NOT NULL DEFAULT 0,
                first_send_at BIGINT, last_send_at BIGINT, updated_at BIGINT NOT NULL)""",
            f"""CREATE TABLE IF NOT EXISTS {p}timers (
                session_id TEXT NOT NULL, timer_id BIGINT NOT NULL, due_at BIGINT NOT NULL,
                note TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'armed', interval_s BIGINT,
                fire_count BIGINT NOT NULL DEFAULT 0, last_fired_at BIGINT,
                created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL,
                PRIMARY KEY (session_id, timer_id))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_timers_due ON {p}timers(status, due_at)",
            f"""CREATE TABLE IF NOT EXISTS {p}notes (
                session_id TEXT NOT NULL, note_id BIGINT NOT NULL, content TEXT NOT NULL,
                pinned SMALLINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, note_id))""",
            *self.project_messages_ddl(p),
            *self.hook_events_ddl(p),
        ]

    def project_messages_ddl(self, prefix: str) -> list[str]:
        p = prefix
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}project_messages (
                session_id TEXT NOT NULL, send_index BIGINT NOT NULL, kind TEXT NOT NULL,
                content_json TEXT NOT NULL, rendered_text TEXT,
                source_seq_lo BIGINT, source_seq_hi BIGINT,
                compact_from_send BIGINT, compact_to_send BIGINT,
                projector_version BIGINT NOT NULL DEFAULT 0, token_estimate BIGINT,
                created_at BIGINT NOT NULL, PRIMARY KEY (session_id, send_index, kind))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_project_messages_session_kind "
            f"ON {p}project_messages(session_id, kind, send_index)",
        ]

    def hook_events_ddl(self, prefix: str) -> list[str]:
        p = prefix
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}hook_events (
                session_id TEXT NOT NULL, event_id BIGINT NOT NULL, message_seq BIGINT,
                round_index BIGINT, send_index BIGINT, hook_point TEXT NOT NULL, hook TEXT,
                position TEXT, kind TEXT NOT NULL, payload_json TEXT, created_at BIGINT NOT NULL,
                PRIMARY KEY (session_id, event_id))""",
            f"CREATE INDEX IF NOT EXISTS {p}idx_hook_events_session_msg "
            f"ON {p}hook_events(session_id, message_seq)",
        ]

    def upsert(self, table, key_cols, val_cols, *, add_cols=(), insert_only_cols=()):
        return _onconflict_upsert(table, key_cols, val_cols, add_cols, insert_only_cols)

    async def lock_state(self, tx, state_table, session_id):
        # Real multi-writer safety: lock the session_state row for the rest of the txn.
        row = await tx.fetchone(
            f"SELECT * FROM {state_table} WHERE session_id=? FOR UPDATE", (session_id,)
        )
        if row is None:
            raise ValueError(f"unknown session: {session_id}")
        return row


# ── MySQL ─────────────────────────────────────────────────────────────────────
# Pure SQL rendering only (no driver import) — the aiomysql-backed Database lives in
# backends/mysql.py. Differences from PG that live here: paramstyle is ``%s`` (and a
# literal ``%`` must be escaped to ``%%`` for the driver's %-formatting); string PK/index
# columns must be VARCHAR (MySQL can't index/PK a TEXT without a prefix length); indexes
# are declared INLINE in CREATE TABLE (MySQL has no ``CREATE INDEX IF NOT EXISTS``); and
# upsert is ``ON DUPLICATE KEY UPDATE … VALUES(col)`` rather than ``ON CONFLICT … excluded``.


class MySQLDialect:
    name = "mysql"

    def translate(self, sql: str) -> str:
        # The driver (aiomysql/PyMySQL) uses ``%s`` and runs ``query % args``, so a literal
        # ``%`` must be doubled here and is collapsed back by that pass. This is only correct
        # because backends/mysql.py:_args ALWAYS passes a tuple (never None), so the
        # ``query % args`` collapse runs even for parameterless statements — otherwise a
        # doubled ``%%`` would leak to MySQL verbatim. Escape ``%`` first, then qmark → ``%s``.
        return sql.replace("%", "%%").replace("?", "%s")

    def ddl(self, prefix: str) -> list[str]:
        p = prefix
        # utf8mb4 + InnoDB. INTEGER→BIGINT (epoch-ms time + counters); string PK/index
        # columns → VARCHAR(255) (≤ the 3072-byte large-prefix limit even for the 2-col
        # composite PKs); small enum-like columns → VARCHAR(32); JSON/free text → TEXT.
        # Indexes are declared inline (no CREATE INDEX IF NOT EXISTS in MySQL).
        opts = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}sessions (
                session_id VARCHAR(255) NOT NULL, created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL, system_prompt TEXT, model VARCHAR(255),
                config_json TEXT, status VARCHAR(32) NOT NULL DEFAULT 'active',
                kind VARCHAR(32) NOT NULL DEFAULT 'root', parent_session_id VARCHAR(255),
                spawn_tool_call_id VARCHAR(255), spawn_depth BIGINT NOT NULL DEFAULT 0,
                lifecycle VARCHAR(32) NOT NULL DEFAULT 'ephemeral', metadata_json TEXT,
                PRIMARY KEY (session_id),
                KEY {p}idx_sessions_parent (parent_session_id)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}messages (
                session_id VARCHAR(255) NOT NULL, seq BIGINT NOT NULL, role VARCHAR(32) NOT NULL,
                name VARCHAR(255), content TEXT, tool_calls_json TEXT, tool_call_id VARCHAR(255),
                round_index BIGINT, state VARCHAR(32) NOT NULL DEFAULT 'active', meta_json TEXT,
                send_index BIGINT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, seq),
                KEY {p}idx_messages_session_state (session_id, state, seq)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}compactions (
                session_id VARCHAR(255) NOT NULL, compact_seq BIGINT NOT NULL, note_seq BIGINT NOT NULL,
                from_seq BIGINT NOT NULL, to_seq BIGINT NOT NULL, before_tokens BIGINT,
                after_tokens BIGINT, round_index BIGINT, created_at BIGINT NOT NULL,
                PRIMARY KEY (session_id, compact_seq)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}usage_rounds (
                session_id VARCHAR(255) NOT NULL, round_index BIGINT NOT NULL, prompt_tokens BIGINT,
                completion_tokens BIGINT, total_tokens BIGINT, model VARCHAR(255),
                created_at BIGINT NOT NULL, PRIMARY KEY (session_id, round_index)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}session_state (
                session_id VARCHAR(255) NOT NULL, next_seq BIGINT NOT NULL DEFAULT 1,
                round_index BIGINT NOT NULL DEFAULT 0, last_compact_seq BIGINT NOT NULL DEFAULT 0,
                pending_json TEXT, PRIMARY KEY (session_id)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}session_runtime_state (
                session_id VARCHAR(255) NOT NULL, state_key VARCHAR(255) NOT NULL, value_json TEXT,
                updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, state_key)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}shared_state (
                owner VARCHAR(255) NOT NULL, state_key VARCHAR(255) NOT NULL, value_json TEXT,
                updated_at BIGINT NOT NULL, PRIMARY KEY (owner, state_key)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}background_tasks (
                session_id VARCHAR(255) NOT NULL, task_id VARCHAR(255) NOT NULL, command TEXT NOT NULL,
                status VARCHAR(32) NOT NULL, return_code BIGINT, output_tail TEXT, output_path TEXT,
                last_seen_at BIGINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, task_id),
                KEY {p}idx_background_tasks_session_status (session_id, status, updated_at)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}session_stats (
                session_id VARCHAR(255) NOT NULL, sends BIGINT NOT NULL DEFAULT 0,
                rounds BIGINT NOT NULL DEFAULT 0, llm_calls BIGINT NOT NULL DEFAULT 0,
                tool_calls BIGINT NOT NULL DEFAULT 0, prompt_tokens BIGINT NOT NULL DEFAULT 0,
                completion_tokens BIGINT NOT NULL DEFAULT 0, total_tokens BIGINT NOT NULL DEFAULT 0,
                first_send_at BIGINT, last_send_at BIGINT, updated_at BIGINT NOT NULL,
                PRIMARY KEY (session_id)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}timers (
                session_id VARCHAR(255) NOT NULL, timer_id BIGINT NOT NULL, due_at BIGINT NOT NULL,
                note TEXT NOT NULL, status VARCHAR(32) NOT NULL DEFAULT 'armed', interval_s BIGINT,
                fire_count BIGINT NOT NULL DEFAULT 0, last_fired_at BIGINT,
                created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL,
                PRIMARY KEY (session_id, timer_id),
                KEY {p}idx_timers_due (status, due_at)) {opts}""",
            f"""CREATE TABLE IF NOT EXISTS {p}notes (
                session_id VARCHAR(255) NOT NULL, note_id BIGINT NOT NULL, content TEXT NOT NULL,
                pinned TINYINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, note_id)) {opts}""",
            *self.project_messages_ddl(p),
            *self.hook_events_ddl(p),
        ]

    def project_messages_ddl(self, prefix: str) -> list[str]:
        p = prefix
        opts = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}project_messages (
                session_id VARCHAR(255) NOT NULL, send_index BIGINT NOT NULL, kind VARCHAR(32) NOT NULL,
                content_json TEXT NOT NULL, rendered_text TEXT,
                source_seq_lo BIGINT, source_seq_hi BIGINT,
                compact_from_send BIGINT, compact_to_send BIGINT,
                projector_version BIGINT NOT NULL DEFAULT 0, token_estimate BIGINT,
                created_at BIGINT NOT NULL, PRIMARY KEY (session_id, send_index, kind),
                KEY {p}idx_project_messages_session_kind (session_id, kind, send_index)) {opts}""",
        ]

    def hook_events_ddl(self, prefix: str) -> list[str]:
        p = prefix
        opts = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        return [
            f"""CREATE TABLE IF NOT EXISTS {p}hook_events (
                session_id VARCHAR(255) NOT NULL, event_id BIGINT NOT NULL, message_seq BIGINT,
                round_index BIGINT, send_index BIGINT, hook_point VARCHAR(32) NOT NULL, hook VARCHAR(255),
                position VARCHAR(32), kind VARCHAR(32) NOT NULL, payload_json TEXT, created_at BIGINT NOT NULL,
                PRIMARY KEY (session_id, event_id),
                KEY {p}idx_hook_events_session_msg (session_id, message_seq)) {opts}""",
        ]

    def upsert(self, table, key_cols, val_cols, *, add_cols=(), insert_only_cols=()):
        # MySQL: INSERT … AS new_row ON DUPLICATE KEY UPDATE col=new_row.col; accumulate via
        # col=col+new_row.col; insert_only_cols are inserted but omitted from the UPDATE (so
        # they are preserved on conflict). The row-alias form (MySQL 8.0.19+) replaces the
        # deprecated VALUES() function.
        cols = [*key_cols, *val_cols, *add_cols, *insert_only_cols]
        placeholders = ",".join("?" * len(cols))
        sets = [f"{c}=new_row.{c}" for c in val_cols]
        sets += [f"{c}={table}.{c}+new_row.{c}" for c in add_cols]
        return (
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) AS new_row "
            f"ON DUPLICATE KEY UPDATE {','.join(sets)}"
        )

    async def lock_state(self, tx, state_table, session_id):
        # Real multi-writer safety: lock the session_state row for the rest of the txn.
        row = await tx.fetchone(
            f"SELECT * FROM {state_table} WHERE session_id=? FOR UPDATE", (session_id,)
        )
        if row is None:
            raise ValueError(f"unknown session: {session_id}")
        return row


__all__ = ["Dialect", "SqliteDialect", "PostgresDialect", "MySQLDialect"]
