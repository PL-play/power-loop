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

    def upsert(
        self,
        table: str,
        key_cols: Sequence[str],
        val_cols: Sequence[str],
        *,
        add_cols: Sequence[str] = (),
    ) -> str:
        """Render an idempotent INSERT-or-UPDATE. ``val_cols`` are overwritten;
        ``add_cols`` are accumulated (``col = col + new``). ``?`` placeholders for
        ``key_cols + val_cols + add_cols`` in that order."""
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
                created_at INTEGER NOT NULL, PRIMARY KEY (session_id, seq))""",
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
                session_id TEXT NOT NULL, key TEXT NOT NULL, value_json TEXT,
                updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, key))""",
            f"""CREATE TABLE IF NOT EXISTS {p}shared_state (
                owner TEXT NOT NULL, key TEXT NOT NULL, value_json TEXT,
                updated_at INTEGER NOT NULL, PRIMARY KEY (owner, key))""",
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
        ]

    def upsert(self, table, key_cols, val_cols, *, add_cols=()):
        cols = [*key_cols, *val_cols, *add_cols]
        placeholders = ",".join("?" * len(cols))
        sets = [f"{c}=excluded.{c}" for c in val_cols]
        sets += [f"{c}={table}.{c}+excluded.{c}" for c in add_cols]
        conflict = ",".join(key_cols)
        return (
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {','.join(sets)}"
        )

    async def lock_state(self, tx, state_table, session_id):
        # The SQLite backend serializes writers (one connection under a lock), so a
        # plain SELECT is atomic enough; PG/MySQL add `FOR UPDATE`.
        row = await tx.fetchone(f"SELECT * FROM {state_table} WHERE session_id=?", (session_id,))
        if row is None:
            raise ValueError(f"unknown session: {session_id}")
        return row


__all__ = ["Dialect", "SqliteDialect"]
