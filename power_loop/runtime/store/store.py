"""Backend-neutral async ``SessionStore`` facade.

Written ONCE against the ``Database`` + ``Dialect`` ports; the backend (SQLite today,
PostgreSQL/MySQL next) is an implementation detail. Every multi-statement write runs in
one transaction so it is atomic on any engine; the per-session ``seq`` counter is
allocated under the dialect's ``lock_state`` so it is correct even with multiple writers.

VERTICAL SLICE: this currently implements the lifecycle + the hardest paths (sessions,
messages, transactional seq allocation, compaction, session_state). The remaining ~40
methods (timers, notes, stats, runtime/shared state, background tasks, retention,
export/import, cross-session timer scans) translate from the legacy ``session_store.py``
against this same contract and land next.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any

from power_loop.runtime.store.db import Database, Row
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION, ensure_schema
from power_loop.runtime.store.types import (
    CompactionRow,
    MessageRow,
    MessageState,
    SessionKind,
    SessionRow,
    SessionStateRow,
    SessionStatus,
    SubagentLifecycle,
)

DEFAULT_TABLE_PREFIX = "pl_"
DEFAULT_MAX_SPAWN_DEPTH = 3


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _new_session_id() -> str:
    return "sess_" + secrets.token_hex(12)


def _dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _loads(s: str | None) -> Any:
    if s is None or s == "":
        return None
    return json.loads(s)


class _Tables:
    """Prefixed table names (default ``pl_``) so power-loop's tables stay isolated on a
    database shared with another app."""

    def __init__(self, prefix: str) -> None:
        self.sessions = f"{prefix}sessions"
        self.messages = f"{prefix}messages"
        self.compactions = f"{prefix}compactions"
        self.usage_rounds = f"{prefix}usage_rounds"
        self.session_state = f"{prefix}session_state"
        self.session_runtime_state = f"{prefix}session_runtime_state"
        self.shared_state = f"{prefix}shared_state"
        self.background_tasks = f"{prefix}background_tasks"
        self.session_stats = f"{prefix}session_stats"
        self.timers = f"{prefix}timers"
        self.notes = f"{prefix}notes"


class SessionStore:
    """Async, backend-neutral session store. Construct via :meth:`open`."""

    def __init__(
        self,
        db: Database,
        *,
        max_spawn_depth: int = DEFAULT_MAX_SPAWN_DEPTH,
        table_prefix: str = DEFAULT_TABLE_PREFIX,
    ) -> None:
        self._db = db
        self.max_spawn_depth = int(max_spawn_depth)
        self.table_prefix = table_prefix
        self.t = _Tables(table_prefix)

    @classmethod
    async def open(
        cls,
        path: str = ":memory:",
        *,
        max_spawn_depth: int = DEFAULT_MAX_SPAWN_DEPTH,
        table_prefix: str = DEFAULT_TABLE_PREFIX,
        create_schema: bool = True,
    ) -> SessionStore:
        """Open a SQLite-backed store (the default backend). Postgres/MySQL get their
        own ``open``/DSN factory in later phases."""
        from power_loop.runtime.store.backends.sqlite import SqliteDatabase

        db = SqliteDatabase.open(path)
        await ensure_schema(db, table_prefix, create_schema=create_schema)
        return cls(db, max_spawn_depth=max_spawn_depth, table_prefix=table_prefix)

    @property
    def schema_version(self) -> int:
        return CURRENT_SCHEMA_VERSION

    async def close(self) -> None:
        await self._db.close()

    async def __aenter__(self) -> SessionStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── sessions ────────────────────────────────────────────────────────────
    async def create_session(
        self,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        config: dict[str, Any] | None = None,
        parent_session_id: str | None = None,
        spawn_tool_call_id: str | None = None,
        kind: SessionKind = SessionKind.ROOT,
        lifecycle: SubagentLifecycle = SubagentLifecycle.EPHEMERAL,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        spawn_depth = 0
        if parent_session_id is not None:
            parent = await self.get_session(parent_session_id)
            if parent is None:
                raise ValueError(f"parent session not found: {parent_session_id}")
            spawn_depth = parent.spawn_depth + 1
            if spawn_depth > self.max_spawn_depth:
                raise ValueError(f"spawn depth {spawn_depth} exceeds max {self.max_spawn_depth}")
            kind = SessionKind.SUBAGENT

        sid = session_id or _new_session_id()
        now = _now_ms()
        async with self._db.transaction() as tx:
            await tx.execute(
                f"INSERT INTO {self.t.sessions} ("
                "session_id, created_at, updated_at, system_prompt, model, config_json, "
                "status, kind, parent_session_id, spawn_tool_call_id, spawn_depth, "
                "lifecycle, metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid, now, now, system_prompt, model, _dumps(config or {}),
                    SessionStatus.ACTIVE.value, kind.value, parent_session_id,
                    spawn_tool_call_id, spawn_depth, lifecycle.value, _dumps(metadata or {}),
                ),
            )
            await tx.execute(
                f"INSERT INTO {self.t.session_state} (session_id) VALUES (?)", (sid,)
            )
        return sid

    async def get_session(self, session_id: str) -> SessionRow | None:
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.sessions} WHERE session_id=?", (session_id,)
        )
        return _row_to_session(row) if row else None

    async def get_state(self, session_id: str) -> SessionStateRow | None:
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.session_state} WHERE session_id=?", (session_id,)
        )
        if row is None:
            return None
        return SessionStateRow(
            session_id=row["session_id"],
            next_seq=row["next_seq"],
            round_index=row["round_index"],
            last_compact_seq=row["last_compact_seq"],
            pending=_loads(row["pending_json"]),
        )

    async def set_round_index(self, session_id: str, round_index: int) -> None:
        async with self._db.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.t.session_state} SET round_index=? WHERE session_id=?",
                (round_index, session_id),
            )

    async def set_pending(self, session_id: str, pending: dict[str, Any] | None) -> None:
        async with self._db.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.t.session_state} SET pending_json=? WHERE session_id=?",
                (_dumps(pending), session_id),
            )

    # ── messages ──────────────────────────────────────────────────────────────
    async def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        round_index: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        """Append one message and return its allocated per-session ``seq`` (allocated +
        inserted atomically in one transaction)."""
        now = _now_ms()
        async with self._db.transaction() as tx:
            st = await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            seq = int(st["next_seq"])
            await tx.execute(
                f"INSERT INTO {self.t.messages} ("
                "session_id, seq, role, name, content, tool_calls_json, tool_call_id, "
                "round_index, state, meta_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id, seq, role, name, content,
                    _dumps(tool_calls) if tool_calls else None, tool_call_id, round_index,
                    MessageState.ACTIVE.value, _dumps(meta or {}), now,
                ),
            )
            await tx.execute(
                f"UPDATE {self.t.session_state} SET next_seq=? WHERE session_id=?",
                (seq + 1, session_id),
            )
            await tx.execute(
                f"UPDATE {self.t.sessions} SET updated_at=? WHERE session_id=?", (now, session_id)
            )
        return seq

    async def load_active_messages(self, session_id: str) -> list[MessageRow]:
        """Active messages in **logical** order (a ``compact_note`` sorts at its
        ``meta['ord']``, not its high identity ``seq``)."""
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.messages} WHERE session_id=? AND state=? ORDER BY seq ASC",
            (session_id, MessageState.ACTIVE.value),
        )
        messages = [_row_to_message(r) for r in rows]
        messages.sort(key=_logical_order_key)
        return messages

    async def load_all_messages(self, session_id: str) -> list[MessageRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.messages} WHERE session_id=? ORDER BY seq ASC", (session_id,)
        )
        return [_row_to_message(r) for r in rows]

    async def get_message(self, session_id: str, seq: int) -> MessageRow | None:
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.messages} WHERE session_id=? AND seq=?", (session_id, seq)
        )
        return _row_to_message(row) if row else None

    # ── compaction ──────────────────────────────────────────────────────────────
    async def record_compaction(
        self,
        session_id: str,
        *,
        from_seq: int,
        to_seq: int,
        note_content: str,
        before_tokens: int | None,
        after_tokens: int | None,
        round_index: int | None,
        note_meta: dict[str, Any] | None = None,
        fold_seqs: list[int] | None = None,
        order_key: int | None = None,
    ) -> tuple[int, int]:
        """Mark the folded messages ``compacted_out`` and append the ``compact_note`` in
        one transaction. ``fold_seqs`` marks the EXACT set (never a BETWEEN range, which
        could invert under a non-monotonic identity map); ``order_key`` is the note's
        logical position (``meta['ord']``). Returns ``(compact_seq, note_seq)``."""
        now = _now_ms()
        async with self._db.transaction() as tx:
            st = await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            note_seq = int(st["next_seq"])
            compact_seq = int(st["last_compact_seq"]) + 1

            if fold_seqs is not None:
                uniq = [int(s) for s in dict.fromkeys(fold_seqs)]
                if uniq:
                    placeholders = ",".join("?" * len(uniq))
                    await tx.execute(
                        f"UPDATE {self.t.messages} SET state=? "
                        f"WHERE session_id=? AND seq IN ({placeholders})",
                        (MessageState.COMPACTED_OUT.value, session_id, *uniq),
                    )
                span_from = min(uniq) if uniq else from_seq
                span_to = max(uniq) if uniq else to_seq
            else:
                await tx.execute(
                    f"UPDATE {self.t.messages} SET state=? "
                    f"WHERE session_id=? AND seq BETWEEN ? AND ?",
                    (MessageState.COMPACTED_OUT.value, session_id, from_seq, to_seq),
                )
                span_from, span_to = from_seq, to_seq

            note_ord = int(order_key) if order_key is not None else int(span_from)
            meta = dict(note_meta or {})
            meta.update({
                "compacted_at_round": round_index, "from_seq": span_from, "to_seq": span_to,
                "ord": note_ord, "original_tokens": before_tokens, "summary_tokens": after_tokens,
            })
            await tx.execute(
                f"INSERT INTO {self.t.messages} ("
                "session_id, seq, role, name, content, tool_calls_json, tool_call_id, "
                "round_index, state, meta_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id, note_seq, "system", "compact_note", note_content, None, None,
                    round_index, MessageState.ACTIVE.value, _dumps(meta), now,
                ),
            )
            await tx.execute(
                f"INSERT INTO {self.t.compactions} ("
                "session_id, compact_seq, note_seq, from_seq, to_seq, before_tokens, "
                "after_tokens, round_index, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    session_id, compact_seq, note_seq, span_from, span_to, before_tokens,
                    after_tokens, round_index, now,
                ),
            )
            await tx.execute(
                f"UPDATE {self.t.session_state} SET next_seq=?, last_compact_seq=? "
                "WHERE session_id=?",
                (note_seq + 1, compact_seq, session_id),
            )
            await tx.execute(
                f"UPDATE {self.t.sessions} SET updated_at=? WHERE session_id=?", (now, session_id)
            )
        return compact_seq, note_seq

    async def list_compactions(self, session_id: str) -> list[CompactionRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.compactions} WHERE session_id=? ORDER BY compact_seq",
            (session_id,),
        )
        return [
            CompactionRow(
                session_id=r["session_id"], compact_seq=r["compact_seq"], note_seq=r["note_seq"],
                from_seq=r["from_seq"], to_seq=r["to_seq"], before_tokens=r["before_tokens"],
                after_tokens=r["after_tokens"], round_index=r["round_index"], created_at=r["created_at"],
            )
            for r in rows
        ]


# ── row converters (dict rows; backend-agnostic) ────────────────────────────────


def _row_to_session(row: Row) -> SessionRow:
    return SessionRow(
        session_id=row["session_id"], created_at=row["created_at"], updated_at=row["updated_at"],
        system_prompt=row["system_prompt"], model=row["model"],
        config=_loads(row["config_json"]) or {}, status=SessionStatus(row["status"]),
        kind=SessionKind(row["kind"]), parent_session_id=row["parent_session_id"],
        spawn_tool_call_id=row["spawn_tool_call_id"], spawn_depth=row["spawn_depth"],
        lifecycle=SubagentLifecycle(row["lifecycle"]), metadata=_loads(row["metadata_json"]) or {},
    )


def _row_to_message(row: Row) -> MessageRow:
    return MessageRow(
        session_id=row["session_id"], seq=row["seq"], role=row["role"], name=row["name"],
        content=row["content"], tool_calls=_loads(row["tool_calls_json"]),
        tool_call_id=row["tool_call_id"], round_index=row["round_index"],
        state=MessageState(row["state"]), meta=_loads(row["meta_json"]) or {},
        created_at=row["created_at"],
    )


def _logical_order_key(m: MessageRow) -> tuple[int, int]:
    if m.name == "compact_note":
        ord_val = m.meta.get("ord")
        if ord_val is not None:
            return (int(ord_val), m.seq)
    return (m.seq, m.seq)


__all__ = ["SessionStore", "DEFAULT_TABLE_PREFIX", "DEFAULT_MAX_SPAWN_DEPTH"]
