"""Backend-neutral async ``SessionStore`` facade.

Written ONCE against the ``Database`` + ``Dialect`` ports; the backend (SQLite today,
PostgreSQL/MySQL next) is an implementation detail. Every multi-statement write runs in
one transaction so it is atomic on any engine; the per-session ``seq`` counter is
allocated under the dialect's ``lock_state`` so it is correct even with multiple writers.

Covers the full legacy ``session_store.py`` surface: sessions + lifecycle/cascade,
messages + transactional seq allocation, compaction, session_state, timers, notes,
usage/stats, runtime + shared state, background tasks, retention/prune, and
export/import. SQLite-only maintenance (checkpoint/vacuum/backup) is an optional
:class:`~power_loop.runtime.store.capabilities.Maintenance` capability that no-ops on
backends that lack it.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Mapping
from typing import Any

from power_loop.runtime.store.capabilities import Maintenance
from power_loop.runtime.store.db import Database, Row, Transaction
from power_loop.runtime.store.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaPolicy,
    ensure_schema,
    validate_table_prefix,
)
from power_loop.runtime.store.types import (
    BackgroundTaskRow,
    CompactionRow,
    MessageRow,
    MessageState,
    NoteRow,
    ProjectMessageRow,
    SessionKind,
    SessionRow,
    SessionStateRow,
    SessionStatsRow,
    SessionStatus,
    SubagentLifecycle,
    TimerRow,
)

DEFAULT_TABLE_PREFIX = "pl_"
DEFAULT_MAX_SPAWN_DEPTH = 3

# Back-compat public aliases (the names the rest of power-loop and its callers
# import). The async store is the canonical source for these now.
MAX_SPAWN_DEPTH = DEFAULT_MAX_SPAWN_DEPTH
DEFAULT_DB_PATH = "./power_loop_sessions.db"


class _NoWrite:
    """Sentinel: a :meth:`SessionStore.mutate_runtime_state` callback returns this to
    leave the row untouched (no write, no ``updated_at`` bump)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "MUTATE_SKIP"


MUTATE_SKIP = _NoWrite()

# Conversation order WITHIN one send for projection rows: the user turn before the assistant
# (project) summary; a compact row (keyed at its own send_index) sorts last at that index.
_PROJECT_KIND_ORDER = {"user": 0, "project": 1, "compact": 2}


def _coerce_max_spawn_depth(value: int) -> int:
    """Validate a spawn-depth ceiling: a positive int (≥1). Raises ValueError."""
    try:
        depth = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"max_spawn_depth must be an int ≥ 1, got {value!r}") from None
    if depth < 1:
        raise ValueError(f"max_spawn_depth must be ≥ 1, got {depth}")
    return depth


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
        self.project_messages = f"{prefix}project_messages"


# Logical export schema: (logical_name, physical(t)->table, explicit_columns). Explicit
# column lists (no SELECT *) keep the export wire format backend-neutral; background_tasks
# (transient), shared_state (not session-scoped), and project_messages (a DERIVED projection
# rebuildable from messages) are intentionally excluded.
_EXPORT_TABLES: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("sessions", lambda t: t.sessions, (
        "session_id", "created_at", "updated_at", "system_prompt", "model", "config_json",
        "status", "kind", "parent_session_id", "spawn_tool_call_id", "spawn_depth",
        "lifecycle", "metadata_json")),
    ("session_state", lambda t: t.session_state, (
        "session_id", "next_seq", "round_index", "last_compact_seq", "pending_json")),
    ("messages", lambda t: t.messages, (
        "session_id", "seq", "role", "name", "content", "tool_calls_json", "tool_call_id",
        "round_index", "state", "meta_json", "send_index", "created_at")),
    ("compactions", lambda t: t.compactions, (
        "session_id", "compact_seq", "note_seq", "from_seq", "to_seq", "before_tokens",
        "after_tokens", "round_index", "created_at")),
    ("usage_rounds", lambda t: t.usage_rounds, (
        "session_id", "round_index", "prompt_tokens", "completion_tokens", "total_tokens",
        "model", "created_at")),
    ("session_runtime_state", lambda t: t.session_runtime_state, (
        "session_id", "state_key", "value_json", "updated_at")),
    ("timers", lambda t: t.timers, (
        "session_id", "timer_id", "due_at", "note", "status", "interval_s", "fire_count",
        "last_fired_at", "created_at", "updated_at")),
    ("notes", lambda t: t.notes, (
        "session_id", "note_id", "content", "pinned", "created_at", "updated_at")),
    ("session_stats", lambda t: t.session_stats, (
        "session_id", "sends", "rounds", "llm_calls", "tool_calls", "prompt_tokens",
        "completion_tokens", "total_tokens", "first_send_at", "last_send_at", "updated_at")),
)


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
        self.max_spawn_depth = max_spawn_depth  # validated by the property setter
        self.table_prefix = validate_table_prefix(table_prefix)
        self.t = _Tables(self.table_prefix)

    @property
    def max_spawn_depth(self) -> int:
        """Ceiling on sub-agent nesting depth for sessions created by this store.

        A per-process config knob, not per-request state: set it once before the store
        is used concurrently (it is read in ``create_session`` outside any DB lock, so
        mutating it mid-flight is a last-writer-wins ordering question — harmless on a
        set-once value)."""
        return self._max_spawn_depth

    @max_spawn_depth.setter
    def max_spawn_depth(self, value: int) -> None:
        self._max_spawn_depth = _coerce_max_spawn_depth(value)

    @classmethod
    async def open(
        cls,
        path: str = ":memory:",
        *,
        max_spawn_depth: int = DEFAULT_MAX_SPAWN_DEPTH,
        table_prefix: str = DEFAULT_TABLE_PREFIX,
        schema: SchemaPolicy | str | None = None,
        create_schema: bool | None = None,
    ) -> SessionStore:
        """Open a SQLite-backed store (the default backend). For Postgres/MySQL use
        :func:`power_loop.runtime.store.factory.open_store` with a DSN.

        ``schema`` is a :class:`SchemaPolicy` (default AUTO_CREATE). ``create_schema`` (bool)
        is a deprecated alias kept for the 1.x line (True→AUTO_CREATE, False→VERIFY)."""
        from power_loop.runtime.store.backends.sqlite import SqliteDatabase

        db = SqliteDatabase.open(path)
        await ensure_schema(db, table_prefix, policy=schema, create_schema=create_schema)
        return cls(db, max_spawn_depth=max_spawn_depth, table_prefix=table_prefix)

    @property
    def schema_version(self) -> int:
        return CURRENT_SCHEMA_VERSION

    # ── maintenance (optional capability; no-op on backends that lack it) ──────
    async def checkpoint(self, *, mode: str = "TRUNCATE") -> None:
        if isinstance(self._db, Maintenance):
            await self._db.checkpoint(mode=mode)

    async def vacuum(self, *, incremental: bool = True) -> None:
        if isinstance(self._db, Maintenance):
            await self._db.vacuum(incremental=incremental)

    async def backup(self, dest_path: str) -> None:
        if isinstance(self._db, Maintenance):
            await self._db.backup(dest_path)

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
                # Normalize a falsy-but-non-None pending ({}) to SQL NULL, matching the
                # legacy oracle so get_state(...).pending round-trips to None, not {}.
                (_dumps(pending) if pending else None, session_id),
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
        send_index: int | None = None,
    ) -> int:
        """Append one message and return its allocated per-session ``seq`` (allocated +
        inserted atomically in one transaction). ``send_index`` is the authoritative per-session
        send index (a real column, NULL outside a send)."""
        now = _now_ms()
        async with self._db.transaction() as tx:
            st = await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            seq = int(st["next_seq"])
            await tx.execute(
                f"INSERT INTO {self.t.messages} ("
                "session_id, seq, role, name, content, tool_calls_json, tool_call_id, "
                "round_index, state, meta_json, send_index, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id, seq, role, name, content,
                    _dumps(tool_calls) if tool_calls else None, tool_call_id, round_index,
                    MessageState.ACTIVE.value, _dumps(meta or {}),
                    (int(send_index) if send_index is not None else None), now,
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

    async def load_active_messages(
        self, session_id: str, *, after_seq: int | None = None
    ) -> list[MessageRow]:
        """Active messages in **logical** order (a ``compact_note`` sorts at its
        ``meta['ord']``, not its high identity ``seq``).

        ``after_seq`` (inclusive) returns only the active tail with ``seq >= after_seq`` — a
        cheap O(delta) read for incrementally extending a cached window after the caller's own
        appends (valid only when no compaction reshuffled the older active set; the caller
        must reload in full otherwise)."""
        if after_seq is None:
            rows = await self._db.fetchall(
                f"SELECT * FROM {self.t.messages} WHERE session_id=? AND state=? ORDER BY seq ASC",
                (session_id, MessageState.ACTIVE.value),
            )
        else:
            rows = await self._db.fetchall(
                f"SELECT * FROM {self.t.messages} WHERE session_id=? AND state=? AND seq>=? "
                "ORDER BY seq ASC",
                (session_id, MessageState.ACTIVE.value, int(after_seq)),
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

    # ── project_messages (send-context projection layer) ──────────────────────
    # The DERIVED per-send view fed to the LLM, kept entirely separate from the immutable
    # pl_messages audit log. Append-only / idempotent-upsert; rebuildable from pl_messages.
    async def _upsert_project_message_tx(
        self,
        tx: Transaction,
        session_id: str,
        *,
        send_index: int,
        kind: str,
        content: Any,
        now: int,
        rendered_text: str | None = None,
        source_seq_lo: int | None = None,
        source_seq_hi: int | None = None,
        compact_from_send: int | None = None,
        compact_to_send: int | None = None,
        projector_version: int = 0,
        token_estimate: int | None = None,
    ) -> None:
        """Upsert one projection row on an OPEN transaction (so several rows + a compact fold can
        share one atomic, lockable transaction). See :meth:`upsert_project_message`."""
        sql = self._db.dialect.upsert(
            self.t.project_messages,
            ("session_id", "send_index", "kind"),
            (
                "content_json", "rendered_text", "source_seq_lo", "source_seq_hi",
                "compact_from_send", "compact_to_send", "projector_version", "token_estimate",
            ),
            insert_only_cols=("created_at",),
        )
        await tx.execute(
            sql,
            (
                session_id, int(send_index), kind,
                _dumps(content if content is not None else {}), rendered_text,
                source_seq_lo, source_seq_hi, compact_from_send, compact_to_send,
                int(projector_version), token_estimate, now,
            ),
        )

    async def upsert_project_message(
        self,
        session_id: str,
        *,
        send_index: int,
        kind: str,
        content: Any,
        rendered_text: str | None = None,
        source_seq_lo: int | None = None,
        source_seq_hi: int | None = None,
        compact_from_send: int | None = None,
        compact_to_send: int | None = None,
        projector_version: int = 0,
        token_estimate: int | None = None,
    ) -> None:
        """Insert (or replace, by ``(session_id, send_index, kind)``) one projection row.
        Idempotent so a resume/re-finalize of the same send is a no-op-equivalent rewrite;
        ``created_at`` is preserved across re-finalize (insert-only)."""
        async with self._db.transaction() as tx:
            await self._upsert_project_message_tx(
                tx, session_id, send_index=send_index, kind=kind, content=content,
                rendered_text=rendered_text, source_seq_lo=source_seq_lo,
                source_seq_hi=source_seq_hi, compact_from_send=compact_from_send,
                compact_to_send=compact_to_send, projector_version=projector_version,
                token_estimate=token_estimate, now=_now_ms(),
            )

    async def _query_project_messages(
        self, fetchall: Callable[..., Any], session_id: str, after_send_index: int | None
    ) -> list[ProjectMessageRow]:
        if after_send_index is None:
            rows = await fetchall(
                f"SELECT * FROM {self.t.project_messages} WHERE session_id=? ORDER BY send_index ASC",
                (session_id,),
            )
        else:
            rows = await fetchall(
                f"SELECT * FROM {self.t.project_messages} WHERE session_id=? AND send_index>? "
                "ORDER BY send_index ASC",
                (session_id, int(after_send_index)),
            )
        out = [_row_to_project_message(r) for r in rows]
        out.sort(key=lambda m: (m.send_index, _PROJECT_KIND_ORDER.get(m.kind, 9)))
        return out

    async def load_project_messages(
        self, session_id: str, *, after_send_index: int | None = None
    ) -> list[ProjectMessageRow]:
        """Projection rows for a session in CONVERSATION order: by ``send_index``, and WITHIN a
        send ``user`` before ``project`` (a kind-alphabetical sort would put the assistant reply
        before the user turn — reversed). ``after_send_index`` (exclusive) returns only rows with
        ``send_index > after_send_index`` — the read cursor a ``compact`` row's ``compact_to_send``
        provides (everything newer than the latest fold)."""
        return await self._query_project_messages(self._db.fetchall, session_id, after_send_index)

    async def _query_latest_project_compact(
        self, fetchone: Callable[..., Any], session_id: str
    ) -> ProjectMessageRow | None:
        row = await fetchone(
            f"SELECT * FROM {self.t.project_messages} WHERE session_id=? AND kind='compact' "
            "ORDER BY send_index DESC LIMIT 1",
            (session_id,),
        )
        return _row_to_project_message(row) if row is not None else None

    async def latest_project_compact(self, session_id: str) -> ProjectMessageRow | None:
        """The most recent ``compact`` projection row (highest ``send_index``), or None.
        Its ``compact_to_send`` is the cursor for :meth:`load_project_messages`."""
        return await self._query_latest_project_compact(self._db.fetchone, session_id)

    async def write_send_projection_locked(
        self,
        session_id: str,
        *,
        send_index: int,
        rows: list[tuple[str, Any, str | None]],
        source_seq_lo: int | None,
        source_seq_hi: int | None,
        projector_version: int,
        plan_compaction: Callable[
            [ProjectMessageRow | None, list[ProjectMessageRow]],
            tuple[Any, str | None, int, int] | None,
        ],
    ) -> None:
        """Persist a finished send's projection ``rows`` AND an optional compact fold in ONE
        transaction, under the ``session_state`` row lock.

        Two guarantees:

        * **Atomic multi-row write** — all of a send's projection rows (user + project) commit
          together, so a crash can't leave the next-send reader a half-projected send.
        * **Serialized compaction across loops** — the lock makes two ``StatefulAgentLoop``
          instances sharing this store on the same session take turns; without it both could
          read the same pre-fold state, compute the same fold, and have the second UPSERT clobber
          the first (benign for the deterministic projector, divergent for a non-idempotent one).

        ``rows`` is ``(kind, content, rendered_text)`` per projection row. ``plan_compaction`` is
        called INSIDE the lock with ``(prior_compact, project_rows_after_prior)`` — a consistent
        snapshot taken after this send's rows are written — and returns
        ``(content, rendered_text, from_send, to_send)`` to fold, or ``None`` to skip. It MUST be
        pure (no I/O): it only runs the projector's in-memory trigger + ``compact()`` logic.
        """
        now = _now_ms()
        async with self._db.transaction() as tx:
            # Lock the session row first (PG/MySQL FOR UPDATE; SQLite's single-writer connection
            # already serializes) so the read-decide-write below is atomic against a concurrent loop.
            await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            for kind, content, rendered_text in rows:
                await self._upsert_project_message_tx(
                    tx, session_id, send_index=send_index, kind=kind, content=content,
                    rendered_text=rendered_text, source_seq_lo=source_seq_lo,
                    source_seq_hi=source_seq_hi, projector_version=projector_version, now=now,
                )
            prior = await self._query_latest_project_compact(tx.fetchone, session_id)
            cutoff = prior.compact_to_send if prior is not None else None
            proj_rows = await self._query_project_messages(tx.fetchall, session_id, cutoff)
            plan = plan_compaction(prior, proj_rows)
            if plan is not None:
                content, rendered_text, from_send, to_send = plan
                await self._upsert_project_message_tx(
                    tx, session_id, send_index=to_send, kind="compact", content=content,
                    rendered_text=rendered_text, compact_from_send=from_send,
                    compact_to_send=to_send, projector_version=projector_version, now=now,
                )


    # ── session lifecycle ─────────────────────────────────────────────────────
    async def list_children(self, parent_session_id: str) -> list[SessionRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.sessions} WHERE parent_session_id=? ORDER BY created_at",
            (parent_session_id,),
        )
        return [_row_to_session(r) for r in rows]

    async def archive_session(self, session_id: str) -> None:
        async with self._db.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.t.sessions} SET status=?, updated_at=? WHERE session_id=?",
                (SessionStatus.ARCHIVED.value, _now_ms(), session_id),
            )

    async def close_session(self, session_id: str, *, cascade: bool = True) -> int:
        """Physically delete the session's rows across all tables.

        With ``cascade=True`` (default), also deletes every descendant whose
        lifecycle is ``LINKED``. ``DETACHED`` descendants are preserved and
        re-parented to ``NULL``. Returns the number of sessions removed.
        """
        return len(await self.close_session_tree(session_id, cascade=cascade))

    async def close_session_tree(self, session_id: str, *, cascade: bool = True) -> list[str]:
        """Like :meth:`close_session`, but return the id of EVERY session deleted (the named
        session plus any cascaded ``LINKED`` descendants), so a caller can drop per-session
        in-memory bookkeeping (caches/locks/queues) for each. Re-parented ``DETACHED``
        descendants are NOT deleted and are NOT included."""
        async with self._db.transaction() as tx:
            deleted: list[str] = []
            await self._delete_session_tree(tx, session_id, cascade=cascade, deleted=deleted)
            return deleted

    async def _delete_session_tree(
        self, tx: Any, session_id: str, *, cascade: bool, deleted: list[str]
    ) -> None:
        if cascade:
            children = await tx.fetchall(
                f"SELECT session_id, lifecycle FROM {self.t.sessions} WHERE parent_session_id=?",
                (session_id,),
            )
            for child in children:
                if child["lifecycle"] == SubagentLifecycle.DETACHED.value:
                    await tx.execute(
                        f"UPDATE {self.t.sessions} SET parent_session_id=NULL WHERE session_id=?",
                        (child["session_id"],),
                    )
                else:
                    await self._delete_session_tree(
                        tx, child["session_id"], cascade=True, deleted=deleted
                    )
        await tx.execute(f"DELETE FROM {self.t.messages} WHERE session_id=?", (session_id,))
        await tx.execute(f"DELETE FROM {self.t.compactions} WHERE session_id=?", (session_id,))
        await tx.execute(f"DELETE FROM {self.t.usage_rounds} WHERE session_id=?", (session_id,))
        await tx.execute(f"DELETE FROM {self.t.session_stats} WHERE session_id=?", (session_id,))
        await tx.execute(f"DELETE FROM {self.t.timers} WHERE session_id=?", (session_id,))
        await tx.execute(
            f"DELETE FROM {self.t.session_runtime_state} WHERE session_id=?", (session_id,)
        )
        await tx.execute(
            f"DELETE FROM {self.t.background_tasks} WHERE session_id=?", (session_id,)
        )
        await tx.execute(f"DELETE FROM {self.t.notes} WHERE session_id=?", (session_id,))
        await tx.execute(
            f"DELETE FROM {self.t.project_messages} WHERE session_id=?", (session_id,)
        )
        await tx.execute(f"DELETE FROM {self.t.session_state} WHERE session_id=?", (session_id,))
        affected = await tx.execute(
            f"DELETE FROM {self.t.sessions} WHERE session_id=?", (session_id,)
        )
        if affected:  # the session row existed → count it as removed
            deleted.append(session_id)

    async def update_session_prompt(self, session_id: str, system_prompt: str | None) -> None:
        async with self._db.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.t.sessions} SET system_prompt=?, updated_at=? WHERE session_id=?",
                (system_prompt, _now_ms(), session_id),
            )

    # ── timers (durable wake-ups; see runtime/timers.py) ──────────────────

    async def create_timer(
        self, session_id: str, *, due_at: int, note: str, interval_s: int | None = None
    ) -> TimerRow:
        now = _now_ms()
        ivl = int(interval_s) if interval_s else None
        async with self._db.transaction() as tx:
            # Per-session MAX+1 id alloc. Serialize it against concurrent writers the
            # same way append_message does — take the session_state row lock (FOR UPDATE
            # on a server backend) so two concurrent create_timer calls can't read the
            # same MAX and collide on the (session_id, timer_id) PK. Legacy tolerated
            # timers without a session row, so a missing state row is not fatal here:
            # there is simply nothing to lock (single-writer SQLite is gap-free anyway).
            try:
                await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            except ValueError:
                pass
            row = await tx.fetchone(
                f"SELECT COALESCE(MAX(timer_id), 0) + 1 AS tid FROM {self.t.timers} "
                "WHERE session_id=?",
                (session_id,),
            )
            assert row is not None  # aggregate always returns one row
            timer_id = int(row["tid"])
            await tx.execute(
                f"INSERT INTO {self.t.timers} (session_id, timer_id, due_at, note, status, "
                "interval_s, created_at, updated_at) VALUES (?,?,?,?,'armed',?,?,?)",
                (session_id, timer_id, int(due_at), note, ivl, now, now),
            )
        return TimerRow(
            session_id=session_id,
            timer_id=timer_id,
            due_at=int(due_at),
            note=note,
            status="armed",
            interval_s=ivl,
            fire_count=0,
            last_fired_at=None,
            created_at=now,
            updated_at=now,
        )

    async def get_timer(self, session_id: str, timer_id: int) -> TimerRow | None:
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.timers} WHERE session_id=? AND timer_id=?",
            (session_id, int(timer_id)),
        )
        return _row_to_timer(row) if row is not None else None

    async def list_timers(
        self, session_id: str, *, statuses: tuple[str, ...] = ("armed", "firing")
    ) -> list[TimerRow]:
        marks = ",".join("?" for _ in statuses)
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.timers} WHERE session_id=? AND status IN ({marks}) "
            "ORDER BY due_at ASC",
            (session_id, *statuses),
        )
        return [_row_to_timer(r) for r in rows]

    async def due_timers(self, *, now: int | None = None, limit: int = 50) -> list[TimerRow]:
        """Armed timers whose due_at has passed, oldest first (cross-session scan)."""
        ts = _now_ms() if now is None else int(now)
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.timers} WHERE status='armed' AND due_at<=? "
            "ORDER BY due_at ASC LIMIT ?",
            (ts, int(limit)),
        )
        return [_row_to_timer(r) for r in rows]

    async def transition_timer(
        self,
        session_id: str,
        timer_id: int,
        *,
        from_status: str,
        to_status: str,
        due_at: int | None = None,
    ) -> bool:
        """Compare-and-set status transition (claims are race-free even with several
        runners on one store). Optionally moves due_at (postpone). Returns False when
        from_status no longer matches."""
        async with self._db.transaction() as tx:
            affected = await tx.execute(
                f"UPDATE {self.t.timers} SET status=?, due_at=COALESCE(?, due_at), "
                "updated_at=? WHERE session_id=? AND timer_id=? AND status=?",
                (to_status, due_at, _now_ms(), session_id, int(timer_id), from_status),
            )
        return affected > 0

    async def finish_firing_timer(self, session_id: str, timer_id: int) -> bool:
        """Complete a delivery: one-shot firing -> fired (due_at unchanged); recurring
        firing -> armed at now + interval (fixed-delay — missed periods collapse). Bumps
        fire_count / last_fired_at either way. CAS on status='firing': returns False if
        the caller no longer holds the firing claim."""
        now = _now_ms()
        async with self._db.transaction() as tx:
            affected = await tx.execute(
                f"UPDATE {self.t.timers} SET "
                "status = CASE WHEN interval_s IS NULL THEN 'fired' ELSE 'armed' END, "
                "due_at = CASE WHEN interval_s IS NULL THEN due_at "
                "              ELSE ? + interval_s * 1000 END, "
                "fire_count = fire_count + 1, last_fired_at = ?, updated_at = ? "
                "WHERE session_id=? AND timer_id=? AND status='firing'",
                (now, now, now, session_id, int(timer_id)),
            )
        return affected > 0

    async def heartbeat_firing_timer(self, session_id: str, timer_id: int) -> bool:
        """Re-stamp a 'firing' row's ``updated_at`` so a slow-but-live delivery isn't
        reclaimed as stale by :meth:`recover_stale_firing_timers` and double-fired.
        Returns False if the row is no longer 'firing' (already finished, re-armed, or
        cancelled) — the caller has lost the claim."""
        async with self._db.transaction() as tx:
            affected = await tx.execute(
                f"UPDATE {self.t.timers} SET updated_at=? "
                "WHERE session_id=? AND timer_id=? AND status='firing'",
                (_now_ms(), session_id, int(timer_id)),
            )
        return affected > 0

    async def recover_stale_firing_timers(self, *, older_than_ms: int) -> int:
        """Re-arm 'firing' rows that never finished (process died mid-fire), cross-session.
        At-least-once: a re-armed timer may deliver twice; the TIMER_FIRE hook is the place
        to dedupe if that matters. A *live* slow delivery keeps its row fresh via
        :meth:`heartbeat_firing_timer`, so only genuinely stuck rows (older than
        ``older_than_ms``) are reclaimed. Returns the number of rows reclaimed."""
        cutoff = _now_ms() - int(older_than_ms)
        async with self._db.transaction() as tx:
            affected = await tx.execute(
                f"UPDATE {self.t.timers} SET status='armed', updated_at=? "
                "WHERE status='firing' AND updated_at<?",
                (_now_ms(), cutoff),
            )
        return affected

    async def prune_timers(
        self,
        session_id: str,
        *,
        statuses: tuple[str, ...] = ("fired", "cancelled"),
        older_than_ms: int | None = None,
    ) -> int:
        """Delete timers in terminal ``statuses`` (default fired/cancelled), optionally
        only those whose ``updated_at`` is older than ``older_than_ms``. Armed/recurring
        timers in other statuses are never touched. Returns deletions."""
        if not statuses:
            return 0
        placeholders = ",".join("?" * len(statuses))
        sql = f"DELETE FROM {self.t.timers} WHERE session_id=? AND status IN ({placeholders})"
        params: list[Any] = [session_id, *statuses]
        if older_than_ms is not None:
            sql += " AND updated_at < ?"
            params.append(_now_ms() - int(older_than_ms))
        async with self._db.transaction() as tx:
            return await tx.execute(sql, params)

    # ── notes (agent-authored persistent memory) ──────────────────────────
    async def add_note(self, session_id: str, content: str, *, pinned: bool = False) -> NoteRow:
        """Insert a note with the next per-session ``note_id`` and return it. The
        ``COALESCE(MAX(note_id),0)+1`` allocation and the INSERT run in ONE transaction,
        serialized against concurrent writers by the session_state row lock (FOR UPDATE on
        a server backend) so two concurrent add_note calls can't read the same MAX and
        collide on the ``(session_id, note_id)`` composite PK. (Legacy does NOT check
        session existence here, so a missing state row is tolerated — nothing to lock.)"""
        now = _now_ms()
        async with self._db.transaction() as tx:
            try:
                await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            except ValueError:
                pass
            row = await tx.fetchone(
                f"SELECT COALESCE(MAX(note_id), 0) + 1 AS nid FROM {self.t.notes} "
                "WHERE session_id=?",
                (session_id,),
            )
            assert row is not None  # aggregate always returns one row
            note_id = int(row["nid"])
            await tx.execute(
                f"INSERT INTO {self.t.notes} "
                "(session_id, note_id, content, pinned, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, note_id, content, 1 if pinned else 0, now, now),
            )
        return NoteRow(
            session_id=session_id,
            note_id=note_id,
            content=content,
            pinned=pinned,
            created_at=now,
            updated_at=now,
        )

    async def update_note(
        self,
        session_id: str,
        note_id: int,
        *,
        content: str | None = None,
        pinned: bool | None = None,
    ) -> bool:
        """Update content and/or pinned flag (``updated_at`` always bumped). Returns
        ``False`` if the note doesn't exist (CAS via rowcount)."""
        sets: list[str] = ["updated_at=?"]
        params: list[Any] = [_now_ms()]
        if content is not None:
            sets.append("content=?")
            params.append(content)
        if pinned is not None:
            sets.append("pinned=?")
            params.append(1 if pinned else 0)
        params.extend([session_id, note_id])
        async with self._db.transaction() as tx:
            affected = await tx.execute(
                f"UPDATE {self.t.notes} SET {', '.join(sets)} "
                "WHERE session_id=? AND note_id=?",
                params,
            )
        return affected > 0

    async def delete_note(self, session_id: str, note_id: int) -> bool:
        async with self._db.transaction() as tx:
            affected = await tx.execute(
                f"DELETE FROM {self.t.notes} WHERE session_id=? AND note_id=?",
                (session_id, note_id),
            )
        return affected > 0

    async def list_notes(self, session_id: str) -> list[NoteRow]:
        """All notes for a session in ``note_id`` (= creation) order."""
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.notes} WHERE session_id=? ORDER BY note_id",
            (session_id,),
        )
        return [
            NoteRow(
                session_id=r["session_id"],
                note_id=int(r["note_id"]),
                content=r["content"],
                pinned=bool(r["pinned"]),
                created_at=int(r["created_at"]),
                updated_at=int(r["updated_at"]),
            )
            for r in rows
        ]

    async def count_notes(self, session_id: str) -> int:
        row = await self._db.fetchone(
            f"SELECT COUNT(*) AS n FROM {self.t.notes} WHERE session_id=?", (session_id,)
        )
        assert row is not None  # COUNT always returns one row
        return int(row["n"])

    # ── usage ─────────────────────────────────────────────────────────────
    async def record_usage(
        self,
        session_id: str,
        *,
        round_index: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        model: str | None = None,
    ) -> None:
        """Record per-round token usage. Legacy used ``INSERT OR REPLACE`` keyed on
        ``(session_id, round_index)``; since every non-key column is supplied, that is
        an overwrite-all upsert — all columns become ``val_cols``."""
        sql = self._db.dialect.upsert(
            self.t.usage_rounds,
            ("session_id", "round_index"),
            ("prompt_tokens", "completion_tokens", "total_tokens", "model", "created_at"),
        )
        async with self._db.transaction() as tx:
            await tx.execute(
                sql,
                (
                    session_id, round_index, prompt_tokens, completion_tokens,
                    total_tokens, model, _now_ms(),
                ),
            )

    # ── session statistics (cumulative, accounting-grade) ─────────────────
    async def bump_session_stats(
        self,
        session_id: str,
        usage: Mapping[str, Any],
        *,
        rounds: int = 0,
        tool_calls: int = 0,
    ) -> None:
        """Cumulative per-session accounting, bumped once per finished send. On conflict
        the seven counters ACCUMULATE (``col = col + new``), ``last_send_at``/``updated_at``
        OVERWRITE, and ``first_send_at`` is INSERT-ONLY (preserved on conflict, matching
        legacy which omits it from the ON CONFLICT SET). ``sends`` accumulates by 1."""
        now = _now_ms()
        sql = self._db.dialect.upsert(
            self.t.session_stats,
            ("session_id",),
            ("last_send_at", "updated_at"),
            add_cols=(
                "sends", "rounds", "llm_calls", "tool_calls",
                "prompt_tokens", "completion_tokens", "total_tokens",
            ),
            insert_only_cols=("first_send_at",),
        )
        # Param order matches the rendered column order:
        #   key_cols + val_cols + add_cols + insert_only_cols
        params = (
            session_id,                              # key: session_id
            now, now,                                # val: last_send_at, updated_at
            1,                                       # add: sends
            int(rounds),                             # add: rounds
            int(usage.get("calls") or 0),            # add: llm_calls
            int(tool_calls),                         # add: tool_calls
            int(usage.get("prompt_tokens") or 0),    # add: prompt_tokens
            int(usage.get("completion_tokens") or 0),  # add: completion_tokens
            int(usage.get("total_tokens") or 0),     # add: total_tokens
            now,                                     # insert-only: first_send_at
        )
        async with self._db.transaction() as tx:
            await tx.execute(sql, params)

    async def get_session_stats(self, session_id: str) -> SessionStatsRow | None:
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.session_stats} WHERE session_id=?", (session_id,)
        )
        return _row_to_stats(row) if row is not None else None

    async def list_session_stats(self) -> list[SessionStatsRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.session_stats} ORDER BY updated_at DESC"
        )
        return [_row_to_stats(r) for r in rows]

    # ── retention: usage rounds ───────────────────────────────────────────
    async def prune_usage_rounds(
        self,
        session_id: str,
        *,
        keep_last: int | None = None,
        older_than_ms: int | None = None,
    ) -> int:
        """Delete per-round usage accounting rows. ``keep_last`` retains the N most
        recent (by ``round_index``); ``older_than_ms`` prunes rows older than that age
        (by ``created_at``). Cumulative ``session_stats`` is untouched. Returns deletions
        (the DELETE's affected-row count)."""
        sql = f"DELETE FROM {self.t.usage_rounds} WHERE session_id=?"
        params: list[Any] = [session_id]
        if older_than_ms is not None:
            sql += " AND created_at < ?"
            params.append(_now_ms() - int(older_than_ms))
        if keep_last and keep_last > 0:
            # Wrap the ORDER BY/LIMIT subquery in a derived table: MySQL rejects both a
            # LIMIT inside IN(...) (err 1235) and modifying a table referenced in its own
            # subquery (err 1093); materializing it first is accepted by SQLite/PG/MySQL.
            sql += (
                f" AND round_index NOT IN (SELECT r FROM (SELECT round_index AS r "
                f"FROM {self.t.usage_rounds} "
                "WHERE session_id=? ORDER BY round_index DESC LIMIT ?) AS _keep)"
            )
            params += [session_id, int(keep_last)]
        async with self._db.transaction() as tx:
            return int(await tx.execute(sql, params))

    # ── runtime state ─────────────────────────────────────────────────────
    async def get_runtime_state(self, session_id: str, key: str, default: Any = None) -> Any:
        row = await self._db.fetchone(
            f"SELECT value_json FROM {self.t.session_runtime_state} "
            "WHERE session_id=? AND state_key=?",
            (session_id, key),
        )
        if row is None:
            return default
        value = _loads(row["value_json"])
        return default if value is None else value

    async def set_runtime_state(self, session_id: str, key: str, value: Any) -> None:
        now = _now_ms()
        async with self._db.transaction() as tx:
            exists = await tx.fetchone(
                f"SELECT 1 FROM {self.t.sessions} WHERE session_id=?", (session_id,)
            )
            if exists is None:
                raise ValueError(f"unknown session: {session_id}")
            sql = self._db.dialect.upsert(
                self.t.session_runtime_state,
                ("session_id", "state_key"),
                ("value_json", "updated_at"),
            )
            await tx.execute(sql, (session_id, key, _dumps(value), now))
            await tx.execute(
                f"UPDATE {self.t.sessions} SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )

    async def mutate_runtime_state(
        self,
        session_id: str,
        key: str,
        fn: Callable[[Any], Any],
        *,
        default: Any = None,
    ) -> Any:
        """Atomically read-modify-write a ``session_runtime_state`` value.

        ``fn(current)`` receives the deserialized current value (``default`` when the key
        is absent) and returns the new value to persist, or :data:`MUTATE_SKIP` to leave
        the row untouched. The whole read → ``fn`` → write runs under the session's row
        lock (``dialect.lock_state`` → ``SELECT … FOR UPDATE`` on a server engine; the
        SQLite backend already serializes writers), so concurrent mutators of the same key
        never clobber one another — unlike a bare ``get_runtime_state`` + ``set_runtime_state``
        pair, whose two awaits yield the event loop between read and write and so lose
        updates when parallel coroutines interleave.

        Locking the always-present ``session_state`` row (not the possibly-absent
        runtime-state row) also serializes the *first* writer of a brand-new key. ``fn``
        must be a plain (non-coroutine) callable. Returns the persisted new value, or the
        unchanged current value when ``fn`` skips. Raises ``ValueError`` for an unknown
        session."""
        now = _now_ms()
        async with self._db.transaction() as tx:
            await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            row = await tx.fetchone(
                f"SELECT value_json FROM {self.t.session_runtime_state} "
                "WHERE session_id=? AND state_key=?",
                (session_id, key),
            )
            current = default if row is None else _loads(row["value_json"])
            if current is None:
                current = default
            new_value = fn(current)
            if new_value is MUTATE_SKIP:
                return current
            sql = self._db.dialect.upsert(
                self.t.session_runtime_state,
                ("session_id", "state_key"),
                ("value_json", "updated_at"),
            )
            await tx.execute(sql, (session_id, key, _dumps(new_value), now))
            await tx.execute(
                f"UPDATE {self.t.sessions} SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )
        return new_value

    async def delete_runtime_state(self, session_id: str, key: str) -> None:
        async with self._db.transaction() as tx:
            await tx.execute(
                f"DELETE FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?",
                (session_id, key),
            )

    # ── shared_state: keyed JSON owned by an arbitrary scope (not a session) ──
    async def get_shared_state(self, owner: str, key: str, default: Any = None) -> Any:
        row = await self._db.fetchone(
            f"SELECT value_json FROM {self.t.shared_state} WHERE owner=? AND state_key=?",
            (owner, key),
        )
        if row is None:
            return default
        value = _loads(row["value_json"])
        return default if value is None else value

    async def set_shared_state(self, owner: str, key: str, value: Any) -> None:
        now = _now_ms()
        async with self._db.transaction() as tx:
            sql = self._db.dialect.upsert(
                self.t.shared_state,
                ("owner", "state_key"),
                ("value_json", "updated_at"),
            )
            await tx.execute(sql, (owner, key, _dumps(value), now))

    async def delete_shared_state(self, owner: str, key: str) -> None:
        async with self._db.transaction() as tx:
            await tx.execute(
                f"DELETE FROM {self.t.shared_state} WHERE owner=? AND state_key=?",
                (owner, key),
            )

    # ── background tasks ──────────────────────────────────────────────────
    async def upsert_background_task(
        self,
        session_id: str,
        *,
        task_id: str,
        command: str,
        status: str,
        return_code: int | None = None,
        output_tail: str | None = None,
        output_path: str | None = None,
    ) -> None:
        now = _now_ms()
        async with self._db.transaction() as tx:
            exists = await tx.fetchone(
                f"SELECT 1 FROM {self.t.sessions} WHERE session_id=?", (session_id,)
            )
            if exists is None:
                raise ValueError(f"unknown session: {session_id}")
            existing = await tx.fetchone(
                f"SELECT last_seen_at, created_at FROM {self.t.background_tasks} "
                "WHERE session_id=? AND task_id=?",
                (session_id, task_id),
            )
            created_at = now
            if existing is not None:
                created_at = int(existing["created_at"])
                # Monotonic bump: updated_at must advance past the last seen marker so
                # the row reappears in list_unseen_background_updates (updated_at >
                # last_seen_at) even if wall-clock ms did not move since mark_background_seen.
                now = max(now, int(existing["last_seen_at"]) + 1)
            # Upsert where the INSERT carries `last_seen_at` and `created_at` but the
            # conflict path must NOT overwrite either (last_seen_at is the reader's
            # cursor; created_at is immutable) — that's exactly `insert_only_cols`, so the
            # generic dialect.upsert expresses it backend-neutrally. Param order matches
            # the renderer: key_cols + val_cols + insert_only_cols.
            sql = self._db.dialect.upsert(
                self.t.background_tasks,
                ("session_id", "task_id"),
                ("command", "status", "return_code", "output_tail", "output_path", "updated_at"),
                insert_only_cols=("last_seen_at", "created_at"),
            )
            await tx.execute(
                sql,
                (
                    session_id, task_id, command, status, return_code,
                    output_tail, output_path, now, 0, created_at,
                ),
            )
            await tx.execute(
                f"UPDATE {self.t.sessions} SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )

    async def get_background_task(
        self, session_id: str, task_id: str
    ) -> BackgroundTaskRow | None:
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.background_tasks} WHERE session_id=? AND task_id=?",
            (session_id, task_id),
        )
        return _row_to_background_task(row) if row else None

    async def list_background_tasks(self, session_id: str) -> list[BackgroundTaskRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.background_tasks} WHERE session_id=? "
            "ORDER BY created_at ASC",
            (session_id,),
        )
        return [_row_to_background_task(r) for r in rows]

    async def list_unseen_background_updates(
        self, session_id: str
    ) -> list[BackgroundTaskRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.background_tasks} "
            "WHERE session_id=? AND updated_at > last_seen_at "
            "ORDER BY updated_at ASC",
            (session_id,),
        )
        return [_row_to_background_task(r) for r in rows]

    async def mark_background_seen(self, session_id: str, task_ids: list[str]) -> None:
        if not task_ids:
            return
        now = _now_ms()
        placeholders = ",".join("?" for _ in task_ids)
        async with self._db.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.t.background_tasks} SET last_seen_at=? "
                f"WHERE session_id=? AND task_id IN ({placeholders})",
                (now, session_id, *task_ids),
            )

    # ── retention & reclamation (OPS-2 / OPS-3) ───────────────────────────
    #
    # All retention is OPT-IN and caller-driven — the store never deletes on its own.
    # Pruning the folded ``compacted_out`` originals is IRREVERSIBLE and removes what
    # ``recall_compacted`` can surface; ``compact_note`` rows are ``state='active'`` and
    # are never touched, so ``load_active_messages`` is unaffected.

    async def prune_compacted_messages(
        self,
        session_id: str,
        *,
        older_than_ms: int | None = None,
        keep_recent: int = 0,
    ) -> int:
        """Delete folded-out (``state='compacted_out'``) message rows for a session.

        ``older_than_ms`` — only prune rows older than this many ms (by ``created_at``);
        ``None`` prunes regardless of age. ``keep_recent`` — always retain the N most
        recent compacted rows (by ``seq``). Returns the number of rows deleted.
        Irreversible; ``compact_note`` (active) rows are never deleted.
        """
        sql = f"DELETE FROM {self.t.messages} WHERE session_id=? AND state=?"
        params: list[Any] = [session_id, MessageState.COMPACTED_OUT.value]
        if older_than_ms is not None:
            sql += " AND created_at < ?"
            params.append(_now_ms() - int(older_than_ms))
        if keep_recent and keep_recent > 0:
            # Derived table (see prune_usage_rounds): MySQL rejects LIMIT-in-IN (1235) and
            # self-referential DELETE subqueries (1093); materializing first is portable.
            sql += (
                f" AND seq NOT IN (SELECT s FROM (SELECT seq AS s FROM {self.t.messages} "
                "WHERE session_id=? AND state=? ORDER BY seq DESC LIMIT ?) AS _keep)"
            )
            params += [session_id, MessageState.COMPACTED_OUT.value, int(keep_recent)]
        async with self._db.transaction() as tx:
            return int(await tx.execute(sql, params))

    # ── export / archival (OPS-4) ─────────────────────────────────────────

    async def export_session(self, session_id: str) -> dict[str, Any]:
        """Serialize a session's FULL durable state (the session row + all messages
        incl. compacted, compactions, usage rounds, runtime state, timers, notes, stats)
        into a JSON-serializable dict stamped with the current ``schema_version``.

        Pairs with :meth:`import_session` for archive-then-prune and cross-store moves.
        Raises ``ValueError`` for an unknown session.

        Backend-neutral: rows are read with an EXPLICIT column list per logical table
        (never ``SELECT *`` / column reflection), and keyed by the LOGICAL table name
        (unprefixed, e.g. ``"sessions"``) so an export round-trips across backends with
        different physical prefixes."""
        version = self.schema_version
        tables: dict[str, list[dict[str, Any]]] = {}
        for logical, physical, cols in _EXPORT_TABLES:
            collist = ",".join(cols)
            rows = await self._db.fetchall(
                f"SELECT {collist} FROM {physical(self.t)} WHERE session_id=?",
                (session_id,),
            )
            tables[logical] = [{c: r[c] for c in cols} for r in rows]
        if not tables["sessions"]:
            raise ValueError(f"unknown session: {session_id}")
        return {"schema_version": version, "session_id": session_id, "tables": tables}

    async def import_session(
        self, data: Mapping[str, Any], *, new_session_id: str | None = None
    ) -> str:
        """Insert a session previously produced by :meth:`export_session` under a new
        (or supplied) id, in one transaction. Returns the new ``session_id``.

        Refuses an export whose ``schema_version`` is newer than this build supports, or
        a target id that already exists. Columns absent from an older export default.

        An imported session is an INDEPENDENT root: its lineage columns
        (``parent_session_id`` / ``spawn_tool_call_id`` / ``spawn_depth`` / ``kind``)
        are reset rather than copied from the source. Otherwise a re-imported subagent
        would stay linked to the source's *original* parent and be silently
        cascade-deleted when that unrelated parent is closed — destroying the
        just-restored session (C2)."""
        version = int(data.get("schema_version", 0))
        if version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"export schema_version {version} is newer than this power_loop build "
                f"supports (max {CURRENT_SCHEMA_VERSION})"
            )
        tables = data["tables"]
        new_id = new_session_id or _new_session_id()
        async with self._db.transaction() as tx:
            if await tx.fetchone(
                f"SELECT 1 FROM {self.t.sessions} WHERE session_id=?", (new_id,)
            ):
                raise ValueError(f"session already exists: {new_id}")
            for logical, physical, cols in _EXPORT_TABLES:
                for raw_row in tables.get(logical, []):
                    row = dict(raw_row)
                    row["session_id"] = new_id
                    if logical == "sessions":
                        # Detach lineage so the import is an independent root (see
                        # docstring), never silently re-linked to the source's parent.
                        row["parent_session_id"] = None
                        row["spawn_tool_call_id"] = None
                        row["spawn_depth"] = 0
                        row["kind"] = SessionKind.ROOT.value
                    # Only insert columns the export actually carries (an older export
                    # may omit columns added later); the rest take their schema defaults.
                    present = [c for c in cols if c in row]
                    collist = ",".join(present)
                    placeholders = ",".join("?" * len(present))
                    await tx.execute(
                        f"INSERT INTO {physical(self.t)} ({collist}) "
                        f"VALUES ({placeholders})",
                        [row[c] for c in present],
                    )
        return new_id

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
        created_at=row["created_at"], send_index=row["send_index"],
    )


def _row_to_project_message(row: Row) -> ProjectMessageRow:
    return ProjectMessageRow(
        session_id=row["session_id"], send_index=row["send_index"], kind=row["kind"],
        content=_loads(row["content_json"]), rendered_text=row["rendered_text"],
        source_seq_lo=row["source_seq_lo"], source_seq_hi=row["source_seq_hi"],
        compact_from_send=row["compact_from_send"], compact_to_send=row["compact_to_send"],
        projector_version=row["projector_version"], token_estimate=row["token_estimate"],
        created_at=row["created_at"],
    )


def _logical_order_key(m: MessageRow) -> tuple[int, int]:
    if m.name == "compact_note":
        ord_val = m.meta.get("ord")
        if ord_val is not None:
            return (int(ord_val), m.seq)
    return (m.seq, m.seq)


def _row_to_timer(row: Row) -> TimerRow:
    return TimerRow(
        session_id=row["session_id"], timer_id=int(row["timer_id"]), due_at=int(row["due_at"]),
        note=row["note"], status=row["status"],
        interval_s=(int(row["interval_s"]) if row["interval_s"] is not None else None),
        fire_count=int(row["fire_count"]),
        last_fired_at=(int(row["last_fired_at"]) if row["last_fired_at"] is not None else None),
        created_at=int(row["created_at"]), updated_at=int(row["updated_at"]),
    )


def _row_to_stats(row: Row) -> SessionStatsRow:
    return SessionStatsRow(
        session_id=row["session_id"], sends=row["sends"], rounds=row["rounds"],
        llm_calls=row["llm_calls"], tool_calls=row["tool_calls"],
        prompt_tokens=row["prompt_tokens"], completion_tokens=row["completion_tokens"],
        total_tokens=row["total_tokens"], first_send_at=row["first_send_at"],
        last_send_at=row["last_send_at"], updated_at=row["updated_at"],
    )


def _row_to_background_task(row: Row) -> BackgroundTaskRow:
    return BackgroundTaskRow(
        session_id=row["session_id"], task_id=row["task_id"], command=row["command"],
        status=row["status"], return_code=row["return_code"], output_tail=row["output_tail"],
        output_path=row["output_path"], last_seen_at=int(row["last_seen_at"]),
        created_at=int(row["created_at"]), updated_at=int(row["updated_at"]),
    )


__all__ = [
    "SessionStore",
    "DEFAULT_TABLE_PREFIX",
    "DEFAULT_MAX_SPAWN_DEPTH",
    "MAX_SPAWN_DEPTH",
    "DEFAULT_DB_PATH",
]
