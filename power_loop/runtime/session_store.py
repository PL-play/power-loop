"""SQLite-backed session store for power-loop.

Owns the persistence layer for stateful agent loops:
  - sessions       : metadata + parent linking + spawn depth + lifecycle
  - messages       : ordered (session_id, seq) message log with state flag
  - compactions    : audit of every compaction (which seq range → which note)
  - usage_rounds   : per-round token usage
  - session_state  : single-row per-session mutable state (next_seq, pending, …)
  - session_runtime_state : per-session JSON state for tool/runtime protocols
  - background_tasks : persisted background command task status
  - notes            : agent-authored persistent notes (self-managed memory)

The store is the **only** place that writes to disk. Callers (StatefulAgentLoop,
the Sink that the pipeline drives, the subagent runtime) all go through here.

Threading
---------
A single :class:`sqlite3.Connection` is opened with ``check_same_thread=False``
and guarded by a re-entrant lock. SQLite is set to WAL so concurrent readers
from other processes still work. For asyncio callers, wrap calls in
``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "./power_loop_sessions.db"
MAX_SPAWN_DEPTH = 3


class SessionKind(str, Enum):
    ROOT = "root"
    SUBAGENT = "subagent"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class SubagentLifecycle(str, Enum):
    """How long a subagent's session persists relative to its parent."""

    EPHEMERAL = "ephemeral"  # delete child immediately after it returns
    LINKED = "linked"        # keep, cascade-delete when parent is closed
    DETACHED = "detached"    # keep, independent of parent's lifecycle


class MessageState(str, Enum):
    ACTIVE = "active"
    COMPACTED_OUT = "compacted_out"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    created_at          INTEGER NOT NULL,
    updated_at          INTEGER NOT NULL,
    system_prompt       TEXT,
    model               TEXT,
    config_json         TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    kind                TEXT NOT NULL DEFAULT 'root',
    parent_session_id   TEXT,
    spawn_tool_call_id  TEXT,
    spawn_depth         INTEGER NOT NULL DEFAULT 0,
    lifecycle           TEXT NOT NULL DEFAULT 'ephemeral',
    metadata_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_parent
    ON sessions(parent_session_id);

CREATE TABLE IF NOT EXISTS messages (
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    name            TEXT,
    content         TEXT,
    tool_calls_json TEXT,
    tool_call_id    TEXT,
    round_index     INTEGER,
    state           TEXT NOT NULL DEFAULT 'active',
    meta_json       TEXT,
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_session_state
    ON messages(session_id, state, seq);

CREATE TABLE IF NOT EXISTS compactions (
    session_id      TEXT NOT NULL,
    compact_seq     INTEGER NOT NULL,
    note_seq        INTEGER NOT NULL,
    from_seq        INTEGER NOT NULL,
    to_seq          INTEGER NOT NULL,
    before_tokens   INTEGER,
    after_tokens    INTEGER,
    round_index     INTEGER,
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (session_id, compact_seq)
);

CREATE TABLE IF NOT EXISTS usage_rounds (
    session_id        TEXT NOT NULL,
    round_index       INTEGER NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    model             TEXT,
    created_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, round_index)
);

CREATE TABLE IF NOT EXISTS session_state (
    session_id        TEXT PRIMARY KEY,
    next_seq          INTEGER NOT NULL DEFAULT 1,
    round_index       INTEGER NOT NULL DEFAULT 0,
    last_compact_seq  INTEGER NOT NULL DEFAULT 0,
    pending_json      TEXT
);

CREATE TABLE IF NOT EXISTS session_runtime_state (
    session_id        TEXT NOT NULL,
    key               TEXT NOT NULL,
    value_json        TEXT,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, key)
);

CREATE TABLE IF NOT EXISTS background_tasks (
    session_id        TEXT NOT NULL,
    task_id           TEXT NOT NULL,
    command           TEXT NOT NULL,
    status            TEXT NOT NULL,
    return_code       INTEGER,
    output_tail       TEXT,
    output_path       TEXT,
    last_seen_at      INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_background_tasks_session_status
    ON background_tasks(session_id, status, updated_at);

CREATE TABLE IF NOT EXISTS notes (
    session_id        TEXT NOT NULL,
    note_id           INTEGER NOT NULL,
    content           TEXT NOT NULL,
    pinned            INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, note_id)
);
"""


@dataclass
class SessionRow:
    session_id: str
    created_at: int
    updated_at: int
    system_prompt: str | None
    model: str | None
    config: dict[str, Any]
    status: SessionStatus
    kind: SessionKind
    parent_session_id: str | None
    spawn_tool_call_id: str | None
    spawn_depth: int
    lifecycle: SubagentLifecycle
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageRow:
    session_id: str
    seq: int
    role: str
    name: str | None
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None
    round_index: int | None
    state: MessageState
    meta: dict[str, Any]
    created_at: int


@dataclass
class SessionStateRow:
    session_id: str
    next_seq: int
    round_index: int
    last_compact_seq: int
    pending: dict[str, Any] | None


@dataclass
class CompactionRow:
    session_id: str
    compact_seq: int
    note_seq: int
    from_seq: int
    to_seq: int
    before_tokens: int | None
    after_tokens: int | None
    round_index: int | None
    created_at: int


@dataclass
class BackgroundTaskRow:
    session_id: str
    task_id: str
    command: str
    status: str
    return_code: int | None
    output_tail: str | None
    output_path: str | None
    last_seen_at: int
    created_at: int
    updated_at: int


@dataclass
class NoteRow:
    """One agent-authored note. ``note_id`` is a short per-session integer so
    the model can reference notes naturally ("update note 3")."""

    session_id: str
    note_id: int
    content: str
    pinned: bool
    created_at: int
    updated_at: int


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


class SessionStore:
    """Owns the SQLite connection and exposes typed CRUD over the schema."""

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self._lock = threading.RLock()
        self.path = path

    # ── lifecycle ─────────────────────────────────────────────────────────

    @classmethod
    def open(cls, path: str | Path = DEFAULT_DB_PATH) -> SessionStore:
        path_str = str(path)
        # ":memory:" stays as-is; file paths get expanded.
        if path_str != ":memory:":
            p = Path(path_str).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            path_str = str(p)
        conn = sqlite3.connect(path_str, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(SCHEMA_SQL)
        return cls(conn, path_str)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── session CRUD ──────────────────────────────────────────────────────

    def create_session(
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
        """Insert a new session row and its empty session_state row.

        For subagents, supply ``parent_session_id``; ``spawn_depth`` is computed
        from the parent and enforced against :data:`MAX_SPAWN_DEPTH`.
        """
        spawn_depth = 0
        if parent_session_id is not None:
            parent = self.get_session(parent_session_id)
            if parent is None:
                raise ValueError(f"parent session not found: {parent_session_id}")
            spawn_depth = parent.spawn_depth + 1
            if spawn_depth > MAX_SPAWN_DEPTH:
                raise ValueError(
                    f"spawn depth {spawn_depth} exceeds max {MAX_SPAWN_DEPTH}"
                )
            kind = SessionKind.SUBAGENT

        sid = session_id or _new_session_id()
        now = _now_ms()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    session_id, created_at, updated_at,
                    system_prompt, model, config_json,
                    status, kind, parent_session_id, spawn_tool_call_id,
                    spawn_depth, lifecycle, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sid, now, now,
                    system_prompt, model, _dumps(config or {}),
                    SessionStatus.ACTIVE.value, kind.value,
                    parent_session_id, spawn_tool_call_id,
                    spawn_depth, lifecycle.value,
                    _dumps(metadata or {}),
                ),
            )
            self._conn.execute(
                "INSERT INTO session_state (session_id) VALUES (?)",
                (sid,),
            )
        return sid

    def get_session(self, session_id: str) -> SessionRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return _row_to_session(row) if row else None

    def list_children(self, parent_session_id: str) -> list[SessionRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE parent_session_id=? ORDER BY created_at",
                (parent_session_id,),
            ).fetchall()
        return [_row_to_session(r) for r in rows]

    def archive_session(self, session_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET status=?, updated_at=? WHERE session_id=?",
                (SessionStatus.ARCHIVED.value, _now_ms(), session_id),
            )

    def close_session(self, session_id: str, *, cascade: bool = True) -> int:
        """Physically delete the session's rows across all tables.

        With ``cascade=True`` (default), also deletes every descendant whose
        lifecycle is ``LINKED``. ``DETACHED`` descendants are preserved and
        re-parented to ``NULL``. Returns the number of sessions removed.
        """
        with self._lock, self._conn:
            return self._delete_session_tree(session_id, cascade=cascade)

    def _delete_session_tree(self, session_id: str, *, cascade: bool) -> int:
        deleted = 0
        if cascade:
            children = self._conn.execute(
                "SELECT session_id, lifecycle FROM sessions WHERE parent_session_id=?",
                (session_id,),
            ).fetchall()
            for child in children:
                if child["lifecycle"] == SubagentLifecycle.DETACHED.value:
                    self._conn.execute(
                        "UPDATE sessions SET parent_session_id=NULL WHERE session_id=?",
                        (child["session_id"],),
                    )
                else:
                    deleted += self._delete_session_tree(child["session_id"], cascade=True)
        self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM compactions WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM usage_rounds WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM session_runtime_state WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM background_tasks WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM notes WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM session_state WHERE session_id=?", (session_id,))
        cur = self._conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        deleted += cur.rowcount
        return deleted

    def update_session_prompt(self, session_id: str, system_prompt: str | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET system_prompt=?, updated_at=? WHERE session_id=?",
                (system_prompt, _now_ms(), session_id),
            )

    # ── messages ──────────────────────────────────────────────────────────

    def append_message(
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
        """Append one message and return its allocated ``seq``.

        Allocation is atomic: ``session_state.next_seq`` is read+incremented
        under the same transaction as the INSERT.
        """
        now = _now_ms()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT next_seq FROM session_state WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown session: {session_id}")
            seq = int(row["next_seq"])
            self._conn.execute(
                """
                INSERT INTO messages (
                    session_id, seq, role, name, content, tool_calls_json,
                    tool_call_id, round_index, state, meta_json, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id, seq, role, name, content,
                    _dumps(tool_calls) if tool_calls else None,
                    tool_call_id, round_index,
                    MessageState.ACTIVE.value,
                    _dumps(meta or {}), now,
                ),
            )
            self._conn.execute(
                "UPDATE session_state SET next_seq=? WHERE session_id=?",
                (seq + 1, session_id),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )
        return seq

    def load_active_messages(self, session_id: str) -> list[MessageRow]:
        """Return all messages in ``state='active'`` order by seq."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id=? AND state=?
                ORDER BY seq ASC
                """,
                (session_id, MessageState.ACTIVE.value),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def load_all_messages(self, session_id: str) -> list[MessageRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY seq ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def get_message(self, session_id: str, seq: int) -> MessageRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM messages WHERE session_id=? AND seq=?",
                (session_id, seq),
            ).fetchone()
        return _row_to_message(row) if row else None

    # ── compaction ────────────────────────────────────────────────────────

    def record_compaction(
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
    ) -> tuple[int, int]:
        """Mark ``[from_seq, to_seq]`` as ``compacted_out`` and append a
        ``compact_note`` message in one transaction.

        Returns ``(compact_seq, note_seq)``.
        """
        now = _now_ms()
        with self._lock, self._conn:
            state_row = self._conn.execute(
                "SELECT next_seq, last_compact_seq FROM session_state WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if state_row is None:
                raise ValueError(f"unknown session: {session_id}")
            note_seq = int(state_row["next_seq"])
            compact_seq = int(state_row["last_compact_seq"]) + 1

            self._conn.execute(
                "UPDATE messages SET state=? WHERE session_id=? AND seq BETWEEN ? AND ?",
                (MessageState.COMPACTED_OUT.value, session_id, from_seq, to_seq),
            )
            meta = dict(note_meta or {})
            meta.update({
                "compacted_at_round": round_index,
                "from_seq": from_seq,
                "to_seq": to_seq,
                "original_tokens": before_tokens,
                "summary_tokens": after_tokens,
            })
            self._conn.execute(
                """
                INSERT INTO messages (
                    session_id, seq, role, name, content, tool_calls_json,
                    tool_call_id, round_index, state, meta_json, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id, note_seq, "system", "compact_note", note_content,
                    None, None, round_index,
                    MessageState.ACTIVE.value, _dumps(meta), now,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO compactions (
                    session_id, compact_seq, note_seq, from_seq, to_seq,
                    before_tokens, after_tokens, round_index, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id, compact_seq, note_seq, from_seq, to_seq,
                    before_tokens, after_tokens, round_index, now,
                ),
            )
            self._conn.execute(
                """
                UPDATE session_state
                SET next_seq=?, last_compact_seq=?
                WHERE session_id=?
                """,
                (note_seq + 1, compact_seq, session_id),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )
        return compact_seq, note_seq

    def list_compactions(self, session_id: str) -> list[CompactionRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM compactions WHERE session_id=? ORDER BY compact_seq",
                (session_id,),
            ).fetchall()
        return [
            CompactionRow(
                session_id=r["session_id"],
                compact_seq=r["compact_seq"],
                note_seq=r["note_seq"],
                from_seq=r["from_seq"],
                to_seq=r["to_seq"],
                before_tokens=r["before_tokens"],
                after_tokens=r["after_tokens"],
                round_index=r["round_index"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── state ─────────────────────────────────────────────────────────────

    def get_state(self, session_id: str) -> SessionStateRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_state WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return SessionStateRow(
            session_id=row["session_id"],
            next_seq=row["next_seq"],
            round_index=row["round_index"],
            last_compact_seq=row["last_compact_seq"],
            pending=_loads(row["pending_json"]),
        )

    def set_round_index(self, session_id: str, round_index: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE session_state SET round_index=? WHERE session_id=?",
                (round_index, session_id),
            )

    def set_pending(self, session_id: str, pending: dict[str, Any] | None) -> None:
        """Mark a session as having unresolved tool_calls (or clear)."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE session_state SET pending_json=? WHERE session_id=?",
                (_dumps(pending) if pending else None, session_id),
            )

    # ── runtime state ─────────────────────────────────────────────────────

    def get_runtime_state(self, session_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT value_json FROM session_runtime_state
                WHERE session_id=? AND key=?
                """,
                (session_id, key),
            ).fetchone()
        if row is None:
            return default
        value = _loads(row["value_json"])
        return default if value is None else value

    def set_runtime_state(self, session_id: str, key: str, value: Any) -> None:
        now = _now_ms()
        with self._lock, self._conn:
            if self.get_session(session_id) is None:
                raise ValueError(f"unknown session: {session_id}")
            self._conn.execute(
                """
                INSERT INTO session_runtime_state (session_id, key, value_json, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(session_id, key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (session_id, key, _dumps(value), now),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )

    def delete_runtime_state(self, session_id: str, key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM session_runtime_state WHERE session_id=? AND key=?",
                (session_id, key),
            )

    # ── background tasks ──────────────────────────────────────────────────

    def upsert_background_task(
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
        with self._lock, self._conn:
            if self.get_session(session_id) is None:
                raise ValueError(f"unknown session: {session_id}")
            existing = self._conn.execute(
                """
                SELECT last_seen_at, created_at FROM background_tasks
                WHERE session_id=? AND task_id=?
                """,
                (session_id, task_id),
            ).fetchone()
            created_at = now
            if existing is not None:
                created_at = int(existing["created_at"])
                now = max(now, int(existing["last_seen_at"]) + 1)
            self._conn.execute(
                """
                INSERT INTO background_tasks (
                    session_id, task_id, command, status, return_code,
                    output_tail, output_path, last_seen_at, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, task_id) DO UPDATE SET
                    command=excluded.command,
                    status=excluded.status,
                    return_code=excluded.return_code,
                    output_tail=excluded.output_tail,
                    output_path=excluded.output_path,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    task_id,
                    command,
                    status,
                    return_code,
                    output_tail,
                    output_path,
                    0,
                    created_at,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )

    def get_background_task(self, session_id: str, task_id: str) -> BackgroundTaskRow | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM background_tasks
                WHERE session_id=? AND task_id=?
                """,
                (session_id, task_id),
            ).fetchone()
        return _row_to_background_task(row) if row else None

    def list_background_tasks(self, session_id: str) -> list[BackgroundTaskRow]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM background_tasks
                WHERE session_id=?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [_row_to_background_task(r) for r in rows]

    def list_unseen_background_updates(self, session_id: str) -> list[BackgroundTaskRow]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM background_tasks
                WHERE session_id=? AND updated_at > last_seen_at
                ORDER BY updated_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [_row_to_background_task(r) for r in rows]

    def mark_background_seen(self, session_id: str, task_ids: list[str]) -> None:
        if not task_ids:
            return
        now = _now_ms()
        placeholders = ",".join("?" for _ in task_ids)
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                UPDATE background_tasks
                SET last_seen_at=?
                WHERE session_id=? AND task_id IN ({placeholders})
                """,
                (now, session_id, *task_ids),
            )

    # ── notes (agent-authored persistent memory) ──────────────────────────

    def add_note(self, session_id: str, content: str, *, pinned: bool = False) -> NoteRow:
        """Insert a note with the next per-session note_id and return it."""
        now = _now_ms()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(note_id), 0) + 1 AS nid FROM notes WHERE session_id=?",
                (session_id,),
            ).fetchone()
            note_id = int(row["nid"])
            self._conn.execute(
                """
                INSERT INTO notes (session_id, note_id, content, pinned, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
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

    def update_note(
        self,
        session_id: str,
        note_id: int,
        *,
        content: str | None = None,
        pinned: bool | None = None,
    ) -> bool:
        """Update content and/or pinned flag. Returns False if the note doesn't exist."""
        sets: list[str] = ["updated_at=?"]
        params: list[Any] = [_now_ms()]
        if content is not None:
            sets.append("content=?")
            params.append(content)
        if pinned is not None:
            sets.append("pinned=?")
            params.append(1 if pinned else 0)
        params.extend([session_id, note_id])
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"UPDATE notes SET {', '.join(sets)} WHERE session_id=? AND note_id=?",
                params,
            )
        return cur.rowcount > 0

    def delete_note(self, session_id: str, note_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM notes WHERE session_id=? AND note_id=?",
                (session_id, note_id),
            )
        return cur.rowcount > 0

    def list_notes(self, session_id: str) -> list[NoteRow]:
        """All notes for a session in note_id (= creation) order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM notes WHERE session_id=? ORDER BY note_id",
                (session_id,),
            ).fetchall()
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

    def count_notes(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM notes WHERE session_id=?", (session_id,)
            ).fetchone()
        return int(row["n"])

    # ── usage ─────────────────────────────────────────────────────────────

    def record_usage(
        self,
        session_id: str,
        *,
        round_index: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        model: str | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO usage_rounds (
                    session_id, round_index, prompt_tokens, completion_tokens,
                    total_tokens, model, created_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    session_id, round_index, prompt_tokens, completion_tokens,
                    total_tokens, model, _now_ms(),
                ),
            )


def _row_to_session(row: sqlite3.Row) -> SessionRow:
    return SessionRow(
        session_id=row["session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        system_prompt=row["system_prompt"],
        model=row["model"],
        config=_loads(row["config_json"]) or {},
        status=SessionStatus(row["status"]),
        kind=SessionKind(row["kind"]),
        parent_session_id=row["parent_session_id"],
        spawn_tool_call_id=row["spawn_tool_call_id"],
        spawn_depth=row["spawn_depth"],
        lifecycle=SubagentLifecycle(row["lifecycle"]),
        metadata=_loads(row["metadata_json"]) or {},
    )


def _row_to_message(row: sqlite3.Row) -> MessageRow:
    return MessageRow(
        session_id=row["session_id"],
        seq=row["seq"],
        role=row["role"],
        name=row["name"],
        content=row["content"],
        tool_calls=_loads(row["tool_calls_json"]),
        tool_call_id=row["tool_call_id"],
        round_index=row["round_index"],
        state=MessageState(row["state"]),
        meta=_loads(row["meta_json"]) or {},
        created_at=row["created_at"],
    )


def _row_to_background_task(row: sqlite3.Row) -> BackgroundTaskRow:
    return BackgroundTaskRow(
        session_id=row["session_id"],
        task_id=row["task_id"],
        command=row["command"],
        status=row["status"],
        return_code=row["return_code"],
        output_tail=row["output_tail"],
        output_path=row["output_path"],
        last_seen_at=row["last_seen_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
