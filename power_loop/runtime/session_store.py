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
import logging
import queue
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./power_loop_sessions.db"
#: Default ceiling on sub-agent nesting (a chain of at most this many levels).
#: This is only the *default*; the effective limit is per-:class:`SessionStore`
#: (``store.max_spawn_depth``), configurable via :meth:`SessionStore.open` or the
#: ``max_spawn_depth`` argument of ``StatefulAgentLoop``.
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

-- Keyed JSON owned by an arbitrary scope key (NOT a session): the blackboard
-- and other cross-session shared state. No session existence check / no cascade
-- with any session — the owner is opaque (e.g. a workflow run id, a board id).
CREATE TABLE IF NOT EXISTS shared_state (
    owner             TEXT NOT NULL,
    key               TEXT NOT NULL,
    value_json        TEXT,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (owner, key)
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

CREATE TABLE IF NOT EXISTS session_stats (
    session_id        TEXT PRIMARY KEY,
    sends             INTEGER NOT NULL DEFAULT 0,
    rounds            INTEGER NOT NULL DEFAULT 0,
    llm_calls         INTEGER NOT NULL DEFAULT 0,
    tool_calls        INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    first_send_at     INTEGER,
    last_send_at      INTEGER,
    updated_at        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS timers (
    session_id        TEXT NOT NULL,
    timer_id          INTEGER NOT NULL,
    due_at            INTEGER NOT NULL,
    note              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'armed',
    interval_s        INTEGER,
    fire_count        INTEGER NOT NULL DEFAULT 0,
    last_fired_at     INTEGER,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, timer_id)
);
CREATE INDEX IF NOT EXISTS idx_timers_due ON timers (status, due_at);

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
class TimerRow:
    """A durable per-session wake-up. One-shot (interval_s is None):
    armed -> firing -> fired | cancelled. Recurring (interval_s set):
    armed -> firing -> armed again at fire-time + interval (fixed-delay,
    missed periods while down collapse into one) until cancelled.
    The row is the source of truth; in-process scheduling is just an
    accelerator over it."""

    session_id: str
    timer_id: int
    due_at: int  # epoch ms
    note: str
    status: str
    interval_s: int | None
    fire_count: int
    last_fired_at: int | None
    created_at: int
    updated_at: int


@dataclass
class SessionStatsRow:
    """Cumulative per-session accounting (bumped once per finished send)."""

    session_id: str
    sends: int
    rounds: int
    llm_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    first_send_at: int | None
    last_send_at: int | None
    updated_at: int


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


#: Current on-disk schema version, stamped into ``PRAGMA user_version``. Bump this
#: AND append a ``(N, fn)`` step to :data:`MIGRATIONS` for ANY change to ``SCHEMA_SQL``
#: (a new column/index/table or a backfill) so existing ``.db`` files are upgraded on
#: open instead of silently keeping the old shape (``CREATE TABLE IF NOT EXISTS`` never
#: alters an existing table).
CURRENT_SCHEMA_VERSION = 1


def _migration_0001_timers_columns(conn: sqlite3.Connection) -> None:
    """Additive ``timers`` columns for stores created before they existed. Idempotent
    (ALTERs only the columns still missing), so it safely upgrades a legacy
    ``user_version=0`` DB — this folds in the old hardcoded ``_micro_migrate``."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(timers)")}
    for name, ddl in (
        ("interval_s", "ALTER TABLE timers ADD COLUMN interval_s INTEGER"),
        ("fire_count", "ALTER TABLE timers ADD COLUMN fire_count INTEGER NOT NULL DEFAULT 0"),
        ("last_fired_at", "ALTER TABLE timers ADD COLUMN last_fired_at INTEGER"),
    ):
        if name not in cols:
            conn.execute(ddl)


#: Ordered migration ladder: ``(target_version, fn)``. Each ``fn`` brings a DB from
#: ``target_version - 1`` to ``target_version`` and MUST be idempotent (guard with
#: ``PRAGMA table_info`` / ``sqlite_master`` checks) so a crashed/retried upgrade and a
#: legacy ``user_version=0`` DB both converge. Append, never reorder or mutate shipped steps.
MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (1, _migration_0001_timers_columns),
)

#: Per-session tables (all keyed by ``session_id``) captured by
#: :meth:`SessionStore.export_session` / replayed by ``import_session``. ``sessions``
#: is first so the parent row lands before its children. ``shared_state`` is excluded
#: (owner-scoped, not session-scoped); ``background_tasks`` is excluded (transient).
_EXPORT_TABLES: tuple[str, ...] = (
    "sessions",
    "session_state",
    "messages",
    "compactions",
    "usage_rounds",
    "session_runtime_state",
    "timers",
    "notes",
    "session_stats",
)


def _apply_migrations(conn: sqlite3.Connection, *, is_fresh: bool) -> None:
    """Bring an opened DB to :data:`CURRENT_SCHEMA_VERSION` via the ``PRAGMA
    user_version`` ladder.

    * A *fresh* DB (``SCHEMA_SQL`` just created the latest shape) is stamped directly.
    * An *existing* DB runs every :data:`MIGRATIONS` step with ``target > user_version``,
      in order, inside one transaction (a failing step rolls the DDL back and the version
      is NOT advanced — the next open() retries the idempotent steps).
    * A DB whose ``user_version`` is *newer* than this build supports is refused, rather
      than silently operated on with a mismatched code path.
    """
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if is_fresh:
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
        return
    if current > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"session store schema version {current} is newer than this power_loop "
            f"build supports (max {CURRENT_SCHEMA_VERSION}); upgrade power_loop to open "
            "this database."
        )
    if current >= CURRENT_SCHEMA_VERSION:
        return  # already current — nothing to do
    with conn:  # atomic DDL: a failing step rolls back, version stays unchanged
        for target, fn in MIGRATIONS:
            if target > current:
                fn(conn)
    # Stamp only after every step committed, so a crash before this leaves the version
    # at `current` and the idempotent steps simply re-run on the next open().
    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _new_session_id() -> str:
    return "sess_" + secrets.token_hex(12)


def _coerce_max_spawn_depth(value: int) -> int:
    """Validate a spawn-depth ceiling: a positive int (≥1). Raises ValueError."""
    try:
        depth = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"max_spawn_depth must be an int ≥ 1, got {value!r}") from None
    if depth < 1:
        raise ValueError(f"max_spawn_depth must be ≥ 1, got {depth}")
    return depth


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

    def __init__(
        self,
        conn: sqlite3.Connection,
        path: str,
        *,
        max_spawn_depth: int = MAX_SPAWN_DEPTH,
        read_pool_size: int = 0,
    ) -> None:
        self._conn = conn
        self._lock = threading.RLock()
        self.path = path
        self.max_spawn_depth = max_spawn_depth  # validated by the property setter
        # SCALE-2: optional pool of read-only connections. The single writer (self._conn
        # under self._lock) stays the sole source of truth for next_seq + write atomicity;
        # WAL lets these readers run concurrently with it WITHOUT taking the write lock, so
        # large concurrent reads (e.g. many sessions loading history) stop serializing
        # behind one writer. query_only=ON makes them physically unable to write. Disabled
        # by default; impossible for ":memory:" (each connect is a SEPARATE database).
        self._read_pool: queue.Queue[sqlite3.Connection] | None = None
        if read_pool_size and read_pool_size > 0:
            if path == ":memory:":
                logger.warning(
                    "read_pool_size=%d ignored for an in-memory store: each :memory: "
                    "connection is a separate database, so a pooled reader could not see "
                    "the writer's data. Reads use the write connection.", read_pool_size,
                )
            else:
                pool: queue.Queue[sqlite3.Connection] = queue.Queue()
                for _ in range(int(read_pool_size)):
                    rc = sqlite3.connect(path, check_same_thread=False, isolation_level="")  # type: ignore[call-overload]
                    rc.row_factory = sqlite3.Row
                    rc.execute("PRAGMA busy_timeout=5000")
                    rc.execute("PRAGMA query_only=ON")  # belt-and-suspenders: cannot write
                    pool.put(rc)
                self._read_pool = pool

    @contextmanager
    def _read_cursor(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for a read-only query.

        With a read pool, checks one out (concurrent reads, no contention on the write
        RLock) and returns it after; otherwise yields the write connection under the
        RLock (the default — identical to the historical behavior). A pooled reader in
        WAL mode sees every transaction committed before the read begins, which is all a
        post-write load needs."""
        pool = self._read_pool
        if pool is not None:
            rc = pool.get()
            try:
                yield rc
            finally:
                pool.put(rc)
        else:
            with self._lock:
                yield self._conn

    @property
    def max_spawn_depth(self) -> int:
        """Ceiling on sub-agent nesting depth for sessions created by this store.

        A per-process config knob, not per-request state: set it once before
        the store is used concurrently. (It is read in ``create_session``
        outside the DB lock, so mutating it while spawns are in flight is a
        last-writer-wins ordering question — harmless on a set-once value.)
        """
        return self._max_spawn_depth

    @max_spawn_depth.setter
    def max_spawn_depth(self, value: int) -> None:
        self._max_spawn_depth = _coerce_max_spawn_depth(value)

    # ── lifecycle ─────────────────────────────────────────────────────────

    @classmethod
    def open(
        cls,
        path: str | Path = DEFAULT_DB_PATH,
        *,
        max_spawn_depth: int = MAX_SPAWN_DEPTH,
        read_pool_size: int = 0,
    ) -> SessionStore:
        """Open (creating if needed) the SQLite-backed store at ``path``.

        ``max_spawn_depth`` caps how deeply sub-agents may nest (a chain of at
        most this many levels); it is enforced in :meth:`create_session` and
        pre-checked in ``run_agent_spec``. Defaults to :data:`MAX_SPAWN_DEPTH`.
        The limit lives on the store instance (not in the DB), so it is a
        property of *this* process's handle, not persisted.

        ``read_pool_size`` (SCALE-2, default 0 = off) opens that many extra read-only
        connections so reads run concurrently with the single writer instead of
        serializing behind its lock — worth it under read-heavy fan-out. Ignored for
        ``:memory:`` (connections can't share an in-memory database).
        """
        path_str = str(path)
        # ":memory:" stays as-is; file paths get expanded.
        if path_str != ":memory:":
            p = Path(path_str).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            path_str = str(p)
        # isolation_level="" (deferred) — NOT None (autocommit). With autocommit,
        # `with self._conn:` is a no-op (no transaction is ever open), so the
        # multi-statement WRITES (append_message's INSERT + next_seq bump,
        # create_session's two inserts, record_compaction, the cascade delete, …)
        # were not atomic: a failure between statements left corrupt state (e.g. a
        # message persisted but next_seq not bumped → permanent UNIQUE wedge).
        # Deferred mode auto-BEGINs before the first DML, so the `with` block's
        # commit/rollback makes each method's multi-statement WRITE atomic.
        #
        # NOTE on the read-modify-write methods (SELECT next_seq → INSERT → UPDATE):
        # deferred mode begins the transaction at the first DML, NOT at the leading
        # SELECT, so that SELECT reads in autocommit. The read-modify-write is still
        # atomic IN-PROCESS because every accessor holds ``self._lock`` (an RLock)
        # across the whole `with self._lock, self._conn:` block, serializing readers
        # and writers — that lock, not the transaction, is what prevents an
        # interleaved next_seq race. Cross-process safety is NOT this store's model
        # (one file = one process); a stray second writer would instead be caught by
        # the messages ``(session_id, seq)`` primary key raising IntegrityError.
        # isolation_level="" is the legacy deferred sentinel (auto-BEGIN before DML);
        # typeshed only types the Literal[...]|None variants, so this overload is a
        # stub gap, not a runtime issue.
        conn = sqlite3.connect(  # type: ignore[call-overload]
            path_str, check_same_thread=False, isolation_level=""
        )
        conn.row_factory = sqlite3.Row
        # auto_vacuum must be chosen BEFORE the database's first table is created and
        # before WAL initializes the header — hence the very first pragma. On an EXISTING
        # file it is a harmless no-op (changing it would need a full VACUUM), so set it
        # unconditionally; fresh stores thus get INCREMENTAL free-page reclamation,
        # reclaimable cheaply via vacuum(incremental=True) after prune/close_session.
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        # Detect a brand-new DB (no tables yet) so the migration ladder can stamp a fresh
        # file at CURRENT_SCHEMA_VERSION vs. upgrade an existing one. ":memory:" is always
        # fresh (new connection per open()).
        is_fresh = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()[0] == 0
        conn.executescript(SCHEMA_SQL)
        _apply_migrations(conn, is_fresh=is_fresh)
        return cls(
            conn, path_str, max_spawn_depth=max_spawn_depth, read_pool_size=read_pool_size
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            if self._read_pool is not None:
                while not self._read_pool.empty():
                    try:
                        self._read_pool.get_nowait().close()
                    except queue.Empty:  # pragma: no cover - guarded by empty()
                        break
                self._read_pool = None

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
        from the parent and enforced against this store's ``max_spawn_depth``
        (default :data:`MAX_SPAWN_DEPTH`).
        """
        spawn_depth = 0
        if parent_session_id is not None:
            parent = self.get_session(parent_session_id)
            if parent is None:
                raise ValueError(f"parent session not found: {parent_session_id}")
            spawn_depth = parent.spawn_depth + 1
            if spawn_depth > self.max_spawn_depth:
                raise ValueError(
                    f"spawn depth {spawn_depth} exceeds max {self.max_spawn_depth}"
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
        self._conn.execute("DELETE FROM session_stats WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM timers WHERE session_id=?", (session_id,))
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
        """Return active messages in **logical** order.

        Ordinarily ``seq`` *is* logical order, but a ``compact_note`` is allocated
        a fresh high identity ``seq`` while logically belonging where the folded
        range began (it summarizes older turns). Ordering purely by ``seq`` would
        sink the note below the kept tail it should precede. We therefore order by
        each row's logical position — ``meta['ord']`` for a ``compact_note`` (set
        by :meth:`record_compaction`), else its ``seq`` — with ``seq`` as a stable
        tiebreak. A note that predates the ``ord`` field falls back to its ``seq``.
        """
        with self._read_cursor() as c:
            rows = c.execute(
                """
                SELECT * FROM messages
                WHERE session_id=? AND state=?
                ORDER BY seq ASC
                """,
                (session_id, MessageState.ACTIVE.value),
            ).fetchall()
        messages = [_row_to_message(r) for r in rows]
        messages.sort(key=_logical_order_key)
        return messages

    def load_all_messages(self, session_id: str) -> list[MessageRow]:
        with self._read_cursor() as c:
            rows = c.execute(
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
        fold_seqs: list[int] | None = None,
        order_key: int | None = None,
    ) -> tuple[int, int]:
        """Mark the folded messages ``compacted_out`` and append a
        ``compact_note`` message in one transaction.

        Two marking modes:

        * ``fold_seqs`` given — mark the **exact set** of those seqs. This is
          the correct mode and what :class:`SQLiteSink` uses. It is robust when
          the in-memory history's index→seq map is non-monotonic: a previous
          ``compact_note`` carries a high *identity* ``seq`` at a low *logical*
          position, so a contiguous ``BETWEEN from_seq AND to_seq`` over those
          translated seqs could **invert** (``from_seq > to_seq`` → mark nothing
          while the in-memory fold proceeds → in-memory/DB divergence) or sweep
          active kept-tail rows. An explicit ``seq IN (…)`` set folds precisely
          what was folded, with no ordering assumption.
        * ``fold_seqs`` omitted — legacy ``BETWEEN from_seq AND to_seq`` range
          (kept for direct callers/tests that fold a known contiguous range).

        ``order_key`` is the note's **logical** position (decoupled from its
        identity ``seq``, which is monotonic-allocated and high): the note logically
        sits where the folded range began, not where its row id lands. It is stored
        as ``meta['ord']`` and used by :meth:`load_active_messages` to order the
        note correctly on reload. Defaults to the span's low end.

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

            if fold_seqs is not None:
                uniq = [int(s) for s in dict.fromkeys(fold_seqs)]
                if uniq:
                    placeholders = ",".join("?" * len(uniq))
                    self._conn.execute(
                        f"UPDATE messages SET state=? "
                        f"WHERE session_id=? AND seq IN ({placeholders})",
                        (MessageState.COMPACTED_OUT.value, session_id, *uniq),
                    )
                span_from = min(uniq) if uniq else from_seq
                span_to = max(uniq) if uniq else to_seq
            else:
                self._conn.execute(
                    "UPDATE messages SET state=? WHERE session_id=? AND seq BETWEEN ? AND ?",
                    (MessageState.COMPACTED_OUT.value, session_id, from_seq, to_seq),
                )
                span_from, span_to = from_seq, to_seq

            note_ord = int(order_key) if order_key is not None else int(span_from)
            meta = dict(note_meta or {})
            meta.update({
                "compacted_at_round": round_index,
                "from_seq": span_from,
                "to_seq": span_to,
                "ord": note_ord,
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
                    session_id, compact_seq, note_seq, span_from, span_to,
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

    # ── retention & reclamation (OPS-2 / OPS-3) ───────────────────────────
    #
    # All retention is OPT-IN and caller-driven — the store never deletes on its own
    # (matching its "only place that writes, nothing magic" contract). Pruning the
    # folded ``compacted_out`` originals is IRREVERSIBLE and removes what
    # ``recall_compacted`` can surface; ``compact_note`` rows are ``state='active'``
    # and are never touched, so ``load_active_messages`` is unaffected.

    def prune_compacted_messages(
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
        sql = "DELETE FROM messages WHERE session_id=? AND state=?"
        params: list[Any] = [session_id, MessageState.COMPACTED_OUT.value]
        if older_than_ms is not None:
            sql += " AND created_at < ?"
            params.append(_now_ms() - int(older_than_ms))
        if keep_recent and keep_recent > 0:
            sql += (
                " AND seq NOT IN (SELECT seq FROM messages WHERE session_id=? AND state=? "
                "ORDER BY seq DESC LIMIT ?)"
            )
            params += [session_id, MessageState.COMPACTED_OUT.value, int(keep_recent)]
        with self._lock, self._conn:
            return int(self._conn.execute(sql, params).rowcount)

    def prune_usage_rounds(
        self,
        session_id: str,
        *,
        keep_last: int | None = None,
        older_than_ms: int | None = None,
    ) -> int:
        """Delete per-round usage accounting rows. ``keep_last`` retains the N most
        recent (by ``round_index``); ``older_than_ms`` prunes rows older than that age
        (by ``created_at``). Cumulative ``session_stats`` is untouched. Returns deletions."""
        sql = "DELETE FROM usage_rounds WHERE session_id=?"
        params: list[Any] = [session_id]
        if older_than_ms is not None:
            sql += " AND created_at < ?"
            params.append(_now_ms() - int(older_than_ms))
        if keep_last and keep_last > 0:
            sql += (
                " AND round_index NOT IN (SELECT round_index FROM usage_rounds "
                "WHERE session_id=? ORDER BY round_index DESC LIMIT ?)"
            )
            params += [session_id, int(keep_last)]
        with self._lock, self._conn:
            return int(self._conn.execute(sql, params).rowcount)

    def prune_timers(
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
        sql = f"DELETE FROM timers WHERE session_id=? AND status IN ({placeholders})"
        params: list[Any] = [session_id, *statuses]
        if older_than_ms is not None:
            sql += " AND updated_at < ?"
            params.append(_now_ms() - int(older_than_ms))
        with self._lock, self._conn:
            return int(self._conn.execute(sql, params).rowcount)

    def checkpoint(self, *, mode: str = "TRUNCATE") -> None:
        """Run a WAL checkpoint (default ``TRUNCATE`` — flush the -wal back into the
        main file and shrink the -wal to zero). Cheap; safe to call periodically."""
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            raise ValueError(f"invalid checkpoint mode: {mode!r}")
        with self._lock:
            self._conn.execute(f"PRAGMA wal_checkpoint({mode})")

    def vacuum(self, *, incremental: bool = True) -> None:
        """Reclaim free pages. ``incremental=True`` runs ``PRAGMA incremental_vacuum``
        (cheap, only effective when the DB was created with ``auto_vacuum=INCREMENTAL``
        — the default for fresh stores); ``incremental=False`` runs a full ``VACUUM``
        (rewrites the whole file: reclaims everything but is blocking and needs free
        disk). Call after :meth:`prune_compacted_messages` / :meth:`close_session` to
        actually shrink the file on disk."""
        with self._lock:
            if incremental:
                self._conn.execute("PRAGMA incremental_vacuum")
            else:
                # VACUUM cannot run inside a transaction; this connection is in deferred
                # mode and no `with self._conn:` is open here, so we are in autocommit.
                self._conn.execute("VACUUM")

    # ── export / archival (OPS-4) ─────────────────────────────────────────

    def export_session(self, session_id: str) -> dict[str, Any]:
        """Serialize a session's FULL durable state (the session row + all messages
        incl. compacted, compactions, usage rounds, runtime state, timers, notes, stats)
        into a JSON-serializable dict stamped with the current ``schema_version``.

        Pairs with :meth:`import_session` for archive-then-prune and cross-store moves.
        Raises ``ValueError`` for an unknown session."""
        with self._lock:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            tables: dict[str, list[dict[str, Any]]] = {}
            for table in _EXPORT_TABLES:
                rows = self._conn.execute(
                    f"SELECT * FROM {table} WHERE session_id=?", (session_id,)  # noqa: S608 (fixed table list)
                ).fetchall()
                tables[table] = [dict(r) for r in rows]
        if not tables["sessions"]:
            raise ValueError(f"unknown session: {session_id}")
        return {"schema_version": version, "session_id": session_id, "tables": tables}

    def import_session(
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
        just-restored session (C2).
        """
        version = int(data.get("schema_version", 0))
        if version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"export schema_version {version} is newer than this power_loop build "
                f"supports (max {CURRENT_SCHEMA_VERSION})"
            )
        tables = data["tables"]
        new_id = new_session_id or _new_session_id()
        with self._lock, self._conn:
            if self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (new_id,)
            ).fetchone():
                raise ValueError(f"session already exists: {new_id}")
            for table in _EXPORT_TABLES:
                for raw_row in tables.get(table, []):
                    row = dict(raw_row)
                    row["session_id"] = new_id
                    if table == "sessions":
                        # Detach lineage so the import is an independent root (see
                        # docstring), never silently re-linked to the source's parent.
                        row["parent_session_id"] = None
                        row["spawn_tool_call_id"] = None
                        row["spawn_depth"] = 0
                        row["kind"] = SessionKind.ROOT.value
                    cols = list(row.keys())
                    collist = ",".join(cols)
                    placeholders = ",".join("?" * len(cols))
                    self._conn.execute(
                        f"INSERT INTO {table} ({collist}) VALUES ({placeholders})",  # noqa: S608
                        [row[c] for c in cols],
                    )
        return new_id

    def backup(self, dest_path: str | Path) -> None:
        """Write a consistent snapshot of the WHOLE store to ``dest_path`` using SQLite's
        online backup API (safe while the store is in use). Overwrites ``dest_path``."""
        dest = sqlite3.connect(str(dest_path))
        try:
            with self._lock:
                self._conn.backup(dest)
        finally:
            dest.close()

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

    # ── shared_state: keyed JSON owned by an arbitrary scope (not a session) ──

    def get_shared_state(self, owner: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM shared_state WHERE owner=? AND key=?",
                (owner, key),
            ).fetchone()
        if row is None:
            return default
        value = _loads(row["value_json"])
        return default if value is None else value

    def set_shared_state(self, owner: str, key: str, value: Any) -> None:
        now = _now_ms()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO shared_state (owner, key, value_json, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(owner, key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (owner, key, _dumps(value), now),
            )

    def delete_shared_state(self, owner: str, key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM shared_state WHERE owner=? AND key=?",
                (owner, key),
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


    # ── session statistics (cumulative, accounting-grade) ────────────────
    # ``usage_rounds`` is keyed (session, round_index) and round indexes
    # restart at 0 every send, so it overwrites itself across sends — fine
    # for inspecting the latest run, useless for accounting. This table is
    # the cumulative source of truth, bumped once per finished send.

    def bump_session_stats(
        self,
        session_id: str,
        usage: Mapping[str, Any],
        *,
        rounds: int = 0,
        tool_calls: int = 0,
    ) -> None:
        now = _now_ms()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO session_stats (
                    session_id, sends, rounds, llm_calls, tool_calls,
                    prompt_tokens, completion_tokens, total_tokens,
                    first_send_at, last_send_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    sends             = sends + 1,
                    rounds            = rounds + excluded.rounds,
                    llm_calls         = llm_calls + excluded.llm_calls,
                    tool_calls        = tool_calls + excluded.tool_calls,
                    prompt_tokens     = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    total_tokens      = total_tokens + excluded.total_tokens,
                    last_send_at      = excluded.last_send_at,
                    updated_at        = excluded.updated_at
                """,
                (
                    session_id, 1,
                    int(rounds),
                    int(usage.get("calls") or 0),
                    int(tool_calls),
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    int(usage.get("total_tokens") or 0),
                    now, now, now,
                ),
            )

    def get_session_stats(self, session_id: str) -> SessionStatsRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_stats WHERE session_id=?", (session_id,)
            ).fetchone()
        return _row_to_stats(row) if row is not None else None

    def list_session_stats(self) -> list[SessionStatsRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_stats ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_stats(r) for r in rows]

    # ── timers (durable wake-ups; see runtime/timers.py) ──────────────────

    def create_timer(
        self, session_id: str, *, due_at: int, note: str, interval_s: int | None = None
    ) -> TimerRow:
        now = _now_ms()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(timer_id), 0) + 1 AS tid FROM timers WHERE session_id=?",
                (session_id,),
            ).fetchone()
            timer_id = int(row["tid"])
            self._conn.execute(
                """
                INSERT INTO timers (session_id, timer_id, due_at, note, status,
                                    interval_s, created_at, updated_at)
                VALUES (?,?,?,?, 'armed', ?, ?, ?)
                """,
                (session_id, timer_id, int(due_at), note,
                 int(interval_s) if interval_s else None, now, now),
            )
        return TimerRow(session_id, timer_id, int(due_at), note, "armed",
                        int(interval_s) if interval_s else None, 0, None, now, now)

    def get_timer(self, session_id: str, timer_id: int) -> TimerRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM timers WHERE session_id=? AND timer_id=?",
                (session_id, int(timer_id)),
            ).fetchone()
        return _row_to_timer(row) if row is not None else None

    def list_timers(
        self, session_id: str, *, statuses: tuple[str, ...] = ("armed", "firing")
    ) -> list[TimerRow]:
        marks = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM timers WHERE session_id=? AND status IN ({marks}) "
                "ORDER BY due_at ASC",
                (session_id, *statuses),
            ).fetchall()
        return [_row_to_timer(r) for r in rows]

    def due_timers(self, *, now: int | None = None, limit: int = 50) -> list[TimerRow]:
        """Armed timers whose due_at has passed, oldest first."""
        ts = _now_ms() if now is None else int(now)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM timers WHERE status='armed' AND due_at<=? "
                "ORDER BY due_at ASC LIMIT ?",
                (ts, int(limit)),
            ).fetchall()
        return [_row_to_timer(r) for r in rows]

    def transition_timer(
        self,
        session_id: str,
        timer_id: int,
        *,
        from_status: str,
        to_status: str,
        due_at: int | None = None,
    ) -> bool:
        """Compare-and-set status transition (claims are race-free even with
        several runners on one store within the process). Optionally moves
        due_at (postpone). Returns False when from_status no longer matches."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE timers SET status=?, due_at=COALESCE(?, due_at), updated_at=?
                WHERE session_id=? AND timer_id=? AND status=?
                """,
                (to_status, due_at, _now_ms(), session_id, int(timer_id), from_status),
            )
            return cur.rowcount > 0

    def finish_firing_timer(self, session_id: str, timer_id: int) -> bool:
        """Complete a delivery: one-shot firing -> fired; recurring firing ->
        armed at now + interval (fixed-delay — missed periods collapse).
        Bumps fire_count / last_fired_at either way."""
        now = _now_ms()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE timers SET
                    status = CASE WHEN interval_s IS NULL THEN 'fired' ELSE 'armed' END,
                    due_at = CASE WHEN interval_s IS NULL THEN due_at
                                  ELSE ? + interval_s * 1000 END,
                    fire_count = fire_count + 1,
                    last_fired_at = ?,
                    updated_at = ?
                WHERE session_id=? AND timer_id=? AND status='firing'
                """,
                (now, now, now, session_id, int(timer_id)),
            )
            return cur.rowcount > 0

    def heartbeat_firing_timer(self, session_id: str, timer_id: int) -> bool:
        """Re-stamp a 'firing' row's ``updated_at`` so a slow-but-live delivery
        isn't reclaimed as stale by :meth:`recover_stale_firing_timers` and
        double-fired. Returns False if the row is no longer 'firing' (already
        finished, re-armed, or cancelled) — the caller has lost the claim."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE timers SET updated_at=? "
                "WHERE session_id=? AND timer_id=? AND status='firing'",
                (_now_ms(), session_id, int(timer_id)),
            )
            return cur.rowcount > 0

    def recover_stale_firing_timers(self, *, older_than_ms: int) -> int:
        """Re-arm 'firing' rows that never finished (process died mid-fire).
        At-least-once: a re-armed timer may deliver twice; the TIMER_FIRE hook
        is the place to dedupe if that matters. A *live* slow delivery keeps its
        row fresh via :meth:`heartbeat_firing_timer`, so only genuinely stuck
        rows (older than ``older_than_ms``) are reclaimed."""
        cutoff = _now_ms() - int(older_than_ms)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE timers SET status='armed', updated_at=? "
                "WHERE status='firing' AND updated_at<?",
                (_now_ms(), cutoff),
            )
            return cur.rowcount


def _row_to_timer(row: sqlite3.Row) -> TimerRow:
    return TimerRow(
        session_id=row["session_id"],
        timer_id=row["timer_id"],
        due_at=row["due_at"],
        note=row["note"],
        status=row["status"],
        interval_s=row["interval_s"],
        fire_count=row["fire_count"],
        last_fired_at=row["last_fired_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_stats(row: sqlite3.Row) -> SessionStatsRow:
    return SessionStatsRow(
        session_id=row["session_id"],
        sends=row["sends"],
        rounds=row["rounds"],
        llm_calls=row["llm_calls"],
        tool_calls=row["tool_calls"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        total_tokens=row["total_tokens"],
        first_send_at=row["first_send_at"],
        last_send_at=row["last_send_at"],
        updated_at=row["updated_at"],
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


def _logical_order_key(m: MessageRow) -> tuple[int, int]:
    """Sort key putting each active message at its logical position.

    A ``compact_note`` logically sits where its folded range began
    (``meta['ord']``), not at its high identity ``seq``. Everything else sorts
    by ``seq``. ``seq`` is the stable tiebreak.
    """
    if m.name == "compact_note":
        ord_val = m.meta.get("ord")
        if ord_val is not None:
            return (int(ord_val), m.seq)
    return (m.seq, m.seq)


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
