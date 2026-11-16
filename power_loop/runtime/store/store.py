"""Backend-neutral async ``SessionStore`` facade.

Written ONCE against the ``Database`` + ``Dialect`` ports; the backend (SQLite today,
PostgreSQL/MySQL next) is an implementation detail. Every multi-statement write runs in
one transaction so it is atomic on any engine; the per-session ``seq`` counter is
allocated under the dialect's ``lock_state`` so it is correct even with multiple writers.

Schema 17 storage: sessions + lifecycle/cascade,
messages + transactional seq allocation, compaction, session_state, timers, notes,
usage/stats, runtime + shared state, background tasks, retention/prune, and
export/import. SQLite-only maintenance (checkpoint/vacuum/backup) is an optional
:class:`~power_loop.runtime.store.capabilities.Maintenance` capability that no-ops on
backends that lack it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from power_loop.contracts.errors import InboxClaimChanged, SessionLeaseLost
from power_loop.runtime.message_ref import integer_id
from power_loop.runtime.store import change_signals, message_usage, send_reads
from power_loop.runtime.store.capabilities import Maintenance
from power_loop.runtime.store.db import Database, Row, Transaction
from power_loop.runtime.store.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaPolicy,
    ensure_schema,
    validate_table_prefix,
)
from power_loop.runtime.store.types import (
    CompactionRow,
    HookEventRow,
    MessageRow,
    MessageState,
    NoteRow,
    ProjectMessageRow,
    SessionRow,
    SessionStateRow,
    SessionStatsRow,
    TimerRow,
)

DEFAULT_TABLE_PREFIX = "pl_"
DEFAULT_MAX_SPAWN_DEPTH = 3


@dataclass(frozen=True)
class ProjectionWindow:
    """One bounded history-entry read: latest compact, live projection rows, first send read.

    ``window_lo`` is the first send index that must be present in ``rows``; a gap inside
    ``[window_lo, current_send_index)`` is a missing projection and must be healed or stopped.
    """

    compact: ProjectMessageRow | None
    rows: list[ProjectMessageRow]
    window_lo: int

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


_KEEP = object()
_TOUCH_INTERVAL_MS = 1000


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _new_session_id() -> str:
    return "sess_" + secrets.token_hex(12)


#: 🔴 Postgres 的 text/jsonb **存不下 0x00**（`invalid byte sequence for encoding "UTF8": 0x00`），
#: 这是硬约束不是配置。而模型的输出偶尔真的会带一个 NUL 进来——真实事故：一次正常的
#: assistant 回复里夹了 0x00，落库直接抛，**整个 run 失败**，用户那边看到的是会话自己停了。
#: 所以在落库这个汇点统一剔除：一个控制字符不值得让一整程工作丢掉。
#: 只去 NUL，别的字符原样保留（制表符、换行、各种 Unicode 都是合法内容）。
def _nul_safe(v: str | None) -> str | None:
    return v.replace("\x00", "") if isinstance(v, str) and "\x00" in v else v


def _dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return _nul_safe(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


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
        self.session_state = f"{prefix}session_state"
        self.session_runtime_state = f"{prefix}session_runtime_state"
        self.project_messages = f"{prefix}project_messages"
        self.follow_up_queue = f"{prefix}follow_up_queue"


# Logical schema 17 export. Derived projections and hook audit bodies are excluded.
_EXPORT_TABLES: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("sessions", lambda t: t.sessions, (
        "session_id", "parent_session_id", "created_by_session_id", "session_kind",
        "descriptor_json", "descriptor_hash", "creation_identity_hash",
        "creation_call_ref_json", "creation_tool_call_id", "start_input_message_id",
        "tenant_id", "principal_id", "result_route_json", "cancel_propagation",
        "lifecycle", "created_at", "updated_at")),
    ("session_state", lambda t: t.session_state, (
        "session_id", "next_seq", "send_index", "active_send_index", "execution_state",
        "owner_epoch", "owner_token", "owner_node_id", "owner_boot_id",
        "owner_lease_until_ms", "inbox_version", "pending_json", "cancel_send_index",
        "cancel_reason", "round_index", "last_compact_seq", "background_version",
        "runtime_version")),
    ("messages", lambda t: t.messages, (
        "id", "session_id", "seq", "role", "name", "content", "tool_calls_json",
        "tool_call_id", "round_index", "state", "meta_json", "send_index", "created_at")),
    ("follow_up_queue", lambda t: t.follow_up_queue, (
        "id", "session_id", "item_id", "kind", "priority", "status", "content_json",
        "meta_json", "request_hash", "result_kind", "created_at", "expires_at",
        "delivered_at", "delivered_seq", "delivered_send_index", "terminal_status",
        "result_seq", "result_send_index", "error_code", "error_detail",
        "result_delivery_error", "completed_at", "continuation_of", "continuation_index")),
    ("session_runtime_state", lambda t: t.session_runtime_state, (
        "session_id", "state_key", "value_json", "updated_at")),
    ("project_messages", lambda t: t.project_messages, (
        "id", "session_id", "send_index", "kind", "content_json", "rendered_text",
        "source_seq_lo", "source_seq_hi", "completed_through_send_index",
        "compact_from_send", "compact_to_send", "projector_version", "token_estimate",
        "created_at")),
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
        self._schema_version = CURRENT_SCHEMA_VERSION
        self.store_id: str | None = None
        # Identity of rows this process appended. A message's id/role/indices never
        # change after insert and seqs are never reused, so a ref can be served
        # without rereading the row; deletes in this process drop the session.
        self._appended_refs: OrderedDict[tuple[str, int], tuple] = OrderedDict()
        self._session_holdings: OrderedDict[str, send_reads.SessionHolding] = OrderedDict()

    def _holding(self, session_id):
        scoped = send_reads.ACTIVE.get()
        if scoped is not None and scoped.store is self and scoped.sid == session_id:
            return scoped
        holding = self._session_holdings.get(session_id)
        if holding is None:
            holding = send_reads.SessionHolding(self, session_id)
            self._session_holdings[session_id] = holding
        self._session_holdings.move_to_end(session_id)
        while len(self._session_holdings) > 4096:
            self._session_holdings.popitem(last=False)
        return holding

    def known_pending(self, session_id):
        """``(True, pending)`` when this holding committed the pending state."""
        holding = self._holding(session_id)
        return holding.pending_known, _loads(holding.pending_json) if holding.pending_known else None

    def forget_pending(self, session_id):
        holding = self._holding(session_id)
        holding.pending_known, holding.pending_json = False, None
        send_reads.forget(self, session_id, "state")

    def defer_round_index(self, session_id: str, round_index: int) -> None:
        """Written with the round's first append instead of its own transaction."""
        send_reads.lease_writer(self, session_id)
        self._holding(session_id).deferred_round = int(round_index)

    def _remember_appended(self, session_id, seq, row_id, role, tool_call_id, send_index,
                           round_index, tool_calls_json):
        key = (session_id, seq)
        self._appended_refs[key] = (row_id, role, tool_call_id, send_index, round_index)
        reads = send_reads.current(self, session_id)
        if reads is not None and role == "assistant":
            reads.assistant_send_index = send_index
            reads.assistant_ref = (row_id, seq, role, tool_call_id, send_index, round_index)
        self._appended_refs.move_to_end(key)
        while len(self._appended_refs) > 4096:
            self._appended_refs.popitem(last=False)
        if tool_calls_json is not None:
            self._holding(session_id).appended_call = (seq, tool_calls_json)

    def _forget_appended(self, session_id):
        self._holding(session_id).change_snapshot = None
        send_reads.forget(self, session_id, "session", "state")
        reads = send_reads.current(self, session_id)
        if reads is not None:
            reads.assistant_send_index, reads.assistant_ref = None, None
        self._holding(session_id).appended_call = None
        for key in [k for k in self._appended_refs if k[0] == session_id]:
            del self._appended_refs[key]

    def appended_tool_calls(self, session_id: str, seq: int) -> str | None:
        """Stored tool_calls JSON of the latest assistant row this process appended."""
        found = self._holding(session_id).appended_call
        return found[1] if found is not None and found[0] == seq else None

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
        ) -> SessionStore:
        """Open a SQLite-backed store (the default backend). For Postgres/MySQL use
        :func:`power_loop.runtime.store.factory.open_store` with a DSN.

        ``schema`` is a :class:`SchemaPolicy` (default AUTO_CREATE); VERIFY never writes DDL."""
        from power_loop.runtime.store.backends.sqlite import SqliteDatabase

        db = await SqliteDatabase.connect(path)
        try:
            version = await ensure_schema(db, table_prefix, policy=schema)
            store = cls(db, max_spawn_depth=max_spawn_depth, table_prefix=table_prefix)
            await store._initialize_identity(version)
            return store
        except BaseException:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await db.close()
            raise

    async def _initialize_identity(self, version: int) -> None:
        self._schema_version = version
        from .task_schema import TASK_SCHEMA_VERSION
        if version == TASK_SCHEMA_VERSION:
            row = await self._db.fetchone(
                f"SELECT store_id FROM {self.table_prefix}schema_migrations WHERE id=1"
            )
            if row is None:
                raise ValueError("session store identity missing")
            self.store_id = str(row["store_id"])

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def has_message_ids(self) -> bool:
        from .task_schema import TASK_SCHEMA_VERSION
        return self.schema_version == TASK_SCHEMA_VERSION and self.store_id is not None

    # ── maintenance (optional capability; no-op on backends that lack it) ──────
    async def purge_expired(self, now: int, *, limit: int = 1000,
                            retention_days: dict[str, float] | None = None,
                            session_id: str | None = None,
                            categories: Iterable[str] | None = None) -> dict[str, int]:
        """Prune terminal inputs and closed session data within one row allowance.

        Categories: sessions, messages, inbox, runtime, projections. Sessions
        counts both identity and state rows. Open sessions retain context and graphs.
        """
        from .retention import purge_expired
        return await purge_expired(self, now, limit=limit, retention_days=retention_days,
                                   session_id=session_id, categories=categories)

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
        for holding in self._session_holdings.values():
            holding.retire()
        self._session_holdings.clear()
        await self._db.close()

    async def __aenter__(self) -> SessionStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── sessions ────────────────────────────────────────────────────────────
    async def create_session(
        self,
        descriptor,
        *,
        actor,
        parent_session_id: str | None = None,
        created_by_session_id: str | None = None,
        result_route: dict[str, Any] | None = None,
        cancel_propagation: str = "linked",
        session_id: str | None = None,
        creation_identity_hash: str | None = None,
        creation_call_ref: dict[str, Any] | None = None,
        creation_tool_call_id: str | None = None,
        start_input_message_id: str | None = None,
    ) -> str:
        from power_loop.agent.runtime_types import canonical, plain, sha256_canonical

        sid = session_id or _new_session_id()
        now = _now_ms()
        descriptor_json = canonical(descriptor)
        descriptor_hash = sha256_canonical(descriptor)
        identity = creation_identity_hash or sha256_canonical([
            self.store_id, parent_session_id, plain(creation_call_ref),
            creation_tool_call_id, descriptor, start_input_message_id,
        ])
        async with self._db.transaction() as tx:
            if parent_session_id is not None:
                await self._db.dialect.lock_state(tx, self.t.session_state, parent_session_id)
                parent = await tx.fetchone(
                    f"WITH RECURSIVE ancestors AS ("
                    f"SELECT session_id,parent_session_id,tenant_id,principal_id,lifecycle,1 AS depth "
                    f"FROM {self.t.sessions} WHERE session_id=? UNION ALL "
                    f"SELECT s.session_id,s.parent_session_id,s.tenant_id,s.principal_id,s.lifecycle,a.depth+1 "
                    f"FROM {self.t.sessions} s JOIN ancestors a ON s.session_id=a.parent_session_id "
                    "WHERE a.depth<=?) SELECT tenant_id,principal_id,lifecycle,"
                    "(SELECT MAX(depth) FROM ancestors) AS child_depth FROM ancestors WHERE depth=1",
                    (parent_session_id, self.max_spawn_depth),
                )
                if parent is None:
                    from power_loop.agent.runtime_types import RuntimeRequestError
                    raise RuntimeRequestError("not_found", "parent session")
                if parent["lifecycle"] != "open":
                    from power_loop.agent.runtime_types import RuntimeRequestError
                    raise RuntimeRequestError("session_closed")
                if (parent["tenant_id"], parent["principal_id"]) != (
                    actor.tenant_id, actor.principal_id
                ):
                    from power_loop.agent.runtime_types import RuntimeRequestError
                    raise RuntimeRequestError("forbidden")
                if parent["child_depth"] > self.max_spawn_depth:
                    from power_loop.agent.runtime_types import RuntimeRequestError
                    raise RuntimeRequestError("invalid_options", f"child depth exceeds max {self.max_spawn_depth}")
            await tx.execute(
                f"INSERT INTO {self.t.sessions} ("
                "session_id,parent_session_id,created_by_session_id,session_kind,descriptor_json,"
                "descriptor_hash,creation_identity_hash,creation_call_ref_json,creation_tool_call_id,"
                "start_input_message_id,tenant_id,principal_id,result_route_json,cancel_propagation,"
                "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid, parent_session_id, created_by_session_id, descriptor.kind, descriptor_json,
                    descriptor_hash, identity, canonical(creation_call_ref) if creation_call_ref else None,
                    creation_tool_call_id, start_input_message_id, actor.tenant_id, actor.principal_id,
                    canonical(result_route) if result_route is not None else None,
                    cancel_propagation, now, now,
                ),
            )
            await tx.execute(f"INSERT INTO {self.t.session_state} (session_id) VALUES (?)", (sid,))
        return sid

    async def get_session(self, session_id: str) -> SessionRow | None:
        holding = self._holding(session_id)
        cached = getattr(self, "_cached_sessions", {}).get(session_id)
        if cached is not None:
            if holding.touched < cached.updated_at:
                holding.touched = cached.updated_at
            return cached
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.sessions} WHERE session_id=?", (session_id,)
        )
        value = _row_to_session(row) if row else None
        if value is not None:
            if holding.touched < value.updated_at:
                holding.touched = value.updated_at
            sessions = getattr(self, "_cached_sessions", None)
            if sessions is None:
                self._cached_sessions = sessions = {}
            sessions[session_id] = value
        return value

    @contextlib.asynccontextmanager
    async def send_read_scope(self, session_id, *, ownership=None):
        existing = send_reads.current(self, session_id)
        if existing is not None and ownership is None:
            yield existing
            return
        snapshot = send_reads.SessionHolding(self, session_id, ownership=ownership)
        if existing is not None and "session" in existing.rows:
            snapshot.rows["session"] = existing.rows["session"]
            snapshot.touched = existing.rows["session"].updated_at
        self._session_holdings[session_id] = snapshot
        token = send_reads.ACTIVE.set(snapshot)
        try:
            yield snapshot
        finally:
            if snapshot.next_seq is not None:
                sequences = getattr(self, "_next_seqs", None)
                if sequences is None:
                    self._next_seqs = sequences = {}
                sequences[session_id] = snapshot.next_seq
            snapshot.retire()
            if self._session_holdings.get(session_id) is snapshot:
                del self._session_holdings[session_id]
            send_reads.ACTIVE.reset(token)

    async def session_snapshot(self, session_id):
        snapshot = send_reads.trusted(self, session_id)
        if snapshot is not None and "session" in snapshot.rows:
            return snapshot.rows["session"]
        value = await self.get_session(session_id)
        snapshot = send_reads.current(self, session_id)
        if snapshot is not None:
            snapshot.rows["session"] = value
        return value

    async def state_snapshot(self, session_id):
        snapshot = send_reads.trusted(self, session_id)
        if snapshot is not None and "state" in snapshot.rows:
            row = snapshot.rows["state"]
            return replace(row, round_index=snapshot.deferred_round if snapshot.deferred_round is not None else row.round_index)
        value = await self.get_state(session_id)
        send_reads.remember_state(self, session_id, value)
        return value

    async def runtime_snapshot(self, session_id, key, default=None):
        snapshot = send_reads.trusted(self, session_id)
        if snapshot is not None and key in snapshot.runtime:
            value = snapshot.runtime[key]
        else:
            value = await self.get_runtime_state(session_id, key)
            snapshot = send_reads.current(self, session_id)
            if snapshot is not None:
                snapshot.runtime[key] = value
        return default if value is None else value

    async def get_state(self, session_id: str) -> SessionStateRow | None:
        holding = self._holding(session_id)
        row = await self._db.fetchone(
            f"SELECT * FROM {self.t.session_state} WHERE session_id=?", (session_id,)
        )
        if row is None:
            return None
        holding.pending_known, holding.pending_json = True, row["pending_json"]
        holding.next_seq = int(row["next_seq"])
        return SessionStateRow(
            session_id=row["session_id"],
            next_seq=row["next_seq"],
            round_index=holding.deferred_round if holding.deferred_round is not None else row["round_index"],
            last_compact_seq=row["last_compact_seq"],
            pending=_loads(row["pending_json"]),
        )

    def begin_polling(self, session_id):
        changes = change_signals.ChangeReader(self, session_id)
        return changes, change_signals.ACTIVE.set(changes)

    def end_polling(self, token):
        change_signals.ACTIVE.reset(token)

    def note_model_versions(self, session_id):
        changes = change_signals.reader(self, session_id)
        if changes is not None:
            changes.model_versions = changes.versions

    async def background_changed_during_model(self, session_id):
        changes = change_signals.reader(self, session_id)
        return (changes is not None and changes.versions[1] != changes.model_versions[1]
                and bool(await self.poll_background_updates(session_id)))

    async def signal_changes(self, tx, session_id, kind):
        from ..session_ownership import require_current_ownership
        require_current_ownership(self)
        self._holding(session_id).change_snapshot = None
        if kind == "runtime":
            snapshot = send_reads.current(self, session_id)
            if snapshot is not None:
                snapshot.runtime.clear()
        await change_signals.bump(self, tx, session_id, kind)

    async def set_round_index(self, session_id: str, round_index: int) -> None:
        send_reads.lease_writer(self, session_id)
        prior = send_reads.take_state(self, session_id)
        self._holding(session_id).deferred_round = None
        async with self._db.transaction() as tx:
            sql, arguments, guarded = self._state_update(session_id, ['round_index=?'], (round_index,))
            changed = await tx.execute(sql, arguments)
            send_reads.lease_writer(self, session_id)
            if guarded and changed != 1:
                raise SessionLeaseLost(session_id)
        if prior is not None:
            send_reads.remember_state(self, session_id, replace(prior, round_index=round_index))

    async def set_pending(self, session_id: str, pending: dict[str, Any] | None) -> None:
        holding = self._holding(session_id)
        send_reads.lease_writer(self, session_id)
        prior = send_reads.take_state(self, session_id)
        async with self._db.transaction() as tx:
            sql, arguments, guarded = self._state_update(session_id, ['pending_json=?'],
                # Normalize a falsy-but-non-None pending ({}) to SQL NULL, matching the
                # empty-state representation so get_state(...).pending round-trips to None, not {}.
                (_dumps(pending) if pending else None,))
            changed = await tx.execute(sql, arguments)
            send_reads.lease_writer(self, session_id)
            if guarded and changed != 1:
                raise SessionLeaseLost(session_id)
        if not holding.active:
            return
        holding.pending_known, holding.pending_json = True, _dumps(pending) if pending else None
        if prior is not None:
            send_reads.remember_state(self, session_id, replace(prior, pending=_loads(_dumps(pending)) if pending else None))

    async def session_owner_epoch(self, session_id):
        row = await self._db.fetchone(
            f"SELECT st.owner_epoch,s.lifecycle FROM {self.t.session_state} st "
            f"JOIN {self.t.sessions} s ON s.session_id=st.session_id WHERE st.session_id=?",
            (session_id,))
        if row is None:
            raise ValueError(f"unknown session: {session_id}")
        if row['lifecycle'] != 'open':
            from power_loop.agent.runtime_types import RuntimeRequestError
            raise RuntimeRequestError('session_closed')
        return row["owner_epoch"]

    async def install_session_owner(self, session_id, lease, expected_epoch):
        lease.require_owned()
        async with self._db.transaction() as tx:
            lease.require_owned()
            changed = await tx.execute(
                f"UPDATE {self.t.session_state} SET owner_epoch=owner_epoch+1,owner_token=?,"
                "owner_node_id=?,owner_boot_id=? WHERE session_id=? AND owner_epoch=? "
                f"AND EXISTS (SELECT 1 FROM {self.t.sessions} s "
                f"WHERE s.session_id={self.t.session_state}.session_id AND s.lifecycle='open')",
                (lease.token, lease.node_id, lease.boot_id, session_id, expected_epoch))
            lease.require_owned()
        if changed:
            lease.epoch = expected_epoch + 1
        return bool(changed)

    async def release_session_owner(self, session_id, lease):
        await self._db.execute(
            f"UPDATE {self.t.session_state} SET owner_token=NULL,owner_node_id=NULL,"
            "owner_boot_id=NULL,owner_lease_until_ms=NULL WHERE session_id=? AND owner_epoch=? AND owner_token=?",
            (session_id, lease.epoch, lease.token))

    # ── messages ──────────────────────────────────────────────────────────────
    def _lease_write_guard(self, session_id):
        # Recheck after awaiting transaction entry: its turn may have ended.
        reads = send_reads.lease_writer(self, session_id)
        from ..session_ownership import current_ownership
        lease = reads.ownership if reads is not None else current_ownership(self, session_id)
        if lease is None:
            return "", ()
        lease.require_owned()
        return " AND owner_epoch=? AND owner_token=?", (lease.epoch, lease.token)

    def _state_update(self, session_id, sets, values, *, condition='', condition_values=()):
        """Declare an ordinary mutation with its owner condition in the DML."""
        guard, lease_arguments = self._lease_write_guard(session_id)
        statement = f'UPDATE {self.t.session_state} SET '+', '.join(sets)+' WHERE session_id=?'+condition+guard
        arguments = (*values, session_id, *condition_values, *lease_arguments)
        return statement, arguments, bool(guard)

    async def fence_owner_tx(self, tx, session_id, ownership):
        """Owner fence for a derived write (design 215 §8.2), inside its transaction.

        A conditional locking read: the row is locked AND the epoch/token must match in
        the same statement, so a stale owner's late projection/journal/state write is
        rejected before any derived row is touched. No read-then-compare race.
        """
        if ownership is None:
            return
        suffix = "" if self._db.dialect.name == "sqlite" else " FOR UPDATE"
        row = await tx.fetchone(
            f"SELECT 1 AS ok FROM {self.t.session_state} WHERE session_id=? "
            "AND owner_epoch=? AND owner_token=?" + suffix,
            (session_id, ownership.epoch, ownership.token),
        )
        if row is None:
            raise SessionLeaseLost(session_id)

    @staticmethod
    def fence_owner_row(row: Row, session_id: str, ownership) -> None:
        """Fence a write whose transaction already locked and returned the state row.

        Reuses the same locked row instead of a second ``SELECT ... FOR UPDATE`` on it; the
        values are read under the session lock in the caller's transaction.
        """
        if ownership is None:
            return
        if int(row["owner_epoch"]) != int(ownership.epoch) or row["owner_token"] != ownership.token:
            raise SessionLeaseLost(session_id)

    async def last_assistant_ref(self, session_id: str, send_index: int):
        """Reuse this send's committed append; recovery reads its stored tail."""
        from ..message_ref import MessageRef
        reads = send_reads.trusted(self, session_id)
        if reads is not None and reads.assistant_send_index == send_index:
            return MessageRef(self.store_id, session_id, *reads.assistant_ref) if reads.assistant_ref is not None else None
        row = await self._db.fetchone(f"SELECT * FROM {self.t.messages} WHERE session_id=? "
            "AND role='assistant' AND send_index=? ORDER BY seq DESC LIMIT 1", (session_id, send_index))
        value = None if row is None else (int(row["id"]) if self.has_message_ids else None,
            int(row["seq"]), row["role"], row["tool_call_id"], row["send_index"], row["round_index"])
        if reads is not None:
            reads.assistant_send_index, reads.assistant_ref = send_index, value
        return MessageRef(self.store_id, session_id, *value) if value is not None else None

    async def message_ref(self, session_id: str, seq: int | str):
        from power_loop.runtime.message_ref import MessageRef

        known = self._appended_refs.get((session_id, integer_id(seq, "seq")))
        if known is not None:
            return MessageRef(self.store_id, session_id, known[0], integer_id(seq, "seq"), *known[1:])
        row = await self._db.fetchone(f"SELECT * FROM {self.t.messages} WHERE session_id=? AND seq=?",
                                     (session_id, integer_id(seq, "seq")))
        if row is None:
            raise ValueError("message_not_found")
        return MessageRef(self.store_id, session_id,
                          int(row["id"]) if "id" in row.keys() else None,
                          int(row["seq"]), row["role"], row["tool_call_id"],
                          row["send_index"], row["round_index"])

    async def message_by_id(self, session_id: str, message_id: int | str) -> MessageRow | None:

        if not self.has_message_ids:
            raise ValueError("message_ids_unsupported: legacy schema")
        row = await self._db.fetchone(f"SELECT * FROM {self.t.messages} WHERE session_id=? AND id=?",
                                     (session_id, integer_id(message_id)))
        return _row_to_message(row) if row else None

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
        hook_injected: dict[str, Any] | None = None,
        pending_tool_calls: bool = False,
        pending: Any = _KEEP,
        round_usage: Mapping[str, Any] | None = None,
    ) -> int:
        """Append one message and return its allocated per-session ``seq`` (allocated +
        inserted atomically in one transaction). ``send_index`` is the authoritative per-session
        send index (a real column, NULL outside a send).

        ``hook_injected`` (audit-only) records the EPHEMERAL context an ``LLM_BEFORE`` hook injected
        into this round's LLM call (e.g. recalled memory) in ``session_runtime_state`` in the SAME
        transaction — linked to this message's ``seq``. It NEVER touches the ``messages`` row itself,
        so it can't reach history or the LLM request. Shape:
        ``{hook_point, hook, position, kind, payload}``.

        ``pending`` (a dict or ``None``) replaces the pending state in the same statement
        that allocates ``seq``; it cannot depend on that seq."""
        holding = self._holding(session_id)
        send_reads.lease_writer(self, session_id)
        if type(pending_tool_calls) is not bool or pending_tool_calls and (role != "assistant" or not tool_calls):
            raise ValueError("invalid_pending_tool_message")
        if pending is not _KEEP and pending_tool_calls:
            raise ValueError("invalid_pending_tool_message")
        reads = send_reads.current(self, session_id)
        if reads is not None and role == "assistant":
            # An unknown append outcome requires a stored read on close.
            reads.assistant_send_index, reads.assistant_ref = None, None
        now = _now_ms()
        if round_usage is not None:
            # One response original contains its usage;no separate per-round
            # write or failure window between the response and its accounting.
            meta = {**(meta or {}), message_usage.META_KEY: dict(round_usage)}
        pending_json = _KEEP if pending is _KEEP else (_dumps(pending) if pending else None)
        deferred_round, holding.deferred_round = holding.deferred_round, None
        sets, params = [], []
        if pending_json is not _KEEP:
            sets.append("pending_json=?")
            params.append(pending_json)
        if deferred_round is not None:
            sets.append("round_index=?")
            params.append(deferred_round)
        mysql = self._db.dialect.name == "mysql"
        expected = holding.next_seq

        def atomic_pending(seq):
            return _dumps({"assistant_seq": seq, "round_index": round_index,
                           "tool_call_ids": [str(call.get("id") or "") for call in tool_calls
                                             if isinstance(call, Mapping) and call.get("id")],
                           "tool_calls": tool_calls})
        committed, lease_lost, claim_changed = False, False, False
        prior = send_reads.take_state(self, session_id)
        signal_columns = ','.join(change_signals.COLUMNS)
        signal_values = None
        try:
            async with self._db.transaction() as tx:
                seq, state_written = None, False
                if expected is not None:
                    # The lease holder knows its next seq: one conditional UPDATE allocates it and
                    # applies every state change; a miss (another writer) takes the locking path.
                    fast_sets, fast_params = list(sets), list(params)
                    if pending_tool_calls:
                        fast_sets.append("pending_json=?")
                        fast_params.append(atomic_pending(expected))
                    guard = " AND pending_json IS NULL" if pending_tool_calls else ""
                    statement, arguments, guarded = self._state_update(session_id,
                        ['next_seq=?', *fast_sets], (expected+1, *fast_params),
                        condition=' AND next_seq=?'+guard, condition_values=(expected,))
                    if mysql:
                        matched = await tx.execute(statement, arguments) == 1
                    else:
                        snapshot = await tx.fetchone(statement + f" RETURNING {signal_columns}", arguments)
                        matched = snapshot is not None
                        if matched:
                            signal_values = tuple(snapshot[column] for column in change_signals.COLUMNS)
                    if matched:
                        seq, state_written = expected, True
                    elif guarded:
                        # An owned send cannot allocate past another writer.
                        raise SessionLeaseLost(session_id)
                if seq is not None:
                    pass
                elif mysql:
                    st = await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
                    seq = int(st["next_seq"])
                    if pending_tool_calls and st["pending_json"] is not None:
                        raise ValueError("pending_tool_calls_exist")
                else:
                    # One statement locks the state row, allocates the seq and applies state changes.
                    guard = " AND pending_json IS NULL" if pending_tool_calls else ""
                    statement, arguments, guarded = self._state_update(session_id,
                        ['next_seq=next_seq+1', *sets], params, condition=guard)
                    st = await tx.fetchone(statement+f' RETURNING next_seq,{signal_columns}', arguments)
                    if st is None:
                        if pending_tool_calls and await tx.fetchone(
                                f"SELECT 1 AS present FROM {self.t.session_state} WHERE session_id=?", (session_id,)):
                            raise ValueError("pending_tool_calls_exist")
                        if guarded:
                            raise SessionLeaseLost(session_id)
                        raise ValueError(f"unknown session: {session_id}")
                    seq = int(st["next_seq"]) - 1
                    signal_values = tuple(st[column] for column in change_signals.COLUMNS)
                    state_written = None  # Allocation and plain state changes only.
                send_reads.lease_writer(self, session_id)
                calls_json = _dumps(tool_calls) if tool_calls else None
                stored_call_id = _nul_safe(tool_call_id)
                stored_send = int(send_index) if send_index is not None else None
                insert = (
                    f"INSERT INTO {self.t.messages} ("
                    "session_id, seq, role, name, content, tool_calls_json, tool_call_id, "
                    "round_index, state, meta_json, send_index, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
                )
                values = (
                    session_id, seq, role, _nul_safe(name), _nul_safe(content),
                    calls_json, stored_call_id, round_index,
                    MessageState.ACTIVE.value, _dumps(meta or {}), stored_send, now,
                )
                if not self.has_message_ids:
                    await tx.execute(insert, values)
                    row_id = None
                elif self._db.dialect.name == "mysql":
                    await tx.execute(insert, values)
                    snapshot = await tx.fetchone(f"SELECT LAST_INSERT_ID() AS id,{signal_columns} "
                                                 f"FROM {self.t.session_state} WHERE session_id=?", (session_id,))
                    row_id = snapshot["id"]
                    signal_values = tuple(snapshot[column] for column in change_signals.COLUMNS)
                else:
                    row_id = (await tx.fetchone(insert + " RETURNING id", values))["id"]
                if state_written is False:
                    statement, arguments, guarded = self._state_update(session_id,
                        ['next_seq=?', *sets], (seq+1, *params))
                    changed = await tx.execute(statement, arguments)
                    if guarded and changed != 1:
                        raise SessionLeaseLost(session_id)
                if now - holding.touched >= _TOUCH_INTERVAL_MS:
                    # Activity time, not ordering: at most one write per session per interval.
                    await tx.execute(
                        f"UPDATE {self.t.sessions} SET updated_at=? WHERE session_id=?", (now, session_id)
                    )
                    touched = now
                else:
                    touched = None
                if pending_tool_calls:
                    pending_json = atomic_pending(seq)
                    if state_written is not True:
                        await tx.execute(f"UPDATE {self.t.session_state} SET pending_json=? WHERE session_id=?",
                                         (pending_json, session_id))
                if hook_injected:
                    await self._put_aux(tx, session_id, f'hook:{seq}', {
                        'session_id': session_id, 'event_id': seq, 'message_seq': seq,
                        'round_index': round_index, 'send_index': send_index,
                        'hook_point': str(hook_injected.get('hook_point') or ''),
                        'hook': hook_injected.get('hook'), 'position': hook_injected.get('position'),
                        'kind': str(hook_injected.get('kind') or ''),
                        'payload_json': _dumps(hook_injected.get('payload') or {}), 'created_at': now,
                    })
            committed = True
        except InboxClaimChanged:
            # The transaction rolled back its allocation. Retain the known seq:
            # NULL would make this live owner's next conditional write miss.
            claim_changed = True
            raise
        except SessionLeaseLost:
            lease_lost = True
            raise
        finally:
            if not committed and not lease_lost and holding.active:
                if not claim_changed:
                    holding.next_seq = None
                holding.change_snapshot = None
                if deferred_round is not None:
                    holding.deferred_round = holding.deferred_round if holding.deferred_round is not None else deferred_round
        if not holding.active:
            # The SQL committed before release, but its ACK arrived in a later
            # period. Keep that period's mutable hints and pending marker.
            return seq
        holding.next_seq = seq + 1
        if prior is not None:
            send_reads.remember_state(self, session_id, replace(prior, next_seq=seq + 1,
                pending=prior.pending if pending_json is _KEEP else _loads(pending_json),
                round_index=deferred_round if deferred_round is not None else prior.round_index))
        if signal_values is not None:
            holding.change_snapshot = seq, signal_values
        if touched is not None:
            holding.touched = touched
        if pending_json is not _KEEP:
            holding.pending_known, holding.pending_json = True, pending_json
        if row_id is not None:
            self._remember_appended(session_id, seq, int(row_id), role, stored_call_id, stored_send,
                                    round_index, calls_json)
        return seq

    async def list_hook_events(
        self, session_id: str, *, message_seq: int | None = None
    ) -> list[HookEventRow]:
        """Audit rows of hook augmentations for a session (chronological by ``event_id``).
        ``message_seq`` filters to the augmentations that fed into one specific message."""
        where, params = ("state_key LIKE 'hook:%'", ()) if message_seq is None else ('state_key=?', (f'hook:{message_seq}',))
        rows = await self._db.fetchall(
            f'SELECT value_json FROM {self.t.session_runtime_state} WHERE session_id=? AND {where}',
            (session_id, *params))
        return sorted((_row_to_hook_event(_loads(r['value_json'])) for r in rows), key=lambda r: r.event_id)

    async def load_active_messages(
        self, session_id: str, *, after_seq: int | None = None, send_index: int | None = None,
        before_send_index: int | None = None,
    ) -> list[MessageRow]:
        """Active messages in **logical** order (a ``compact_note`` sorts at its
        ``meta['ord']``, not its high identity ``seq``).

        ``after_seq`` (inclusive) returns only the active tail with ``seq >= after_seq`` — a
        cheap O(delta) read for incrementally extending a cached window after the caller's own
        appends (valid only when no compaction reshuffled the older active set; the caller
        must reload in full otherwise).

        ``send_index`` returns only that one send's rows. The end-of-send projection needs
        exactly this and used to read the WHOLE session then filter in Python — an O(session)
        read (and O(session) row objects) on every send, for a slice that is O(one send).
        In projection mode ``pl_messages`` is never compacted, so that table only grows:
        the largest real session was 2332 rows / 2.7 MB. Rows with a NULL ``send_index``
        (compact notes) are excluded, matching the Python filter this replaces.

        ``before_send_index`` returns only history older than that send (plus
        NULL-``send_index`` rows). A send entry that already has its current merged user
        row from the claim uses this to read history without selecting its own
        just-written row."""
        where = "WHERE session_id=? AND state=?"
        params: list[object] = [session_id, MessageState.ACTIVE.value]
        if after_seq is not None:
            where += " AND seq>=?"
            params.append(int(after_seq))
        if send_index is not None:
            where += " AND send_index=?"
            params.append(int(send_index))
        if before_send_index is not None:
            where += " AND (send_index IS NULL OR send_index<?)"
            params.append(int(before_send_index))
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.messages} {where} ORDER BY seq ASC", tuple(params)
        )
        messages = [_row_to_message(r) for r in rows]
        messages.sort(key=_logical_order_key)
        return messages

    async def load_projection_window(
        self, session_id: str, *, current_send_index: int, send_limit: int | None = None,
    ) -> ProjectionWindow:
        """The compact row plus the live projection sends after its cursor, in ONE bounded read.

        Replaces the former ``latest_project_compact`` + full ``load_project_messages`` pair on the
        normal history entry (~two statements) and no longer reads NULL-``send_index`` rows: the
        unique projection contract does not load pre-send-index legacy history.

        ``send_limit`` bounds the read by whole SENDS (not rows), so the caller's row cap can never
        split a send. The absolute recent lower bound is ``current_send_index - send_limit``: it
        determines both the read and the caller's repair window even when no projection rows exist
        at all. Without a cap the read is bounded only by the fold cursor, which is the honest
        limit of an uncompacted history (reported by S07 measurements).
        """
        table = self.t.project_messages
        if int(current_send_index) <= 1:
            # No finished send can exist before send 1, so there is nothing to read.
            return ProjectionWindow(None, [], 1)
        # Split the tiny compact set from the recent send range so the range predicate is an
        # actual index search, not a residual filter over every row of the session. SQLite must
        # materialize the compact CTE; otherwise the optimizer expands it and falls back to the
        # session-prefix scan (verified by the R1 window-work probe). MySQL has no AS MATERIALIZED
        # and uses an ordinary CTE + UNION ALL; PG supports the same shape.
        cte = ("WITH compact AS (" if self._db.dialect.name == "mysql"
               else "WITH compact AS MATERIALIZED (")
        send_limit = None if send_limit is None else max(1, int(send_limit))
        lo = (max(1, int(current_send_index) - send_limit)
              if send_limit is not None else None)
        params: list[Any] = [session_id, session_id]
        range_filter = ""
        if lo is not None:
            range_filter = " AND send_index>=?"
            params.append(lo)
        rows = await self._db.fetchall(
            f"{cte}"
            f"SELECT * FROM {table} WHERE session_id=? AND kind='compact') "
            "SELECT * FROM compact UNION ALL "
            f"SELECT * FROM {table} WHERE session_id=?{range_filter} "
            "AND send_index > COALESCE((SELECT MAX(send_index) FROM compact), -1)",
            tuple(params),
        )
        out = [_row_to_project_message(r) for r in rows]
        out.sort(key=lambda m: (m.send_index, _PROJECT_KIND_ORDER.get(m.kind, 9)))
        compact = max((r for r in out if r.kind == "compact"), key=lambda r: r.send_index, default=None)
        cutoff = compact.compact_to_send if compact is not None else None
        window_lo = (cutoff or 0) + 1
        if lo is not None:
            window_lo = max(window_lo, lo)
        live = [r for r in out if r.kind != "compact" and int(r.send_index) < int(current_send_index)]
        return ProjectionWindow(compact, live, window_lo)

    async def load_all_messages(self, session_id: str) -> list[MessageRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.messages} WHERE session_id=? ORDER BY seq ASC", (session_id,)
        )
        return [_row_to_message(r) for r in rows]

    async def history_records(
        self, session_id: str, *, source: str = "messages", lo: int = 0,
        hi: int | None = None, after: int = 0, limit: int = 20,
        send_index: int | None = None, _fetchall: Any = None,
        record_ids: list[int] | None = None, exclude_compact: bool = False,
    ) -> list[dict[str, Any]]:
        """Bounded, session-scoped audit read. Never substitute active history for originals."""
        if source not in {"messages", "projection"}:
            raise ValueError("source must be messages or projection")
        from dataclasses import asdict

        table = self.t.messages if source == "messages" else self.t.project_messages
        coord = "seq" if source == "messages" else "id"
        where = f"session_id=? AND {coord}>=? AND {coord}>?"
        params: list[Any] = [session_id, lo, after]
        if hi is not None:
            where += f" AND {coord}<=?"
            params.append(hi)
        if send_index is not None:
            where += " AND send_index=?"
            params.append(send_index)
        if record_ids is not None:
            if not record_ids or len(record_ids) > 100:
                raise ValueError("record_ids must contain 1..100 exact IDs")
            where += " AND id IN (" + ",".join("?" for _ in record_ids) + ")"
            params.extend(record_ids)
        if source == "projection" and exclude_compact:
            where += " AND kind IN ('user','project')"
        rows = await (_fetchall or self._db.fetchall)(
            f"SELECT * FROM {table} WHERE {where} ORDER BY {coord} ASC LIMIT ?",
            (*params, max(1, min(limit, 1000))),
        )
        decode = _row_to_message if source == "messages" else _row_to_project_message
        return [asdict(decode(r)) for r in rows]

    async def history_bounds(
        self, session_id: str, *, source: str = "messages", after_seq: int | None = None,
        _fetchone: Any = None,
    ) -> list[int] | None:
        """Coordinates for one execution's tail, including all its repair sends."""
        if source == "messages":
            row = await (_fetchone or self._db.fetchone)(
                f"SELECT min(seq) AS lo, max(seq) AS hi FROM {self.t.messages} "
                "WHERE session_id=? AND seq>?", (session_id, after_seq or 0),
            )
        elif source == "projection":
            where = "session_id=?"
            params: tuple[Any, ...] = (session_id,)
            if after_seq is not None:
                where += " AND kind IN ('user','project') AND source_seq_hi>?"
                params += (after_seq,)
            row = await (_fetchone or self._db.fetchone)(
                f"SELECT min(id) AS lo, max(id) AS hi FROM {self.t.project_messages} "
                f"WHERE {where}", params,
            )
        else:
            raise ValueError("source must be messages or projection")
        return [int(row["lo"]), int(row["hi"])] if row and row["lo"] is not None else None

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
        holding = self._holding(session_id)
        send_reads.forget(self, session_id, "state")
        now = _now_ms()
        async with self._db.transaction() as tx:
            st = await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            send_reads.lease_writer(self, session_id)
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
                    session_id, note_seq, "system", "compact_note", _nul_safe(note_content), None, None,
                    round_index, MessageState.ACTIVE.value, _dumps(meta), now,
                ),
            )
            await self._put_aux(tx,session_id,f'compaction:{compact_seq}',dict(
                session_id=session_id,compact_seq=compact_seq,note_seq=note_seq,from_seq=span_from,
                to_seq=span_to,before_tokens=before_tokens,after_tokens=after_tokens,
                round_index=round_index,created_at=now))
            statement, arguments, guarded = self._state_update(session_id,
                ['next_seq=?', 'last_compact_seq=?'], (note_seq + 1, compact_seq))
            if await tx.execute(statement, arguments) != 1 and guarded:
                raise SessionLeaseLost(session_id)
            await tx.execute(
                f"UPDATE {self.t.sessions} SET updated_at=? WHERE session_id=?", (now, session_id)
            )
        if holding.active:
            holding.next_seq = note_seq + 1
        return compact_seq, note_seq

    async def deactivate_messages(self, session_id: str, seqs: list[int]) -> int:
        """Mark the given message rows ``DROPPED`` so they no longer load into the active history.
        Used by the history-repair backstop to physically remove an orphan tool-result row that
        would make the provider reject the prompt — distinct from ``compacted_out`` so it stays
        auditable as a repair. Idempotent; returns the number of rows affected (best-effort)."""
        uniq = [int(s) for s in dict.fromkeys(seqs)]
        if not uniq:
            return 0
        placeholders = ",".join("?" for _ in uniq)
        async with self._db.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.t.messages} SET state=? WHERE session_id=? AND seq IN ({placeholders})",
                (MessageState.DROPPED.value, session_id, *uniq),
            )
        return len(uniq)

    async def list_compactions(self, session_id: str) -> list[CompactionRow]:
        rows = await self._db.fetchall(
            f"SELECT value_json FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key LIKE 'compaction:%' ORDER BY {self._json_field('compact_seq',number=True)}",
            (session_id,),
        )
        rows = [_loads(row["value_json"]) for row in rows]
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
        completed_through_send_index: int = 0,
    ) -> None:
        """Upsert one projection row on an OPEN transaction (so several rows + a compact fold can
        share one atomic, lockable transaction). See :meth:`upsert_project_message`."""
        await self._upsert_project_messages_tx(tx, [(
            session_id, int(send_index), kind,
            _dumps(content if content is not None else {}), _nul_safe(rendered_text),
            source_seq_lo, source_seq_hi, compact_from_send, compact_to_send,
            int(projector_version), token_estimate, int(completed_through_send_index), now,
        )])

    async def _upsert_project_messages_tx(self, tx: Transaction, rows: Sequence[tuple]) -> None:
        """One real multi-row UPSERT for a send's user/project rows.

        The rows tuple order is ``(session_id, send_index, kind, content_json, rendered_text,
        source_seq_lo, source_seq_hi, compact_from_send, compact_to_send, projector_version,
        token_estimate, completed_through_send_index, created_at)``. Callers already hold the
        session lock; this is one ``execute``, not a per-row loop pretending to be one statement.
        """
        if not rows:
            return
        sql = self._db.dialect.upsert_many(
            self.t.project_messages,
            ("session_id", "send_index", "kind"),
            (
                "content_json", "rendered_text", "source_seq_lo", "source_seq_hi",
                "compact_from_send", "compact_to_send", "projector_version", "token_estimate",
                "completed_through_send_index",
            ),
            rows=len(rows),
            insert_only_cols=("created_at",),
        )
        params: list[Any] = []
        for row in rows:
            params.extend(row)
        await tx.execute(sql, tuple(params))

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
        ownership=None,
        completed_through_send_index: int = 0,
    ) -> None:
        """Insert (or replace, by ``(session_id, send_index, kind)``) one projection row.
        Idempotent so a resume/re-finalize of the same send is a no-op-equivalent rewrite;
        ``created_at`` is preserved across re-finalize (insert-only)."""
        send_reads.forget(self, session_id, "state")
        async with self._db.transaction() as tx:
            # Standalone CRUD participates in the same revision fence. Low-level
            # callers may store unattached projections, so this row can be absent.
            suffix = "" if self._db.dialect.name == "sqlite" else " FOR UPDATE"
            await tx.fetchone(f"SELECT next_seq FROM {self.t.session_state} WHERE session_id=?" + suffix,
                              (session_id,))
            await self.fence_owner_tx(tx, session_id, ownership)
            from .projection_fold import invalidate
            await invalidate(self, tx, session_id)
            await self._upsert_project_message_tx(
                tx, session_id, send_index=send_index, kind=kind, content=content,
                rendered_text=rendered_text, source_seq_lo=source_seq_lo,
                source_seq_hi=source_seq_hi, compact_from_send=compact_from_send,
                compact_to_send=compact_to_send, projector_version=projector_version,
                token_estimate=token_estimate, now=_now_ms(),
                completed_through_send_index=completed_through_send_index,
            )

    async def _query_project_messages(
        self, fetchall: Callable[..., Any], session_id: str, after_send_index: int | None,
        last_rows: int | None = None,
    ) -> list[ProjectMessageRow]:
        where, params = "WHERE session_id=?", [session_id]
        if after_send_index is not None:
            where += " AND send_index>?"
            params.append(int(after_send_index))
        if last_rows is None:
            rows = await fetchall(
                f"SELECT * FROM {self.t.project_messages} {where} ORDER BY send_index ASC",
                tuple(params),
            )
        else:
            rows = await fetchall(
                f"SELECT * FROM {self.t.project_messages} {where} ORDER BY send_index DESC LIMIT ?",
                (*params, max(0, int(last_rows))),
            )
        out = [_row_to_project_message(r) for r in rows]
        out.sort(key=lambda m: (m.send_index, _PROJECT_KIND_ORDER.get(m.kind, 9)))
        return out

    async def load_project_messages(
        self, session_id: str, *, after_send_index: int | None = None, last_rows: int | None = None
    ) -> list[ProjectMessageRow]:
        """Projection rows for a session in CONVERSATION order: by ``send_index``, and WITHIN a
        send ``user`` before ``project`` (a kind-alphabetical sort would put the assistant reply
        before the user turn — reversed). ``after_send_index`` (exclusive) returns only rows with
        ``send_index > after_send_index`` — the read cursor a ``compact`` row's ``compact_to_send``
        provides (everything newer than the latest fold). ``last_rows`` keeps only the newest
        that many rows (still returned in conversation order); a send may then be partial."""
        return await self._query_project_messages(
            self._db.fetchall, session_id, after_send_index, last_rows)

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

    async def write_send_projection_rows(
        self,
        session_id: str,
        *,
        send_index: int,
        rows: list[tuple[str, Any, str | None]],
        source_seq_lo: int | None,
        source_seq_hi: int | None,
        projector_version: int,
        load_snapshot: bool = True,
        ownership=None,
    ) -> tuple[ProjectMessageRow | None, list[ProjectMessageRow]]:
        """Atomically persist one send; the optional fenced snapshot is read OUTSIDE locks.

        Repairs use ``load_snapshot=False`` because they do not run a fold. Normal
        finalization uses :meth:`prepare_send_projection_fold` to maintain incremental
        statistics before deciding whether historical projection bodies are needed.
        """
        from .projection_fold import persist, read_snapshot

        await persist(self, session_id, send_index=send_index, rows=rows, source_seq_lo=source_seq_lo,
                      source_seq_hi=source_seq_hi, projector_version=projector_version, ownership=ownership)
        if not load_snapshot:
            return None, []
        for _ in range(3):
            snapshot = await read_snapshot(self, session_id)
            if snapshot is not None:
                return snapshot.prior, snapshot.rows
        raise RuntimeError("projection changed while reading fold snapshot")

    async def prepare_send_projection_fold(
        self, session_id: str, *, send_index: int, rows: list[tuple[str, Any, str | None]],
        source_seq_lo: int | None, source_seq_hi: int | None, projector_version: int,
        projector: Any, keep_last_sends: int, threshold: int, row_threshold: float | None = None,
        ownership=None,
    ):
        """Persist projections and statistics; return a fenced fold candidate only when needed.

        Missing statistics and renderer changes rebuild from a paged snapshot outside
        the session lock. Healthy finalization reads affected sends and small state.
        """
        from .projection_fold import prepare

        return await prepare(
            self, session_id, send_index=send_index, rows=rows, source_seq_lo=source_seq_lo,
            source_seq_hi=source_seq_hi, projector_version=projector_version, projector=projector,
            keep_last_sends=keep_last_sends, threshold=threshold, row_threshold=row_threshold,
            ownership=ownership,
        )

    async def commit_projection_fold(
        self,
        session_id: str,
        *,
        content: Any,
        rendered_text: str | None,
        from_send: int,
        to_send: int,
        projector_version: int,
        expected_prior_to_send: int | None,
        expected_projection_revision: str | None = None,
        ownership=None,
    ) -> bool:
        """Persist a ``compact`` fold computed OUTSIDE the lock, under OPTIMISTIC CONCURRENCY: re-take
        the session lock, re-read the latest compact, and UPSERT the new compact ONLY if the cursor
        still equals ``expected_prior_to_send`` (the value the fold's snapshot was taken at). If a
        concurrent loop folded first (cursor advanced), this snapshot/fold is stale → skip (return
        False) rather than clobber the newer compact. Returns True iff the compact was written. This
        replaces the old "hold the lock across the fold" serialization without any in-tx network call.
        """
        now = _now_ms()
        async with self._db.transaction() as tx:
            from .projection_fold import invalidate, read_state
            # Lock the state row first; the statistics are a mutable fact and must be read
            # with a fresh statement AFTER the lock, not from a join snapshot taken while
            # waiting for it (PostgreSQL keeps the pre-wait row version of an unlocked join).
            locked = await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            self.fence_owner_row(locked, session_id, ownership)
            state = await read_state(self, tx, session_id)
            if expected_projection_revision is not None:
                if state is None or state["revision"] != expected_projection_revision:
                    return False
            cur = await self._query_latest_project_compact(tx.fetchone, session_id)
            cur_to = cur.compact_to_send if cur is not None else None
            if cur_to != expected_prior_to_send:
                return False  # a concurrent fold advanced the cursor; our snapshot/fold is stale
            await invalidate(self, tx, session_id, state=state)
            # Delete any SUPERSEDED compact rows (send_index < the new compact): a rolled-forward
            # fold folds the prior compact's content INTO the new one, so older compacts are never
            # read again (the reader uses the latest compact + after-cursor rows). Without this they
            # accumulate one-per-fold over a long session (storage leak / audit clutter).
            await tx.execute(
                f"DELETE FROM {self.t.project_messages} "
                "WHERE session_id=? AND kind='compact' AND send_index<?",
                (session_id, int(to_send)),
            )
            await self._upsert_project_message_tx(
                tx, session_id, send_index=to_send, kind="compact", content=content,
                rendered_text=rendered_text, compact_from_send=from_send,
                compact_to_send=to_send, projector_version=projector_version, now=now,
                completed_through_send_index=to_send,
            )
        return True



    # ── session lifecycle ─────────────────────────────────────────────────────
    async def list_children(self, parent_session_id: str) -> list[SessionRow]:
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.t.sessions} WHERE parent_session_id=? ORDER BY created_at",
            (parent_session_id,),
        )
        return [_row_to_session(r) for r in rows]


    # ── cross-process session leases ──────────────────────────────────────────
    #
    # The in-process lock in StatefulAgentLoop serializes one interpreter. When several agent
    # processes share this database, it cannot see the others — so the lease row is the arbiter
    # they all consult. A holder renews while it works; if it dies, the lease simply expires and
    # the next process takes over.







    async def close_session(self, session_id: str, *, cascade: bool = True) -> int:
        """Physically delete the session's rows across all tables.

        With ``cascade=True`` (default), also deletes every descendant whose
        lifecycle is ``LINKED``. ``DETACHED`` descendants are preserved and
        re-parented to ``NULL``. With ``cascade=False``, ONLY the named session
        is deleted and ALL surviving children (any lifecycle) are re-parented to
        ``NULL`` — no dangling parent refs. Use this to close a scaffold session
        (e.g. a workflow driver) while keeping its subtree as an audit trail.
        Returns the number of sessions removed.
        """
        return len(await self.close_session_tree(session_id, cascade=cascade))

    async def close_if_idle(self, session_id: str, *, cascade: bool = True, limit: int = 1000, resource_guard=None) -> str:
        """Close admission globally; busy leaves previously open sessions open."""
        from .closing import close_if_idle
        return await close_if_idle(self, [session_id], cascade=cascade, limit=limit, resource_guard=resource_guard)

    async def close_sessions_if_idle(self, session_ids: Sequence[str], *, limit: int = 1000, resource_guard=None) -> str:
        """Close a host's memory/seat group without partially closing a busy group."""
        from .closing import close_if_idle
        return await close_if_idle(self, session_ids, limit=limit, resource_guard=resource_guard)

    async def close_session_tree(self, session_id: str, *, cascade: bool = True, limit: int = 1000, resource_guard=None) -> list[str]:
        """Delete at most limit rows; an unfinished tree keeps its session roots."""
        from .closing import delete_sessions
        return await delete_sessions(self, session_id, cascade=cascade, limit=limit, resource_guard=resource_guard)

    # Normal timers and notes share the existing runtime-state table in schema 17.
    def _json_field(self, name, *, number=False):
        dialect = self._db.dialect.name
        if dialect == 'postgres':
            value = f"(CAST(value_json AS JSONB)->>'{name}')"
        elif dialect == 'mysql':
            value = f"JSON_UNQUOTE(JSON_EXTRACT(value_json,'$.{name}'))"
        else:
            value = f"json_extract(value_json,'$.{name}')"
        return f"CAST({value} AS {'SIGNED' if dialect == 'mysql' else 'BIGINT'})" if number else value

    async def _put_aux(self, tx, sid, key, value):
        await tx.execute(self._db.dialect.upsert(self.t.session_runtime_state,
            ('session_id','state_key'), ('value_json','updated_at')),
            (sid,key,_dumps(value),_now_ms()))

    async def create_timer(self, session_id, *, due_at, note, interval_s=None, tx=None):
        if tx is None:
            async with self._db.transaction() as tx:
                await self._db.dialect.lock_state(tx,self.t.session_state,session_id)
                return await self.create_timer(session_id,due_at=due_at,note=note,interval_s=interval_s,tx=tx)
        row = await tx.fetchone(f"SELECT COALESCE(MAX({self._json_field('timer_id',number=True)}),0)+1 AS id "
            f"FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key LIKE 'timer:%'", (session_id,))
        now, timer_id = _now_ms(), int(row['id'])
        value = dict(session_id=session_id,timer_id=timer_id,due_at=int(due_at),note=note,
            status='armed',interval_s=interval_s,fire_count=0,last_fired_at=None,created_at=now,updated_at=now)
        await self._put_aux(tx,session_id,f'timer:{timer_id}',value)
        return TimerRow(**value)

    async def get_timer(self, session_id, timer_id):
        value = await self.get_runtime_state(session_id,f'timer:{int(timer_id)}')
        return TimerRow(**value) if value is not None else None

    async def list_timers(self, session_id, *, statuses=('armed','firing'), reader=None):
        rows = await (reader or self._db).fetchall(f"SELECT value_json FROM {self.t.session_runtime_state} "
            f"WHERE session_id=? AND state_key LIKE 'timer:%' AND {self._json_field('status')} "
            f"IN ({','.join('?' for _ in statuses)}) ORDER BY {self._json_field('due_at',number=True)}", (session_id,*statuses))
        return [TimerRow(**_loads(row['value_json'])) for row in rows]

    async def due_timers(self, *, now=None, limit=50):
        rows = await self._db.fetchall(f"SELECT value_json FROM {self.t.session_runtime_state} "
            f"WHERE state_key LIKE 'timer:%' AND {self._json_field('status')}='armed' "
            f"AND {self._json_field('due_at',number=True)}<=? ORDER BY {self._json_field('due_at',number=True)} LIMIT ?",
            (_now_ms() if now is None else int(now),int(limit)))
        return [TimerRow(**_loads(row['value_json'])) for row in rows]

    async def _change_timer(self, sid, tid, change):
        key = f'timer:{int(tid)}'
        async with self._db.transaction() as tx:
            row = await tx.fetchone(f'SELECT value_json FROM {self.t.session_runtime_state} '
                'WHERE session_id=? AND state_key=?', (sid,key))
            if row is None:
                return False
            value = _loads(row['value_json'])
            if not change(value):
                return False
            value['updated_at'] = _now_ms()
            return bool(await tx.execute(f'UPDATE {self.t.session_runtime_state} SET value_json=?,updated_at=? '
                'WHERE session_id=? AND state_key=? AND value_json=?',
                (_dumps(value),value['updated_at'],sid,key,row['value_json'])))

    async def transition_timer(self, session_id,timer_id, *, from_status,to_status,due_at=None,
                               expected_due_at=None,expected_fire_count=None):
        def change(value):
            if (value['status'] != from_status
                or expected_due_at is not None and value['due_at'] != expected_due_at
                or expected_fire_count is not None and value['fire_count'] != expected_fire_count):
                return False
            value['status'] = to_status
            if due_at is not None:
                value['due_at'] = due_at
            return True
        return await self._change_timer(session_id,timer_id,change)

    async def finish_firing_timer(self,session_id,timer_id, *, unknown=False):
        def change(value):
            if value['status'] != 'firing':
                return False
            now = _now_ms()
            value['status'] = ('unknown' if unknown else 'fired') if value['interval_s'] is None else 'armed'
            if value['interval_s'] is not None:
                value['due_at'] = now+value['interval_s']*1000
            value['fire_count'] += 1
            if not unknown:
                value['last_fired_at'] = now
            return True
        return await self._change_timer(session_id,timer_id,change)

    async def recover_stale_firing_timers(self, *, older_than_ms,limit=50):
        rows = await self._db.fetchall(f"SELECT session_id,value_json FROM {self.t.session_runtime_state} "
            f"WHERE state_key LIKE 'timer:%' AND {self._json_field('status')}='firing' AND updated_at<? "
            'ORDER BY updated_at LIMIT ?', (_now_ms()-int(older_than_ms),limit))
        count = 0
        for row in rows:
            saved = _loads(row['value_json'])
            def change(value, saved=saved):
                if value != saved:
                    return False
                value['status'] = 'cancelled' if value['interval_s'] is None else 'armed'
                if value['interval_s'] is not None:
                    value['due_at'] = _now_ms()+value['interval_s']*1000
                value['fire_count'] += 1
                return True
            count += await self._change_timer(row['session_id'],saved['timer_id'],change)
        return count

    async def prune_timers(self,session_id, *, statuses=('fired','cancelled','unknown'),older_than_ms=None,now=None,limit=1000):
        from .retention import batch_limit, delete_batch
        where = f"state_key LIKE 'timer:%' AND {self._json_field('status')} IN ({','.join('?' for _ in statuses)})"
        params = list(statuses)
        if session_id is not None:
            where += ' AND session_id=?'
            params.append(session_id)
        if older_than_ms is not None:
            where += ' AND updated_at<?'
            params.append((_now_ms() if now is None else now)-int(older_than_ms))
        async with self._db.transaction() as tx:
            return await delete_batch(tx,self.t.session_runtime_state,('session_id','state_key'),where,tuple(params),batch_limit(limit))

    async def add_note(self, session_id,content, *, pinned=False):
        async with self._db.transaction() as tx:
            await self._db.dialect.lock_state(tx,self.t.session_state,session_id)
            row = await tx.fetchone(f"SELECT COALESCE(MAX({self._json_field('note_id',number=True)}),0)+1 AS id "
                f"FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key LIKE 'note:%'", (session_id,))
            now, nid = _now_ms(), int(row['id'])
            value = dict(session_id=session_id,note_id=nid,content=content,pinned=bool(pinned),created_at=now,updated_at=now)
            await self._put_aux(tx,session_id,f'note:{nid}',value)
        return NoteRow(**value)

    async def update_note(self,session_id,note_id, *, content=None,pinned=None):
        async with self._db.transaction() as tx:
            await self._db.dialect.lock_state(tx,self.t.session_state,session_id)
            key = f'note:{int(note_id)}'
            row = await tx.fetchone(f'SELECT value_json FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?',(session_id,key))
            if row is None:
                return False
            value = _loads(row['value_json'])
            if content is not None:
                value['content'] = content
            if pinned is not None:
                value['pinned'] = bool(pinned)
            value['updated_at'] = _now_ms()
            await self._put_aux(tx,session_id,key,value)
        return True

    async def delete_note(self,session_id,note_id):
        return bool(await self._db.execute(f'DELETE FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?',
            (session_id,f'note:{int(note_id)}')))

    async def list_notes(self,session_id):
        rows = await self._db.fetchall(f"SELECT value_json FROM {self.t.session_runtime_state} "
            f"WHERE session_id=? AND state_key LIKE 'note:%' ORDER BY {self._json_field('note_id',number=True)}",(session_id,))
        return [NoteRow(**_loads(row['value_json'])) for row in rows]

    async def count_notes(self,session_id):
        return int((await self._db.fetchone(f"SELECT COUNT(*) AS n FROM {self.t.session_runtime_state} "
            "WHERE session_id=? AND state_key LIKE 'note:%'",(session_id,)))['n'])

    # ── usage ─────────────────────────────────────────────────────────────
    async def list_usage_rounds(self, session_id: str, *, transaction: Transaction | None = None) -> list[Row]:
        """Read per-round originals once,including rounds without a message.

        A committed response and a round-end observation following a lost ACK
        describe one round. Response usage takes precedence instead of summing
        these observations. Cumulative session statistics are independent.
        """
        reader = self._db if transaction is None else transaction
        rows = await reader.fetchall(
            f'SELECT send_index,round_index,meta_json,created_at FROM {self.t.messages} '
            'WHERE session_id=? ORDER BY seq', (session_id,))
        observations = await reader.fetchall(
            f"SELECT value_json FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key LIKE 'usage:%'",
            (session_id,))
        details = {(v['send_index'], v['round_index']): v for row in observations if (v := _loads(row['value_json']))}
        for row in rows:
            usage = (_loads(row['meta_json']) or {}).get(message_usage.META_KEY)
            if usage is not None:
                key = (row['send_index'] or 0, row['round_index'])
                details[key] = dict(session_id=session_id, send_index=key[0], round_index=key[1],
                                    created_at=row['created_at'], **usage)
        return [details[key] for key in sorted(details)]

    async def record_usage(
        self,
        session_id: str,
        *,
        round_index: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        cached_tokens: int | None = None,
        model: str | None = None,
        send_index: int | None = None,
        transaction: Transaction | None = None,
    ) -> None:
        """Record a response-less round in the session journal; messages win on read."""
        value = dict(session_id=session_id, send_index=int(send_index or 0), round_index=round_index,
                     prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens,
                     cached_tokens=cached_tokens, model=model, created_at=_now_ms())
        key = f"usage:{value['send_index']}:{round_index}"
        if transaction is not None:
            await self._put_aux(transaction, session_id, key, value)
        else:
            async with self._db.transaction() as tx:
                await self._put_aux(tx, session_id, key, value)

    async def _bump_stats(self, tx, session_id, values):
        found = await tx.fetchone(f'SELECT value_json FROM {self.t.session_runtime_state} '
                                 "WHERE session_id=? AND state_key='stats'", (session_id,))
        now = _now_ms()
        data = _loads(found['value_json']) if found else dict(session_id=session_id, first_send_at=now)
        for key in ('sends','rounds','llm_calls','tool_calls','prompt_tokens','completion_tokens','total_tokens','cached_tokens'):
            data[key] = data.get(key, 0) + values.get(key, 0)
        data.update(last_send_at=now, updated_at=now)
        await self._put_aux(tx, session_id, 'stats', data)

    async def bump_session_stats(
        self,
        session_id: str,
        usage: Mapping[str, Any],
        *,
        rounds: int = 0,
        tool_calls: int = 0,
    ) -> None:
        """Accumulate once per finished send while holding the session lock."""
        values = dict(sends=1, rounds=rounds, tool_calls=tool_calls, llm_calls=int(usage.get('calls') or 0))
        values.update({k: int(usage.get(k) or 0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
        values['cached_tokens'] = int(usage.get('cache_read_tokens') or usage.get('cached_tokens') or 0)
        async with self._db.transaction() as tx:
            await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            await self._bump_stats(tx, session_id, values)

    async def record_auxiliary_usage(
        self, session_id: str, attempt_id: str, usage: Mapping[str, Any], *, revision: int = 0,
    ) -> None:
        """Apply a versioned receipt delta; corrections never count a second call."""
        now = _now_ms()
        key = "tool_context_usage:" + attempt_id
        async with self._db.transaction() as tx:
            await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            found = await tx.fetchone(
                f"SELECT value_json FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?",
                (session_id, key),
            )
            old = _loads(found['value_json']) if found is not None else None
            if old is not None and ("revision" not in old or int(old["revision"]) >= revision):
                return  # legacy receipts remain immutable; new receipts support corrections
            await self.signal_changes(tx, session_id, "runtime")
            values = {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                      "completion_tokens": int(usage.get("completion_tokens") or 0),
                      "total_tokens": int(usage.get("total_tokens") or 0),
                      "cached_tokens": int(usage.get("prompt_cached_tokens") or usage.get("cached_tokens")
                                           or usage.get("cache_hit_tokens") or 0)}
            previous = old.get("usage", {}) if old else {}
            delta = {key: value - int(previous.get(key, 0)) for key, value in values.items()}
            await self._bump_stats(tx, session_id, dict(llm_calls=0 if old else 1, **delta))
            sql = self._db.dialect.upsert(
                self.t.session_runtime_state, ("session_id", "state_key"), ("value_json", "updated_at"),
            )
            await tx.execute(sql, (session_id, key, _dumps({"recorded": True, "revision": revision, "usage": values}), now))

    async def get_session_stats(self, session_id: str) -> SessionStatsRow | None:
        value = await self.get_runtime_state(session_id, 'stats')
        return _row_to_stats(value) if value is not None else None

    async def list_session_stats(self) -> list[SessionStatsRow]:
        rows = await self._db.fetchall(f"SELECT value_json FROM {self.t.session_runtime_state} "
                                      "WHERE state_key='stats' ORDER BY updated_at DESC")
        return [_row_to_stats(_loads(r['value_json'])) for r in rows]

    # ── retention: usage rounds ───────────────────────────────────────────
    async def prune_usage_rounds(
        self,
        session_id: str | None,
        *,
        keep_last: int | None = None,
        older_than_ms: int | None = None,
        now: int | None = None,
        limit: int = 1000,
    ) -> int:
        """Prune per-round details from both kinds of original observation.

        Keep the newest rounds across sends. Message bodies and cumulative
        session statistics stay intact. Returns distinct round details removed.
        """
        from .retention import batch_limit
        limit = batch_limit(limit)
        if session_id is None:
            raise ValueError('usage pruning requires a session')
        cutoff = None if older_than_ms is None else (_now_ms() if now is None else now)-int(older_than_ms)
        async with self._db.transaction() as tx:
            await self._db.dialect.lock_state(tx, self.t.session_state, session_id)
            rows = await self.list_usage_rounds(session_id, transaction=tx)
            if keep_last and keep_last > 0:
                rows = rows[:-keep_last]
            selected = [r for r in rows if cutoff is None or r['created_at'] < cutoff][:limit]
            for row in selected:
                await tx.execute(f'DELETE FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?',
                    (session_id, f"usage:{row['send_index']}:{row['round_index']}"))
                await tx.execute(f'UPDATE {self.t.messages} '
                    f'SET meta_json={message_usage.remove_sql(self._db.dialect.name)} '
                    'WHERE session_id=? AND COALESCE(send_index,0)=? AND round_index=?',
                    (session_id, row['send_index'], row['round_index']))
            return len(selected)

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

    async def poll_runtime_state(self, session_id, key, default=None):
        changes = change_signals.reader(self, session_id)
        if changes is None:
            return await self.get_runtime_state(session_id, key, default)
        version = changes.versions[2]
        if key not in changes.runtime or changes.runtime[key][0] != version:
            value = await self.get_runtime_state(session_id, key) if version else None
            changes.runtime[key] = version, value
        value = changes.runtime[key][1]
        return default if value is None else value

    async def set_runtime_state(self, session_id: str, key: str, value: Any) -> None:
        holding = self._holding(session_id)
        now = _now_ms()
        async with self._db.transaction() as tx:
            exists = await tx.fetchone(
                f"SELECT 1 FROM {self.t.sessions} WHERE session_id=?", (session_id,)
            )
            if exists is None:
                raise ValueError(f"unknown session: {session_id}")
            await self.signal_changes(tx, session_id, "runtime")
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
        holding.touched = now

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
        holding = self._holding(session_id)
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
            await self.signal_changes(tx, session_id, "runtime")
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
        holding.touched = now
        return new_value

    async def delete_runtime_state(self, session_id: str, key: str) -> None:
        async with self._db.transaction() as tx:
            await self.signal_changes(tx, session_id, "runtime")
            await tx.execute(
                f"DELETE FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?",
                (session_id, key),
            )

    # ── shared_state: keyed JSON owned by an arbitrary scope (not a session) ──
    async def get_shared_state(self, owner: str, key: str, default: Any = None) -> Any:
        row = await self._db.fetchone(
            f"SELECT value_json FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?",
            (owner, "shared:"+key),
        )
        if row is None:
            return default
        value = _loads(row["value_json"])
        return default if value is None else value

    async def set_shared_state(self, owner: str, key: str, value: Any) -> None:
        now = _now_ms()
        async with self._db.transaction() as tx:
            sql = self._db.dialect.upsert(
                self.t.session_runtime_state,
                ("session_id", "state_key"),
                ("value_json", "updated_at"),
            )
            await tx.execute(sql, (owner, "shared:"+key, _dumps(value), now))

    async def delete_shared_state(self, owner: str, key: str) -> None:
        async with self._db.transaction() as tx:
            await tx.execute(
                f"DELETE FROM {self.t.session_runtime_state} WHERE session_id=? AND state_key=?",
                (owner, "shared:"+key),
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
        self._forget_appended(session_id)
        if older_than_ms is not None:
            sql += " AND created_at < ?"
            params.append(_now_ms() - int(older_than_ms))
        async with self._db.transaction() as tx:
            if keep_recent and keep_recent > 0:
                cutoff = await tx.fetchone(f"SELECT seq FROM {self.t.messages} WHERE session_id=? AND state=? "
                    "ORDER BY seq DESC LIMIT 1 OFFSET ?",
                    (session_id, MessageState.COMPACTED_OUT.value, int(keep_recent)-1))
                if cutoff is None:
                    return 0
                sql += " AND seq < ?"
                params.append(cutoff['seq'])
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
            if logical == "session_runtime_state":
                # Projection bodies are not exported, so their derived statistics
                # and key-owner cache cannot be valid in the restored session.
                from .projection_fold import OWNER_PREFIX, STATE_KEY
                rows = [r for r in rows if r["state_key"] != STATE_KEY
                        and not r["state_key"].startswith((OWNER_PREFIX, "hook:"))]
            tables[logical] = [{c: r[c] for c in cols} for r in rows]
        if not tables["sessions"]:
            raise ValueError(f"unknown session: {session_id}")
        return {"schema_version": version, "session_id": session_id, "tables": tables}

    async def import_session(
        self, data: Mapping[str, Any], *, new_session_id: str | None = None
    ) -> str:
        """Unsupported under schema 17 (design 215). Raises ``capability_unavailable``.

        ``export_session`` remains available for audit; a schema-native import is a
        later-phase concern and is not silently emulated with the old id-copying logic."""
        # Schema 17 exports carry the raw messages.id and the session-native lineage
        # columns; the pre-215 import copied ids verbatim (UNIQUE conflict) and reset
        # spawn columns that no longer exist. Cross-store import is not part of the
        # session-native main path; it is explicitly unsupported rather than silently
        # producing a corrupt restore (design 215 G1-R1 residual).
        from power_loop.agent.runtime_types import RuntimeRequestError
        raise RuntimeRequestError(
            "capability_unavailable", "import_session is not supported by schema 17"
        )

# ── row converters (dict rows; backend-agnostic) ────────────────────────────────


def _row_to_session(row: Row) -> SessionRow:
    from power_loop.agent.runtime_types import RuntimeDescriptor

    return SessionRow(
        session_id=row["session_id"], created_at=row["created_at"], updated_at=row["updated_at"],
        session_kind=row["session_kind"], parent_session_id=row["parent_session_id"],
        created_by_session_id=row["created_by_session_id"], lifecycle=row["lifecycle"],
        cancel_propagation=row["cancel_propagation"], tenant_id=row["tenant_id"],
        principal_id=row["principal_id"], descriptor=RuntimeDescriptor(**json.loads(row["descriptor_json"])),
        result_route=_loads(row["result_route_json"]),
    )


def _row_to_message(row: Row) -> MessageRow:
    meta = _loads(row['meta_json']) or {}
    meta.pop(message_usage.META_KEY, None)
    return MessageRow(
        session_id=row["session_id"], seq=row["seq"], role=row["role"], name=row["name"],
        content=row["content"], tool_calls=_loads(row["tool_calls_json"]),
        tool_call_id=row["tool_call_id"], round_index=row["round_index"],
        state=MessageState(row["state"]), meta=meta,
        created_at=row["created_at"], send_index=row["send_index"],
        id=int(row["id"]) if "id" in row.keys() and row["id"] is not None else None,
    )


def _row_to_project_message(row: Row) -> ProjectMessageRow:
    return ProjectMessageRow(
        session_id=row["session_id"], send_index=row["send_index"], kind=row["kind"],
        content=_loads(row["content_json"]), rendered_text=row["rendered_text"],
        source_seq_lo=row["source_seq_lo"], source_seq_hi=row["source_seq_hi"],
        compact_from_send=row["compact_from_send"], compact_to_send=row["compact_to_send"],
        projector_version=row["projector_version"], token_estimate=row["token_estimate"],
        created_at=row["created_at"], id=int(row["id"]) if "id" in row.keys() else None,
        completed_through_send_index=(
            row["completed_through_send_index"] if "completed_through_send_index" in row.keys() else 0
        ),
    )


def _row_to_hook_event(row: Row) -> HookEventRow:
    return HookEventRow(
        session_id=row["session_id"], event_id=int(row["event_id"]),
        message_seq=(int(row["message_seq"]) if row["message_seq"] is not None else None),
        round_index=(int(row["round_index"]) if row["round_index"] is not None else None),
        send_index=(int(row["send_index"]) if row["send_index"] is not None else None),
        hook_point=row["hook_point"], hook=row["hook"], position=row["position"],
        kind=row["kind"], payload=_loads(row["payload_json"]), created_at=row["created_at"],
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
        total_tokens=row["total_tokens"], cached_tokens=row["cached_tokens"],
        first_send_at=row["first_send_at"],
        last_send_at=row["last_send_at"], updated_at=row["updated_at"],
    )




__all__ = [
    "SessionStore",
    "DEFAULT_TABLE_PREFIX",
    "DEFAULT_MAX_SPAWN_DEPTH",
    "MAX_SPAWN_DEPTH",
    "DEFAULT_DB_PATH",
]
