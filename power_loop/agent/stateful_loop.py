"""StatefulAgentLoop — the single public entry point for power-loop.

Owns a :class:`SessionStore` and gives callers a stateful,
``new_session()`` + ``send(user_input, session_id=...)`` interface, plus
:meth:`follow_up` for steering an in-flight loop without blocking.
Everything else — pipeline orchestration, hooks, events, tool invocation,
persistence, pending-state machine — is wired up internally.

Failure model
-------------
* If a session has unresolved tool_calls from a previous run, :meth:`send`
  raises :class:`SessionPendingError`. Caller decides:
    - :meth:`resume` to finish executing those tool_calls and continue, or
    - :meth:`abort_pending` to synthesize ``<aborted>`` tool messages and
      proceed with the new input.
* :meth:`close_session` physically deletes the session and its data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from power_loop._vendor.llm_client.interface import LLMService
from power_loop.agent.follow_up import FollowUpQueued, merge_follow_up_inputs
from power_loop.agent.sink import SQLiteSink
from power_loop.agent.system_prompt import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    SystemPromptContext,
    format_tool_catalog,
    section_skills,
)
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.errors import SessionNotFoundError, SessionPendingError
from power_loop.core.agent_context import get_ctx, reset_current_loop, set_current_loop
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.pipeline import (
    AgentPipeline,
    _tool_call_args,
    _tool_call_name,
    _truncate_result,
)
from power_loop.core.runner import AgentRunner
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.cancellation import CancellationLike
from power_loop.runtime.skills import SkillLoader
from power_loop.runtime.store.schema import SchemaPolicy
from power_loop.runtime.store.store import (
    DEFAULT_DB_PATH,
    DEFAULT_TABLE_PREFIX,
    MAX_SPAWN_DEPTH,
    SessionStore,
)
from power_loop.runtime.store.types import (
    MessageRow,
    MessageState,
    SessionRow,
    SubagentLifecycle,
)
from power_loop.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from power_loop.runtime.history_projector import HistoryProjector

logger = logging.getLogger(__name__)


@dataclass
class _SessionCache:
    """A loop-local, rebuildable accelerator for one session's active window.

    ``rows`` is EXACTLY what ``store.load_active_messages(session_id)`` returns at the moment
    ``next_seq`` was observed — the DURABLE projection, never the pipeline's mutated working
    copy (recall placeholders / microcompacted content are re-applied fresh each send, never
    cached). The validity token is the PAIR ``(next_seq, last_compact_seq)``: a send reuses
    ``rows`` iff BOTH still match the live ``session_state``. ``next_seq`` alone is insufficient
    — a fold (compaction) reshuffles the OLDER active set into ``compacted_out`` while only
    bumping ``next_seq`` by the note, so an out-of-band fold during a send could leave a stale
    delta-extended window whose ``next_seq`` happens to match. ``last_compact_seq`` advances on
    every fold, so pairing it in makes any fold — this loop's or another writer/process's —
    invalidate the window. A cold loop with an empty cache reproduces identical behavior."""

    next_seq: int
    last_compact_seq: int
    rows: list[MessageRow]


@dataclass
class StatefulResult:
    """Result of a single :meth:`StatefulAgentLoop.send` call."""

    session_id: str
    status: str
    final_text: str = ""
    rounds: int = 0
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending_interactions: list[dict[str, Any]] = field(default_factory=list)
    #: Cumulative token usage for this send (summed over every LLM call):
    #: {prompt_tokens, completion_tokens, cache_read_tokens, reasoning_tokens,
    #:  total_tokens, calls}. Empty when the run never reached the LLM.
    usage: dict[str, int] = field(default_factory=dict)
    #: Tool invocations executed during this send.
    tool_calls: int = 0


class _SyncLoopRunner:
    """A persistent event loop on a daemon thread that drives the blocking sync API.

    ``send_sync`` / ``follow_up_sync`` / ``close`` must NOT spin a fresh ``asyncio.run`` per
    call: an asyncpg/aiomysql connection pool binds to the event loop it was created on, so
    a second ``asyncio.run`` (a new loop) finds the loop's cached store pool bound to the
    now-closed first loop and raises ``InterfaceError`` / ``Event loop is closed``. One
    long-lived loop keeps the pool valid for the whole lifetime of the StatefulAgentLoop —
    matching the legacy synchronous store's "call it as often as you like" contract. (SQLite
    is loop-agnostic but shares this path for uniformity.)
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, name="power-loop-sync", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Submit a coroutine to the dedicated loop and block until it completes."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._thread.is_alive():
            self._loop.close()


class StatefulAgentLoop:
    """The only public entry point for running an agent loop.

    A single instance can drive any number of sessions concurrently on one event
    loop: the store is async, and its SQLite backend runs every statement in a worker
    thread (``asyncio.to_thread``) under a single writer lock, so one session's
    contended write does not freeze the loop and stall the others. PostgreSQL/MySQL
    backends are natively async. The store is owned by the loop (opened lazily on first
    async use); callers may share an already-opened store across multiple
    StatefulAgentLoop instances if they need different configs.
    """

    def __init__(
        self,
        *,
        llm: LLMService,
        store: SessionStore | None = None,
        db_path: str | None = None,
        dsn: str | None = None,
        table_prefix: str | None = None,
        schema: SchemaPolicy | str | None = None,
        config: AgentLoopConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        hooks: AgentHooks | None = None,
        event_bus: AgentEventBus | None = None,
        max_spawn_depth: int | None = None,
        session_cache_size: int = 256,
    ) -> None:
        """Bind the loop to a store. Pass EITHER a pre-opened ``store=`` (e.g. shared across
        loops) OR a store config to lazily open one: ``dsn=`` (a DSN or sqlite path; ``db_path``
        is an accepted alias) + ``table_prefix=`` + ``schema=`` (a :class:`SchemaPolicy`). The
        loop holds no authoritative session state — it can be freely (re)created to resume any
        session by id: ``send(user_input, session_id=sid)`` loads that session from the store.

        ``session_cache_size`` bounds an LRU of per-session active-window caches (0 disables);
        the cache only accelerates long-lived multi-send loops and is always a rebuildable
        accelerator, never a source of truth.
        """
        self.llm = llm
        if store is not None and any(x is not None for x in (db_path, dsn, table_prefix, schema)):
            raise ValueError(
                "pass EITHER store= (a pre-opened store) OR store-config "
                "(dsn=/db_path=/table_prefix=/schema=) — not both"
            )
        # The store is async (``await open_store(...)``) but construction is sync, so an owned
        # store is opened LAZILY on first async use via :meth:`_ensure_store`. A passed store is
        # already open.
        self.store: SessionStore | None = store
        self._dsn = dsn if dsn is not None else (db_path if db_path is not None else DEFAULT_DB_PATH)
        self._table_prefix = table_prefix if table_prefix is not None else DEFAULT_TABLE_PREFIX
        self._schema = schema
        self._explicit_max_spawn_depth = max_spawn_depth
        if store is not None and max_spawn_depth is not None:
            # An explicit limit overrides the (possibly shared) store's setting.
            store.max_spawn_depth = max_spawn_depth
        self._owns_store = store is None
        self._store_open_lock = asyncio.Lock()
        # Dedicated event loop (daemon thread) for the blocking sync API; opened lazily so
        # the store pool stays bound to ONE loop across send_sync/follow_up_sync/close calls.
        self._sync_runner: _SyncLoopRunner | None = None
        self._sync_runner_lock = threading.Lock()
        # Strong ref to a best-effort store.close() scheduled when sync close() is called
        # from inside a running loop (keeps the task from being GC'd before it runs).
        self._orphaned_close_task: asyncio.Future[None] | None = None
        self.config = config if config is not None else AgentLoopConfig()
        self.tool_registry = tool_registry
        self._runner = AgentRunner(event_bus=event_bus, hooks=hooks)
        self._locks: dict[str, asyncio.Lock] = {}
        self._follow_up_queues: dict[str, list[str | LoopMessage]] = {}
        self._follow_up_queue_locks: dict[str, asyncio.Lock] = {}
        self._closing = False
        # ── per-session active-window cache (rebuildable accelerator; see _SessionCache) ──
        self._session_cache_size = int(session_cache_size)
        self._session_cache: OrderedDict[str, _SessionCache] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    async def ensure_store(self) -> SessionStore:
        """Public accessor: return this loop's store, opening an owned one on first use.

        Construction is sync but the store opens lazily on first async use, so ``loop.store``
        is ``None`` until then. Host integrations that need the store up front — e.g. building
        a :class:`~power_loop.runtime.blackboard.SqliteBlackboard` to share with the loop —
        must ``await loop.ensure_store()`` rather than reading ``loop.store`` directly (which
        would capture ``None``).
        """
        return await self._ensure_store()

    async def _ensure_store(self) -> SessionStore:
        """Return the loop's store, opening an owned one on first use.

        Construction is sync but the store is async, so the backend (SQLite/PG/MySQL, by DSN
        scheme) and its schema are opened the first time any async method needs it, routed
        through :func:`power_loop.runtime.store.factory.open_store`. A supplied store is
        returned unchanged.
        """
        if self.store is not None:
            return self.store
        async with self._store_open_lock:
            if self.store is None:
                from power_loop.runtime.store.factory import open_store

                self.store = await open_store(
                    self._dsn,
                    max_spawn_depth=(
                        MAX_SPAWN_DEPTH if self._explicit_max_spawn_depth is None
                        else self._explicit_max_spawn_depth
                    ),
                    table_prefix=self._table_prefix,
                    schema=self._schema,
                )
        return self.store

    # ── per-session active-window cache helpers ─────────────────────────────

    def _cache_get(self, sid: str, next_seq: int, last_compact_seq: int) -> list[MessageRow] | None:
        """Return the cached active rows iff the ``(next_seq, last_compact_seq)`` token still
        matches; else None. The fold counter must match too — a fold reshuffles the older
        active set, so a matching ``next_seq`` alone can still front a stale window."""
        if self._session_cache_size <= 0:
            return None
        entry = self._session_cache.get(sid)
        if entry is not None and entry.next_seq == next_seq \
                and entry.last_compact_seq == last_compact_seq:
            self._session_cache.move_to_end(sid)  # LRU touch
            self._cache_hits += 1
            return entry.rows
        self._cache_misses += 1
        return None

    def _cache_put(
        self, sid: str, next_seq: int, rows: list[MessageRow], last_compact_seq: int
    ) -> None:
        if self._session_cache_size <= 0:
            return
        self._session_cache[sid] = _SessionCache(
            next_seq=next_seq, last_compact_seq=last_compact_seq, rows=list(rows)
        )
        self._session_cache.move_to_end(sid)
        while len(self._session_cache) > self._session_cache_size:
            self._session_cache.popitem(last=False)  # evict LRU
            self._cache_evictions += 1

    def _cache_append(self, sid: str, row: MessageRow, *, new_next_seq: int) -> None:
        """Extend a live cache entry with one row the loop itself just appended (keeping the
        durable projection current without a reload). No-op if there's no live entry.

        CONTIGUITY GUARD: ``row.seq`` is the store's pre-append ``next_seq``, which equals the
        cached token ONLY if the cache saw every write since it was built. A mismatch means some
        out-of-band writer advanced the durable ``next_seq`` past our window —
        ``resume()`` / ``submit_input()`` / ``abort_pending()`` / ``heal_pending`` (which append
        via their own sink), or another loop/process sharing the store. Re-syncing the token
        then would paper over the gap and let the next ``_cache_get`` HIT a row-missing window
        (the bug the code review caught); instead we drop the entry so the next send MISSes and
        full-reloads. This is what makes the validity check sound across ALL writers."""
        entry = self._session_cache.get(sid)
        if entry is None:
            return
        if entry.next_seq != row.seq:
            self._cache_invalidate(sid)
            return
        entry.rows.append(row)
        entry.next_seq = new_next_seq

    def _cache_invalidate(self, sid: str) -> None:
        self._session_cache.pop(sid, None)

    async def _refresh_window_cache_after_send(self, sid: str, store: SessionStore) -> None:
        """Fold this send's appended tail into the live window entry — UNLESS a fold
        reshuffled the older active set, in which case drop the entry so the next send
        full-reloads.

        The fold check compares the durable ``last_compact_seq`` against the entry's, so it
        fires for ANY fold since the entry was built — this send's own compaction OR an
        out-of-band one by another writer/process. A bare ``next_seq`` delta-extend would
        otherwise keep the now-``compacted_out`` rows in the window and, because a fold also
        advances ``next_seq`` (the note), leave the entry's token matching the live state —
        so the next send would HIT a corrupt window mixing folded-out rows with the note."""
        entry = self._session_cache.get(sid)
        if entry is None:
            return
        post_state = await store.get_state(sid)
        if post_state is None:
            return
        if post_state.last_compact_seq != entry.last_compact_seq:
            self._cache_invalidate(sid)
        elif post_state.next_seq != entry.next_seq:
            # Pure append tail (incl. follow-ups drained mid-run): cheap O(delta) extend.
            delta = await store.load_active_messages(sid, after_seq=entry.next_seq)
            entry.rows.extend(delta)
            entry.next_seq = post_state.next_seq

    @property
    def cache_stats(self) -> dict[str, int]:
        """Observability for the per-session window cache: hits / misses / evictions /
        live entry count. (100% misses ⇒ the cache never helps this workload, e.g. spawn /
        workflow child loops that send once and are discarded.)"""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "entries": len(self._session_cache),
        }

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying store (if owned) and the dedicated sync event loop.

        Synchronous and abrupt: it does NOT wait for in-flight sends or pending async
        event-bus tasks. Prefer :meth:`aclose` (or ``async with loop:``) for graceful
        shutdown. When the sync API was used, the store/pool live on the dedicated sync
        loop and are torn down on it (a fresh ``asyncio.run`` could not close a pool bound
        to another loop — the bug this avoids); otherwise the close is driven via
        ``asyncio.run``. Called from inside a running loop it only schedules + warns.
        """
        runner = self._sync_runner
        # Let in-flight background tasks finish + persist their terminal status before the
        # store/loop is torn down (a finishing task's write-back targets the runner loop,
        # which is still alive here); then recover any already-deferred ones.
        if self.store is not None and runner is not None:
            from power_loop.tools.default_tools import BG

            try:
                BG.join_pending(timeout=5.0)
                runner.run(BG.flush_orphaned(self.store))
            except Exception:  # pragma: no cover - drain must never block teardown
                logger.warning("close: background-task drain failed; continuing", exc_info=True)
        store = self.store if self._owns_store else None
        if store is not None:
            if runner is not None:
                # Store/pool were opened on the dedicated loop; close them there.
                runner.run(store.close())
                self.store = None
            else:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(store.close())
                    self.store = None
                else:
                    logger.warning(
                        "StatefulAgentLoop.close() called inside a running event loop; "
                        "use 'await loop.aclose()' for graceful async shutdown"
                    )
                    # Keep a strong reference: a bare ensure_future() returns a task nothing
                    # holds, which the GC can collect mid-flight ('Task was destroyed but it
                    # is pending') so store.close() never runs and the connection/pool leaks.
                    self._orphaned_close_task = asyncio.ensure_future(store.close())
                    self.store = None
        if runner is not None:
            runner.close()
            self._sync_runner = None

    async def aclose(self, *, drain_timeout_s: float = 30.0) -> None:
        """Graceful, async shutdown: quiesce, then stop.

        1. Flip a closing flag so new :meth:`send`/:meth:`follow_up` raise immediately.
        2. Wait for every in-flight send to finish by acquiring each per-session lock
           (a running send holds its lock until its background store writes complete —
           this is what prevents the ``close()`` race that could close the connection
           out from under an ``asyncio.to_thread`` write → ``ProgrammingError``).
        3. Drain pending async event-bus subscriber tasks.
        4. Checkpoint the WAL and close the store (only if this loop owns it).

        Idempotent and bounded by ``drain_timeout_s`` (per wait phase). Safe to call via
        ``async with StatefulAgentLoop(...) as loop:``.
        """
        self._closing = True
        # (2) wait for in-flight sends to drain. Acquiring then releasing each lock
        # blocks until any holder (a running send) finishes; new sends are already
        # blocked by the closing flag, so no lock can be re-taken behind us.
        locks = list(self._locks.values())
        if locks:
            async def _wait_idle(lock: asyncio.Lock) -> None:
                async with lock:
                    return

            try:
                await asyncio.wait_for(
                    asyncio.gather(*(_wait_idle(lock) for lock in locks)),
                    timeout=drain_timeout_s,
                )
            except asyncio.TimeoutError:  # noqa: UP041 — distinct from builtin on py3.10
                logger.warning(
                    "aclose: timed out after %.1fs waiting for in-flight sends to drain",
                    drain_timeout_s,
                )
        # (3) let queued async subscribers finish (best-effort, bounded).
        try:
            await self.event_bus.drain(timeout=drain_timeout_s)
        except Exception:  # pragma: no cover - drain must never block teardown
            logger.warning("aclose: event-bus drain raised; continuing", exc_info=True)
        # (4) let in-flight background tasks finish so their terminal status write-back
        # lands on the still-open store/loop, then recover any that were already deferred —
        # otherwise closing the store here would strand them at 'running' forever.
        if self.store is not None:
            from power_loop.tools.default_tools import BG

            try:
                await asyncio.to_thread(BG.join_pending, drain_timeout_s)
                await BG.flush_orphaned(self.store)
            except Exception:  # pragma: no cover - drain must never block teardown
                logger.warning("aclose: background-task drain failed; continuing", exc_info=True)
        # (5) checkpoint + close the owned store (only if it was ever opened).
        if self._owns_store and self.store is not None:
            try:
                await self.store.checkpoint(mode="TRUNCATE")
            except Exception:  # pragma: no cover - checkpoint is best-effort
                logger.warning("aclose: WAL checkpoint failed; closing anyway", exc_info=True)
            await self.store.close()

    async def __aenter__(self) -> StatefulAgentLoop:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _raise_if_closing(self) -> None:
        if self._closing:
            raise RuntimeError("StatefulAgentLoop is closing; no new sends accepted")

    async def close_session(self, session_id: str, *, cascade: bool = True) -> int:
        """Physically delete the session and (by default) its LINKED subtree."""
        store = await self._ensure_store()
        deleted_ids = await store.close_session_tree(session_id, cascade=cascade)
        # Drop the per-session in-memory bookkeeping for EVERY removed session so a long-lived
        # loop that cycles through many sessions doesn't leak a Lock/queue/cache entry per id —
        # for the directly-closed session (C12) AND each cascade-deleted descendant (C4).
        for sid in {session_id, *deleted_ids}:
            self._locks.pop(sid, None)
            self._follow_up_queue_locks.pop(sid, None)
            self._follow_up_queues.pop(sid, None)
            self._cache_invalidate(sid)
        return len(deleted_ids)

    # ── primary API ───────────────────────────────────────────────────────

    async def new_session(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Create an empty session and return its id.

        Call this before the first :meth:`send`. Keeping session creation
        explicit makes ownership clear for web handlers, CLIs, background
        jobs, and tests: every send targets an existing session id.
        """
        return await self._create_session(metadata=metadata, system_prompt=system_prompt)

    async def prewarm(self, session_id: str) -> bool:
        """Pre-load a session's active window into the loop cache so the FIRST :meth:`send`
        skips its initial reload. Returns ``False`` if the session does not exist.

        Pure optimization for the cold-start/resume case (a freshly-created loop pointed at an
        existing session): sending without ``prewarm`` is behaviorally identical — it just pays
        one reload on the first send. (To *finish pending tool-calls* on resume, use
        :meth:`resume`, not this.)"""
        store = await self._ensure_store()
        if await store.get_session(session_id) is None:
            return False
        if self._session_cache_size > 0:
            state = await store.get_state(session_id)
            rows = await store.load_active_messages(session_id)
            if state is not None:
                self._cache_put(session_id, state.next_seq, rows, state.last_compact_seq)
        return True

    async def send(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        heal_pending: bool = False,
    ) -> StatefulResult:
        """Append one user input to the session and run the loop.

        ``session_id`` must refer to an existing session created by
        :meth:`new_session` (or by lower-level ``SessionStore`` APIs).

        Per-call overrides (do not mutate loop/session state):

        - ``tools``: restrict this run to a subset of the loop's tools. Pass a
          sequence of tool names (allowlisted from the loop registry) or a
          ``ToolRegistry`` to use directly. The model only *sees* these tools.
        - ``system_prompt``: override the system prompt for this run only
          (precedence: this arg > session system_prompt > config).

        Raises :class:`SessionPendingError` if the session has unresolved
        tool_calls (a previous run died mid tool-call); the caller must call
        :meth:`resume` or :meth:`abort_pending` first — or pass
        ``heal_pending=True`` to have ``send`` abort the stale tool_calls
        itself and proceed (the right default for orchestrators whose runs
        can be killed, e.g. by a human interrupt).
        """
        sid = session_id
        self._raise_if_closing()
        await self._ensure_store()
        async with self._lock_for(sid):
            await self._ensure_session_or_raise(sid)
            if heal_pending:
                healed = await self._heal_pending(sid)
                if healed:
                    logger.warning(
                        "send(heal_pending=True): aborted %d stale tool_call(s) "
                        "in session %s before proceeding", healed, sid,
                    )
            else:
                await self._raise_if_pending(sid)
            await self._persist_user_input(sid, user_input)
            return await self._run_loop(
                sid, stop_event=stop_event, tools=tools, system_prompt=system_prompt
            )

    async def follow_up(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
    ) -> StatefulResult | FollowUpQueued:
        """Steer an in-flight loop, or fall back to :meth:`send`.

        When the session lock is held by a running :meth:`send` / :meth:`resume`
        / :meth:`submit_input`, the input is appended to a per-session queue.
        The pipeline drains that queue at each **round** boundary (before
        ``prepare_round``), injects a wrapped ``<follow_up>`` user message, and
        clears the drained items.

        When the session is idle (lock not held), behaves like :meth:`send`.
        """
        sid = session_id
        self._raise_if_closing()
        await self._ensure_store()
        await self._ensure_session_or_raise(sid)
        session_lock = self._lock_for(sid)
        if session_lock.locked():
            depth = await self._enqueue_follow_up(sid, user_input)
            return FollowUpQueued(session_id=sid, queue_depth=depth)
        return await self.send(
            user_input, sid, stop_event=stop_event, tools=tools, system_prompt=system_prompt
        )

    def _run_sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Drive ``coro`` to completion on the loop's dedicated sync event loop.

        All blocking sync entry points funnel through here so an owned PG/MySQL pool stays
        bound to a single, long-lived loop (see :class:`_SyncLoopRunner`). Raises if called
        from within a running event loop — use the async methods (``await loop.send(...)``)
        in that case.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()  # avoid "coroutine was never awaited"
            raise RuntimeError(
                "sync API (send_sync/follow_up_sync) called from within a running event "
                "loop; await the async method (loop.send / loop.follow_up) instead"
            )
        with self._sync_runner_lock:
            if self._sync_runner is None:
                self._sync_runner = _SyncLoopRunner()
        return self._sync_runner.run(coro)

    def follow_up_sync(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
    ) -> StatefulResult | FollowUpQueued:
        return self._run_sync(
            self.follow_up(
                user_input,
                session_id,
                stop_event=stop_event,
                tools=tools,
                system_prompt=system_prompt,
            )
        )

    def new_session_sync(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Synchronous :meth:`new_session`. Use this (not ``asyncio.run(loop.new_session())``)
        to bootstrap a session for the sync API: it runs on the loop's dedicated sync event
        loop, so an owned PG/MySQL pool opens on the SAME loop that ``send_sync`` later uses
        (a throwaway ``asyncio.run`` would bind the pool to a loop that is then closed)."""
        return self._run_sync(
            self.new_session(metadata=metadata, system_prompt=system_prompt)
        )

    def send_sync(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        heal_pending: bool = False,
    ) -> StatefulResult:
        return self._run_sync(
            self.send(
                user_input,
                session_id,
                stop_event=stop_event,
                tools=tools,
                system_prompt=system_prompt,
                heal_pending=heal_pending,
            )
        )

    async def resume(
        self,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
    ) -> StatefulResult:
        """Finish executing pending tool_calls, then continue the loop.

        No-op (but still valid) if the session has no pending state — equivalent
        to "run one more round with no new user input".
        """
        store = await self._ensure_store()
        async with self._lock_for(session_id):
            await self._ensure_session_or_raise(session_id)
            waiting = await self._waiting_result_if_needed(session_id)
            if waiting is not None:
                return waiting
            sink = SQLiteSink(store, session_id)
            await self._prime_sink_from_pending(session_id, sink)
            async with self._runner.session_async(session_id=session_id):
                loop_token = set_current_loop(self)
                try:
                    await self._execute_pending(session_id, sink)
                finally:
                    reset_current_loop(loop_token)
            return await self._run_loop(session_id, stop_event=stop_event, sink=sink)

    async def submit_input(
        self,
        session_id: str,
        interaction_id: str,
        value: Any,
        *,
        stop_event: CancellationLike = None,
    ) -> StatefulResult:
        """Resolve a paused ``request_user_input`` interaction and continue.

        ``request_user_input`` is persisted as pending session state instead of
        awaiting in-process. The product layer can show the prompt/options to a
        user, wait minutes or days, restart processes, then call this method
        with the collected answer.
        """
        store = await self._ensure_store()
        async with self._lock_for(session_id):
            await self._ensure_session_or_raise(session_id)
            state = await store.get_state(session_id)
            pending = state.pending if state is not None else None
            interactions = list((pending or {}).get("pending_interactions") or [])
            interaction = next(
                (item for item in interactions if str(item.get("interaction_id")) == str(interaction_id)),
                None,
            )
            if interaction is None:
                raise ValueError(f"pending interaction not found: {interaction_id}")

            sink = SQLiteSink(store, session_id)
            await self._prime_sink_from_pending(session_id, sink)
            round_index = int((pending or {}).get("round_index") or 0)
            await sink.on_message_appended(
                {
                    "role": "tool",
                    "tool_call_id": str(interaction["tool_call_id"]),
                    "name": str(interaction.get("tool_name") or "request_user_input"),
                    "content": _as_tool_result_text(value),
                },
                round_index=round_index,
            )
            await self._remove_pending_interaction(session_id, str(interaction_id))

            waiting = await self._waiting_result_if_needed(session_id)
            if waiting is not None:
                return waiting

            return await self._run_loop(session_id, stop_event=stop_event, sink=sink)

    async def _heal_pending(self, sid: str) -> int:
        """abort_pending without the session-existence re-check (callers in
        send already hold the lock and have validated the session)."""
        store = await self._ensure_store()
        state = await store.get_state(sid)
        if state is None or not state.pending:
            return 0
        return await self.abort_pending(sid, reason="auto-healed by send(heal_pending=True)")

    async def abort_pending(self, session_id: str, *, reason: str = "aborted") -> int:
        """Synthesize ``<aborted: reason>`` tool messages for every unresolved
        tool_call, restoring message-protocol validity. Returns the number of
        aborted tool_calls.
        """
        store = await self._ensure_store()
        await self._ensure_session_or_raise(session_id)
        state = await store.get_state(session_id)
        if state is None or not state.pending:
            return 0
        pending = state.pending
        round_index = int(pending.get("round_index") or 0)
        tool_calls = pending.get("tool_calls") or [
            {"id": cid} for cid in pending.get("tool_call_ids", [])
        ]
        sink = SQLiteSink(store, session_id)
        sink._unresolved = {str(tc.get("id") or "") for tc in tool_calls}
        sink._assistant_seq = pending.get("assistant_seq")
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            name = _tool_call_name(tc) if "function" in tc or "name" in tc else None
            await sink.on_message_appended(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "name": name,
                    "content": f"<aborted: {reason}>",
                },
                round_index=round_index,
            )
        if tool_calls:
            # The <aborted> rows were appended out-of-band of the window cache; drop any live
            # entry so a later plain send full-reloads instead of serving a row-missing window.
            self._cache_invalidate(session_id)
        return len(tool_calls)

    # ── timers (durable wake-ups; fired by runtime.timers.TimerRunner) ────

    @property
    def hooks(self):
        return self._runner.hooks

    @property
    def event_bus(self):
        return self._runner.event_bus

    async def schedule_timer(
        self,
        session_id: str,
        *,
        delay_s: float | None = None,
        due_at_ms: int | None = None,
        note: str,
        interval_s: int | None = None,
    ):
        """Create a durable wake-up for this session (external/orchestrator
        path; agents use the ``schedule_wakeup`` tool). Provide exactly one of
        ``delay_s`` / ``due_at_ms``. ``interval_s`` makes it recurring: after
        each delivery it re-arms at fire-time + interval (fixed-delay) until
        cancelled. Fires only while a ``TimerRunner`` (or an external
        scheduler polling ``store.due_timers()``) is running."""
        import time as _time

        store = await self._ensure_store()
        await self._ensure_session_or_raise(session_id)
        if (delay_s is None) == (due_at_ms is None):
            raise ValueError("provide exactly one of delay_s / due_at_ms")
        if not (note or "").strip():
            raise ValueError("note is required — the agent needs to know why it woke up")
        if due_at_ms is not None:
            due = int(due_at_ms)
        else:
            assert delay_s is not None
            due = int(_time.time() * 1000 + float(delay_s) * 1000)
        if interval_s is not None and int(interval_s) < 1:
            raise ValueError("interval_s must be >= 1 second")
        return await store.create_timer(
            session_id, due_at=due, note=note.strip(), interval_s=interval_s
        )

    async def cancel_timer(self, session_id: str, timer_id: int) -> bool:
        """Cancel an armed timer. Returns False when it already fired /
        was cancelled / never existed."""
        store = await self._ensure_store()
        await self._ensure_session_or_raise(session_id)
        return await store.transition_timer(
            session_id, int(timer_id), from_status="armed", to_status="cancelled"
        )

    async def list_timers(self, session_id: str):
        """Live (armed/firing) timers for this session, soonest first."""
        store = await self._ensure_store()
        await self._ensure_session_or_raise(session_id)
        return await store.list_timers(session_id)

    # ── inspection ────────────────────────────────────────────────────────

    async def get_session_stats(self, session_id: str):
        """Cumulative accounting for one session (sends / llm_calls /
        prompt / completion / total tokens), or ``None`` before its first
        completed send. See ``SessionStatsRow``."""
        store = await self._ensure_store()
        await self._ensure_session_or_raise(session_id)
        return await store.get_session_stats(session_id)

    async def list_session_stats(self):
        """Cumulative accounting for every session in this store,
        most-recently-active first."""
        store = await self._ensure_store()
        return await store.list_session_stats()

    async def get_messages(self, session_id: str, *, include_compacted: bool = False) -> list[LoopMessage]:
        store = await self._ensure_store()
        rows = (
            await store.load_all_messages(session_id)
            if include_compacted
            else await store.load_active_messages(session_id)
        )
        return [_row_to_loop_message(r) for r in rows]

    async def get_pending(self, session_id: str) -> dict[str, Any] | None:
        store = await self._ensure_store()
        state = await store.get_state(session_id)
        return state.pending if state else None

    async def resolve_system_prompt(self, *, session_id: str | None = None) -> str:
        """Return the system prompt the pipeline will actually use.

        This mirrors the resolution logic in
        :meth:`AgentPipeline.__init__`: falls back to
        ``DEFAULT_AGENT_SYSTEM_PROMPT`` when ``config.system_prompt`` is
        ``None``, then appends the auto-generated tool catalog when
        ``config.inject_tool_descriptions`` is enabled.

        Parameters
        ----------
        session_id
            Optional session id.  When provided, the session-level
            ``system_prompt`` (set via :meth:`new_session`) takes
            precedence over ``config.system_prompt``, matching the
            behaviour of :meth:`_create_session`.

        Returns
        -------
        str
            The fully resolved prompt string — exactly what the LLM
            will see as the system message on the next :meth:`send`
            call.
        """
        # Session-level prompt wins over config-level prompt.
        base: str | None = None
        if session_id is not None:
            store = await self._ensure_store()
            row = await store.get_session(session_id)
            if row is not None:
                base = row.system_prompt

        if base is None or not base.strip():
            base = self.config.system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT

        base = base.strip()

        if self.config.inject_tool_descriptions and self.tool_registry is not None:
            catalog = format_tool_catalog(
                self.tool_registry,
                header=self.config.tool_catalog_header,
            )
            if catalog:
                base = f"{base}\n\n{catalog}"

        skills = None
        if self.config.skills_dir:
            try:
                loader = SkillLoader(self.config.skills_dir)
                skills = section_skills(
                    SystemPromptContext(
                        skills_dir=str(loader.skills_dir),
                        skill_descriptions=loader.get_descriptions(),
                    )
                )
            except Exception:
                skills = None
        if skills:
            base = f"{base}\n\n{skills}"

        return base

    # ── internals ─────────────────────────────────────────────────────────

    async def _create_session(
        self,
        *,
        metadata: dict[str, Any] | None,
        parent_session_id: str | None = None,
        spawn_tool_call_id: str | None = None,
        lifecycle: SubagentLifecycle = SubagentLifecycle.EPHEMERAL,
        system_prompt: str | None = None,
    ) -> str:
        store = await self._ensure_store()
        return await store.create_session(
            system_prompt=system_prompt or self.config.system_prompt,
            config={
                "max_rounds": self.config.max_rounds,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
            parent_session_id=parent_session_id,
            spawn_tool_call_id=spawn_tool_call_id,
            lifecycle=lifecycle,
            metadata=metadata,
        )

    def _lock_for(self, sid: str) -> asyncio.Lock:
        lock = self._locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[sid] = lock
        return lock

    def _follow_up_queue_lock_for(self, sid: str) -> asyncio.Lock:
        lock = self._follow_up_queue_locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._follow_up_queue_locks[sid] = lock
        return lock

    async def _enqueue_follow_up(self, sid: str, user_input: str | LoopMessage) -> int:
        async with self._follow_up_queue_lock_for(sid):
            queue = self._follow_up_queues.setdefault(sid, [])
            queue.append(user_input)
            return len(queue)

    async def _drain_follow_up_messages(self, sid: str) -> list[LoopMessage]:
        async with self._follow_up_queue_lock_for(sid):
            pending = self._follow_up_queues.pop(sid, [])
        merged = merge_follow_up_inputs(pending)
        return [merged] if merged is not None else []

    async def _ensure_session_or_raise(self, sid: str) -> None:
        store = await self._ensure_store()
        if await store.get_session(sid) is None:
            raise SessionNotFoundError(sid)

    async def _raise_if_pending(self, sid: str) -> None:
        store = await self._ensure_store()
        state = await store.get_state(sid)
        if state is not None and state.pending:
            pending = state.pending
            raise SessionPendingError(
                sid,
                assistant_seq=int(pending.get("assistant_seq") or 0),
                pending_tool_calls=pending.get("tool_calls", []),
            )

    async def _persist_user_input(self, sid: str, user_input: str | LoopMessage) -> None:
        store = await self._ensure_store()
        role: str
        content: str | None
        name: str | None
        if isinstance(user_input, str):
            role, content, name = "user", user_input, None
        else:
            role = str(user_input.get("role", "user"))
            content = _as_text(user_input.get("content"))
            name = user_input.get("name")
        # Allocate the next monotonic SEND index for this session (atomic RMW under the
        # session_state row lock — never resets, unlike round_index). This is the single
        # send-begin point (exactly one user row per send; resume()/follow-up drains do
        # NOT pass through here, so they correctly inherit the in-flight send's index).
        # Stamped into meta so the transcript can delimit sends authoritatively.
        send_index = await store.mutate_runtime_state(
            sid, "send_index", lambda v: int(v or 0) + 1, default=0
        )
        seq = await store.append_message(
            sid, role=role, content=content, name=name, send_index=send_index
        )
        # Keep a live cache entry current with the loop's OWN append (no reload): the next
        # send's next_seq token will then match and reuse the cached window. No-op if this
        # session isn't cached. The row mirrors what append_message persisted (only
        # seq/role/content/name/send_index are consumed when rebuilding the working history).
        self._cache_append(
            sid,
            MessageRow(
                session_id=sid, seq=seq, role=role, name=name, content=content,
                tool_calls=None, tool_call_id=None, round_index=None,
                state=MessageState.ACTIVE, meta={}, created_at=0, send_index=send_index,
            ),
            new_next_seq=seq + 1,
        )

    async def _execute_pending(self, sid: str, sink: SQLiteSink) -> None:
        """Replay leftover tool_calls. Idempotent if there is no pending."""
        store = await self._ensure_store()
        state = await store.get_state(sid)
        if state is None or not state.pending:
            return
        pending = state.pending
        if pending.get("pending_interactions"):
            return
        round_index = int(pending.get("round_index") or 0)
        tool_calls = pending.get("tool_calls") or []
        if not tool_calls:
            return
        # Initialize sink's in-memory unresolved set so auto-resolve works.
        await self._prime_sink_from_pending(sid, sink)
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            name = _tool_call_name(tc)
            args = _tool_call_args(tc)
            if self.tool_registry is None:
                output, failed = (
                    f"Error: tool '{name}' has no registry on resume",
                    True,
                )
            else:
                try:
                    raw = await self.tool_registry.invoke_async(name, args)
                    if not isinstance(raw, str):
                        raw = json.dumps(raw, ensure_ascii=False)
                    output, failed = str(raw), False
                except Exception as exc:
                    output, failed = f"Error on resume: {exc}", True
            await sink.on_message_appended(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "name": name,
                    "content": _truncate_result(output),
                },
                round_index=round_index,
            )
            if failed:
                # Still resolved from the protocol's POV — the tool message
                # exists. Surface failure via content text.
                pass

    def _resolve_registry(
        self, tools: Sequence[str] | ToolRegistry | None
    ) -> ToolRegistry | None:
        if tools is None:
            return self.tool_registry
        if isinstance(tools, ToolRegistry):
            return tools
        if self.tool_registry is None:
            return None
        return self.tool_registry.subset(tools)

    async def _run_loop(
        self,
        sid: str,
        *,
        stop_event: CancellationLike,
        sink: SQLiteSink | None = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
    ) -> StatefulResult:
        store = await self._ensure_store()
        projector = self.config.history_projector
        # The current send's authoritative index (set by _persist_user_input; inherited by
        # resume()/follow-up). Read up-front so projection mode can partition history by it.
        si = await store.get_runtime_state(sid, "send_index", default=0)
        current_send_index = int(si) if si else None
        # Cache only the plain-send path: resume()/submit_input() pass a pre-primed sink built
        # from pending state (NOT a full init_history_seqs), so they must neither read from nor
        # write to the window cache — they self-invalidate via the next_seq bump from their own
        # appended tool rows. Projection mode bypasses the window cache too (it caches verbatim
        # rows, but projected history != verbatim). Capture eligibility BEFORE the default sink.
        cache_eligible = sink is None and self._session_cache_size > 0 and projector is None
        sink = sink if sink is not None else SQLiteSink(store, sid)
        if projector is not None:
            # ── Projection mode (v2): history = rendered projections of FINISHED sends + the
            # in-flight send's structured rows. pl_messages is NEVER compacted here (compactor
            # is None, enforced by AgentLoopConfig), so it stays in seq order and we partition
            # it by the meta.send_index stamped on every row. ──
            active_rows = await store.load_active_messages(sid)
            current_rows = (
                [r for r in active_rows if r.send_index == current_send_index]
                if current_send_index is not None
                else list(active_rows)
            )
            compact = await store.latest_project_compact(sid)
            cutoff = compact.compact_to_send if compact is not None else None
            proj_rows = await store.load_project_messages(sid, after_send_index=cutoff)
            if current_send_index is not None:
                proj_rows = [r for r in proj_rows if r.send_index < current_send_index]
            prefix_rows = ([compact] if compact is not None else []) + proj_rows
            prefix_msgs = projector.render(prefix_rows)
            current_msgs = [_row_to_loop_message(r) for r in current_rows]
            history = prefix_msgs + current_msgs
            # Sink index↔seq map: the rendered prefix has no DB rows (None placeholders); the
            # in-flight send carries its real seqs. (Compaction is off, so this map is only kept
            # aligned for the appends, never used to fold.)
            placeholders: list[int | None] = [None] * len(prefix_msgs)
            cur_seqs: list[int | None] = [r.seq for r in current_rows]
            sink.init_history_seqs(placeholders + cur_seqs, placeholders + cur_seqs)
        else:
            # ── Default mode: verbatim history (+ optional window cache + in-place compactor). ──
            # The async store offloads its own blocking I/O (SQLite → threadpool; PG/MySQL → real
            # async). When this session's window cache is current (token = session_state.next_seq,
            # already advanced by the just-persisted user input), reuse it and skip the O(active-
            # history) load entirely; otherwise load in full and (re)populate the cache.
            active_rows = None
            cache_token: tuple[int, int] | None = None
            if cache_eligible:
                state = await store.get_state(sid)
                if state is not None:
                    cache_token = (state.next_seq, state.last_compact_seq)
                    active_rows = self._cache_get(sid, state.next_seq, state.last_compact_seq)
            if active_rows is None:
                active_rows = await store.load_active_messages(sid)
                if cache_eligible and cache_token is not None:
                    self._cache_put(sid, cache_token[0], active_rows, cache_token[1])
            history = [_row_to_loop_message(r) for r in active_rows]
            # Mirror loaded seqs into the sink so the compactor can translate in-memory indices
            # back to store rows when it folds messages. Pass the parallel logical positions too:
            # a compact_note's identity seq is high, but it sits at its logical ``ord`` (set when
            # it was folded) — load order and the sink's index map must agree on that.
            sink.init_history_seqs(
                [r.seq for r in active_rows],
                [
                    int(r.meta["ord"]) if r.name == "compact_note" and r.meta.get("ord") is not None
                    else r.seq
                    for r in active_rows
                ],
            )
        session_row = await store.get_session(sid)
        # System prompt precedence: per-call > session > config.
        effective_sp = system_prompt
        if effective_sp is None and session_row is not None and session_row.system_prompt:
            effective_sp = session_row.system_prompt
        runtime_config = self.config
        if effective_sp is not None and effective_sp != self.config.system_prompt:
            runtime_config = replace(self.config, system_prompt=effective_sp)
        effective_registry = self._resolve_registry(tools)

        async with self._runner.session_async(session_id=sid):
            loop_token = set_current_loop(self)
            try:
                async def _drain_follow_ups() -> list[LoopMessage]:
                    return await self._drain_follow_up_messages(sid)

                pipeline = AgentPipeline(
                    llm=self.llm,
                    config=runtime_config,
                    tool_registry=effective_registry,
                    hooks=self._runner.hooks,
                    bus=self._runner.event_bus,
                    ctx=get_ctx(),
                    session_id=sid,
                    stop_event=stop_event,
                    sink=sink,
                    store=store,
                    drain_follow_ups=_drain_follow_ups,
                )
                # Every row this run appends (assistant/tool/system, plus follow-up and
                # trailing-notice rows drained mid-run) inherits the current send index, so
                # the whole send shares one index. _persist_user_input bumped it for a fresh
                # send; resume()/submit_input() leave it, correctly attaching to the prior send.
                pipeline.send_index = current_send_index
                try:
                    result: AgentLoopResult = await pipeline.run(history)
                except Exception as exc:
                    # An unexpected error escaped the pipeline (a raising hook, sink,
                    # prepare_round, store I/O …). Emit the advertised AGENT_ERROR
                    # channel + a terminal SESSION_ENDED so subscribers that saw
                    # SESSION_STARTED aren't stranded, then re-raise unchanged (H1.5).
                    await pipeline._emit_error_terminal(exc)
                    raise
            finally:
                reset_current_loop(loop_token)
        # ── Maintain the window cache from the DURABLE store, never the pipeline's mutated
        # working `history` (recall placeholders / microcompacted content would diverge). A
        # fold reshuffled the older active set → invalidate (next send full-reloads). Otherwise
        # extend with the active tail this send appended (a cheap O(delta) read, incl. any
        # follow-up rows drained mid-run) so back-to-back sends stay on the fast path. ──
        if cache_eligible:
            await self._refresh_window_cache_after_send(sid, store)
        else:
            # A pre-primed sink (resume()/submit_input()) durably appended rows out-of-band of
            # the cache; drop any live entry now so it can't be served stale. (The contiguity
            # guard in _cache_append is the correctness backstop; this is prompt cleanup.)
            self._cache_invalidate(sid)
        try:
            # The async store offloads its own blocking I/O, so this no longer
            # needs an explicit thread hop (H1.9/C8).
            await store.bump_session_stats(
                sid, result.usage, rounds=result.rounds, tool_calls=result.tool_calls,
            )
        except Exception:
            logger.exception("session_stats bump failed for %s (continuing)", sid)
        # Send-context projection (v2): at end-of-send, project the finished send's pl_messages
        # rows into pl_project_messages so the NEXT send reads them as plain-text history. Derived
        # + best-effort: a failure never affects the send result (pl_messages is the source of truth).
        if projector is not None and current_send_index is not None:
            try:
                await self._write_send_projection(
                    store, sid, send_index=current_send_index, status=result.status,
                    session_row=session_row, projector=projector, tool_registry=effective_registry,
                )
            except Exception:
                logger.exception("send-context projection write failed for %s (continuing)", sid)
        return StatefulResult(
            session_id=sid,
            status=result.status,
            final_text=result.final_text,
            rounds=result.rounds,
            pending_tool_calls=result.pending_tool_calls,
            pending_interactions=result.pending_interactions,
            usage=result.usage,
            tool_calls=result.tool_calls,
        )

    async def _write_send_projection(
        self,
        store: SessionStore,
        sid: str,
        *,
        send_index: int,
        status: str,
        session_row: SessionRow | None,
        projector: HistoryProjector,
        tool_registry: ToolRegistry | None,
    ) -> None:
        """Project the just-finished send's ``pl_messages`` rows into ``pl_project_messages`` (v2).

        Skips sub-agent CHILD sessions (their transcript lives in their own pl_session) by
        ``parent_session_id`` — NOT ``scope``, which ``_finalize`` hardcodes to ``'main'`` even
        for children. Defers on a still-in-flight status (the resume re-finalizes under the SAME
        ``send_index``; the idempotent UPSERT then overwrites). Best-effort — the caller swallows
        failures, since ``pl_messages`` remains the source of truth."""
        if session_row is not None and session_row.parent_session_id is not None:
            return  # a spawned sub-agent run, not a top-level send
        if status in ("waiting_for_input", "pending_tools"):
            return  # send not finished; the resume will re-finalize under the same send_index
        rows = [
            r for r in await store.load_active_messages(sid) if r.send_index == send_index
        ]
        if not rows:
            return
        projected = projector.project_send(rows, send_index=send_index, tool_registry=tool_registry)
        version = int(getattr(projector, "version", 0) or 0)
        for pr in projected.rows:
            await store.upsert_project_message(
                sid,
                send_index=send_index,
                kind=pr.kind,
                content=pr.content,
                rendered_text=pr.rendered_text,
                source_seq_lo=projected.source_seq_lo,
                source_seq_hi=projected.source_seq_hi,
                projector_version=version,
            )
        await self._maybe_compact_projection(store, sid, projector=projector, version=version)

    async def _maybe_compact_projection(
        self, store: SessionStore, sid: str, *, projector: HistoryProjector, version: int
    ) -> None:
        """Projection-layer compaction (append-only, never touches pl_messages): once the rendered
        projected prefix reaches ``max_tokens × trigger_ratio`` (mirrors DefaultCompactor's
        token-driven policy), fold the OLDEST uncompacted sends — always keeping the most-recent
        ``projector.keep_last_sends`` — into a single ``compact`` row, rolling any prior compact
        forward so nothing is lost. The reader's ``latest compact + after compact_to_send`` window
        then serves the bounded prefix; the folded user/project rows REMAIN (recoverable via
        recall_send)."""
        keep = int(getattr(projector, "keep_last_sends", 0) or 0)
        if keep <= 0:
            return
        prior = await store.latest_project_compact(sid)
        cutoff = prior.compact_to_send if prior is not None else None
        rows = await store.load_project_messages(sid, after_send_index=cutoff)
        live_sends = sorted({r.send_index for r in rows if r.kind in ("user", "project")})
        if len(live_sends) <= keep:
            return  # nothing foldable beyond the keep-recent floor
        # Token-driven trigger (reuse the default compactor's policy): only fold once the rendered
        # prefix is actually large. Below the threshold, small per-send projections just accumulate.
        trigger_ratio = float(getattr(projector, "trigger_ratio", 0.75) or 0.75)
        threshold = int((self.config.max_tokens or 8000) * trigger_ratio)
        rendered_prefix = projector.render(([prior] if prior is not None else []) + rows)
        if estimate_tokens(rendered_prefix) < threshold:
            return
        fold_sends = set(live_sends[: len(live_sends) - keep])
        fold_rows = [
            r for r in rows if r.kind in ("user", "project") and r.send_index in fold_sends
        ]
        to_compact = ([prior] if prior is not None else []) + fold_rows  # roll prior forward
        folded = projector.compact(to_compact)
        if folded is None:
            return
        from_send = (
            prior.compact_from_send
            if (prior is not None and prior.compact_from_send is not None)
            else min(fold_sends)
        )
        to_send = max(fold_sends)
        await store.upsert_project_message(
            sid,
            send_index=to_send,
            kind="compact",
            content=folded.content,
            rendered_text=folded.rendered_text,
            compact_from_send=from_send,
            compact_to_send=to_send,
            projector_version=version,
        )

    async def _prime_sink_from_pending(self, sid: str, sink: SQLiteSink) -> None:
        store = await self._ensure_store()
        state = await store.get_state(sid)
        if state is None or not state.pending:
            return
        pending = state.pending
        tool_calls = list(pending.get("tool_calls") or [])
        ids = {str(tc.get("id") or "") for tc in tool_calls if tc.get("id")}
        ids.update(str(cid) for cid in pending.get("tool_call_ids", []) if cid)
        sink._unresolved = ids
        sink._assistant_seq = pending.get("assistant_seq")
        sink._tool_calls = tool_calls

    async def _remove_pending_interaction(self, sid: str, interaction_id: str) -> None:
        store = await self._ensure_store()
        state = await store.get_state(sid)
        if state is None or not state.pending:
            return
        pending = dict(state.pending)
        interactions = [
            item
            for item in list(pending.get("pending_interactions") or [])
            if str(item.get("interaction_id")) != str(interaction_id)
        ]
        if interactions:
            pending["pending_interactions"] = interactions
            await store.set_pending(sid, pending)
            return
        pending.pop("pending_interactions", None)
        if pending.get("tool_call_ids") or pending.get("tool_calls"):
            await store.set_pending(sid, pending)

    async def _waiting_result_if_needed(self, sid: str) -> StatefulResult | None:
        store = await self._ensure_store()
        state = await store.get_state(sid)
        if state is None or not state.pending:
            return None
        pending = state.pending
        interactions = list(pending.get("pending_interactions") or [])
        if not interactions:
            return None
        return StatefulResult(
            session_id=sid,
            status="waiting_for_input",
            pending_tool_calls=list(pending.get("tool_calls") or []),
            pending_interactions=interactions,
        )


# ── helpers ──────────────────────────────────────────────────────────────


def _row_to_loop_message(row: MessageRow) -> LoopMessage:
    msg: LoopMessage = {"role": row.role}
    if row.content is not None:
        msg["content"] = row.content
    if row.tool_calls:
        msg["tool_calls"] = list(row.tool_calls)
    if row.tool_call_id:
        msg["tool_call_id"] = row.tool_call_id
    if row.name:
        msg["name"] = row.name
    return msg


def _as_text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _as_tool_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


__all__ = ["StatefulAgentLoop", "StatefulResult", "MessageState", "FollowUpQueued"]
