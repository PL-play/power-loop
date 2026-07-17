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
from power_loop.agent.sink import (
    CONTENT_ENCODING_JSON,
    CONTENT_ENCODING_META_KEY,
    SQLiteSink,
    _encode_content,
    _meta_with_content_encoding,
)
from power_loop.agent.system_prompt import (
    resolve_runtime_system_prompt,
)
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.errors import SessionNotFoundError, SessionPendingError
from power_loop.contracts.protocols import ChildRunGuard
from power_loop.core.agent_context import (
    get_ctx,
    reset_current_loop,
    reset_effective_tools,
    set_current_loop,
    set_effective_tools,
)
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
from power_loop.runtime.history_sanitize import align_tool_calls
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
    ProjectMessageRow,
    SessionRow,
    SubagentLifecycle,
)
from power_loop.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from power_loop.runtime.representation import Representation

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
        # Own a FRESH AgentHooks when the caller supplies none — NOT the shared
        # module-level DEFAULT_HOOKS singleton — so per-loop built-in hooks (e.g.
        # the memory recall hook) don't stack across loops or leak config.
        self._runner = AgentRunner(
            event_bus=event_bus, hooks=hooks if hooks is not None else AgentHooks()
        )
        self._register_builtin_hooks()
        self._locks: dict[str, asyncio.Lock] = {}
        self._follow_up_queues: dict[str, list[str | LoopMessage]] = {}
        self._follow_up_queue_locks: dict[str, asyncio.Lock] = {}
        self._closing = False
        # Host-registered guards entered around every inline child run
        # (run_agent_spec) — see register_child_run_guard. (name, factory) pairs
        # in registration order.
        self._child_run_guards: list[tuple[str | None, ChildRunGuard]] = []
        # ── per-session active-window cache (rebuildable accelerator; see _SessionCache) ──
        self._session_cache_size = int(session_cache_size)
        self._session_cache: OrderedDict[str, _SessionCache] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def _register_builtin_hooks(self) -> None:
        """Register power-loop's default functional hooks on this loop's own
        AgentHooks. They carry a ``builtin.*`` name so a host can override them
        (``hooks.replace(..., name=...)``) or disable them (``hooks.remove(...)``).
        """
        cfg = self.config
        if cfg.memory is not None and getattr(cfg, "builtin_memory_hook", True):
            from power_loop.contracts.hooks import HookPoint
            from power_loop.runtime.memory import MemoryRecallHook

            hook = MemoryRecallHook(
                cfg.memory,
                budget_tokens=int(cfg.memory_budget_tokens or 0),
                position=getattr(cfg, "memory_position", "tail"),
                hooks=self._runner.hooks,
                event_bus=self._runner.event_bus,
            )
            # order=100 → runs AFTER host LLM_BEFORE hooks (default order 0) so
            # memory lands at the true request tail. Skip if the host already
            # registered one under this name (their override wins); a host can
            # also override/disable post-construction via loop.hooks.replace /
            # .remove(MemoryRecallHook.NAME).
            if not self._runner.hooks.has(MemoryRecallHook.NAME):
                self._runner.hooks.register(
                    HookPoint.LLM_BEFORE, hook, order=100, name=MemoryRecallHook.NAME,
                )

    def register_child_run_guard(
        self, guard: ChildRunGuard, *, name: str | None = None
    ) -> None:
        """Register a context-manager factory entered around every INLINE child
        run spawned under this loop (``run_agent_spec``: spawn_agent / run_agent
        delegations and in-process workflow leaves).

        An inline child shares the parent's hooks object and runs in the same
        task (same contextvars), so per-send hook state kept by the host —
        reminder counters, turn flags, same-send finalize claims — would
        otherwise tick during the child's rounds and pollute the parent (or,
        worse, fire parent finalization INTO the child session). A guard's
        typical body: snapshot/suspend that state on ``__enter__``, restore on
        ``__exit__``.

        Guards are entered in registration order and exited in reverse, around
        the child's whole run (exceptions included). They must be RE-ENTRANT:
        a grandchild run enters the same guards again, nested. Out-of-process
        leaves (``SubprocessExecutor``) never enter guards — nothing is shared.

        ``name`` enables targeted removal via :meth:`remove_child_run_guard`.
        PROVISIONAL (3.14).
        """
        self._child_run_guards.append((name, guard))

    def remove_child_run_guard(self, name: str) -> bool:
        """Remove the guard registered under ``name``. Returns True if found."""
        for i, (n, _g) in enumerate(self._child_run_guards):
            if n == name:
                del self._child_run_guards[i]
                return True
        return False

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
        max_rounds: int | None = None,
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
                sid, stop_event=stop_event, tools=tools, system_prompt=system_prompt,
                max_rounds=max_rounds,
            )

    async def follow_up(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_rounds: int | None = None,
    ) -> StatefulResult | FollowUpQueued:
        """Steer an in-flight loop, or fall back to :meth:`send`.

        When the session lock is held by a running :meth:`send` / :meth:`resume`
        / :meth:`submit_input`, the input is appended to a per-session queue.
        The pipeline drains that queue at each **round** boundary (before
        ``prepare_round``), injects a wrapped ``<follow_up>`` user message, and
        clears the drained items.

        When the session is idle (lock not held), behaves like :meth:`send`.

        ``max_rounds`` (per-call, idle path only): run this continuation with a different round
        budget than ``config.max_rounds`` — e.g. a short bounded "finalize" turn. Ignored on the
        STEERED path (an in-flight loop's own budget governs the drained follow-up).
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
            user_input, sid, stop_event=stop_event, tools=tools, system_prompt=system_prompt,
            max_rounds=max_rounds,
        )

    def pending_follow_up_count(self, session_id: str) -> int:
        """Number of queued (not yet drained) follow-up items for ``session_id``.

        Steering accepted in the terminal window of a run (after the loop's last
        round-boundary drain) stays queued on the now-idle session. Hosts use this
        after a run returns to detect stranded steering and hand it to
        :meth:`flush_follow_ups` instead of leaving it silently parked.
        """
        return len(self._follow_up_queues.get(session_id, []))

    async def flush_follow_ups(
        self,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_rounds: int | None = None,
    ) -> StatefulResult | None:
        """Run queued follow-up steering left stranded on an IDLE session.

        Returns ``None`` when there is nothing to do: the queue is empty, or the
        session lock is held (the running owner drains the queue itself at its
        round boundaries). Otherwise drains the queue, merges the items into one
        ``<follow_up>`` user message and runs it via :meth:`send`, returning that
        run's result. Call in a loop until it returns ``None`` to also cover items
        enqueued during the flush run's own terminal window.
        """
        sid = session_id
        self._raise_if_closing()
        await self._ensure_store()
        await self._ensure_session_or_raise(sid)
        if self._lock_for(sid).locked():
            return None
        async with self._follow_up_queue_lock_for(sid):
            pending = self._follow_up_queues.pop(sid, [])
        merged = merge_follow_up_inputs(pending)
        if merged is None:
            return None
        return await self.send(
            merged, sid, stop_event=stop_event, tools=tools,
            system_prompt=system_prompt, max_rounds=max_rounds,
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
            # Stamp the in-flight send's index so projection mode keeps this answer in the
            # active send's current_rows (else it lands in the NULL-send_index legacy prefix,
            # renders before its own tool_call, and is dropped as an orphan). See H1.
            send_index = await self._current_send_index(store, session_id)
            await sink.on_message_appended(
                {
                    "role": "tool",
                    "tool_call_id": str(interaction["tool_call_id"]),
                    "name": str(interaction.get("tool_name") or "request_user_input"),
                    "content": _as_tool_result_text(value),
                    "send_index": send_index,
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
        # Prime _tool_calls so a crash mid-abort (after some but not all <aborted> rows land)
        # persists a CONSISTENT intermediate pending — on_message_appended rebuilds the still-pending
        # tool_calls from self._tool_calls (sink.py:171-174); left empty it would write
        # {tool_call_ids:[…], tool_calls:[]}, a self-inconsistent pending.
        sink._tool_calls = list(tool_calls)
        # Stamp the pending send's index (runtime_state still holds it — abort runs before the
        # next _persist_user_input bumps it) so projection keeps these <aborted> rows paired with
        # their assistant tool_call instead of orphaning them in the legacy prefix. See H1.
        send_index = await self._current_send_index(store, session_id)
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            name = _tool_call_name(tc) if "function" in tc or "name" in tc else None
            await sink.on_message_appended(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "name": name,
                    "content": f"<aborted: {reason}>",
                    "send_index": send_index,
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
            The fully resolved prompt for a :meth:`send` with **no per-call
            overrides** — base (session/config) + auto-injected tool catalog
            (full registry) + skill section, via the same
            ``resolve_runtime_system_prompt`` helper the live pipeline uses.

            A per-call ``send(system_prompt=...)`` or ``send(tools=[...])`` is
            applied at send time and is NOT reflected here (this previews the
            no-override case; pass nothing at ``send`` for a byte-identical match).
        """
        # Session-level prompt wins over config-level prompt.
        base: str | None = None
        if session_id is not None:
            store = await self._ensure_store()
            row = await store.get_session(session_id)
            if row is not None:
                base = row.system_prompt
        if base is None or not base.strip():
            base = self.config.system_prompt

        # Shared assembly — the SAME helper AgentPipeline.__init__ uses — so this
        # preview is byte-identical to what the LLM actually receives.
        return resolve_runtime_system_prompt(
            base,
            inject_tool_descriptions=self.config.inject_tool_descriptions,
            tool_catalog_header=self.config.tool_catalog_header,
            tool_registry=self.tool_registry,
            skills_dir=self.config.skills_dir,
        )

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

    @staticmethod
    def _coerce_send_index(raw: Any) -> int | None:
        """The current send's authoritative index, or None when unallocated/legacy.

        send_index is allocated >= 1 by _persist_user_input and persists across
        resume()/submit_input()/follow-up (they inherit, never re-bump). 0 is the
        unallocated/legacy default. A corrupted runtime_state value (non-numeric /
        inf / nan) must degrade to "unallocated", never crash int()."""
        try:
            v = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        return v if v >= 1 else None

    async def _current_send_index(self, store: Any, sid: str) -> int | None:
        """Read the in-flight send index from runtime state (None if unallocated).

        Out-of-band tool appends (submit_input/resume/abort_pending) MUST stamp this
        onto every row so projection mode partitions them into the active send's
        ``current_rows`` rather than the legacy (NULL send_index) prefix — otherwise
        the tool result renders BEFORE its own assistant tool_call and align_tool_calls
        drops it as an orphan, silently losing the answer."""
        raw = await store.get_runtime_state(sid, "send_index", default=0)
        return self._coerce_send_index(raw)

    async def _persist_user_input(self, sid: str, user_input: str | LoopMessage) -> None:
        store = await self._ensure_store()
        role: str
        content: str | None
        name: str | None
        # Encode multimodal (list/dict) content losslessly: JSON in the text column + a meta
        # marker so the reload path reconstructs the original structure rather than handing the
        # model a literal JSON string (vision would otherwise silently break). See H6.
        if isinstance(user_input, str):
            role, content, name, structured = "user", user_input, None, False
        else:
            role = str(user_input.get("role", "user"))
            content, structured = _encode_content(user_input.get("content"))
            name = user_input.get("name")
        meta = _meta_with_content_encoding(None, structured=structured)
        # Allocate the next monotonic SEND index for this session (atomic RMW under the
        # session_state row lock — never resets, unlike round_index). This is the single
        # send-begin point (exactly one user row per send; resume()/follow-up drains do
        # NOT pass through here, so they correctly inherit the in-flight send's index).
        # Stamped into meta so the transcript can delimit sends authoritatively.
        send_index = await store.mutate_runtime_state(
            sid, "send_index", lambda v: int(v or 0) + 1, default=0
        )
        seq = await store.append_message(
            sid, role=role, content=content, name=name, send_index=send_index, meta=meta
        )
        # Keep a live cache entry current with the loop's OWN append (no reload): the next
        # send's next_seq token will then match and reuse the cached window. No-op if this
        # session isn't cached. The row mirrors what append_message persisted (only
        # seq/role/content/name/send_index/meta are consumed when rebuilding the working history).
        self._cache_append(
            sid,
            MessageRow(
                session_id=sid, seq=seq, role=role, name=name, content=content,
                tool_calls=None, tool_call_id=None, round_index=None,
                state=MessageState.ACTIVE, meta=meta or {}, created_at=0, send_index=send_index,
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
        # Fall back to tool_call_ids (as abort_pending / _prime_sink_from_pending do): a pending that
        # carries only ids (e.g. a crash mid-abort left {tool_call_ids:[…], tool_calls:[]}) must
        # still be resolved here, else resume() returns "completed" while the pending stays set and
        # the session is permanently stranded.
        tool_calls = pending.get("tool_calls") or [
            {"id": cid} for cid in (pending.get("tool_call_ids") or [])
        ]
        if not tool_calls:
            return
        # Initialize sink's in-memory unresolved set so auto-resolve works.
        await self._prime_sink_from_pending(sid, sink)
        # The in-flight send's index (inherited, not re-bumped on resume): stamp it on every
        # replayed tool row so projection mode pairs the result with its assistant tool_call
        # instead of orphaning it in the NULL-send_index legacy prefix. See H1.
        send_index = await self._current_send_index(store, sid)
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            name = _tool_call_name(tc)
            if name is None:
                # Reconstructed from ids only — no name/args to replay. Resolve the protocol with an
                # aborted marker (clears unresolved → pending cleared) instead of stranding.
                await sink.on_message_appended(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "name": None,
                        "content": "<aborted: unrecoverable tool_call on resume>",
                        "send_index": send_index,
                    },
                    round_index=round_index,
                )
                continue
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
                    "send_index": send_index,
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
        max_rounds: int | None = None,
    ) -> StatefulResult:
        store = await self._ensure_store()
        # 3.0: projection-style representation drives the derived-layer path; verbatim → None
        # (in-place compactor path). The fold trigger/keep come from config.fold_strategy.
        projector = self.config.projection_representation
        fold_strategy = self.config.fold_strategy
        # The current send's authoritative index (set by _persist_user_input; inherited by
        # resume()/follow-up). Read up-front so projection mode can partition history by it.
        # The current send (>= 1) or None when unallocated/legacy — same coercion the out-of-band
        # tool appends (submit_input/resume/abort_pending) use to stamp send_index, so the reader's
        # partition and the writer's stamp can never disagree.
        current_send_index = await self._current_send_index(store, sid)
        # Cache only the plain-send path: resume()/submit_input() pass a pre-primed sink built
        # from pending state (NOT a full init_history_seqs), so they must neither read from nor
        # write to the window cache — they self-invalidate via the next_seq bump from their own
        # appended tool rows. Projection mode bypasses the window cache too (it caches verbatim
        # rows, but projected history != verbatim). Capture eligibility BEFORE the default sink.
        cache_eligible = sink is None and self._session_cache_size > 0 and projector is None
        sink = sink if sink is not None else SQLiteSink(store, sid)
        # Record the session's ORIGINAL history mode + projector config ONCE, in the session
        # METADATA (inspectable, and the baseline for switch-detection). A later mode switch never
        # fails the run — it logs a warning and renders best-effort (the config-level
        # projector/compactor exclusion can't catch a cross-RUN switch). Loaded here once and
        # reused below for the system-prompt precedence.
        session_row = await store.get_session(sid)
        current_mode = "projection" if projector is not None else "default"
        try:
            recorded = session_row.metadata.get("history_mode") if session_row is not None else None
            if recorded is None:
                await store.merge_session_metadata(sid, {
                    "history_mode": current_mode,
                    "projector_version": int(getattr(projector, "version", 0) or 0) if projector else None,
                    "projector_trigger_ratio": (
                        float(getattr(fold_strategy, "trigger_ratio", 0) or 0) if projector else None
                    ),
                    "projector_keep_last_sends": (
                        int(getattr(fold_strategy, "keep_last_sends", 0) or 0) if projector else None
                    ),
                })
            elif recorded != current_mode:
                logger.warning(
                    "session %s: running in %r history mode but it was originally %r — rendering "
                    "best-effort (prior-mode history shown verbatim or compacted).",
                    sid, current_mode, recorded,
                )
        except Exception:
            logger.exception("session %s: history-mode bookkeeping failed (continuing)", sid)

        projection_active = projector is not None
        active_rows: list[Any] | None = None
        if projection_active:
            active_rows = await store.load_active_messages(sid)
            migrated = (
                session_row is not None and session_row.metadata.get("projection_migrated") is not None
            )
            if current_send_index is None:
                # resume()/submit_input() before any send() — no in-flight send to partition by.
                logger.warning(
                    "session %s: projection mode degrading to verbatim rendering (no allocated "
                    "send_index — resume before any send); best-effort.", sid,
                )
                projection_active = False
            elif not migrated and self.config.migrate_history_on_switch:
                # One-time migration: a session with prior NON-projection history opened in
                # projection mode. Fold that prior history into the projection table so it becomes
                # projection-native (a compact + the recent keep_last_sends as project rows) rather
                # than rendering it verbatim forever. Only when the projection table is EMPTY (a
                # genuine switch / in-place-compacted / never-projected session) — an incidental
                # missing/stale row on an already-projected session is left to the verbatim
                # fallback below.
                has_note = any(r.name == "compact_note" for r in active_rows)
                has_prior = any(
                    r.send_index is not None and r.send_index < current_send_index
                    and r.name != "compact_note"
                    for r in active_rows
                )
                if has_note or has_prior:
                    if await store.load_project_messages(sid):
                        # Already projection-native (normal session, or a version bump handled by
                        # the verbatim fallback) — mark resolved so we don't re-check every send.
                        try:
                            await store.merge_session_metadata(sid, {"projection_migrated": 0})
                        except Exception:
                            logger.exception("session %s: migration bookkeeping failed", sid)
                        migrated = True
                    else:
                        try:
                            # Writes the rows AND the projection_migrated marker in one tx (atomic).
                            await self._migrate_prior_history_to_projection(
                                store, sid, projector,
                                current_send_index=current_send_index, active_rows=active_rows,
                            )
                            migrated = True
                            logger.info(
                                "session %s: migrated prior history into the projection table "
                                "(mode switch).", sid,
                            )
                        except Exception:
                            # Best-effort: fall back to verbatim this send (and retry next send).
                            logger.exception(
                                "session %s: projection migration failed (continuing best-effort)",
                                sid,
                            )
            # A compact_note that migration did NOT fold in (migration off/failed) can't be
            # send_index-partitioned → degrade to verbatim. Once migrated, the note is represented
            # by the projection compact and is excluded from the partition below.
            if (
                projection_active
                and not migrated
                and any(r.name == "compact_note" for r in active_rows)
            ):
                logger.warning(
                    "session %s: projection mode degrading to verbatim rendering (in-place "
                    "compaction history present, not migrated); best-effort.", sid,
                )
                projection_active = False

        if projection_active:
            # ── Projection mode (v2): history = rendered projections of FINISHED sends + the
            # in-flight send's structured rows. pl_messages is NEVER compacted here (compactor is
            # None, enforced by AgentLoopConfig), so it stays in seq order; we partition by the
            # send_index stamped on every row. (active_rows loaded above; current_send_index set.)
            assert current_send_index is not None and active_rows is not None
            # Rows predating the projection era (pre-v2 / export→import) carry send_index=NULL —
            # render them VERBATIM as a temporally-first prefix (never dropped, never lumped into
            # the current send). compact_note rows are in-place-compactor artifacts: in projection
            # mode they are represented by the projection compact (after migration) and must never
            # be rendered verbatim, so they are excluded from both the prefix and the partition.
            legacy_rows = [
                r for r in active_rows if r.send_index is None and r.name != "compact_note"
            ]
            current_rows = [
                r for r in active_rows
                if r.send_index == current_send_index and r.name != "compact_note"
            ]
            # pl_messages is the immutable audit log and is NEVER compacted in projection mode, so
            # every past send's rows are still present here — the FALLBACK source whenever a send's
            # projection row is missing (a best-effort end-of-send projection write failed/crashed)
            # or was written by a DIFFERENT projector version (the user changed the projector).
            # Without this fallback such a send would be silently dropped from context forever.
            version = int(getattr(projector, "version", 0) or 0)
            active_by_send: dict[int, list[Any]] = {}
            for r in active_rows:
                if r.send_index is not None and r.name != "compact_note":
                    active_by_send.setdefault(r.send_index, []).append(r)

            def _verbatim(send_idx: int) -> list[LoopMessage]:
                return [_row_to_loop_message(m) for m in active_by_send.get(send_idx, [])]

            compact = await store.latest_project_compact(sid)
            cutoff = compact.compact_to_send if compact is not None else None
            proj_rows = [
                r
                for r in await store.load_project_messages(sid, after_send_index=cutoff)
                if r.send_index < current_send_index
            ]
            # Only rows written by the CURRENT projector version are faithfully renderable by it;
            # rows from another version fall back to verbatim (below), keyed per send.
            cur_proj_by_send: dict[int, list[Any]] = {}
            for r in proj_rows:
                if int(r.projector_version or 0) == version:
                    cur_proj_by_send.setdefault(r.send_index, []).append(r)

            # Attach recency + cross-send-dedup context ONCE over the full ordered project-row set
            # (the ONLY place with global send ordering + the current cursor). A recency-aware
            # render_project_row then renders the most-recent sends richly and collapses older ones;
            # the isolated old-span render inside the fold path is left unstamped → cold, so its
            # summary still shrinks. No-op for representations without the method (verbatim).
            if hasattr(projector, "stamp_render_context"):
                projector.stamp_render_context(proj_rows, current_send_index)

            prefix_msgs: list[LoopMessage] = []
            # (1) folded-history compact: render via the projector when it matches the current
            # version, else render its covered send range verbatim from the audit log.
            if compact is not None:
                if int(compact.projector_version or 0) == version:
                    prefix_msgs.extend(projector.render([compact]))
                else:
                    for si2 in range(compact.compact_from_send or 0, (compact.compact_to_send or 0) + 1):
                        prefix_msgs.extend(_verbatim(si2))
            # (2) each uncompacted past send in order: projected when a current-version row exists,
            # else verbatim fallback (missing or stale projection).
            lo = cutoff or 0
            for si2 in sorted(s for s in active_by_send if lo < s < current_send_index):
                rows_for_send = cur_proj_by_send.get(si2)
                prefix_msgs.extend(projector.render(rows_for_send) if rows_for_send else _verbatim(si2))

            legacy_msgs = [_row_to_loop_message(r) for r in legacy_rows]
            current_msgs = [_row_to_loop_message(r) for r in current_rows]
            history = legacy_msgs + prefix_msgs + current_msgs
            # Sink index↔seq map: legacy + in-flight rows carry real seqs; the rendered prefix
            # (projected OR verbatim-fallback) has no foldable DB row (None). Compaction is off in
            # projection mode, so this map is only kept aligned for the appends, never used to fold.
            hist_seqs: list[int | None] = (
                [r.seq for r in legacy_rows]
                + [None] * len(prefix_msgs)
                + [r.seq for r in current_rows]
            )
            hist_ords: list[int | None] = list(hist_seqs)
        else:
            # ── Default / degraded-verbatim mode: full verbatim history (+ window cache + in-place
            # compactor). When degrading from projection, active_rows is already loaded above and
            # the cache is bypassed (cache_eligible is False whenever a projector is configured). ──
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
            # back to store rows when it folds. Pass the parallel logical positions too: a
            # compact_note's identity seq is high, but it sits at its logical ``ord``.
            hist_seqs = [r.seq for r in active_rows]
            hist_ords = [
                int(r.meta["ord"]) if r.name == "compact_note" and r.meta.get("ord") is not None
                else r.seq
                for r in active_rows
            ]

        # ── Robustness backstop (mode-agnostic, always on): repair tool-call/result misalignment
        # in the assembled prompt so a corrupt row (crash mid-tool, bad import/edit, projection
        # mismatch) can't make the provider reject the history and brick the session forever. No-op
        # on a healthy history. With repair_corrupt_history=True, the dropped orphan rows are also
        # deactivated in the store so the fix is durable (else the prompt is just re-sanitized). ──
        history, hist_seqs, hist_ords, dropped_seqs, synthesized = align_tool_calls(
            history, hist_seqs, hist_ords
        )
        if dropped_seqs or synthesized:
            logger.warning(
                "session %s: repaired malformed tool-call pairing in the prompt "
                "(dropped %d orphan result(s) seqs=%s, synthesized %d placeholder result(s))",
                sid, len(dropped_seqs), dropped_seqs, synthesized,
            )
        if dropped_seqs and self.config.repair_corrupt_history:
            try:
                await store.deactivate_messages(sid, dropped_seqs)
            except Exception:
                logger.exception("session %s: durable history repair (deactivate) failed", sid)
        sink.init_history_seqs(hist_seqs, hist_ords)
        # System prompt precedence: per-call > session > config. (session_row loaded above.)
        effective_sp = system_prompt
        if effective_sp is None and session_row is not None and session_row.system_prompt:
            effective_sp = session_row.system_prompt
        # Per-call config overrides (system_prompt, max_rounds) → a per-run copy; never mutate
        # self.config (shared across concurrent sessions).
        _overrides: dict[str, Any] = {}
        if effective_sp is not None and effective_sp != self.config.system_prompt:
            _overrides["system_prompt"] = effective_sp
        if max_rounds is not None and int(max_rounds) != self.config.max_rounds:
            _overrides["max_rounds"] = max(1, int(max_rounds))
        runtime_config = replace(self.config, **_overrides) if _overrides else self.config
        effective_registry = self._resolve_registry(tools)
        # Publish this run's per-send allowlist to child spawns (innermost-run
        # semantics: always set — a run WITHOUT tools= resets to unrestricted so a
        # nested child run doesn't inherit an outer run's filter names; its own
        # registry is already the clamped subset). None = unrestricted.
        _eff_tools = (
            frozenset(effective_registry.names())
            if tools is not None and effective_registry is not None
            else None
        )

        async with self._runner.session_async(session_id=sid):
            loop_token = set_current_loop(self)
            tools_token = set_effective_tools(_eff_tools)
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
                # Same index onto the sink: per-round usage rows are keyed by send (store v6).
                # Stamped HERE (the convergence point of send/resume/submit_input) so every
                # entry path gets it without duplicating the lookup.
                if hasattr(sink, "send_index"):
                    sink.send_index = self._coerce_send_index(current_send_index)
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
                reset_effective_tools(tools_token)
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
        # Skipped when projection degraded to verbatim this run (projection_active=False).
        if projection_active and current_send_index is not None:
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

    async def _migrate_prior_history_to_projection(
        self,
        store: SessionStore,
        sid: str,
        projector: Representation,
        *,
        current_send_index: int,
        active_rows: list[Any],
    ) -> None:
        """One-time best-effort migration: fold a session's prior NON-projection history into the
        projection table so it becomes projection-native (starts with a ``compact`` covering the
        old sends, plus the most-recent ``keep_last_sends`` as individual project rows). Handles
        both a clean default→projection switch (no projection rows yet) and a session the in-place
        compactor had folded (its ``compact_note`` summary seeds the projection compact). Folds via
        the configured ``fold_strategy`` (bounded by ``fold_timeout_s``, OUTSIDE the migration write
        lock). Never raises (the caller swallows + sets the migrated marker only on success)."""
        from power_loop.runtime.store.types import ProjectMessageRow

        keep = max(int(getattr(self.config.fold_strategy, "keep_last_sends", 0) or 0), 0)
        version = int(getattr(projector, "version", 0) or 0)
        note = next((r for r in active_rows if r.name == "compact_note"), None)
        by_send: dict[int, list[Any]] = {}
        for r in active_rows:
            if r.send_index is not None and r.send_index < current_send_index and r.name != "compact_note":
                by_send.setdefault(r.send_index, []).append(r)
        prior = sorted(by_send)
        if not prior and note is None:
            return  # nothing to migrate

        projected = {si: projector.project_send(by_send[si], send_index=si, tool_registry=None) for si in prior}
        # Fold all but the most-recent `keep` prior sends; keep the rest as individual project rows.
        if keep > 0 and len(prior) > keep:
            fold, recent = prior[:-keep], prior[-keep:]
        elif keep == 0:
            fold, recent = prior, []
        else:
            fold, recent = [], prior

        def _pm(si: int, kind: str, content: Any) -> ProjectMessageRow:
            return ProjectMessageRow(
                session_id=sid, send_index=si, kind=kind, content=content, rendered_text=None,
                source_seq_lo=None, source_seq_hi=None, compact_from_send=None,
                compact_to_send=None, projector_version=version, token_estimate=None, created_at=0,
            )

        # Build the seed compact: the in-place compactor's note (if any) rolled forward + the
        # folded sends, via the configured fold_strategy (run below with the fold timeout).
        to_compact: list[ProjectMessageRow] = []
        if note is not None:
            to_compact.append(_pm(0, "compact", {"summary": note.content or ""}))
        for si in fold:
            for pr in projected[si].rows:
                to_compact.append(_pm(si, pr.kind, pr.content))
        compact_tuple: tuple[Any, str | None, int, int] | None = None
        migration_note_ops: list[Any] = []
        folded = None
        if any(r.kind in ("user", "project") for r in to_compact):
            from power_loop.runtime.fold import FoldContext

            # Same fold-timeout guard as the end-of-send path (this runs OUTSIDE the migration's
            # write lock — write_projection_migration is called afterwards with the result).
            folded = await self._run_fold_with_timeout(
                self.config.fold_strategy,
                to_compact,
                FoldContext(
                    session_id=sid, round_index=0, representation=projector,
                    llm=self.llm, max_tokens=self.config.max_tokens,
                ),
            )
        fold_as_project: list[int] = []
        if folded is not None:
            from_send = 0 if note is not None else (min(fold) if fold else 0)
            compact_tuple = (folded.content, folded.rendered_text, from_send, folded.folded_to_send)
            migration_note_ops = list(folded.note_ops)
        else:
            # The fold soft-failed (LLM error/timeout/empty) OR nothing was foldable. Do NOT write a
            # compact that claims to COVER sends it never merged — the reader uses compact_to_send as
            # the exclusion cutoff, so an over-claiming range silently drops real history (B4), and a
            # marker-set no-op drops compression forever (B13). Instead preserve everything: keep the
            # note as a standalone compact that covers NO real send (to_send=0), and write the
            # would-be-folded sends as individual project rows. A later end-of-send fold compresses
            # them (rolling this note compact forward) once over budget.
            if note is not None:
                compact_tuple = ({"summary": note.content or ""}, None, 0, 0)
            fold_as_project = fold

        project_rows = [
            (si, pr.kind, pr.content, pr.rendered_text)
            for si in (fold_as_project + recent)
            for pr in projected[si].rows
        ]
        # Mark migrated in the SAME transaction as the rows (atomic): a crash can't leave the
        # migration written with an unset marker.
        await store.write_projection_migration(
            sid, project_rows=project_rows, compact=compact_tuple, projector_version=version,
            metadata_patch={"projection_migrated": current_send_index},
        )
        for op in migration_note_ops:  # agentic-fold facts from the migrated history (best-effort)
            try:
                if getattr(op, "op", None) == "add":
                    await store.add_note(sid, op.content or "", pinned=bool(op.pinned))
                elif getattr(op, "op", None) == "update":
                    await store.update_note(sid, op.note_id, content=op.content, pinned=op.pinned)
            except Exception:
                logger.exception("session %s: migration note op failed (continuing)", sid)

    async def _write_send_projection(
        self,
        store: SessionStore,
        sid: str,
        *,
        send_index: int,
        status: str,
        session_row: SessionRow | None,
        projector: Representation,
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
        proj_rows_to_write = [(pr.kind, pr.content, pr.rendered_text) for pr in projected.rows]

        # power-loop 3.0, THREE phases so the (multi-second / possibly-hung) LLM fold never runs
        # inside a DB transaction or under the session lock:
        #   1) write this send's projection rows under a SHORT lock + snapshot the live rows;
        #   2) decide + run the fold OUTSIDE the lock (bounded by fold_timeout_s, soft-fails);
        #   3) commit the compact under a SHORT lock with optimistic concurrency (skip if a
        #      concurrent loop already advanced the compact cursor).
        prior, snapshot = await store.write_send_projection_rows(
            sid, send_index=send_index, rows=proj_rows_to_write,
            source_seq_lo=projected.source_seq_lo, source_seq_hi=projected.source_seq_hi,
            projector_version=version,
        )
        plan, note_ops = await self._plan_and_run_projection_fold(sid, projector, prior, snapshot)
        if plan is None:
            return
        content, rendered_text, from_send, to_send = plan
        committed = await store.commit_projection_fold(
            sid, content=content, rendered_text=rendered_text, from_send=from_send,
            to_send=to_send, projector_version=version,
            expected_prior_to_send=(prior.compact_to_send if prior is not None else None),
        )
        if committed:
            await self._apply_fold_notes(store, sid, note_ops)

    async def _plan_and_run_projection_fold(
        self, sid: str, projector: Representation,
        prior: ProjectMessageRow | None, snapshot: list[ProjectMessageRow],
    ) -> tuple[tuple[Any, str | None, int, int] | None, tuple[Any, ...]]:
        """Decide whether to fold (token threshold + keep-recent floor) and, if so, run the
        configured ``fold_strategy`` OUTSIDE any lock (bounded by ``fold_timeout_s``). Always keeps
        the most-recent ``keep_last_sends`` whole sends (never splits an atomic tool pair). Rolls any
        prior compact forward so nothing is lost; the folded rows REMAIN (recall_send). Returns
        ``((content, rendered_text, from_send, to_send), note_ops)`` or ``(None, ())``. Soft-fails
        (no fold, rows already written) on any error/timeout — pl_messages stays the source of truth."""
        fold_strategy = self.config.fold_strategy
        try:
            keep = int(getattr(fold_strategy, "keep_last_sends", 0) or 0)
            if keep <= 0:
                return None, ()
            live_sends = sorted({r.send_index for r in snapshot if r.kind in ("user", "project")})
            if len(live_sends) <= keep:
                return None, ()  # nothing foldable beyond the keep-recent floor
            trigger_ratio = float(getattr(fold_strategy, "trigger_ratio", 0.75) or 0.75)
            # Reserve headroom for the ephemeral tail-injected memory block (not
            # part of the projected snapshot, so invisible here) — fold earlier.
            threshold = int((self.config.effective_context_budget() or 8000) * trigger_ratio)
            # Estimate against the SAME hot/cold split the real prompt uses: stamp recency so the kept
            # recent sends render HOT (full) here too. Unstamped they'd all render cold, under-counting
            # the live prompt (whose recent sends ARE hot) → the fold would fire late (recency-fold-1).
            if hasattr(projector, "stamp_render_context"):
                projector.stamp_render_context(snapshot, (max(live_sends) + 1) if live_sends else None)
            rendered_prefix = projector.render(([prior] if prior is not None else []) + snapshot)
            if estimate_tokens(rendered_prefix) < threshold:
                return None, ()  # below threshold — small per-send projections just accumulate
            fold_sends = set(live_sends[: len(live_sends) - keep])
            fold_rows = [
                r for r in snapshot
                if r.kind in ("user", "project") and r.send_index in fold_sends
            ]
            to_compact = ([prior] if prior is not None else []) + fold_rows  # roll prior fwd
            from_send = (
                prior.compact_from_send
                if (prior is not None and prior.compact_from_send is not None)
                else min(fold_sends)
            )
            from power_loop.runtime.fold import FoldContext

            ctx = FoldContext(
                session_id=sid, round_index=0, representation=projector,
                llm=self.llm, max_tokens=self.config.max_tokens,
            )
            fr = await self._run_fold_with_timeout(fold_strategy, to_compact, ctx)
            if fr is None:
                return None, ()
            return (fr.content, fr.rendered_text, from_send, fr.folded_to_send), tuple(fr.note_ops)
        except Exception:
            logger.exception(
                "projection fold planning failed for %s (skipping fold; rows still written)", sid,
            )
            return None, ()

    async def _run_fold_with_timeout(self, fold_strategy: Any, rows: Any, ctx: Any) -> Any | None:
        """Await ``fold_strategy.fold`` bounded by ``config.fold_timeout_s`` (None disables). A
        timeout soft-fails to None (no fold this send; rows already committed)."""
        timeout = self.config.fold_timeout_s
        try:
            if timeout is not None and timeout > 0:
                return await asyncio.wait_for(fold_strategy.fold(rows, context=ctx), timeout)
            return await fold_strategy.fold(rows, context=ctx)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "projection fold timed out after %ss for %s (skipping fold this send)",
                timeout, ctx.session_id,
            )
            return None

    async def _apply_fold_notes(self, store: SessionStore, sid: str, note_ops: tuple[Any, ...]) -> None:
        """Apply an agentic fold's captured NoteOps (best-effort, additive memory — NOT
        transactional with the compact; a rare crash here loses a note, never corrupts context)."""
        for op in note_ops:
            try:
                if getattr(op, "op", None) == "add":
                    await store.add_note(sid, op.content or "", pinned=bool(op.pinned))
                elif getattr(op, "op", None) == "update":
                    await store.update_note(sid, op.note_id, content=op.content, pinned=op.pinned)
            except Exception:
                logger.exception("session %s: applying fold note op failed (continuing)", sid)

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
        content: Any = row.content
        # Reconstruct structured (multimodal) content that was JSON-encoded on persist, so the
        # model receives the original list/dict — not a literal JSON string. See H6. A corrupt
        # marker / unparseable payload degrades to the raw text rather than raising.
        if (row.meta or {}).get(CONTENT_ENCODING_META_KEY) == CONTENT_ENCODING_JSON:
            try:
                content = json.loads(row.content)
            except (ValueError, TypeError):
                content = row.content
        msg["content"] = content
    if row.tool_calls:
        msg["tool_calls"] = list(row.tool_calls)
    if row.tool_call_id:
        msg["tool_call_id"] = row.tool_call_id
    if row.name:
        msg["name"] = row.name
    return msg


def _as_tool_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


__all__ = ["StatefulAgentLoop", "StatefulResult", "MessageState", "FollowUpQueued"]
