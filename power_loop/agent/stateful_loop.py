"""StatefulAgentLoop — the single public entry point for power-loop.

Owns a :class:`SessionStore` and gives callers a stateful,
``new_session()`` + ``send(user_input, session_id=...)`` interface, plus
:meth:`follow_up` for steering an in-flight loop without blocking.
Everything else — pipeline orchestration, hooks, events, tool invocation,
persistence, pending-state machine — is wired up internally.

Failure model
-------------
* If a session has unresolved tool_calls from a previous run, :meth:`send`
  records interrupted/unknown results and continues with the new input.
  :meth:`resume` explicitly continues pending work; :meth:`abort_pending`
  explicitly discards it.
* :meth:`close_session` physically deletes the session and its data.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from power_loop._vendor.llm_client.interface import LLMService
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
from power_loop.contracts.errors import (
    CancellationRequested,
    SessionNotFoundError,
)
from power_loop.core.agent_context import (
    get_ctx,
    reset_current_loop,
    reset_effective_tools,
    set_current_loop,
    set_effective_tools,
)
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.pass_turn import ensure_pass_turn
from power_loop.core.pipeline import (
    AgentPipeline,
    _tool_call_args,
    _tool_call_name,
    _truncate_result,
)
from power_loop.core.runner import AgentRunner
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
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
_ENTRY_INDEX_UNSET = object()


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
    #: Cumulative token usage for this send (summed over every LLM call):
    #: {prompt_tokens, completion_tokens, cache_read_tokens, reasoning_tokens,
    #:  total_tokens, calls}. Empty when the run never reached the LLM.
    usage: dict[str, int] = field(default_factory=dict)
    #: Tool invocations executed during this send.
    tool_calls: int = 0
    #: design/188：调 pass_turn 明确结束的、理由；停的那一刻有没有刚折进来的新输入。
    passed: bool = False
    pass_reason: str | None = None
    new_input_after_pass: bool = False
    #: 每一步写的文字：[{step, text, tools}]，按顺序（不只是最后一段）。
    texts: list[dict[str, Any]] = field(default_factory=list)


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


@dataclass
class _SessionSync:
    """The concurrency primitives for ONE session_id, shared by every loop driving it.

    ``users`` includes lock waiters. Once the last operation exits and the inbox
    is drained, release the entry and every loop's reference to it.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Serializes this process's inbox claims for the session (the claim itself is atomic in the
    # store; this just avoids two same-process drains racing for nothing).
    queue_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # In-process HINT of undelivered inbox items this process accepted (+ on accept, − on claim,
    # + on release). The store is the truth (``inbox_counts``); this only backs the sync
    # ``pending_follow_up_count`` and registry cleanup.
    pending_hint: int = 0
    # Set while a ``steer``-mode item waits undelivered (design/124 §7 interrupt points and
    # waiting tools await it); cleared when the inbox is claimed.
    steer: asyncio.Event = field(default_factory=asyncio.Event)
    refs: weakref.WeakValueDictionary = field(default_factory=weakref.WeakValueDictionary)
    users: int = 0
    # 运行中追加的轮数（6.46，extend_rounds）：正在跑的这个 send 在下一轮开始时把它加到轮数上限上。
    round_bonus: int = 0


# session_id -> _SessionSync. PROCESS-GLOBAL and deliberately NOT per-instance state.
#
# "Sends to a session are serialized" is an API promise. It must not silently depend on how many
# StatefulAgentLoop OBJECTS the host happens to have built: with per-instance locks, two loops over
# one session_id take two DIFFERENT locks, so the mutual exclusion evaporates and both drive the
# same session concurrently — interleaved rounds, an interleaved workspace, and (because
# ``follow_up`` decides to fold by testing ``lock.locked()``) a second OWNER run instead of a fold.
# A host hits this simply by replacing a cached loop while a run is in flight, which is a legitimate
# thing to do (config edit) and is invisible from here. Keying the primitives on the session — the
# thing being protected — makes the guarantee hold no matter how the host manages objects.
#
# The follow-up queue is global for the same reason, and it is NOT optional: globalizing only the
# lock would make the second loop enqueue into a queue that the loop actually running the session
# never drains, turning concurrent corruption into silently stranded steering.
#
# Not keyed on the event loop: an ``asyncio.Lock`` binds to the loop that first awaits it, so a
# session driven from two event loops (e.g. mixing the sync API's runner thread with direct async
# calls) is already broken today with per-instance locks. This changes nothing there.
_SESSION_SYNC: dict[str, _SessionSync] = {}


def take_round_bonus(session_id: str | None) -> int:
    """取走这个会话运行中追加的轮数（6.46，extend_rounds）；流水线每轮开始时调。"""
    if not session_id:
        return 0
    entry = _SESSION_SYNC.get(session_id)
    if entry is None or entry.round_bonus <= 0:
        return 0
    n, entry.round_bonus = entry.round_bonus, 0
    return n
# Guards only registry mutation (dict insert/delete) — instances may be constructed from several
# threads. Never held across an await.
_SESSION_SYNC_GUARD = threading.Lock()


def _session_sync(sid: str, *, owner: StatefulAgentLoop | None = None) -> _SessionSync:
    """Fetch (or create) the shared primitives for ``sid``, registering ``owner`` as a user.

    Registry lookup and reference updates do not await. Only creation needs
    the thread guard; acquisition and idle cleanup happen in the session guard.
    """
    entry = _SESSION_SYNC.get(sid)
    if entry is None:
        with _SESSION_SYNC_GUARD:
            entry = _SESSION_SYNC.get(sid)
            if entry is None:
                entry = _SessionSync()
                _SESSION_SYNC[sid] = entry
    if owner is not None:
        entry.refs[id(owner)] = owner
    return entry


def _release_session_sync(sid: str, owner: int) -> None:
    """Drop ``owner``'s claim on ``sid``; delete the entry once nobody holds it and it's idle."""
    with _SESSION_SYNC_GUARD:
        entry = _SESSION_SYNC.get(sid)
        if entry is None:
            return
        entry.refs.pop(owner, None)
        if not entry.refs and not entry.users and not entry.lock.locked() and entry.pending_hint <= 0:
            del _SESSION_SYNC[sid]


def _discard_session_sync(sid: str) -> None:
    """Forget ``sid`` entirely (the session was physically deleted)."""
    with _SESSION_SYNC_GUARD:
        _SESSION_SYNC.pop(sid, None)


class _ExecutionLoop:
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
        task_runtime: Any,
        timer_backend: Any = None,
        session_coordinator: Any = None,
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
        if store is None or task_runtime.manager.store is not store:
            raise ValueError("task_runtime requires its pre-opened authoritative store")
        from power_loop.runtime.timer_backend import TimerBackend
        if timer_backend is not None and not isinstance(timer_backend, TimerBackend):
            raise ValueError('timer_backend must implement create/list/cancel')
        self.timer_backend = timer_backend
        self.task_runtime = task_runtime
        from power_loop.runtime.session_ownership import DEFAULT_SESSION_COORDINATOR
        self.session_coordinator = session_coordinator or DEFAULT_SESSION_COORDINATOR
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
        if self.config.delegation_policy != "single_level":
            raise ValueError("task_runtime requires single_level delegation")
        self.tool_registry = tool_registry
        # Own a FRESH AgentHooks when the caller supplies none — NOT the shared
        # module-level DEFAULT_HOOKS singleton — so per-loop built-in hooks (e.g.
        # the memory recall hook) don't stack across loops or leak config.
        self._runner = AgentRunner(
            event_bus=event_bus, hooks=hooks if hooks is not None else AgentHooks()
        )
        self._register_builtin_hooks()
        # Session locks/queues are PROCESS-GLOBAL (see _SESSION_SYNC): the guarantee is per
        # session, not per loop object. This set records which sessions THIS loop has touched, so
        # aclose() drains only its own and teardown can release its claims.
        self._sync_sids: set[str] = set()
        self._closing = False
        # Identity passed to the coordinator. Unique per process and loop instance,
        # and must NOT survive a restart: a crashed process's lease has to look foreign to its
        # replacement so the replacement can take it over once it expires.
        self._owner_id = f"{os.getpid()}:{uuid.uuid4().hex[:12]}"
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

    async def ensure_store(self) -> SessionStore:
        """Public accessor: return this loop's store, opening an owned one on first use.

        Construction is sync but the store opens lazily on first async use, so ``loop.store``
        is ``None`` until then. Host integrations that need the store up front — e.g. building
        a :class:`~power_loop.runtime.blackboard.StoreBlackboard` to share with the loop —
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
        ``resume()`` / ``submit_input()`` / ``abort_pending()`` (which append
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
        post_state = await store.state_snapshot(sid)
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
        # Only THIS loop's sessions: the locks are shared process-wide now, so waiting on the
        # whole registry would block teardown on runs belonging to other loops.
        locks = [_session_sync(sid).lock for sid in list(self._sync_sids)]
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
        # Release this loop's claims on the shared per-session primitives so the registry doesn't
        # accumulate an entry per session across loop objects. Entries still held by another live
        # loop (or still locked / holding queued steering) survive.
        for sid in list(self._sync_sids):
            _release_session_sync(sid, id(self))
        self._sync_sids.clear()
        # (3) let queued async subscribers finish (best-effort, bounded).
        try:
            await self.event_bus.drain(timeout=drain_timeout_s)
        except Exception:  # pragma: no cover - drain must never block teardown
            logger.warning("aclose: event-bus drain raised; continuing", exc_info=True)
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
            _discard_session_sync(sid)
            self._sync_sids.discard(sid)
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




    def _note_committed_inbox(self, session_id: str) -> None:
        """Re-read pending inputs after a rolled-back batch releases its claims."""
        ctx = self._runtime_worker_context
        ctx.append_overflow = True
        ctx.append_cursor = 0

    def pending_follow_up_count(self, session_id: str) -> int:
        """In-process HINT of undelivered inbox items for ``session_id`` (items this process
        accepted and no drain has claimed yet).

        Steering accepted in the terminal window of a run (after the loop's last round-boundary
        drain) stays in the inbox of the now-idle session; hosts use this after a run returns to
        detect it and hand it to :meth:`flush_follow_ups`. Sync and cheap, but blind to other
        processes and to a restart — :meth:`inbox_pending` asks the store."""
        entry = _SESSION_SYNC.get(session_id)
        return max(0, entry.pending_hint) if entry is not None else 0

    async def inbox_pending(self, session_id: str) -> dict[str, int]:
        """Authoritative ``{"pending": n, "steer": m}`` for ``session_id`` from the store."""
        store = await self._ensure_store()
        return await store.inbox_counts(session_id)

    def extend_rounds(self, session_id: str, rounds: int) -> bool:
        """给**正在跑**的会话加轮数（6.46，DeepTalk design/170）：它在下一轮开始时把轮数上限加上 ``rounds``。
        会话没在跑（锁没被占）返回 False——空闲的会话用 follow_up / send 的 max_rounds。"""
        entry = _SESSION_SYNC.get(session_id)
        if entry is None or not entry.lock.locked() or int(rounds) <= 0:
            return False
        entry.round_bonus += int(rounds)
        return True

    def steer_event(self, session_id: str) -> asyncio.Event:
        """Set while an urgent inbox item for the session is undelivered; the tool batch
        starts nothing new until the next round delivers it."""
        return _session_sync(session_id).steer



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






    async def _require_host_continuation(self, sid: str) -> None:
        from power_loop.runtime.execution_roles import session_role
        from power_loop.runtime.jobs import JobError
        from power_loop.runtime.task_runtime import _OWNER, _SEND
        # Check before taking the session lock or writing an answer: a tool must
        # not recursively resume its own running send or drive another owner.
        if _SEND.get() is not None or _OWNER.get() is not None:
            raise JobError("task_owner_required")
        if await session_role(await self._ensure_store(), sid) != "main":
            raise JobError("task_owner_required")

    async def _interrupt_pending(self, sid: str) -> int:
        """Reuse the send-entry snapshot; only stale pending causes writes."""
        from power_loop.runtime.session_ownership import _CURRENT
        period = _CURRENT.get()
        pending = getattr(None if period is None else period.lease, "pending_json", None)
        # An ordinary turn already read pending_json while taking the session.
        # Empty means there is no tool call to close and no task row to scan.
        if period is not None and hasattr(period.lease, "pending_json") and not pending:
            return 0
        store = await self._ensure_store()
        state = await store.state_snapshot(sid)
        cached = self._session_cache.get(sid)
        cache_valid = (cached is not None and state is not None and
                       cached.next_seq == state.next_seq and cached.last_compact_seq == state.last_compact_seq)
        if cached is not None and not cache_valid:
            self._cache_invalidate(sid)
        from power_loop.runtime.session_ownership import recover_acquired_session
        await recover_acquired_session(self, sid, cache_valid=cache_valid)
        if state is None or not state.pending:
            return 0
        return await self.abort_pending(sid, reason="执行中断，结果未知")

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

    # ── work in flight: list + stop by id (design/124 §8.1 / §8.4) ────────────────────────

    async def list_jobs(self, session_id: str) -> list[dict[str, Any]]:
        from power_loop.runtime import task_query
        from power_loop.runtime.child_tasks import tool_actor
        _, actor = await tool_actor(self)
        result = await task_query.list_jobs(self.task_runtime.manager,
            self.task_runtime.actor(actor.session_id), status="active", limit=100)
        return result["nodes"]

    async def stop_job(self, session_id: str, job_id: str) -> dict:
        from power_loop.runtime.task_control import stop_from_current_call
        return await stop_from_current_call(self, session_id, job_id)


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
            # pipeline 会补上永远在的 pass_turn（design/188）：预览也补，才和模型看到的一样
            tool_registry=ensure_pass_turn(self.tool_registry) if self.tool_registry is not None else None,
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

    @contextlib.asynccontextmanager
    async def _session_lease(self, sid: str):
        from power_loop.runtime.session_ownership import hold_session_ownership
        async with hold_session_ownership(self, sid) as lease:
            yield lease

    @staticmethod
    def _session_busy(sid: str) -> bool:
        entry = _SESSION_SYNC.get(sid)
        return entry is not None and entry.lock.locked()

    @staticmethod
    def _release_idle_sync(sid: str) -> None:
        with _SESSION_SYNC_GUARD:
            entry = _SESSION_SYNC.get(sid)
            if (entry is None or entry.users or entry.lock.locked()
                    or entry.queue_lock.locked() or entry.pending_hint > 0):
                return
            for loop in list(entry.refs.values()):
                loop._sync_sids.discard(sid)
            del _SESSION_SYNC[sid]

    @contextlib.asynccontextmanager
    async def _session_guard(self, sid: str):
        entry = _session_sync(sid, owner=self)
        self._sync_sids.add(sid)
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            self._release_idle_sync(sid)

    async def _claim_inbox(self, sid: str) -> list[dict[str, Any]]:
        """Claim everything waiting in ``sid``'s inbox (see ``SessionStore.inbox_claim``)."""
        store = await self._ensure_store()
        entry = _session_sync(sid, owner=self)
        self._sync_sids.add(sid)
        async with entry.queue_lock:
            rows = await store.inbox_claim(sid)
        entry.pending_hint = max(0, entry.pending_hint - len(rows))
        if rows:
            entry.steer.clear()
        self._release_idle_sync(sid)
        return rows

    async def _drain_follow_up_messages(self, sid: str) -> list[LoopMessage]:
        """Round-boundary drain: claim the inbox and render it as ``<follow_up>`` messages (one
        per run of same-kind items), each marked for same-transaction delivery."""
        return await self._runtime_worker.drain_append(self._runtime_worker_context)

    async def _ensure_session_or_raise(self, sid: str) -> None:
        store = await self._ensure_store()
        if await store.session_snapshot(sid) is None:
            raise SessionNotFoundError(sid)

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
        raw = await store.runtime_snapshot(sid, "send_index", default=0)
        return self._coerce_send_index(raw)

    async def _persist_user_input(self, sid: str, user_input: str | LoopMessage, *,
                                  send_meta: Mapping[str, Any] | None = None) -> int:
        store = await self._ensure_store()
        inbox = None
        if isinstance(user_input, dict) and "_inbox" in user_input:
            # A send fed from the inbox (_send_from_inbox): mark those rows delivered in the same
            # transaction as this user row, and record which items it carries.
            user_input = dict(user_input)
            inbox = user_input.pop("_inbox")
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
        if inbox:
            meta = {**(meta or {}), "inbox": {k: inbox[k] for k in ("item_ids", "kinds")
                                              if k in inbox}}
        if send_meta:
            meta = {**(meta or {}), "send": dict(send_meta)}
        # Allocate the next monotonic SEND index for this session (atomic RMW under the
        # session_state row lock — never resets, unlike round_index). This is the single
        # send-begin point (exactly one user row per send; resume()/follow-up drains do
        # NOT pass through here, so they correctly inherit the in-flight send's index).
        # Stamped into meta so the transcript can delimit sends authoritatively.
        send_index = await store.mutate_runtime_state(
            sid, "send_index", lambda v: int(v or 0) + 1, default=0
        )
        seq = await store.append_message(
            sid, role=role, content=content, name=name, send_index=send_index, meta=meta,
            inbox=inbox,
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
        return send_index

    async def _execute_pending(self, sid: str, sink: SQLiteSink) -> None:
        """Replay leftover tool_calls. Idempotent if there is no pending."""
        store = await self._ensure_store()
        state = await store.get_state(sid)
        if state is None or not state.pending:
            return
        pending = state.pending
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
                    from power_loop.runtime.invocation import invocation_context
                    with invocation_context(sid, pending.get("assistant_seq"), cid, name):
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
                    "content": _truncate_result(output, self.config.tool_result_max_chars),
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

    async def _run_loop(self, sid: str, *, _entry_send_index=_ENTRY_INDEX_UNSET, **kwargs: Any) -> StatefulResult:
        store = await self._ensure_store()
        async with store.send_read_scope(sid) as reads:
            if _entry_send_index is not _ENTRY_INDEX_UNSET:
                reads.runtime["send_index"] = _entry_send_index
                reads.assistant_send_index = _entry_send_index
                reads.assistant_ref = None
            return await self._run_loop_owned(sid, _entry_send_index=_entry_send_index, **kwargs)

    async def _run_loop_owned(self, sid: str, *, _entry_send_index,
                              _runtime_start, **kwargs: Any) -> StatefulResult:
        store = await self._ensure_store()
        # This send can run on a task also hosting its logical node and
        # workflow. Its lease heartbeat must stop the send's operation,
        # not cancel the shared task and bypass its parent's bookkeeping.
        stop = CancellationToken.from_any(kwargs.get("stop_event")).child()
        kwargs["stop_event"] = stop
        send_index = (_entry_send_index if _entry_send_index is not _ENTRY_INDEX_UNSET
                      else await self._current_send_index(store, sid))
        async with self.task_runtime.model_send(self, sid, send_index, cancel_token=stop,
                                                round_limit=kwargs.get('max_rounds'), input_start=_runtime_start) as evidence:
            result = await self._run_with_tool_context(sid, **kwargs)
            _runtime_start.completed(result)
            evidence.update(exit=result.status, usage=dict(result.usage or {}), rounds=result.rounds,
                            new_input_after_pass=bool(result.new_input_after_pass))
            from power_loop.runtime.jobs import JobError
            from power_loop.runtime.send_receipt import KEY, capture
            try:
                evidence[KEY] = capture(result)
            except JobError as exc:
                # Evidence limits cannot turn an already completed model
                # call into a retry. Such exits remain explicitly unreadable.
                evidence["stateful_result_error"] = exc.code
            return result


    async def _run_with_tool_context(self, sid: str, **kwargs: Any) -> StatefulResult:
        from power_loop.runtime.tool_context import ToolContextEngine, active_engine

        store = await self._ensure_store()
        engine = None
        token = active_engine.set(None)  # child must never use its parent's ledger
        try:
            if self.config.tool_context is not None:
                registry = kwargs.get("tools")
                if not hasattr(registry, "get"):
                    registry = getattr(self, "tool_registry", None)
                engine = ToolContextEngine(
                    config=self.config.tool_context, store=store, owner_session_id=sid,
                    owner_send_index=await self._current_send_index(store, sid),
                    llm=self.llm, model=self.config.model, registry=registry,
                    rule=getattr(self.config.representation, "distill_tool_call", None),
                )
                await engine.start()
                active_engine.set(engine)
            result = await self._run_loop_impl(sid, **kwargs)
            try:
                await store.bump_session_stats(
                    sid, result.usage, rounds=result.rounds, tool_calls=result.tool_calls,
                )
            except Exception:
                logger.exception("session_stats bump failed for %s (continuing)", sid)
            return result
        finally:
            active_engine.reset(token)

    async def _run_loop_impl(
        self,
        sid: str,
        *,
        stop_event: CancellationLike,
        sink: SQLiteSink | None = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_rounds: int | None = None,
        response_format: dict[str, Any] | None = None,
        with_history: bool = True,
        _initial_round_index: int = 0,
    ) -> StatefulResult:
        store = await self._ensure_store()
        # 3.0: projection-style representation drives the derived-layer path; verbatim → None
        # (in-place compactor path). The fold trigger/keep come from config.fold_strategy.
        projector = self.config.projection_representation
        from power_loop.runtime.tool_context import active_engine
        engine = active_engine.get()
        if engine is not None and projector is not None:
            import copy
            projector = copy.copy(projector)
            projector.version = engine.projection_version(int(getattr(projector, "version", 0) or 0))
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
        # METADATA for inspection. Per-send originals supply a missing/stale
        # projection; no mode switch performs a runtime migration. Loaded once and
        # reused below for the system-prompt precedence.
        session_row = await store.session_snapshot(sid)
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
                    "per-send originals supply missing or version-mismatched projections.",
                    sid, current_mode, recorded,
                )
        except Exception:
            logger.exception("session %s: history-mode bookkeeping failed (continuing)", sid)

        if not with_history:
            # send(history=False): this run sees only its own send — no past sends, no compact, no
            # projection read — and writes no projection or fold at the end (nothing will read it).
            projection_active = False
            cache_eligible = False
            own = ([r for r in await store.load_active_messages(sid, send_index=current_send_index)
                    if r.name != "compact_note"] if current_send_index is not None else [])
            history = [_row_to_loop_message(r) for r in own]
            hist_seqs = [r.seq for r in own]
            hist_ords = list(hist_seqs)
        else:
            projection_active = projector is not None
            active_rows: list[Any] | None = None
            if projection_active:
                assert current_send_index is not None
                active_rows = [
                    *await store.load_unsent_messages(sid),
                    *await store.load_active_messages(sid, send_index=current_send_index),
                ]

                # ── Projection mode (v2): history = rendered projections of FINISHED sends + the
                # in-flight send's structured rows. pl_messages is NEVER compacted here (compactor is
                # None, enforced by AgentLoopConfig), so it stays in seq order; we partition by the
                # send_index stamped on every row. (active_rows loaded above; current_send_index set.)
                # Out-of-send messages stay a separate prefix. In-place compact
                # notes are converted only by explicit offline maintenance.
                legacy_rows = [
                    r for r in active_rows if r.send_index is None and r.name != "compact_note"
                ]
                current_rows = [
                    r for r in active_rows
                    if r.send_index == current_send_index and r.name != "compact_note"
                ]
                # pl_messages is the immutable audit log. Read a past send only when its
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
                cap = getattr(self.config, "max_context_rows", None)
                limit = int(cap) + 2 if cap and int(cap) > 0 else None

                async def _load_proj_rows() -> list[Any]:
                    raw = await store.load_project_messages(
                        sid, after_send_index=cutoff,
                        **({"last_rows": limit} if limit is not None else {}))
                    rows = [r for r in raw if r.send_index < current_send_index]
                    if limit is not None and len(raw) >= limit and rows:
                        # The oldest send in a LIMIT window may be partial: drop it (the cap below
                        # only ever keeps whole sends from the newest end anyway).
                        oldest = min(r.send_index for r in rows)
                        rows = [r for r in rows if r.send_index != oldest]
                    return rows

                proj_rows = await _load_proj_rows()
                # Past sends inside the window whose current-version projection is missing need
                # their raw rows (heal or verbatim fallback); nothing older is read.
                window_lo = (min(r.send_index for r in proj_rows)
                             if limit is not None and proj_rows else (cutoff or 0) + 1)
                if limit is not None:  # every send renders at least one row
                    window_lo = max(window_lo, current_send_index - limit)
                have = {r.send_index for r in proj_rows if int(r.projector_version or 0) == version}
                for si in range(window_lo, current_send_index):
                    if si not in have:
                        rows = [r for r in await store.load_active_messages(sid, send_index=si)
                                if r.name != "compact_note"]
                        if rows:
                            active_by_send[si] = rows
                # Only rows written by the CURRENT projector version are faithfully renderable by it;
                # rows from another version fall back to verbatim (below), keyed per send.
                cur_proj_by_send: dict[int, list[Any]] = {}
                for r in proj_rows:
                    if int(r.projector_version or 0) == version:
                        cur_proj_by_send.setdefault(r.send_index, []).append(r)

                # ── 6.9.0 self-heal: project past sends that have NO current-version projection ──
                # A send killed mid-flight (process restart / crash) never reaches its end-of-send
                # projection, so its raw rows used to ride VERBATIM into every later send forever
                # (真实事故: one interrupted 80-round send = 182K chars ≈ 115K tokens, every following
                # round took 2–3 minutes and the agent looked dead). Projection is deterministic and
                # LLM-free, so heal lazily HERE: project + persist (idempotent upsert, short lock),
                # then render like any other send. The verbatim fallback below now only covers a
                # projection that itself fails.
                _lo0 = compact.compact_to_send if compact is not None else 0
                _missing = [
                    si for si in sorted(active_by_send)
                    if (_lo0 or 0) < si < current_send_index and si not in cur_proj_by_send
                ]
                if _missing:
                    _healed_any = False
                    for si in _missing:
                        healed = await self._heal_send_projection(
                            store, sid, send_index=si, rows=active_by_send[si], projector=projector,
                        )
                        if healed:
                            _healed_any = True
                    if _healed_any:
                        proj_rows = await _load_proj_rows()
                        cur_proj_by_send = {}
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

                # (1) folded-history compact: always rendered by the projector. A compact row is summary
                # text, not tied to the per-send rendering rules, so a projector version change must not
                # replace it with the verbatim originals of its whole span. Kept in its OWN list — the
                # recent-rows cap below must never drop the compact summary.
                compact_msgs: list[LoopMessage] = []
                if compact is not None:
                    compact_msgs.extend(projector.render([compact]))
                # (2) each uncompacted past send in order: projected when a current-version row exists,
                # else verbatim fallback (missing or stale projection). Collected as PER-SEND chunks so
                # the cap can drop whole sends only (a verbatim-fallback send carries raw tool-protocol
                # rows — cutting inside one would orphan tool results).
                lo = cutoff or 0
                send_chunks: list[list[LoopMessage]] = []
                for si2 in sorted(s for s in {*active_by_send, *cur_proj_by_send} if lo < s < current_send_index):
                    rows_for_send = cur_proj_by_send.get(si2)
                    send_chunks.append(
                        list(projector.render(rows_for_send)) if rows_for_send else _verbatim(si2)
                    )

                legacy_msgs = [_row_to_loop_message(r) for r in legacy_rows]
                current_msgs = [_row_to_loop_message(r) for r in current_rows]
                # ── History-rows cap (max_context_rows) ──
                # 上限管的是**历史**消息的条数，**不含当前 send**：当前 send 内的膨胀由 send 内
                # 保险丝（insend_distill）与切 send 阈值负责，两件事分开才可预期。
                # 6.24.0 之前当前 send 也计入总数，于是一个 120 轮的 send 会把历史整段挤掉——
                # 历史的预算随着本轮跑多久而变，没法解释。
                # compact 行永远保留；当前 send 永远完整；更早的按「整块」从最老端丢（先 legacy，
                # 再一个 send 一块），单块超预算也整块保留，绝不切开一个 send。
                kept_legacy_rows = legacy_rows
                if cap and int(cap) > 0:
                    # chunks[0] = the legacy prefix (oldest), then one chunk per past send.
                    chunks: list[list[LoopMessage]] = [legacy_msgs, *send_chunks]
                    total = sum(len(c) for c in chunks)
                    drop = 0
                    while drop < len(chunks) and total > int(cap):
                        total -= len(chunks[drop])
                        drop += 1
                    if drop > 0:
                        legacy_msgs, kept_legacy_rows = [], []
                        send_chunks = chunks[drop:]

                prefix_msgs = [m for chunk in send_chunks for m in chunk]
                history = legacy_msgs + compact_msgs + prefix_msgs + current_msgs
                # Sink index↔seq map: legacy + in-flight rows carry real seqs; the rendered prefix
                # (projected OR verbatim-fallback) has no foldable DB row (None). Compaction is off in
                # projection mode, so this map is only kept aligned for the appends, never used to fold.
                hist_seqs: list[int | None] = (
                    [r.seq for r in kept_legacy_rows]
                    + [None] * (len(compact_msgs) + len(prefix_msgs))
                    + [r.seq for r in current_rows]
                )
                hist_ords: list[int | None] = list(hist_seqs)
            else:
                # ── Default / degraded-verbatim mode: full verbatim history (+ window cache + in-place
                # compactor). When degrading from projection, active_rows is already loaded above and
                # the cache is bypassed (cache_eligible is False whenever a projector is configured). ──
                cache_token: tuple[int, int] | None = None
                if cache_eligible:
                    state = await store.state_snapshot(sid)
                    if state is not None:
                        cache_token = (state.next_seq, state.last_compact_seq)
                        active_rows = self._cache_get(sid, state.next_seq, state.last_compact_seq)
                if active_rows is None:
                    active_rows = await store.load_active_messages(sid)
                    if cache_eligible and cache_token is not None:
                        self._cache_put(sid, cache_token[0], active_rows, cache_token[1])
                # ── History-rows cap，逐字模式（6.24.0）──
                # 与投影分支同一条语义：只管**历史**的条数，当前 send 与 compact 笔记永远保留。
                # 逐字模式一轮可能有多条消息（assistant(tool_calls) + 每个调用一条 tool 结果），
                # 从最老端按条丢会切出孤儿 tool 结果——这不要紧：下面 align_tool_calls 是
                # mode-agnostic 的兜底，孤儿会被清掉/补占位，发出去的历史始终合法。
                _cap = getattr(self.config, "max_context_rows", None)
                if _cap and int(_cap) > 0 and current_send_index is not None:
                    _hist_idx = [
                        i for i, r in enumerate(active_rows)
                        if getattr(r, "name", None) != "compact_note"
                        and r.send_index is not None
                        and r.send_index < current_send_index
                    ]
                    _over = len(_hist_idx) - int(_cap)
                    if _over > 0:
                        _drop = set(_hist_idx[:_over])
                        active_rows = [r for i, r in enumerate(active_rows) if i not in _drop]
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
        for row in active_rows if with_history else own:
            sink.remember_input(row.seq, row.role, row.content, row.meta)
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
        # Per-call structured output. Must be a per-run copy, never a mutation of self.config:
        # one loop object commonly serves many sessions concurrently (a host keyed by agent
        # definition, say), and flipping the shared config into JSON mode would silently force
        # every other in-flight send to answer in JSON too.
        if response_format is not None and response_format != self.config.response_format:
            _overrides["response_format"] = response_format
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
            changes, changes_token = store.begin_polling(sid)
            try:
                async def _drain_follow_ups(*, force=False) -> list[LoopMessage]:
                    if force and getattr(self, "_lightweight_turn", False):
                        force = False
                    await changes.refresh(force=force)
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
                    drain_follow_ups_final=lambda: _drain_follow_ups(force=True),
                    steer_event=_session_sync(sid, owner=self).steer,
                )
                # Every row this run appends (assistant/tool/system, plus follow-up and
                # trailing-notice rows drained mid-run) inherits the current send index, so
                # the whole send shares one index. _persist_user_input bumped it for a fresh
                # send; resume()/submit_input() leave it, correctly attaching to the prior send.
                pipeline.send_index = current_send_index
                pipeline._initial_round_index = _initial_round_index
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
                store.end_polling(changes_token)
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
        # Send-context projection (v2): at end-of-send, project the finished send's pl_messages
        # rows into pl_project_messages so the NEXT send reads them as plain-text history. Derived
        # + best-effort: a failure never affects the send result (pl_messages is the source of truth).
        # Skipped when projection degraded to verbatim this run (projection_active=False).
        if projection_active and current_send_index is not None:
            try:
                await self._write_send_projection(
                    store, sid, send_index=current_send_index, status=result.status,
                    session_row=session_row, projector=projector, tool_registry=effective_registry,
                    stop_event=stop_event,
                )
            except CancellationRequested:
                raise
            except Exception:
                logger.exception("send-context projection write failed for %s (continuing)", sid)
        return StatefulResult(
            session_id=sid,
            status=result.status,
            final_text=result.final_text,
            rounds=result.rounds,
            pending_tool_calls=result.pending_tool_calls,
            usage=result.usage,
            tool_calls=result.tool_calls,
            passed=result.passed,
            pass_reason=result.pass_reason,
            new_input_after_pass=result.new_input_after_pass,
            texts=result.texts,
        )


    async def _heal_send_projection(
        self,
        store: SessionStore,
        sid: str,
        *,
        send_index: int,
        rows: list[Any],
        projector: Representation,
    ) -> bool:
        """Project + persist ONE past send that has no current-version projection rows (6.9.0).

        Used at context-build time for sends whose end-of-send projection never ran (killed
        mid-flight) or was written by another projector version. Deterministic, no LLM, no fold
        (the fold still only runs at end-of-send). Returns True when rows were written. Never
        raises — a failure keeps the caller's verbatim fallback."""
        try:
            projected = projector.project_send(
                rows, send_index=send_index, tool_registry=getattr(self, "tool_registry", None)
            )
            from power_loop.runtime.tool_context import active_engine
            engine = active_engine.get()
            if engine is not None:
                projected = await engine.prepare_projection(projected, rows, send_index)
            if not projected.rows:
                return False
            await store.write_send_projection_rows(
                sid, send_index=send_index,
                rows=[(pr.kind, pr.content, pr.rendered_text) for pr in projected.rows],
                source_seq_lo=projected.source_seq_lo, source_seq_hi=projected.source_seq_hi,
                projector_version=int(getattr(projector, "version", 0) or 0),
                load_snapshot=False,
            )
            logger.info(
                "projection self-heal: session %s send %d projected lazily (%d raw rows)",
                sid, send_index, len(rows),
            )
            return True
        except Exception:  # noqa: BLE001 — heal is best-effort; verbatim fallback still works
            logger.warning(
                "projection self-heal failed for session %s send %d", sid, send_index, exc_info=True
            )
            return False

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
        stop_event: CancellationLike = None,
    ) -> None:
        """Project the just-finished send's ``pl_messages`` rows into ``pl_project_messages`` (v2).

        Includes child sessions regardless of their tool-context/fuse setting.
        Defers on a still-in-flight status (the resume re-finalizes under the SAME
        ``send_index``; the idempotent UPSERT then overwrites). Best-effort — the caller swallows
        failures, since ``pl_messages`` remains the source of truth."""
        if status == "pending_tools":
            return  # send not finished; the resume will re-finalize under the same send_index
        # 只取本 send 的行：这里要投影的就是这一个 send，从前是读全表再在 Python 里过滤，
        # 等于每个 send 都为一小段做一次 O(会话) 的读。投影模式下 pl_messages 永不压缩。
        rows = await store.load_active_messages(sid, send_index=send_index)
        if not rows:
            return
        projected = projector.project_send(rows, send_index=send_index, tool_registry=tool_registry)
        from power_loop.runtime.tool_context import active_engine
        engine = active_engine.get()
        if engine is not None:
            projected = await engine.prepare_projection(projected, rows, send_index)
        version = int(getattr(projector, "version", 0) or 0)
        proj_rows_to_write = [(pr.kind, pr.content, pr.rendered_text) for pr in projected.rows]

        # Update small statistics with the projection writes; below-threshold sends
        # do not snapshot historical bodies. Candidate/repair reads and the model
        # fold run outside locks. Commit checks the cursor AND projection revision.
        strategy = self.config.fold_strategy
        keep = int(getattr(strategy, "keep_last_sends", 0) or 0)
        ratio = float(getattr(strategy, "trigger_ratio", 0.75) or 0.75)
        threshold = int((self.config.effective_context_budget() or 8000) * ratio)
        cap = self.config.max_context_rows
        row_threshold = cap * ratio if cap and cap > 0 else None
        policy_provider = getattr(projector, "fold_statistics_policy", None)
        policy = policy_provider() if policy_provider else None
        candidate = await store.prepare_send_projection_fold(
            sid, send_index=send_index, rows=proj_rows_to_write,
            source_seq_lo=projected.source_seq_lo, source_seq_hi=projected.source_seq_hi,
            projector_version=version, projector=projector, keep_last_sends=keep, threshold=threshold,
            row_threshold=row_threshold,
        )
        if candidate is None:
            return
        prior, snapshot = candidate.prior, candidate.rows
        plan, note_ops = await self._plan_and_run_projection_fold(
            sid, projector, prior, snapshot, stop_event=stop_event)
        if plan is None:
            return
        if policy_provider and policy_provider() != policy:
            return  # render configuration changed during the asynchronous fold
        content, rendered_text, from_send, to_send = plan
        committed = await store.commit_projection_fold(
            sid, content=content, rendered_text=rendered_text, from_send=from_send,
            to_send=to_send, projector_version=version,
            expected_prior_to_send=(prior.compact_to_send if prior is not None else None),
            expected_projection_revision=candidate.revision,
        )
        if committed:
            await self._apply_fold_notes(store, sid, note_ops)

    async def _plan_and_run_projection_fold(
        self, sid: str, projector: Representation,
        prior: ProjectMessageRow | None, snapshot: list[ProjectMessageRow],
        stop_event: CancellationLike = None,
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
            from power_loop.runtime.fold import fold_due
            cap = self.config.max_context_rows
            row_threshold = cap * trigger_ratio if cap and cap > 0 else None
            if not fold_due(estimate_tokens(rendered_prefix), threshold, len(snapshot), row_threshold):
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
            calls = self._fold_calls(sid, stop_event)
            ctx = self._fold_context(sid, projector, calls)
            fr = await self._run_fold_with_timeout(fold_strategy, to_compact, ctx)
            if calls is not None:
                calls.raise_if_cancelled()
            if fr is None:
                return None, ()
            if fr.folded_to_send not in fold_sends:
                logger.warning("projection fold returned an unprocessed cursor for %s; skipping", sid)
                return None, ()
            return (fr.content, fr.rendered_text, from_send, fr.folded_to_send), tuple(fr.note_ops)
        except CancellationRequested:
            raise
        except Exception:
            logger.exception(
                "projection fold planning failed for %s (skipping fold; rows still written)", sid,
            )
            return None, ()

    def _fold_calls(self, sid: str, stop_event: CancellationLike) -> Any:
        """Fold model calls use the same task accounting as ordinary rounds."""
        from power_loop.runtime.fold_calls import FoldCalls
        return FoldCalls(session_id=sid, bus=self._runner.event_bus, cancel_token=stop_event)

    def _fold_context(self, sid: str, projector: Representation, calls: Any) -> Any:
        from power_loop.runtime.fold import FoldContext
        summary = getattr(self.config.fold_strategy, "summary_llm", None) or self.llm
        return FoldContext(session_id=sid, round_index=0, representation=projector,
                           llm=calls.wrap(self.llm), summary_llm=calls.wrap(summary),
                           memory=self.config.memory, tool_registry=self.tool_registry,
                           max_tokens=self.config.max_tokens)

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


# ── helpers ──────────────────────────────────────────────────────────────


def _row_to_loop_message(row: MessageRow) -> LoopMessage:
    msg: LoopMessage = {"role": row.role}
    if row.role == "assistant" and (row.meta or {}).get("reasoning_content"):
        msg["reasoning_content"] = row.meta["reasoning_content"]
        msg["_reasoning_model"] = row.meta["reasoning_model"]
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


class StatefulAgentLoop:
    """A client facade. Execution and resource lifetime belong to RuntimeWorker.

    Construct a Runtime with a registered factory, start its worker, and pass
    that runtime here. Every operation keeps the caller's stable request ID and
    authenticated identity. Closing this facade only closes its own client.
    """

    def __init__(self, *, runtime):
        from .runtime_client import RuntimeClient
        self.runtime = runtime
        self.store = runtime.store
        self.client = RuntimeClient(store=runtime.store, transport=runtime.transport, namespace=runtime.namespace,
                                    hooks=runtime.hooks, hook_timeout_s=runtime.hook_timeout_s)

    async def new_session(self, descriptor, *, actor):
        return await self.runtime.open_session(descriptor, actor=actor)

    async def send(self, target, body, *, message_id, actor, deadline_ms, options=None, descriptor=None):
        receipt = await self.client.send(target, body, message_id=message_id, actor=actor,
                                         options=options, descriptor=descriptor, deadline_ms=deadline_ms)
        return await self.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline_ms)

    async def resume(self, ref, *, message_id, actor, body=None, deadline_ms=None):
        return await self.client.resume(ref, message_id=message_id, actor=actor, body=body, deadline_ms=deadline_ms)

    async def aclose(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


__all__ = ["StatefulAgentLoop", "StatefulResult", "MessageState"]
