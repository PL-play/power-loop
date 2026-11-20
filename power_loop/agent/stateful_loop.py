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
  :meth:`abort_pending` records aborted results for the remaining calls.
* :meth:`close_session` physically deletes the session and its data.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import weakref
from collections import OrderedDict
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from power_loop._vendor.llm_client.interface import LLMService
from power_loop.agent.runtime_types import RuntimeRequestError
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
    _tool_call_name,
)
from power_loop.core.runner import AgentRunner
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.history_sanitize import align_tool_calls
from power_loop.runtime.store.schema import SchemaPolicy
from power_loop.runtime.store.store import (
    _PROJECT_KIND_ORDER,
    DEFAULT_DB_PATH,
    DEFAULT_TABLE_PREFIX,
    MAX_SPAWN_DEPTH,
    SessionStore,
)
from power_loop.runtime.store.types import (
    MessageRow,
    MessageState,
    ProjectMessageRow,
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
    ``rows`` iff BOTH still match the live ``sessions`` row. ``next_seq`` alone is insufficient
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
    #: design 215: seq of this send's last assistant row — the result message final() stamps.
    result_seq: int | None = None
    #: The exact meta persisted for that assistant row (reasoning, round usage). final()
    #: merges the result fields into it instead of reading the row back.
    result_meta: dict[str, Any] | None = None
    #: design 217: highest next_seq reached during this send (persisted back in Final).
    next_seq: int | None = None


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
    """Private executor constructed by RuntimeWorker after a session claim.

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
        timer_backend: Any = None,
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
        if store is None:
            raise ValueError("a pre-opened authoritative store is required")
        from power_loop.runtime.timer_backend import TimerBackend
        if timer_backend is not None and not isinstance(timer_backend, TimerBackend):
            raise ValueError('timer_backend must implement create/list/cancel')
        self.timer_backend = timer_backend
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
            raise ValueError("session runtime requires single_level delegation")
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
        one reload on the first send. Pending tools are aligned as unknown results before new work."""
        store = await self._ensure_store()
        if await store.get_session(session_id) is None:
            return False
        if self._session_cache_size > 0:
            state = await store.get_state(session_id)
            rows = await store.load_active_messages(session_id)
            if state is not None:
                self._cache_put(session_id, state.next_seq, rows, state.last_compact_seq)
        return True







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
        from .pending_tools import read_pending_tools
        source, tool_calls = await read_pending_tools(
            store, store._db, session_id, state.pending, state.next_seq,
        )
        if not tool_calls:
            await store.set_pending(session_id, None)
            return 0
        round_index, send_index = source["round_index"], source["send_index"]
        sink = SQLiteSink(store, session_id)
        sink._unresolved = {str(tc.get("id") or "") for tc in tool_calls}
        sink._assistant_seq = source["seq"]
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
        """Return only the durable assistant-seq marker; do not hydrate call arrays."""
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
        # sessions row lock — never resets, unlike round_index). This is the single
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

    async def _run_session_send(
        self, sid: str, *, batch, ownership, config: AgentLoopConfig,
        tool_registry: ToolRegistry | None, hooks: AgentHooks, event_bus: AgentEventBus,
        stop_event: CancellationLike, system_prompt: str | None = None,
        max_rounds: int | None = None, response_format: dict[str, Any] | None = None,
        with_history: bool = True, boundary=None,
    ) -> StatefulResult:
        """Run one claimed send on the session-native boundary (design 215 §7–§8).

        The claim transaction already wrote the merged user message and installed the
        owner fence. This method reads history ONCE, appends every model/tool/follow-up
        row to memory and the store together, and builds the end-of-send projection from
        the in-memory rows. The caller (worker) writes the result message + input terminal
        states through ``RuntimeInputStore.final``.
        """
        from power_loop.runtime.task_runtime import _SESSION_SEND, SessionSend
        store = await self._ensure_store()
        send = SessionSend(store=store, target=batch.target, batch=batch, ownership=ownership)
        token = _SESSION_SEND.set(send)
        try:
            async with store.send_read_scope(sid, ownership=ownership) as reads:
                reads.runtime["send_index"] = batch.send_index
                reads.assistant_send_index = batch.send_index
                reads.assistant_ref = None
                # The claim already wrote the merged user message at batch.message_seq;
                # seed the cursor and piggyback boundary markers (design 217).
                reads.next_seq = batch.message_seq + 1
                reads.wake_inbox_seq = batch.wake_inbox_seq
                reads.cancel_seq = batch.cancel_seq
                reads.background_version = batch.background_version
                reads.runtime_version = batch.runtime_version
                # The claim's post-lock session row already tells us when the session was last
                # active. Seed the existing touch throttle with that fact; a scope holding a
                # newer value keeps it. No extra read/cache: the claim write fence still runs.
                reads.touched = max(reads.touched, batch.session_updated_at)
                res = await self._run_session_send_impl(
                    sid, batch=batch, ownership=ownership, send=send, config=config,
                    tool_registry=tool_registry, hooks=hooks, event_bus=event_bus,
                    stop_event=stop_event, system_prompt=system_prompt, max_rounds=max_rounds,
                    response_format=response_format, with_history=with_history, boundary=boundary,
                )
                res.next_seq = reads.next_seq
                return res
        finally:
            _SESSION_SEND.reset(token)

    async def _run_session_send_impl(
        self, sid: str, *, batch, ownership, send, config, tool_registry, hooks, event_bus,
        stop_event, system_prompt, max_rounds, response_format, with_history, boundary=None,
    ) -> StatefulResult:
        store = self.store
        projector = config.representation
        current_send_index = batch.send_index
        sink = SQLiteSink(store, sid)
        sink.send_index = current_send_index

        # ── ONE bounded entry read from the projection layer (design 216 §7.4). Past sends
        # reach the model only through their stored projections; the current send's raw rows
        # are the claim's merged user message plus everything appended in memory below. ──
        # The claim already returned the merged user row it wrote in this send's transaction
        # (content, seq, send_index, meta). Seed it instead of selecting the row back.
        claim_row = batch.user_row
        current_raw_rows: list[Any] = [claim_row]
        if not with_history:
            history = [_row_to_loop_message(claim_row)]
            hist_seqs: list[int | None] = [claim_row.seq]
            hist_ords: list[int | None] = list(hist_seqs)
        else:
            version = int(getattr(projector, "version", 0) or 0)
            cap = getattr(config, "max_context_rows", None)
            send_limit = int(cap) + 1 if cap and int(cap) > 0 else None
            window = await store.load_projection_window(
                sid, current_send_index=current_send_index, send_limit=send_limit)
            compact = window.compact
            cutoff = compact.compact_to_send if compact is not None else 0
            # Group the window rows ONCE by send. Rendering then walks each send's rows in
            # order; never scan the whole projection list per send.
            by_send: dict[int, list[Any]] = {}
            for row in window.rows:
                by_send.setdefault(int(row.send_index), []).append(row)
            for rows in by_send.values():
                rows.sort(key=lambda r: _PROJECT_KIND_ORDER.get(r.kind, 9))
            for si in range(int(window.window_lo), current_send_index):
                rows = by_send.get(si, [])
                current_rows = [r for r in rows if int(r.projector_version or 0) == version]
                stale = len(current_rows) != len(rows)  # at least one stale/other-version row
                if current_rows and not stale:
                    continue
                # A send with any stale row must be rebuilt even if a sibling row is current;
                # a send with no rows at all is only rebuilt when original audit rows exist.
                raw = [r for r in await store.load_active_messages(sid, send_index=si)
                       if r.name != "compact_note"]
                if not raw:
                    if stale:
                        raise RuntimeRequestError(
                            "projection_unavailable",
                            f"send {si}: stale projection has no active audit rows to rebuild it",
                        )
                    continue  # truly empty index: no projection and no raw history
                try:
                    repaired = await self._heal_send_projection(
                        store, sid, send_index=si, rows=raw, projector=projector,
                        ownership=ownership)
                except Exception as exc:
                    raise RuntimeRequestError(
                        "projection_unavailable",
                        f"send {si}: projection repair failed: {exc}",
                    ) from exc
                if not repaired:
                    raise RuntimeRequestError(
                        "projection_unavailable",
                        f"send {si}: projection repair produced no rows",
                    )
                # Heal wrote projections/statistics after the window snapshot. Keep repaired
                # rows for model assembly, but discard the old facts used by the later save.
                window = None
                by_send[si] = list(repaired)
            flat_rows = [r for si2 in sorted(by_send) for r in by_send[si2]
                         if cutoff < si2 < current_send_index]
            if hasattr(projector, "stamp_render_context"):
                projector.stamp_render_context(flat_rows, current_send_index)
            compact_msgs: list[LoopMessage] = []
            if compact is not None:
                compact_msgs.extend(projector.render([compact]))
            send_chunks: list[list[LoopMessage]] = [
                list(projector.render(by_send[si2]))
                for si2 in sorted(by_send) if cutoff < si2 < current_send_index
            ]
            current_msgs = [_row_to_loop_message(claim_row)]
            if cap and int(cap) > 0:
                # The cap counts rendered MESSAGES and drops whole sends from the oldest end;
                # a send is never split. The entry read itself is already bounded by send.
                total = len(compact_msgs) + sum(len(c) for c in send_chunks)
                drop = 0
                while drop < len(send_chunks) and total > int(cap):
                    total -= len(send_chunks[drop])
                    drop += 1
                if drop:
                    send_chunks = send_chunks[drop:]
            prefix_msgs = [m for chunk in send_chunks for m in chunk]
            history = compact_msgs + prefix_msgs + current_msgs
            hist_seqs = ([None] * (len(compact_msgs) + len(prefix_msgs))
                         + [claim_row.seq])
            hist_ords = list(hist_seqs)

        current_raw_rows = [_render_batch_row(r) for r in current_raw_rows]
        history = _render_batch_history(history)
        history, hist_seqs, hist_ords, dropped_seqs, synthesized = align_tool_calls(
            history, hist_seqs, hist_ords)
        if dropped_seqs and config.repair_corrupt_history:
            await store.deactivate_messages(sid, dropped_seqs)
        sink.init_history_seqs(hist_seqs, hist_ords)
        sink.send_rows = list(current_raw_rows)
        for row in current_raw_rows:
            sink.remember_input(row.seq, row.role, row.content, row.meta)

        effective_sp = system_prompt if system_prompt is not None else config.system_prompt
        _overrides: dict[str, Any] = {}
        if effective_sp is not None and effective_sp != config.system_prompt:
            _overrides["system_prompt"] = effective_sp
        if max_rounds is not None and int(max_rounds) != config.max_rounds:
            _overrides["max_rounds"] = max(1, int(max_rounds))
        if response_format is not None and response_format != config.response_format:
            _overrides["response_format"] = response_format
        runtime_config = replace(config, **_overrides) if _overrides else config
        effective_registry = self._resolve_registry(tool_registry)
        _eff_tools = (frozenset(effective_registry.names())
                      if tool_registry is not None and effective_registry is not None else None)

        stop = CancellationToken.from_any(stop_event).child()
        send.stop_renewal = stop
        async with self._runner.session_async(session_id=sid):
            loop_token = set_current_loop(self)
            tools_token = set_effective_tools(_eff_tools)
            changes, changes_token = store.begin_polling(sid)
            changes.versions = (
                batch.inbox_version, batch.background_version, batch.runtime_version,
            )
            try:
                async def _drain(*, force=False):
                    if boundary is None:
                        return []
                    return await boundary(final=force)

                pipeline = AgentPipeline(
                    llm=self.llm, config=runtime_config, tool_registry=effective_registry,
                    hooks=self._runner.hooks, bus=self._runner.event_bus, ctx=get_ctx(),
                    session_id=sid, stop_event=stop,
                    sink=sink, store=store, drain_follow_ups=_drain,
                    drain_follow_ups_final=lambda: _drain(force=True),
                    steer_event=_session_sync(sid, owner=self).steer,
                )
                pipeline.send_index = current_send_index
                try:
                    result: AgentLoopResult = await pipeline.run(history)
                except Exception as exc:
                    await pipeline._emit_error_terminal(exc)
                    raise
            finally:
                store.end_polling(changes_token)
                reset_effective_tools(tools_token)
                reset_current_loop(loop_token)

        # ── End-of-send projection FROM MEMORY (design 215 §P2.4): no re-read of the
        # current send's raw rows. The derived write carries the owner fence + coverage. ──
        if with_history:
            try:
                await self._write_session_projection(
                    store, sid, send_index=current_send_index, rows=sink.send_rows,
                    status=result.status, projector=projector, tool_registry=effective_registry,
                    ownership=ownership, stop_event=stop, window=window,
                )
            except CancellationRequested:
                raise
            except Exception:
                logger.exception("session projection write failed for %s (continuing)", sid)
        return StatefulResult(
            session_id=sid, status=result.status, final_text=result.final_text, rounds=result.rounds,
            pending_tool_calls=result.pending_tool_calls, usage=result.usage, tool_calls=result.tool_calls,
            passed=result.passed, pass_reason=result.pass_reason,
            new_input_after_pass=result.new_input_after_pass, texts=result.texts,
            result_seq=sink.last_assistant_seq, result_meta=sink.last_assistant_meta,
        )

    async def _write_session_projection(
        self, store, sid, *, send_index, rows, status, projector, tool_registry, ownership, stop_event,
        window=None,
    ) -> None:
        """Project the finished send from its in-memory raw rows, fenced by the owner."""
        if status == "pending_tools" or not rows:
            return
        projected = projector.project_send(rows, send_index=send_index, tool_registry=tool_registry)
        from power_loop.runtime.tool_context import active_engine
        engine = active_engine.get()
        if engine is not None:
            projected = await engine.prepare_projection(projected, rows, send_index)
        version = int(getattr(projector, "version", 0) or 0)
        proj_rows_to_write = [(pr.kind, pr.content, pr.rendered_text) for pr in projected.rows]
        strategy = self.config.fold_strategy
        keep = int(getattr(strategy, "keep_last_sends", 0) or 0)
        ratio = float(getattr(strategy, "trigger_ratio", 0.75) or 0.75)
        threshold = int((self.config.effective_context_budget() or 8000) * ratio)
        cap = self.config.max_context_rows
        row_threshold = cap * ratio if cap and cap > 0 else None
        candidate = await store.prepare_send_projection_fold(
            sid, send_index=send_index, rows=proj_rows_to_write,
            source_seq_lo=projected.source_seq_lo, source_seq_hi=projected.source_seq_hi,
            projector_version=version, projector=projector, keep_last_sends=keep, threshold=threshold,
            row_threshold=row_threshold, ownership=ownership, window=window,
        )
        if candidate is None:
            return
        prior, snapshot = candidate.prior, candidate.rows
        plan, note_ops = await self._plan_and_run_projection_fold(
            sid, projector, prior, snapshot, stop_event=stop_event)
        if plan is None:
            return
        content, rendered_text, from_send, to_send = plan
        committed = await store.commit_projection_fold(
            sid, content=content, rendered_text=rendered_text, from_send=from_send,
            to_send=to_send, projector_version=version,
            expected_prior_to_send=(prior.compact_to_send if prior is not None else None),
            expected_projection_revision=candidate.revision, ownership=ownership,
        )
        if committed:
            await self._apply_fold_notes(store, sid, note_ops)




    async def _heal_send_projection(
        self,
        store: SessionStore,
        sid: str,
        *,
        send_index: int,
        rows: list[Any],
        projector: Representation,
        ownership=None,
    ) -> list[ProjectMessageRow]:
        """Project + persist ONE past send that has no current-version projection rows.

        Used at context-build time for sends whose end-of-send projection never ran (killed
        mid-flight) or was written by another projector version. Deterministic, no LLM, no fold
        (the fold still only runs at end-of-send). Returns the rows that are now usable for
        rendering. A failure is raised to the caller, which turns it into
        ``projection_unavailable`` before the first model call; it is never replaced with raw
        history."""
        version = int(getattr(projector, "version", 0) or 0)
        projected = projector.project_send(
            rows, send_index=send_index, tool_registry=getattr(self, "tool_registry", None)
        )
        from power_loop.runtime.tool_context import active_engine
        engine = active_engine.get()
        if engine is not None:
            projected = await engine.prepare_projection(projected, rows, send_index)
        if not projected.rows:
            return []
        await store.write_send_projection_rows(
            sid, send_index=send_index,
            rows=[(pr.kind, pr.content, pr.rendered_text) for pr in projected.rows],
            source_seq_lo=projected.source_seq_lo, source_seq_hi=projected.source_seq_hi,
            projector_version=version, load_snapshot=False, ownership=ownership,
        )
        logger.info(
            "projection self-heal: session %s send %d projected lazily (%d raw rows)",
            sid, send_index, len(rows),
        )
        return [
            ProjectMessageRow(
                session_id=sid, send_index=int(send_index), kind=pr.kind, content=pr.content,
                rendered_text=pr.rendered_text, source_seq_lo=projected.source_seq_lo,
                source_seq_hi=projected.source_seq_hi, compact_from_send=None,
                compact_to_send=None, projector_version=version, token_estimate=None,
                created_at=0, completed_through_send_index=int(send_index),
            )
            for pr in projected.rows
        ]


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
            fr = await self._run_fold_with_timeout(fold_strategy, to_compact, ctx, calls=calls)
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

    async def _run_fold_with_timeout(self, fold_strategy: Any, rows: Any, ctx: Any, *, calls: Any) -> Any | None:
        """Discard a timed-out fold while preserving cancellation of the send."""
        timeout = self.config.fold_timeout_s
        deadline = asyncio.timeout(timeout if timeout is not None and timeout > 0 else None)
        try:
            async with deadline:
                result = await fold_strategy.fold(rows, context=ctx)
                # Fold strategies may absorb model errors. Re-raise cancellation
                # inside the deadline so asyncio can consume its own request.
                calls.raise_if_cancelled()
        except TimeoutError:
            timed_out = True
        else:
            timed_out = deadline.expired()
        if calls.cancel_token is not None:
            calls.cancel_token.raise_if_cancelled()
        if timed_out:
            logger.warning(
                "projection fold timed out after %ss for %s (skipping fold this send)",
                timeout, ctx.session_id,
            )
            return None
        return result

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

# ── helpers ──────────────────────────────────────────────────────────────


def _render_batch_row(row: MessageRow) -> MessageRow:
    """Render a merged-batch user row's canonical JSON body to deterministic text.

    The raw message keeps the JSON for audit; the model reads the equivalent text
    (design 215 §7.3). Non-batch rows pass through unchanged."""
    from power_loop.agent.follow_up import is_batch_content, render_batch_content
    if row.role != "user" or (row.meta or {}).get(CONTENT_ENCODING_META_KEY) != CONTENT_ENCODING_JSON:
        return row
    try:
        content = json.loads(row.content) if row.content is not None else None
    except (ValueError, TypeError):
        return row
    if not is_batch_content(content):
        return row
    rendered = render_batch_content(content)
    if isinstance(rendered, list):
        return replace(row, content=json.dumps(rendered, ensure_ascii=False))
    meta = {k: v for k, v in (row.meta or {}).items() if k != CONTENT_ENCODING_META_KEY}
    return replace(row, content=rendered, meta=meta)


def _render_batch_history(history: list[LoopMessage]) -> list[LoopMessage]:
    from power_loop.agent.follow_up import is_batch_content, render_batch_content
    out = []
    for msg in history:
        if msg.get("role") == "user" and is_batch_content(msg.get("content")):
            out.append({**msg, "content": render_batch_content(msg["content"])})
        else:
            out.append(msg)
    return out


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
        self.runtime = runtime
        self.store = runtime.store
        self.client = runtime.client

    async def new_session(self, descriptor, *, actor):
        return await self.runtime.open_session(descriptor, actor=actor)

    async def send(
        self, target, body, *, message_id, actor, priority="normal", deadline_ms=None,
        wake=True, kind="user", reply_route=None,
    ):
        """Accept once, then wait through the subscription opened before submission."""
        output = await self.client.send_and_subscribe(
            target, body, message_id=message_id, actor=actor, priority=priority,
            deadline_ms=deadline_ms, wake=wake, kind=kind, reply_route=reply_route,
        )
        try:
            return await output.wait(deadline_ms=deadline_ms)
        finally:
            await output.aclose()

    async def configure_session(self, target, *, actor, descriptor=None, cancel_propagation=None):
        return await self.client.configure_session(
            target, actor=actor, descriptor=descriptor, cancel_propagation=cancel_propagation,
        )

    async def inspect_input(self, input_ref, *, actor):
        return await self.client.inspect_input(input_ref, actor=actor)

    async def wait(self, input_ref, *, actor, deadline_ms=None):
        return await self.client.wait(input_ref, actor=actor, deadline_ms=deadline_ms)

    async def inspect(self, ref, *, actor):
        return await self.client.inspect(ref, actor=actor)

    async def messages(self, target, *, actor, after_seq=None, limit=20):
        return await self.client.messages(target, actor=actor, after_seq=after_seq, limit=limit)

    async def cancel(self, ref, *, actor, scope="self", reason="cancelled"):
        return await self.client.cancel(ref, actor=actor, scope=scope, reason=reason)

    async def list_children(self, target, *, actor, cursor=None, limit=100):
        return await self.client.list_children(target, actor=actor, cursor=cursor, limit=limit)

    async def aclose(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


__all__ = ["StatefulAgentLoop", "StatefulResult", "MessageState"]
