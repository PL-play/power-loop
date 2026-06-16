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
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

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
from power_loop.runtime.cancellation import CancellationLike
from power_loop.runtime.skills import SkillLoader
from power_loop.runtime.store.store import (
    DEFAULT_DB_PATH,
    MAX_SPAWN_DEPTH,
    SessionStore,
)
from power_loop.runtime.store.types import (
    MessageRow,
    MessageState,
    SubagentLifecycle,
)
from power_loop.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
        db_path: str = DEFAULT_DB_PATH,
        config: AgentLoopConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        hooks: AgentHooks | None = None,
        event_bus: AgentEventBus | None = None,
        max_spawn_depth: int | None = None,
    ) -> None:
        self.llm = llm
        # The store is async (``await SessionStore.open(...)``), but construction is
        # sync — so an owned store is opened LAZILY on the first async use via
        # :meth:`_ensure_store`. An explicitly-passed store is already open.
        self.store: SessionStore | None = store
        self._db_path = db_path
        self._explicit_max_spawn_depth = max_spawn_depth
        if store is not None and max_spawn_depth is not None:
            # An explicit limit overrides the (possibly shared) store's setting.
            store.max_spawn_depth = max_spawn_depth
        self._owns_store = store is None
        self._store_open_lock = asyncio.Lock()
        self.config = config if config is not None else AgentLoopConfig()
        self.tool_registry = tool_registry
        self._runner = AgentRunner(event_bus=event_bus, hooks=hooks)
        self._locks: dict[str, asyncio.Lock] = {}
        self._follow_up_queues: dict[str, list[str | LoopMessage]] = {}
        self._follow_up_queue_locks: dict[str, asyncio.Lock] = {}
        self._closing = False

    async def _ensure_store(self) -> SessionStore:
        """Return the loop's store, opening an owned one on first use.

        Construction is sync but the store is async, so the SQLite backend (and its
        schema) is opened the first time any async method needs it. An explicitly
        supplied store is returned unchanged.
        """
        if self.store is not None:
            return self.store
        async with self._store_open_lock:
            if self.store is None:
                self.store = await SessionStore.open(
                    self._db_path,
                    max_spawn_depth=(
                        MAX_SPAWN_DEPTH if self._explicit_max_spawn_depth is None
                        else self._explicit_max_spawn_depth
                    ),
                )
        return self.store

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying store (if owned). Does NOT delete sessions.

        Synchronous and abrupt: it does NOT wait for in-flight sends or pending async
        event-bus tasks. The store is async, so this can only close cleanly when no
        event loop is running (it drives ``store.close()`` via ``asyncio.run``); when
        called from inside a running loop it schedules the close and warns. Prefer
        :meth:`aclose` (or ``async with loop:``) for graceful shutdown.
        """
        if not self._owns_store or self.store is None:
            return
        store = self.store
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(store.close())
        else:
            logger.warning(
                "StatefulAgentLoop.close() called inside a running event loop; "
                "use 'await loop.aclose()' for graceful async shutdown"
            )
            asyncio.ensure_future(store.close())

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
        # (4) checkpoint + close the owned store (only if it was ever opened).
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
        n = await store.close_session(session_id, cascade=cascade)
        # Drop the per-session in-memory bookkeeping so a long-lived loop that
        # cycles through many sessions doesn't leak a Lock per session id (C12).
        self._locks.pop(session_id, None)
        self._follow_up_queue_locks.pop(session_id, None)
        self._follow_up_queues.pop(session_id, None)
        return n

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

    def follow_up_sync(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
        tools: Sequence[str] | ToolRegistry | None = None,
        system_prompt: str | None = None,
    ) -> StatefulResult | FollowUpQueued:
        return asyncio.run(
            self.follow_up(
                user_input,
                session_id,
                stop_event=stop_event,
                tools=tools,
                system_prompt=system_prompt,
            )
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
        return asyncio.run(
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
        if isinstance(user_input, str):
            await store.append_message(sid, role="user", content=user_input)
            return
        role = user_input.get("role", "user")
        await store.append_message(
            sid,
            role=str(role),
            content=_as_text(user_input.get("content")),
            name=user_input.get("name"),
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
        sink = sink if sink is not None else SQLiteSink(store, sid)
        # The async store offloads its own blocking I/O (SQLite → threadpool; PG/MySQL
        # → real async), so the per-send active-history load no longer blocks the event
        # loop — other sessions run during the read (SCALE-3).
        active_rows = await store.load_active_messages(sid)
        history = [_row_to_loop_message(r) for r in active_rows]
        # Mirror loaded seqs into the sink so the compactor can translate
        # in-memory indices back to store rows when it folds messages. Pass the
        # parallel logical positions too: a compact_note's identity seq is high,
        # but it sits at its logical ``ord`` (set when it was folded) — load order
        # and the sink's index map must agree on that, or the next fold mis-maps.
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
        try:
            # The async store offloads its own blocking I/O, so this no longer
            # needs an explicit thread hop (H1.9/C8).
            await store.bump_session_stats(
                sid, result.usage, rounds=result.rounds, tool_calls=result.tool_calls,
            )
        except Exception:
            logger.exception("session_stats bump failed for %s (continuing)", sid)
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
