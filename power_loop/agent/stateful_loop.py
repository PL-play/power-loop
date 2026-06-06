"""StatefulAgentLoop — the single public entry point for power-loop.

Owns a :class:`SessionStore` and gives callers a stateful,
``new_session()`` + ``send(user_input, session_id=...)`` interface.
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
from dataclasses import dataclass, field
from typing import Any

from llm_client.interface import LLMService
from power_loop.agent.sink import SQLiteSink
from power_loop.agent.system_prompt import DEFAULT_AGENT_SYSTEM_PROMPT, format_tool_catalog
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.errors import SessionNotFoundError, SessionPendingError
from power_loop.core.agent_context import reset_current_loop, set_current_loop
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.pipeline import (
    AgentPipeline,
    _tool_call_args,
    _tool_call_name,
    _truncate_result,
)
from power_loop.core.runner import AgentRunner
from power_loop.core.state import ContextManager
from power_loop.runtime.cancellation import CancellationLike
from power_loop.runtime.session_store import (
    DEFAULT_DB_PATH,
    MessageRow,
    MessageState,
    SessionStore,
    SubagentLifecycle,
)
from power_loop.tools.registry import ToolRegistry


@dataclass
class StatefulResult:
    """Result of a single :meth:`StatefulAgentLoop.send` call."""

    session_id: str
    status: str
    final_text: str = ""
    rounds: int = 0
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)


class StatefulAgentLoop:
    """The only public entry point for running an agent loop.

    A single instance can drive any number of sessions concurrently (one
    session never blocks another beyond SQLite's row-level locking). The
    store is owned by the loop; callers may share it across multiple
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
    ) -> None:
        self.llm = llm
        self.store = store if store is not None else SessionStore.open(db_path)
        self._owns_store = store is None
        self.config = config if config is not None else AgentLoopConfig()
        self.tool_registry = tool_registry
        self._runner = AgentRunner(event_bus=event_bus, hooks=hooks)
        self._locks: dict[str, asyncio.Lock] = {}

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying store (if owned). Does NOT delete sessions."""
        if self._owns_store:
            self.store.close()

    def close_session(self, session_id: str, *, cascade: bool = True) -> int:
        """Physically delete the session and (by default) its LINKED subtree."""
        return self.store.close_session(session_id, cascade=cascade)

    # ── primary API ───────────────────────────────────────────────────────

    def new_session(
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
        return self._create_session(metadata=metadata, system_prompt=system_prompt)

    async def send(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
    ) -> StatefulResult:
        """Append one user input to the session and run the loop.

        ``session_id`` must refer to an existing session created by
        :meth:`new_session` (or by lower-level ``SessionStore`` APIs).

        Raises :class:`SessionPendingError` if the session has unresolved
        tool_calls; the caller must call :meth:`resume` or
        :meth:`abort_pending` first.
        """
        sid = session_id
        async with self._lock_for(sid):
            self._ensure_session_or_raise(sid)
            self._raise_if_pending(sid)
            self._persist_user_input(sid, user_input)
            return await self._run_loop(sid, stop_event=stop_event)

    def send_sync(
        self,
        user_input: str | LoopMessage,
        session_id: str,
        *,
        stop_event: CancellationLike = None,
    ) -> StatefulResult:
        return asyncio.run(
            self.send(user_input, session_id, stop_event=stop_event)
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
        async with self._lock_for(session_id):
            self._ensure_session_or_raise(session_id)
            sink = SQLiteSink(self.store, session_id)
            await self._execute_pending(session_id, sink)
            return await self._run_loop(session_id, stop_event=stop_event, sink=sink)

    def abort_pending(self, session_id: str, *, reason: str = "aborted") -> int:
        """Synthesize ``<aborted: reason>`` tool messages for every unresolved
        tool_call, restoring message-protocol validity. Returns the number of
        aborted tool_calls.
        """
        self._ensure_session_or_raise(session_id)
        state = self.store.get_state(session_id)
        if state is None or not state.pending:
            return 0
        pending = state.pending
        round_index = int(pending.get("round_index") or 0)
        tool_calls = pending.get("tool_calls") or [
            {"id": cid} for cid in pending.get("tool_call_ids", [])
        ]
        sink = SQLiteSink(self.store, session_id)
        sink._unresolved = {str(tc.get("id") or "") for tc in tool_calls}
        sink._assistant_seq = pending.get("assistant_seq")
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            name = _tool_call_name(tc) if "function" in tc or "name" in tc else None
            sink.on_message_appended(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "name": name,
                    "content": f"<aborted: {reason}>",
                },
                round_index=round_index,
            )
        return len(tool_calls)

    # ── inspection ────────────────────────────────────────────────────────

    def get_messages(self, session_id: str, *, include_compacted: bool = False) -> list[LoopMessage]:
        rows = (
            self.store.load_all_messages(session_id)
            if include_compacted
            else self.store.load_active_messages(session_id)
        )
        return [_row_to_loop_message(r) for r in rows]

    def get_pending(self, session_id: str) -> dict[str, Any] | None:
        state = self.store.get_state(session_id)
        return state.pending if state else None

    def resolve_system_prompt(self, *, session_id: str | None = None) -> str:
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
            row = self.store.get_session(session_id)
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

        return base

    # ── internals ─────────────────────────────────────────────────────────

    def _create_session(
        self,
        *,
        metadata: dict[str, Any] | None,
        parent_session_id: str | None = None,
        spawn_tool_call_id: str | None = None,
        lifecycle: SubagentLifecycle = SubagentLifecycle.EPHEMERAL,
        system_prompt: str | None = None,
    ) -> str:
        return self.store.create_session(
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

    def _ensure_session_or_raise(self, sid: str) -> None:
        if self.store.get_session(sid) is None:
            raise SessionNotFoundError(sid)

    def _raise_if_pending(self, sid: str) -> None:
        state = self.store.get_state(sid)
        if state is not None and state.pending:
            pending = state.pending
            raise SessionPendingError(
                sid,
                assistant_seq=int(pending.get("assistant_seq") or 0),
                pending_tool_calls=pending.get("tool_calls", []),
            )

    def _persist_user_input(self, sid: str, user_input: str | LoopMessage) -> None:
        if isinstance(user_input, str):
            self.store.append_message(sid, role="user", content=user_input)
            return
        role = user_input.get("role", "user")
        self.store.append_message(
            sid,
            role=str(role),
            content=_as_text(user_input.get("content")),
            name=user_input.get("name"),
        )

    async def _execute_pending(self, sid: str, sink: SQLiteSink) -> None:
        """Replay leftover tool_calls. Idempotent if there is no pending."""
        state = self.store.get_state(sid)
        if state is None or not state.pending:
            return
        pending = state.pending
        round_index = int(pending.get("round_index") or 0)
        tool_calls = pending.get("tool_calls") or []
        if not tool_calls:
            return
        # Initialize sink's in-memory unresolved set so auto-resolve works.
        sink._unresolved = {str(tc.get("id") or "") for tc in tool_calls}
        sink._assistant_seq = pending.get("assistant_seq")
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
            sink.on_message_appended(
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

    async def _run_loop(
        self,
        sid: str,
        *,
        stop_event: CancellationLike,
        sink: SQLiteSink | None = None,
    ) -> StatefulResult:
        sink = sink if sink is not None else SQLiteSink(self.store, sid)
        active_rows = self.store.load_active_messages(sid)
        history = [_row_to_loop_message(r) for r in active_rows]
        # Mirror loaded seqs into the sink so the compactor can translate
        # in-memory indices back to store rows when it folds messages.
        sink.init_history_seqs([r.seq for r in active_rows])

        async with self._runner.session_async(session_id=sid):
            loop_token = set_current_loop(self)
            try:
                pipeline = AgentPipeline(
                    llm=self.llm,
                    config=self.config,
                    tool_registry=self.tool_registry,
                    hooks=self._runner.hooks,
                    bus=self._runner.event_bus,
                    ctx=ContextManager(role="main"),
                    session_id=sid,
                    stop_event=stop_event,
                    sink=sink,
                )
                result: AgentLoopResult = await pipeline.run(history)
            finally:
                reset_current_loop(loop_token)
        return StatefulResult(
            session_id=sid,
            status=result.status,
            final_text=result.final_text,
            rounds=result.rounds,
            pending_tool_calls=result.pending_tool_calls,
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


__all__ = ["StatefulAgentLoop", "StatefulResult", "MessageState"]
