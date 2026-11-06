from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

RuntimeMessage = dict[str, Any]


@dataclass(frozen=True)
class ToolRuntimeContext:
    """Current tool-call runtime context.

    Tool handlers can call :func:`get_tool_runtime_context` to access the
    active session id, ``SessionStore``, and loop config without importing
    internal contextvars. This is the public primitive used by default
    runtime-bound tools and available to user-defined tools.
    """

    session_id: str | None
    store: Any | None
    loop: Any | None
    config: Any | None
    assistant_seq: int | None = None
    tool_call_id: str | None = None

    @property
    def has_session_store(self) -> bool:
        return self.session_id is not None and self.store is not None


def get_tool_runtime_context(*, required: bool = False) -> ToolRuntimeContext:
    """Return context for the currently executing tool handler.

    When called outside ``StatefulAgentLoop`` execution, fields are ``None``.
    Pass ``required=True`` to raise a clear error instead.
    """
    from power_loop.core.agent_context import get_current_loop, get_session_id

    loop = get_current_loop()
    session_id = get_session_id()
    store = getattr(loop, "store", None) if loop is not None else None
    config = getattr(loop, "config", None) if loop is not None else None
    from power_loop.runtime.invocation import current_invocation
    assistant_seq, tool_call_id = current_invocation(session_id)
    ctx = ToolRuntimeContext(session_id=session_id, store=store, loop=loop, config=config,
                             assistant_seq=assistant_seq, tool_call_id=tool_call_id)
    if required and not ctx.has_session_store:
        raise RuntimeError("No active StatefulAgentLoop session/store is available for this tool call.")
    return ctx


@runtime_checkable
class RuntimeProjector(Protocol):
    """Project persisted runtime state into transient LLM messages.

    Implementations may read/write ``SessionStore`` state. Returned messages
    are sent to the next LLM call but are not persisted in the conversation
    log. This is the extension point for tools whose state must survive
    compaction, process restarts, and custom agent loop instances.

    ``project`` is async because the store is async (its backend offloads I/O);
    the pipeline awaits each projector once per round.
    """

    async def project(
        self,
        *,
        store: Any,
        session_id: str,
        round_index: int,
        context: Any,
    ) -> list[RuntimeMessage]: ...


@dataclass(frozen=True)
class TodoRuntimeProjector(RuntimeProjector):
    state_key: str = "todo"

    async def project(
        self,
        *,
        store: Any,
        session_id: str,
        round_index: int,
        context: Any,
    ) -> list[RuntimeMessage]:
        todo_state = await store.poll_runtime_state(session_id, self.state_key, default={}) or {}
        rendered = todo_state.get("rendered") if isinstance(todo_state, dict) else None
        if not isinstance(rendered, str) or not rendered.strip():
            return []
        return [
            {
                "role": "user",
                "name": "runtime_todo_state",
                "content": f"<current_todos>\n{rendered}\n</current_todos>",
            }
        ]


def default_runtime_projectors() -> tuple[RuntimeProjector, ...]:
    return (TodoRuntimeProjector(),)


__all__ = [
    "RuntimeMessage",
    "RuntimeProjector",
    "ToolRuntimeContext",
    "TodoRuntimeProjector",
    "default_runtime_projectors",
    "get_tool_runtime_context",
]
