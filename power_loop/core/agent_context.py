from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from power_loop.agent.stateful_loop import StatefulAgentLoop
    from power_loop.core.events import AgentEventBus
    from power_loop.core.hooks import AgentHooks
    from power_loop.core.state import ContextManager


_current_event_bus: ContextVar[AgentEventBus | None] = ContextVar("power_loop_event_bus", default=None)
_current_hooks: ContextVar[AgentHooks | None] = ContextVar("power_loop_hooks", default=None)
_current_ctx: ContextVar[ContextManager | None] = ContextVar("power_loop_ctx", default=None)
_current_session_id: ContextVar[str | None] = ContextVar("power_loop_session_id", default=None)
_current_loop: ContextVar[StatefulAgentLoop | None] = ContextVar(
    "power_loop_current_loop", default=None
)
# The innermost run's per-send effective tool allowlist (None = unrestricted).
# Set by StatefulAgentLoop._run_loop for the duration of a run: to the resolved
# registry names when the caller passed ``send(tools=...)``, else to None. Child
# spawns (run_agent_spec) read it to clamp their own registry so a sub-agent
# never exceeds the parent's per-send capabilities (see ``inherit_send_filter``).
_effective_tools: ContextVar[frozenset[str] | None] = ContextVar(
    "power_loop_effective_tools", default=None
)


def get_event_bus() -> AgentEventBus:
    bus = _current_event_bus.get()
    # Local import to avoid import cycles
    from power_loop.core.events import DEFAULT_EVENT_BUS

    return DEFAULT_EVENT_BUS if bus is None else bus


def get_hooks() -> AgentHooks:
    hooks = _current_hooks.get()
    from power_loop.core.hooks import DEFAULT_HOOKS

    return DEFAULT_HOOKS if hooks is None else hooks


def get_ctx() -> ContextManager:
    ctx = _current_ctx.get()
    if ctx is None:
        from power_loop.core.state import ContextManager

        ctx = ContextManager(role="main")
        _current_ctx.set(ctx)
        return ctx
    return ctx


def get_session_id() -> str | None:
    return _current_session_id.get()


def get_effective_tools() -> frozenset[str] | None:
    """The innermost run's per-send tool allowlist; None = unrestricted."""
    return _effective_tools.get()


def set_effective_tools(names: frozenset[str] | None) -> Token:
    return _effective_tools.set(names)


def reset_effective_tools(token: Token) -> None:
    _effective_tools.reset(token)


def get_current_loop() -> StatefulAgentLoop | None:
    return _current_loop.get()


def set_current_loop(loop: StatefulAgentLoop | None) -> Token:
    return _current_loop.set(loop)


def reset_current_loop(token: Token) -> None:
    _current_loop.reset(token)


def set_event_bus(bus: AgentEventBus | None) -> Token:
    return _current_event_bus.set(bus)


def set_hooks(hooks: AgentHooks | None) -> Token:
    return _current_hooks.set(hooks)


def set_ctx(ctx: ContextManager | None) -> Token:
    return _current_ctx.set(ctx)


def set_session_id(session_id: str | None) -> Token:
    return _current_session_id.set(session_id)


def reset_event_bus(token: Token) -> None:
    _current_event_bus.reset(token)


def reset_hooks(token: Token) -> None:
    _current_hooks.reset(token)


def reset_ctx(token: Token) -> None:
    _current_ctx.reset(token)


def reset_session_id(token: Token) -> None:
    _current_session_id.reset(token)

