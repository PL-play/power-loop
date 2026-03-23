from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
import threading
from typing import AsyncIterator, Iterator, Optional, Sequence

from power_loop.contracts.events import AgentEvent
from power_loop.core.agent_context import (
    set_ctx,
    set_event_bus,
    set_hooks,
    set_session_id,
    reset_ctx,
    reset_event_bus,
    reset_hooks,
    reset_session_id,
)
from power_loop.core.events import AgentEventBus, DEFAULT_EVENT_BUS
from power_loop.core.hooks import AgentHooks, DEFAULT_HOOKS
from power_loop.core.state import ContextManager


class AgentRunner:
    """Runner provides per-session isolation for event bus / hooks / state."""

    def __init__(
        self,
        *,
        event_bus: AgentEventBus | None = None,
        hooks: AgentHooks | None = None,
    ) -> None:
        self.event_bus = event_bus if event_bus is not None else DEFAULT_EVENT_BUS
        self.hooks = hooks if hooks is not None else DEFAULT_HOOKS

    @contextmanager
    def session(self, *, session_id: str | None = None) -> Iterator["AgentRunner"]:
        tok_bus = set_event_bus(self.event_bus)
        tok_hooks = set_hooks(self.hooks)
        tok_ctx = set_ctx(ContextManager(role="main"))
        tok_sid = set_session_id(session_id)
        try:
            yield self
        finally:
            reset_session_id(tok_sid)
            reset_ctx(tok_ctx)
            reset_hooks(tok_hooks)
            reset_event_bus(tok_bus)

    @asynccontextmanager
    async def session_async(self, *, session_id: str | None = None) -> AsyncIterator["AgentRunner"]:
        tok_bus = set_event_bus(self.event_bus)
        tok_hooks = set_hooks(self.hooks)
        tok_ctx = set_ctx(ContextManager(role="main"))
        tok_sid = set_session_id(session_id)
        try:
            yield self
        finally:
            reset_session_id(tok_sid)
            reset_ctx(tok_ctx)
            reset_hooks(tok_hooks)
            reset_event_bus(tok_bus)

