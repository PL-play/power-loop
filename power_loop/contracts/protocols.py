from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.handlers import EventHandler, HookHandler
from power_loop.contracts.hook_contexts import BaseHookCtx
from power_loop.contracts.hooks import HookContext, HookPoint, HookResult


class EventBusProtocol(Protocol):
    def subscribe(self, event_type: AgentEventType | None, handler: EventHandler, *, priority: int = 0) -> None:
        ...

    def unsubscribe(self, handler: EventHandler) -> None:
        ...

    def publish(self, event: AgentEvent) -> None:
        ...

    async def publish_async(self, event: AgentEvent) -> None:
        ...


class HookManagerProtocol(Protocol):
    def register(self, hook_point: HookPoint | str, handler: HookHandler, *, order: int = 0) -> None:
        ...

    def clear(self, hook_point: HookPoint | str | None = None) -> None:
        ...

    def run(self, hook_point: HookPoint | str, context: HookContext) -> HookResult:
        ...

    async def run_async(self, hook_point: HookPoint | str, context: HookContext) -> HookResult:
        ...

    def run_typed(self, hook_point: HookPoint | str, ctx: BaseHookCtx) -> None:
        ...

    async def run_typed_async(self, hook_point: HookPoint | str, ctx: BaseHookCtx) -> None:
        ...


class ToolArgsValidator(Protocol):
    """Pre-execution tool-argument validator: return an error string to reject the call, or ``None``
    to allow it (may be async).

    RESERVED / PROVISIONAL (exec-skills-structured-6): this is a typed seam published for forward
    compatibility, but the runtime does NOT yet consume it — there is currently no
    ``ToolRegistry`` / ``AgentLoopConfig`` hook that calls a ToolArgsValidator. Validate tool args
    inside the tool handler itself for now. (Tracked for a future wiring; not STABLE_API.)"""

    def __call__(self, tool_name: str, args: dict[str, Any]) -> str | None | Awaitable[str | None]:
        ...
