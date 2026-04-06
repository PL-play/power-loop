from __future__ import annotations

from typing import Any, Awaitable, Protocol

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
    def __call__(self, tool_name: str, args: dict[str, Any]) -> str | None | Awaitable[str | None]:
        ...
