from __future__ import annotations

from typing import Protocol

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.handlers import EventHandler, HookHandler
from power_loop.contracts.hook_contexts import BaseHookCtx
from power_loop.contracts.hooks import HookPoint


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

    def run_typed(self, hook_point: HookPoint | str, ctx: BaseHookCtx) -> None:
        ...

    async def run_typed_async(self, hook_point: HookPoint | str, ctx: BaseHookCtx) -> None:
        ...
