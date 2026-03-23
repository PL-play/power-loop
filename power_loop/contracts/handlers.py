from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Protocol

from power_loop.contracts.events import AgentEvent
from power_loop.contracts.hooks import HookContext


class EventHandler(Protocol):
    def __call__(self, event: AgentEvent) -> Any:
        ...


class HookHandler(Protocol):
    def __call__(self, context: HookContext) -> HookContext | Dict[str, Any] | Awaitable[HookContext | Dict[str, Any]]:
        ...


@dataclass
class ToolHandlerResult:
    """Normalized tool execution result envelope."""

    ok: bool
    content: str
    data: Dict[str, Any] = field(default_factory=dict)


class ToolHandler(Protocol):
    def __call__(self, args: Dict[str, Any]) -> ToolHandlerResult | Dict[str, Any] | str | Awaitable[ToolHandlerResult | Dict[str, Any] | str]:
        ...


ToolHandlerMap = Dict[str, ToolHandler]
EventSubscriber = Callable[[EventHandler], None]
