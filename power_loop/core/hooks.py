from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Protocol

from power_loop.contracts.events import AgentEventType
from power_loop.contracts.hooks import HookContext, HookPoint
from power_loop.contracts.handlers import HookHandler


class _HookHandlerResult(Protocol):
    def __await__(self) -> Any: ...


HookHandlerFn = Callable[[HookContext], HookContext | Dict[str, Any] | Awaitable[HookContext | Dict[str, Any]]]


@dataclass
class _HookEntry:
    handler: HookHandlerFn
    order: int


class AgentHooks:
    """Hook manager with ordered sync/async handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[_HookEntry]] = {}

    def register(self, hook_point: HookPoint | str, handler: HookHandlerFn, *, order: int = 0) -> None:
        key = str(hook_point)
        self._handlers.setdefault(key, []).append(_HookEntry(handler=handler, order=order))
        self._handlers[key].sort(key=lambda e: e.order)

    def clear(self, hook_point: HookPoint | str | None = None) -> None:
        if hook_point is None:
            self._handlers.clear()
            return
        self._handlers.pop(str(hook_point), None)

    def run(self, hook_point: HookPoint | str, context: HookContext) -> HookContext:
        key = str(hook_point)
        for entry in self._handlers.get(key, []):
            result = entry.handler(context)
            if isinstance(result, dict):
                context.values = result
            elif isinstance(result, HookContext):
                context = result
        return context

    async def run_async(self, hook_point: HookPoint | str, context: HookContext) -> HookContext:
        key = str(hook_point)
        for entry in self._handlers.get(key, []):
            result = entry.handler(context)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment]
            if isinstance(result, dict):
                context.values = result
            elif isinstance(result, HookContext):
                context = result
        return context


DEFAULT_HOOKS = AgentHooks()

