from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Protocol

from power_loop.contracts.events import AgentEventType
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult
from power_loop.contracts.handlers import HookHandler


class _HookHandlerResult(Protocol):
    def __await__(self) -> Any: ...


HookHandlerFn = Callable[
    [HookContext],
    HookContext | HookResult | Dict[str, Any] | Awaitable[HookContext | HookResult | Dict[str, Any]],
]


@dataclass
class _HookEntry:
    handler: HookHandlerFn
    order: int


class AgentHooks:
    """Hook manager with ordered sync/async handlers.

    Handlers may return:
      - ``HookContext`` — replaces the current context, directive stays CONTINUE.
      - ``HookResult`` — replaces both context and directive.
      - ``dict`` — shorthand, assigned to ``context.values``.
      - ``None`` — no change.

    Once any handler sets a non-CONTINUE directive the pipeline **keeps running**
    (later handlers can observe or override) but the final ``HookResult.directive``
    will be the *last* non-CONTINUE directive set.  This lets an "audit" handler
    after a "skip" handler override the skip if needed.
    """

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

    @staticmethod
    def _apply(context: HookContext, directive: HookDirective, result: Any) -> tuple[HookContext, HookDirective]:
        if result is None:
            return context, directive
        if isinstance(result, HookResult):
            # HookResult always carries an explicit directive choice.
            return result.context, result.directive
        if isinstance(result, HookContext):
            return result, directive
        if isinstance(result, dict):
            context.values = result
            return context, directive
        return context, directive

    def run(self, hook_point: HookPoint | str, context: HookContext) -> HookResult:
        key = str(hook_point)
        directive = HookDirective.CONTINUE
        for entry in self._handlers.get(key, []):
            result = entry.handler(context)
            context, directive = self._apply(context, directive, result)
        return HookResult(context=context, directive=directive)

    async def run_async(self, hook_point: HookPoint | str, context: HookContext) -> HookResult:
        key = str(hook_point)
        directive = HookDirective.CONTINUE
        for entry in self._handlers.get(key, []):
            result = entry.handler(context)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment]
            context, directive = self._apply(context, directive, result)
        return HookResult(context=context, directive=directive)


DEFAULT_HOOKS = AgentHooks()

