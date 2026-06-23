from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from power_loop.contracts.hook_contexts import BaseHookCtx
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult

HookHandlerFn = Callable[..., Any]
"""A hook handler callable.

For **typed** hooks (the recommended style) the signature is::

    def handler(ctx: SomeHookCtx) -> None | HookDirective

Handlers mutate *ctx* in-place and optionally return a ``HookDirective``.

Legacy handlers that receive ``HookContext`` and return
``HookContext | HookResult | dict | None`` are still supported via
:meth:`AgentHooks.run` / :meth:`AgentHooks.run_async`.
"""


@dataclass
class _HookEntry:
    handler: HookHandlerFn
    order: int
    name: str | None = None


class AgentHooks:
    """Hook manager with ordered sync/async handlers.

    **Typed API** (recommended — all pipeline hook points use this):

    Handlers receive a strongly-typed ``*Ctx`` dataclass, mutate it in place,
    and optionally return ``HookDirective`` or set ``ctx.directive``.

    **Legacy API** (still supported for unit-test ergonomics):

    Handlers receive ``HookContext`` and may return ``HookContext``,
    ``HookResult``, ``dict``, or ``None``.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[_HookEntry]] = {}

    def register(
        self,
        hook_point: HookPoint | str,
        handler: HookHandlerFn,
        *,
        order: int = 0,
        name: str | None = None,
        replace: bool = False,
    ) -> None:
        """Register a handler at ``hook_point``.

        ``name`` gives the entry a stable identity so it can later be
        :meth:`remove`-d or :meth:`replace`-d — used by built-in hooks
        (``builtin.*``) so hosts can override them. ``replace=True`` (with a
        ``name``) drops any existing entry of the same name at this point first,
        so re-registering is idempotent.
        """
        key = str(hook_point)
        entries = self._handlers.setdefault(key, [])
        if name is not None and replace:
            entries[:] = [e for e in entries if e.name != name]
        entries.append(_HookEntry(handler=handler, order=order, name=name))
        entries.sort(key=lambda e: e.order)

    def replace(
        self, hook_point: HookPoint | str, handler: HookHandlerFn, *, name: str, order: int = 0
    ) -> None:
        """Replace (or add) the named handler at ``hook_point``."""
        self.register(hook_point, handler, order=order, name=name, replace=True)

    def remove(self, name: str, hook_point: HookPoint | str | None = None) -> int:
        """Remove every entry with ``name`` (optionally scoped to one point).

        Returns the number of entries removed. Used to disable a built-in hook:
        ``hooks.remove("builtin.memory_recall")``.
        """
        keys = [str(hook_point)] if hook_point is not None else list(self._handlers)
        removed = 0
        for key in keys:
            entries = self._handlers.get(key)
            if not entries:
                continue
            before = len(entries)
            entries[:] = [e for e in entries if e.name != name]
            removed += before - len(entries)
        return removed

    def has(self, name: str, hook_point: HookPoint | str | None = None) -> bool:
        """Whether a named entry is registered (optionally scoped to one point)."""
        keys = [str(hook_point)] if hook_point is not None else list(self._handlers)
        return any(e.name == name for key in keys for e in self._handlers.get(key, []))

    def clear(self, hook_point: HookPoint | str | None = None) -> None:
        if hook_point is None:
            self._handlers.clear()
            return
        self._handlers.pop(str(hook_point), None)

    # ── Legacy dict-based API ──

    @staticmethod
    def _apply(context: HookContext, directive: HookDirective, result: Any) -> tuple[HookContext, HookDirective]:
        if result is None:
            return context, directive
        if isinstance(result, HookResult):
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
                result = await result
            context, directive = self._apply(context, directive, result)
        return HookResult(context=context, directive=directive)

    # ── Typed context API (all pipeline hooks use this path) ──

    @staticmethod
    def _apply_typed(ctx: BaseHookCtx, result: Any) -> None:
        """Apply handler return value to the typed context."""
        if result is None:
            return
        if isinstance(result, HookDirective):
            ctx.directive = result
            return
        if isinstance(result, HookResult):
            ctx.directive = result.directive
            return

    def run_typed(self, hook_point: HookPoint | str, ctx: BaseHookCtx) -> None:
        """Run handlers with a typed context.  Handlers mutate *ctx* in place."""
        for entry in self._handlers.get(str(hook_point), []):
            result = entry.handler(ctx)
            self._apply_typed(ctx, result)

    async def run_typed_async(self, hook_point: HookPoint | str, ctx: BaseHookCtx) -> None:
        """Async version of :meth:`run_typed`."""
        for entry in self._handlers.get(str(hook_point), []):
            result = entry.handler(ctx)
            if hasattr(result, "__await__"):
                result = await result
            self._apply_typed(ctx, result)


DEFAULT_HOOKS = AgentHooks()

