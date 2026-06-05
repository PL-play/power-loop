from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from power_loop.contracts.events import AgentEvent
from power_loop.contracts.hook_contexts import BaseHookCtx
from power_loop.contracts.hooks import HookDirective


class EventHandler(Protocol):
    def __call__(self, event: AgentEvent) -> Any:
        ...


class HookHandler(Protocol):
    """Hook handler callable.

    The recommended (typed) signature receives a ``BaseHookCtx`` subclass,
    mutates it in-place, and optionally returns ``HookDirective`` or ``None``.

    Legacy handlers that accept ``HookContext`` are still supported.
    """

    def __call__(self, ctx: BaseHookCtx) -> HookDirective | None | Awaitable[HookDirective | None]:
        ...


@dataclass
class ToolHandlerResult:
    """Normalized tool execution result envelope."""

    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)


class ToolHandler(Protocol):
    def __call__(self, args: dict[str, Any]) -> ToolHandlerResult | dict[str, Any] | str | Awaitable[ToolHandlerResult | dict[str, Any] | str]:
        ...


ToolHandlerMap = dict[str, ToolHandler]
EventSubscriber = Callable[[EventHandler], None]
