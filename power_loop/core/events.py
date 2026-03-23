from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, DefaultDict, Dict, List

from power_loop.contracts.events import AgentEvent, AgentEventType

_logger = logging.getLogger(__name__)

EventHandler = Callable[[AgentEvent], Any] | Callable[[AgentEvent], Awaitable[Any]]


@dataclass(frozen=True)
class _SubscribedHandler:
    handler: EventHandler
    priority: int
    order: int


class AgentEventBus:
    """In-process pub/sub event bus.

    - Subscribers are ordered by (priority, registration order).
    - Supports both sync and async handlers.
    - If `suppress_subscriber_errors=True`, subscriber exceptions are logged
      and do not break the publisher.
    """

    def __init__(self, *, suppress_subscriber_errors: bool = False) -> None:
        self._handlers: DefaultDict[AgentEventType, List[_SubscribedHandler]] = defaultdict(list)
        self._global_handlers: List[_SubscribedHandler] = []
        self._counter = 0
        self._suppress_subscriber_errors = suppress_subscriber_errors

    def subscribe(
        self,
        event_type: AgentEventType | None,
        handler: EventHandler,
        *,
        priority: int = 0,
    ) -> None:
        self._counter += 1
        wrapped = _SubscribedHandler(handler=handler, priority=priority, order=self._counter)
        if event_type is None:
            self._global_handlers.append(wrapped)
            self._global_handlers.sort(key=lambda h: (h.priority, h.order))
            return
        self._handlers[event_type].append(wrapped)
        self._handlers[event_type].sort(key=lambda h: (h.priority, h.order))

    def unsubscribe(self, handler: EventHandler) -> None:
        self._global_handlers = [h for h in self._global_handlers if h.handler is not handler]
        for etype, handlers in list(self._handlers.items()):
            self._handlers[etype] = [h for h in handlers if h.handler is not handler]
            if not self._handlers[etype]:
                self._handlers.pop(etype, None)

    def _invoke_handler(self, sub: _SubscribedHandler, event: AgentEvent) -> Any:
        try:
            return sub.handler(event)
        except Exception:
            if self._suppress_subscriber_errors:
                _logger.exception(
                    "AgentEventBus: subscriber raised (event_type=%s, handler=%s)",
                    event.type,
                    getattr(sub.handler, "__qualname__", sub.handler),
                )
                return None
            raise

    async def _await_handler_result(self, result: Any) -> None:
        if not asyncio.iscoroutine(result):
            return
        try:
            await result
        except Exception:
            if self._suppress_subscriber_errors:
                _logger.exception("AgentEventBus: async subscriber raised")
                return
            raise

    def publish(self, event: AgentEvent) -> None:
        """Publish synchronously.

        Async subscriber handlers are scheduled on the running loop when available.
        """

        handlers: List[_SubscribedHandler] = list(self._global_handlers)
        handlers.extend(self._handlers.get(event.type, []))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for sub in handlers:
            result = self._invoke_handler(sub, event)
            if asyncio.iscoroutine(result):
                if loop is not None:
                    async def _run_coro(coro: Any) -> None:
                        await self._await_handler_result(coro)

                    loop.create_task(_run_coro(result))
                else:
                    asyncio.run(self._await_handler_result(result))

    async def publish_async(self, event: AgentEvent) -> None:
        """Publish asynchronously (awaits async subscribers)."""

        handlers: List[_SubscribedHandler] = list(self._global_handlers)
        handlers.extend(self._handlers.get(event.type, []))

        for sub in handlers:
            result = self._invoke_handler(sub, event)
            await self._await_handler_result(result)


# Process-wide default bus instance (used only when user doesn't pass one).
DEFAULT_EVENT_BUS = AgentEventBus()

