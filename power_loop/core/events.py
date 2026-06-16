from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

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

    **Subscriber contract (OBS-3).** A *synchronous* subscriber runs INLINE on the
    publishing thread (usually the agent's event loop), so it MUST be fast and
    non-blocking — do not do network/disk I/O or sleep in one, or you stall the agent
    loop. For slow work, register an ``async def`` handler (scheduled as a task, never
    blocks the loop) or opt into ``sync_dispatch="thread"``: that drains sync subscribers
    on a background thread via a bounded queue with an overflow ``on_overflow`` policy
    (``drop_newest`` default / ``drop_oldest`` / ``block``), so a slow/blocking sync
    subscriber can no longer stall ``publish()``. Call :meth:`shutdown` to stop the
    worker (it flushes the queue first).
    """

    def __init__(
        self,
        *,
        suppress_subscriber_errors: bool = False,
        sync_dispatch: str = "inline",
        queue_maxsize: int = 1000,
        on_overflow: str = "drop_newest",
    ) -> None:
        self._handlers: defaultdict[AgentEventType, list[_SubscribedHandler]] = defaultdict(list)
        self._global_handlers: list[_SubscribedHandler] = []
        self._counter = 0
        self._suppress_subscriber_errors = suppress_subscriber_errors
        # Background tasks spawned for async subscribers from the SYNC publish()
        # path. Retained so CPython can't GC them mid-flight (it only weakly refs a
        # bare create_task); discarded by a done-callback that also surfaces errors.
        self._async_tasks: set[asyncio.Task[Any]] = set()
        # OBS-3 backpressure: optional background-thread dispatch for SYNC subscribers.
        if sync_dispatch not in ("inline", "thread"):
            raise ValueError(f"sync_dispatch must be 'inline' or 'thread', got {sync_dispatch!r}")
        if on_overflow not in ("drop_newest", "drop_oldest", "block"):
            raise ValueError(f"invalid on_overflow: {on_overflow!r}")
        self._sync_dispatch = sync_dispatch
        self._on_overflow = on_overflow
        self.dropped = 0  # count of events dropped by the overflow policy (observability)
        self._queue: queue.Queue[tuple[_SubscribedHandler, AgentEvent] | None] | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        if sync_dispatch == "thread":
            self._queue = queue.Queue(maxsize=max(1, int(queue_maxsize)))
            self._worker = threading.Thread(
                target=self._drain_worker, name="power-loop-eventbus", daemon=True
            )
            self._worker.start()

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

    def _on_async_publish_done(self, task: asyncio.Task[Any]) -> None:
        """Done-callback for an async subscriber scheduled from sync ``publish()``.

        Always retrieve the exception (so it isn't a silent "Task exception was never
        retrieved" GC warning). When ``suppress_subscriber_errors`` is False the
        coroutine re-raised, so surface it loudly at ERROR — async subscribers that
        need inline error handling should use :meth:`publish_async`.
        """
        self._async_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _logger.error(
                "AgentEventBus: async subscriber raised in sync publish() "
                "(use publish_async to handle these inline)",
                exc_info=exc,
            )

    async def drain(self, *, timeout: float | None = None) -> None:
        """Await all in-flight async-subscriber tasks scheduled from sync ``publish()``.

        Graceful shutdown (``StatefulAgentLoop.aclose``) calls this so queued
        observability/side-effect handlers finish — or time out — before the bus's owner
        tears down, instead of being dropped mid-flight. Returns once they settle."""
        pending = [t for t in self._async_tasks if not t.done()]
        if not pending:
            return
        await asyncio.wait(pending, timeout=timeout)

    def _drain_worker(self) -> None:
        """Background worker (thread mode): run offloaded SYNC subscribers off the
        publishing thread. Flushes the queue on shutdown before exiting."""
        assert self._queue is not None
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    return  # stop only once drained
                continue
            if item is None:  # explicit shutdown sentinel
                return
            sub, event = item
            # The worker MUST outlive any single subscriber: with the default
            # suppress_subscriber_errors=False, _invoke_handler re-raises, which —
            # if it escaped here — would kill the worker thread and silently drop
            # every subsequent event (queue fills with no consumer), or deadlock the
            # publisher under on_overflow="block" (C11). Contain it per-item.
            try:
                result = self._invoke_handler(sub, event)
                if asyncio.iscoroutine(result):  # a sync callable that returned a coroutine
                    asyncio.run(self._await_handler_result(result))
            except Exception:
                _logger.exception(
                    "AgentEventBus: offloaded subscriber raised; worker continues "
                    "(event_type=%s)",
                    event.type,
                )

    def _enqueue_sync(self, sub: _SubscribedHandler, event: AgentEvent) -> None:
        """Hand a sync subscriber to the worker thread, applying the overflow policy."""
        q = self._queue
        assert q is not None
        try:
            q.put_nowait((sub, event))
        except queue.Full:
            if self._on_overflow == "block":
                q.put((sub, event))
            elif self._on_overflow == "drop_oldest":
                try:
                    q.get_nowait()
                    self.dropped += 1
                    q.put_nowait((sub, event))
                except (queue.Empty, queue.Full):
                    self.dropped += 1
            else:  # drop_newest
                self.dropped += 1

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Stop the background dispatch worker (thread mode), flushing queued events
        first. No-op in inline mode. Idempotent."""
        if self._worker is None:
            return
        self._stop.set()
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)  # nudge the worker awake
            except queue.Full:
                pass
        self._worker.join(timeout=timeout)
        self._worker = None

    def publish(self, event: AgentEvent) -> None:
        """Publish synchronously.

        Async subscriber handlers are scheduled on the running loop when available. In
        ``sync_dispatch="thread"`` mode, sync handlers are offloaded to the worker thread
        (bounded queue + overflow policy) so a slow one can't block this call.
        """

        handlers: list[_SubscribedHandler] = list(self._global_handlers)
        handlers.extend(self._handlers.get(event.type, []))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        offload = self._sync_dispatch == "thread" and self._queue is not None
        for sub in handlers:
            # In thread mode, predicted-sync handlers go to the worker (non-blocking).
            # Async handlers always schedule on the loop here (the worker has no loop).
            if offload and not asyncio.iscoroutinefunction(sub.handler):
                self._enqueue_sync(sub, event)
                continue
            result = self._invoke_handler(sub, event)
            if asyncio.iscoroutine(result):
                if loop is not None:
                    async def _run_coro(coro: Any) -> None:
                        await self._await_handler_result(coro)

                    task = loop.create_task(_run_coro(result))
                    self._async_tasks.add(task)
                    task.add_done_callback(self._on_async_publish_done)
                else:
                    asyncio.run(self._await_handler_result(result))

    async def publish_async(self, event: AgentEvent) -> None:
        """Publish asynchronously (awaits async subscribers)."""

        handlers: list[_SubscribedHandler] = list(self._global_handlers)
        handlers.extend(self._handlers.get(event.type, []))

        for sub in handlers:
            result = self._invoke_handler(sub, event)
            await self._await_handler_result(result)


# Process-wide default bus instance (used only when user doesn't pass one).
DEFAULT_EVENT_BUS = AgentEventBus()

