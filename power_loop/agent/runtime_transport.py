"""Bounded notification transports; durable consumption belongs to the store."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Protocol

from .runtime_types import PROTOCOL_VERSION, CapacityError, InputRef, RuntimeRequestError, TargetRef, canonical, freeze


@dataclass(frozen=True)
class TransportCapabilities:
    scope: str = "process"
    redelivery: bool = False
    broker_ack: bool = False
    ordered: bool = True
    retention_ms: int = 0
    max_bytes: int = 1_048_576


@dataclass(frozen=True)
class TransportPacket:
    protocol_version: int
    namespace: str
    message_id: str
    target: TargetRef
    channel: str
    input_ref: InputRef | None = None
    payload: dict | None = None

    def __post_init__(self):
        if self.protocol_version != PROTOCOL_VERSION:
            raise RuntimeRequestError("invalid_options", "unsupported message protocol")
        if not self.namespace or self.channel not in {"activation", "work", "control", "output"}:
            raise RuntimeRequestError("invalid_options", "invalid notification binding")
        object.__setattr__(self, "payload", freeze(self.payload))


@dataclass(frozen=True)
class PublishReceipt:
    accepted: bool


@dataclass(frozen=True)
class SubscriptionSpec:
    namespace: str
    store_id: str
    channel: str
    node_id: str
    session_id: str | None = None
    message_id: str | None = None
    # Database discovery only. Live packet matching still uses message_id.
    discovery_message_id: str | None = None


class Delivery(Protocol):
    packet: TransportPacket

    async def ack(self) -> None: ...
    async def retry(self, *, delay_ms: int) -> None: ...


class Subscription(Protocol):
    async def ready(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[Delivery]: ...
    async def aclose(self) -> None: ...


class MessageTransport(Protocol):
    capabilities: TransportCapabilities

    async def publish(self, packet: TransportPacket) -> PublishReceipt: ...
    async def subscribe(self, binding: SubscriptionSpec) -> Subscription: ...
    async def aclose(self) -> None: ...


class _Delivery:
    def __init__(self, packet, subscription):
        self.packet, self.subscription = packet, subscription

    async def ack(self):
        # Acknowledges only this notification, never the input row.
        pass

    async def retry(self, *, delay_ms):
        await asyncio.sleep(delay_ms / 1000)
        self.subscription.offer(self.packet)


class _Subscription:
    def __init__(self, transport, binding):
        self.transport, self.binding = transport, binding
        self.queue = asyncio.Queue(maxsize=256 if binding.channel == "output" else 1024)
        self.closed = False

    async def ready(self):
        if self.closed:
            raise RuntimeRequestError("session_closed")

    def matches(self, packet):
        b = self.binding
        return (b.namespace == packet.namespace and b.store_id == packet.target.store_id
                and b.channel == packet.channel
                and (b.session_id is None or b.session_id == packet.target.session_id)
                and (b.message_id is None or b.message_id == packet.message_id))

    def offer(self, packet):
        if self.closed:
            return False
        if self.queue.full():
            if self.binding.channel != "output":
                return False
            while not self.queue.empty():
                self.queue.get_nowait()
            # Clearing the queue drops queued notifications for every input on
            # this subscription, so the gap must be session-scoped (input_ref=None)
            # rather than attributed to the overflowing packet's input.
            self.queue.put_nowait(replace(
                packet,
                message_id=self.binding.message_id or "gap",
                input_ref=None,
                payload={"kind": "gap"},
            ))
        self.queue.put_nowait(packet)
        return True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        packet = await self.queue.get()
        if packet is None:
            raise StopAsyncIteration
        return _Delivery(packet, self)

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        self.transport._subscriptions.discard(self)
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(None)


class MemoryMessageTransport:
    capabilities = TransportCapabilities()

    def __init__(self):
        self._subscriptions = set()
        self._closed = False
        self._activation_cursor = 0

    async def subscribe(self, binding):
        if self._closed:
            raise RuntimeRequestError("session_closed")
        if len(self._subscriptions) >= 4096:
            raise CapacityError()
        subscription = _Subscription(self, binding)
        self._subscriptions.add(subscription)
        return subscription

    async def publish(self, packet):
        if self._closed:
            raise RuntimeRequestError("session_closed")
        if len(canonical(packet).encode()) > self.capabilities.max_bytes:
            raise CapacityError()
        subscribers = [s for s in self._subscriptions if s.matches(packet)]
        if packet.channel == "activation" and subscribers:
            selected = self._activation_cursor % len(subscribers)
            self._activation_cursor += 1
            subscribers = [subscribers[selected]]
        accepted = True
        for subscription in subscribers:
            accepted = subscription.offer(packet) and accepted
        return PublishReceipt(accepted)

    async def aclose(self):
        self._closed = True
        for subscription in tuple(self._subscriptions):
            await subscription.aclose()


class _PollingSubscription(_Subscription):
    def __init__(self, transport, binding):
        super().__init__(transport, binding)
        self.cursor = None
        self.poller = asyncio.create_task(self._poll(), name="runtime-poll-" + binding.channel)

    async def _poll(self):
        interval = (self.transport.control_interval_s if self.binding.channel == "control"
                    else self.transport.interval_s)
        try:
            while True:
                packets, self.cursor = await self.transport.source.poll_notifications(
                    self.binding, cursor=self.cursor, limit=self.transport.page_size)
                for packet in packets:
                    if (self.offer(packet) and self.binding.channel == "output"
                            and self.binding.discovery_message_id is not None):
                        return
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = exc
            await super().aclose()

    async def __anext__(self):
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            if hasattr(self, "failure"):
                raise self.failure from None
            raise

    async def aclose(self):
        self.poller.cancel()
        await asyncio.gather(self.poller, return_exceptions=True)
        await super().aclose()


class DatabasePollingTransport(MemoryMessageTransport):
    """Discover persisted work, controls and final output in separate bounded pages.

    The source is the runtime store's notification query port. publish does not
    insert another input or a notification journal. Token deltas remain local
    best effort; final results are discoverable through the shared store.
    """
    capabilities = TransportCapabilities(scope="shared", ordered=False)

    def __init__(self, source, *, interval_s=1.0, control_interval_s=0.2, page_size=100):
        super().__init__()
        if interval_s <= 0 or control_interval_s <= 0 or not 1 <= page_size <= 100:
            raise ValueError("positive intervals and page_size in 1..100 required")
        self.source = source
        self.interval_s, self.control_interval_s = interval_s, control_interval_s
        self.page_size = page_size

    async def subscribe(self, binding):
        if self._closed:
            raise RuntimeRequestError("session_closed")
        if len(self._subscriptions) >= 4096:
            raise CapacityError()
        subscription = _PollingSubscription(self, binding)
        self._subscriptions.add(subscription)
        return subscription

    async def publish(self, packet):
        # Only live token output uses the process-local optional notification.
        if packet.channel == "output":
            return await super().publish(packet)
        return PublishReceipt(True)


WakeCallback = Callable[[str, str], Awaitable[None]]
CancelCallback = Callable[[str, str], Awaitable[None]]


class WakeTransport(ABC):
    """Session-Core wake/cancel transport interface (Design 217 §4.5 / P1-6).

    Host integrations (e.g. Kafka/Redis) implement this interface; the SDK ships
    two zero-middleware defaults: ``InMemoryWakeTransport`` (0 SQL, single-process)
    and ``DatabasePollingWakeTransport`` (bounded indexed DB scan, multi-process).
    """

    @abstractmethod
    async def publish_wake(self, session_id: str, *, reason: str = "accept") -> None:
        """Notify workers that ``session_id`` has wakeable pending work."""

    @abstractmethod
    async def publish_cancel(self, session_id: str, *, cancel_by: str = "user") -> None:
        """Hint active workers that ``session_id`` received a cancel/deactivate."""

    @abstractmethod
    async def subscribe(
        self,
        *,
        on_wake: WakeCallback,
        on_cancel: CancelCallback | None = None,
    ) -> None:
        """Register local worker callbacks."""

    async def poll_once(self, *, limit: int = 32) -> list[str]:
        """Pull-based discovery hook (no-op for push transports; bounded scan for DB polling)."""
        return []

    @abstractmethod
    async def aclose(self) -> None:
        """Release resources cleanly."""


class InMemoryWakeTransport(WakeTransport):
    """Default 0-SQL single-process wake transport for embedded SDK mode."""

    _MAX_HISTORY = 1000

    def __init__(self, *, max_history: int = _MAX_HISTORY) -> None:
        self._wake_handlers: list[WakeCallback] = []
        self._cancel_handlers: list[CancelCallback] = []
        cap = max(1, int(max_history))
        self.published_wakes: deque[tuple[str, str]] = deque(maxlen=cap)
        self.published_cancels: deque[tuple[str, str]] = deque(maxlen=cap)
        self._closed = False

    async def publish_wake(self, session_id: str, *, reason: str = "accept") -> None:
        if self._closed:
            return
        self.published_wakes.append((session_id, reason))
        for handler in list(self._wake_handlers):
            await handler(session_id, reason)

    async def publish_cancel(self, session_id: str, *, cancel_by: str = "user") -> None:
        if self._closed:
            return
        self.published_cancels.append((session_id, cancel_by))
        for handler in list(self._cancel_handlers):
            await handler(session_id, cancel_by)

    async def subscribe(
        self,
        *,
        on_wake: WakeCallback,
        on_cancel: CancelCallback | None = None,
    ) -> None:
        self._wake_handlers.append(on_wake)
        if on_cancel is not None:
            self._cancel_handlers.append(on_cancel)

    async def aclose(self) -> None:
        self._closed = True
        self._wake_handlers.clear()
        self._cancel_handlers.clear()
        self.published_wakes.clear()
        self.published_cancels.clear()


class DatabasePollingWakeTransport(WakeTransport):
    """Zero-middleware multi-process wake transport backed by bounded indexed DB scans.

    ``poll_once(limit=32)`` issues 1 bounded index scan on ``sessions`` via
    ``idx_session_recovery`` (``WHERE execution_state='queued' AND lifecycle='open'
    AND wake_pending_count>0 ORDER BY session_id ASC LIMIT ?``).
    """

    def __init__(self, source, *, poll_interval_s: float = 1.0) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if not hasattr(source, "discover_queued"):
            from .runtime_store import RuntimeInputStore
            source = RuntimeInputStore(source)
        self.source = source
        self.poll_interval_s = poll_interval_s
        self._wake_handlers: list[WakeCallback] = []
        self._cancel_handlers: list[CancelCallback] = []
        self._closed = False

    async def publish_wake(self, session_id: str, *, reason: str = "accept") -> None:
        return None

    async def publish_cancel(self, session_id: str, *, cancel_by: str = "user") -> None:
        return None

    async def subscribe(
        self,
        *,
        on_wake: WakeCallback,
        on_cancel: CancelCallback | None = None,
    ) -> None:
        self._wake_handlers.append(on_wake)
        if on_cancel is not None:
            self._cancel_handlers.append(on_cancel)

    async def poll_once(self, *, limit: int = 32) -> list[str]:
        if self._closed:
            return []
        sids = await self.source.discover_queued(limit=limit)
        for sid in sids:
            for handler in list(self._wake_handlers):
                await handler(sid, "db_poll")
        return sids

    async def aclose(self) -> None:
        self._closed = True
        self._wake_handlers.clear()
        self._cancel_handlers.clear()
