"""Bounded notification transports; durable consumption belongs to the store."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Protocol

from .runtime_types import CapacityError, InputRef, RuntimeRequestError, TargetRef, canonical, freeze

PROTOCOL_VERSION = 42


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
            self.queue.put_nowait(replace(packet, payload={"kind": "gap"}))
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
                    self.offer(packet)
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
