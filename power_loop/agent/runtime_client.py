"""The unified session-native client (design 215 §5).

One authenticated surface: create_session / send / cancel / inspect / wait / messages /
list_children. ``send`` is idle→new round or busy→follow-up through the same accept
boundary; results are ordinary assistant messages plus input terminal states, so a
detached client (or a reopened store) can wait without any in-memory result cache.
There is no resume, list_jobs, append/nudge/steer or job generation.
"""
from __future__ import annotations

import asyncio
import logging

from .runtime_store import RuntimeInputStore, now_ms
from .runtime_transport import PROTOCOL_VERSION as TRANSPORT_PROTOCOL
from .runtime_transport import SubscriptionSpec, TransportPacket
from .runtime_types import (
    CapacityError,
    InputRef,
    OutputItem,
    Receipt,
    RuntimeRequestError,
    TargetRef,
    WaitTimeout,
    plain,
)

log = logging.getLogger(__name__)


class RuntimeClient:
    def __init__(self, *, store, transport, namespace="power-loop", hooks=None, hook_timeout_s=1.0):
        if not namespace:
            raise ValueError("namespace must be nonempty")
        self.store, self.transport, self.namespace = store, transport, namespace
        self.hooks, self.hook_timeout_s = hooks, hook_timeout_s
        self.inputs = RuntimeInputStore(store)
        self._waiters, self._outputs = set(), set()
        self._closed = False
        self._slots = 0
        self._accepting = True

    def _require_open(self, *, reading=False):
        if self._closed or (not reading and not self._accepting):
            raise RuntimeRequestError("session_closed")

    # ── unified operations ────────────────────────────────────────────────────

    async def workflow(self, target, spec, *, message_id, actor, descriptor=None, options=None):
        """Create a driver session and its start input through the same client port."""
        from power_loop.workflow.runner import submit

        return await submit(
            self, target, spec, message_id=message_id, actor=actor,
            descriptor=descriptor, options=options,
        )

    async def create_session(self, descriptor, *, actor, parent=None, result_route=None,
                             cancel_propagation="linked", options=None) -> TargetRef:
        self._require_open()
        return await self.inputs.create_session(
            descriptor, actor=actor, parent=parent, result_route=result_route,
            cancel_propagation=cancel_propagation, options=options,
        )

    async def send(self, target, body, *, message_id, actor, priority="normal",
                   deadline_ms=None) -> Receipt:
        """Accept an input: idle starts a new round, busy joins the next loop boundary.

        The durable accept is the authority; the activation notification is a best-effort
        latency optimization. A lost notification never loses the input (the worker's
        periodic unfinished scan and the running loop's boundary read both pick it up).
        """
        self._require_open()
        receipt = await self.inputs.accept(
            target, body, message_id=message_id, actor=actor, priority=priority,
            deadline_ms=deadline_ms,
        )
        if receipt.newly_accepted:
            from power_loop.contracts.hooks import HookPoint
            from power_loop.core.execution_notifications import notify
            await notify(self.hooks, HookPoint.MESSAGE_ACCEPTED, {"receipt": receipt}, timeout_s=self.hook_timeout_s)
        if receipt.state != "accepted":
            return receipt
        published = await self._notify(receipt.target, receipt.message_id, receipt.input_ref,
                                       ["work", "activation"])
        from dataclasses import replace
        return replace(receipt, notification="published" if published else "failed",
                       reason=None if published else "wake_pending")

    async def cancel(self, ref, *, actor, scope="self", reason="cancelled") -> Receipt:
        self._require_open()
        receipt, activations = await self.inputs.cancel(
            ref, actor=actor, scope=scope, reason=reason, _with_activation=True,
        )
        target = receipt.target
        # Wake the owner so it can observe the cancel marker at its next checkpoint; the
        # marker itself is durable and is also read at every loop boundary.
        await self._notify(target, receipt.message_id or "cancel", receipt.input_ref,
                           ["activation", "work"], {"cancelled": True, "reason": reason})
        for parent in dict.fromkeys(activations):
            result_ref = InputRef(parent.store_id, parent.session_id, "cancel-result")
            await self._notify(parent, result_ref.message_id, result_ref, ["work", "activation"])
        return receipt

    async def inspect(self, ref, *, actor):
        self._require_open(reading=True)
        return await self.inputs.inspect(ref, actor=actor)

    async def messages(self, target, *, actor, after_seq=None, limit=20):
        self._require_open(reading=True)
        return await self.inputs.messages(target, actor=actor, after_seq=after_seq, limit=limit)

    async def list_children(self, target, *, actor, cursor=None, limit=100):
        self._require_open(reading=True)
        return await self.inputs.list_children(target, actor=actor, cursor=cursor, limit=limit)

    async def wait(self, input_ref, *, actor, deadline_ms=None):
        """Wait for an input's terminal result. Detached-safe: reads the durable input
        terminal state + result message, never an in-memory result cache."""
        self._require_open(reading=True)
        if self._slots >= 1024:
            raise CapacityError()
        self._slots += 1
        task = asyncio.current_task()
        self._waiters.add(task)
        subscription = None
        try:
            target = TargetRef(input_ref.store_id, input_ref.session_id)
            await self.inputs.authorize(self.inputs.db, target, actor)
            subscription = await self.transport.subscribe(SubscriptionSpec(
                self.namespace, input_ref.store_id, "output", "client", input_ref.session_id))
            await subscription.ready()
            while True:
                result = await self.inputs.get_result(input_ref, actor=actor)
                if result is not None:
                    return result
                remaining = 30.0 if deadline_ms is None else (deadline_ms - now_ms()) / 1000
                if remaining <= 0:
                    raise WaitTimeout(str(input_ref))
                try:
                    # Notifications only shorten latency; a short poll keeps a lost
                    # notification from stalling the durable result read.
                    delivery = await asyncio.wait_for(anext(subscription), min(0.2, remaining))
                    await delivery.ack()
                except TimeoutError:
                    # A lost notification is not a lost durable result; re-check above.
                    continue
        finally:
            self._slots -= 1
            self._waiters.discard(task)
            if subscription is not None:
                await subscription.aclose()

    async def subscribe_output(self, ref, *, actor, before_send=False):
        self._require_open(reading=True)
        if self._slots >= 1024:
            raise CapacityError()
        self._slots += 1
        subscription = None
        try:
            target = ref if isinstance(ref, TargetRef) else TargetRef(ref.store_id, ref.session_id)
            await self.inputs.authorize(self.inputs.db, target, actor)
            if isinstance(ref, InputRef) and await self.inputs.input(self.inputs.db, ref) is not None:
                await self.inputs.inspect(ref, actor=actor)
            self._require_open(reading=True)
            subscription = await self.transport.subscribe(SubscriptionSpec(
                self.namespace, ref.store_id, "output", "client", ref.session_id))
            await subscription.ready()
            self._require_open(reading=True)
            output = OutputSubscription(self, ref, actor, subscription)
            self._outputs.add(output)
            return output
        except BaseException:
            self._slots -= 1
            if subscription is not None:
                await subscription.aclose()
            raise

    # ── notifications ─────────────────────────────────────────────────────────

    async def _notify(self, target, message_id, input_ref, channels, payload=None):
        success = True
        for channel in channels:
            try:
                sent = await self.transport.publish(TransportPacket(
                    TRANSPORT_PROTOCOL, self.namespace, message_id, target, channel, input_ref, payload))
                success = sent.accepted and success
            except Exception:
                log.warning("input saved; %s notification failed for %s", channel, message_id, exc_info=True)
                success = False
        return success

    async def aclose(self):
        self._closed = True
        for task in tuple(self._waiters):
            task.cancel("client closed")
        if self._waiters:
            await asyncio.gather(*tuple(self._waiters), return_exceptions=True)
        for output in tuple(self._outputs):
            await output.aclose()


class OutputSubscription:
    """A lossy observation stream; the durable result is always re-read from the store."""

    def __init__(self, client, ref, actor, subscription):
        self.client, self.ref, self.actor, self.subscription = client, ref, actor, subscription
        self.sequence = 0
        self.closed = False
        self.initial = True

    async def ready(self):
        await self.subscription.ready()

    def __aiter__(self):
        return self

    async def __anext__(self):
        input_ref = self.ref if isinstance(self.ref, InputRef) else None
        while not self.closed:
            payload = None
            if not self.initial:
                try:
                    delivery = await anext(self.subscription)
                except BaseException:
                    await self.aclose()
                    raise
                await delivery.ack()
                if input_ref is not None and delivery.packet.input_ref is not None \
                        and delivery.packet.input_ref != input_ref:
                    continue
                payload = delivery.packet.payload or {}
                if payload.get("kind") == "delta":
                    self.sequence += 1
                    return OutputItem(self.ref, self.sequence, "delta", plain(payload))
            if input_ref is not None:
                result = await self.client.inputs.get_result(input_ref, actor=self.actor)
                if result is not None:
                    self.sequence += 1
                    await self.aclose()
                    return OutputItem(self.ref, self.sequence, "completed", plain(result))
            await self.client.inspect(self.ref, actor=self.actor)
            if self.initial or (payload and payload.get("kind") in {"gap", "error"}):
                self.initial = False
                self.sequence += 1
                kind = payload.get("kind", "gap") if payload else "gap"
                return OutputItem(self.ref, self.sequence, kind, plain(payload or {}))
        raise StopAsyncIteration

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        self.client._slots -= 1
        self.client._outputs.discard(self)
        await self.subscription.aclose()
