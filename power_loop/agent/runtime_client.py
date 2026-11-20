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
        receipt, activate = await self.inputs.accept(
            target, body, message_id=message_id, actor=actor, priority=priority,
            deadline_ms=deadline_ms, _with_activation=True,
        )
        if receipt.newly_accepted:
            from power_loop.contracts.hooks import HookPoint
            from power_loop.core.execution_notifications import notify
            await notify(self.hooks, HookPoint.MESSAGE_ACCEPTED, {"receipt": receipt}, timeout_s=self.hook_timeout_s)
        if receipt.state != "accepted" or not activate:
            return receipt
        published = await self._notify(receipt.target, receipt.message_id, receipt.input_ref,
                                       ["work", "activation"])
        from dataclasses import replace
        return replace(receipt, notification="published" if published else "failed",
                       reason=None if published else "wake_pending")

    async def cancel(self, ref, *, actor, scope="self", reason="cancelled") -> Receipt:
        self._require_open()
        receipt, activations, completed_refs = await self.inputs.cancel(
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
        for completed_ref in dict.fromkeys(completed_refs):
            await self._notify(
                TargetRef(completed_ref.store_id, completed_ref.session_id),
                completed_ref.message_id, completed_ref, ["output"],
                {"kind": "completed", "status": "cancelled",
                 "session_id": completed_ref.session_id},
            )
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

    async def _durable_result(self, input_ref, actor):
        """One authorized persistent read; only a durable terminal returns non-None."""
        return (await self.inputs.read_result_state(input_ref, actor=actor)).result

    async def wait(self, input_ref, *, actor, deadline_ms=None):
        """Wait for an input's terminal result through one output observation."""
        output = await self.subscribe_output(input_ref, actor=actor)
        try:
            return await output.wait(deadline_ms=deadline_ms)
        finally:
            await output.aclose()

    async def subscribe_output(self, ref, *, actor, before_send=False):
        """Authorize and read the initial input/result before returning the subscription.

        ``before_send=True`` is the only case where the target may be authorized while
        its input does not exist yet; after-send subscriptions keep the public
        not_found contract.
        """
        return await self._open_output(
            ref, actor=actor, read_initial=True, allow_missing=before_send,
        )

    async def send_and_subscribe(
        self, target, body, *, message_id, actor, priority="normal",
        deadline_ms=None, on_submit_start=None,
    ) -> OutputSubscription:
        """Accept once and observe that input without a post-accept initial read.

        The subscription is ready before submission. A newly accepted input reuses the
        accept facts; an idempotent retry still performs the normal authorized initial
        read because it is not known to be this call's fresh input.
        """
        async def submit():
            if on_submit_start is not None:
                on_submit_start()
            receipt = await self.send(
                target, body, message_id=message_id, actor=actor,
                priority=priority, deadline_ms=deadline_ms,
            )
            ref = receipt.input_ref
            initial = None
            if not receipt.newly_accepted:
                initial = await self.inputs.read_result_state(ref, actor=actor)
                if initial.record is None:
                    raise RuntimeRequestError("not_found")
            return ref, initial
        return await self._open_output(
            target, actor=actor, submit=submit, discovery_message_id=message_id,
        )

    async def _open_output(
        self, ref, *, actor, read_initial=False, allow_missing=False, submit=None,
        discovery_message_id=None,
    ):
        """Own one output from slot acquisition through registration or cleanup."""
        self._require_open(reading=True)
        if self._slots >= 1024:
            raise CapacityError()
        self._slots += 1
        task = asyncio.current_task()
        self._waiters.add(task)
        subscription = None
        output = None
        try:
            target = ref if isinstance(ref, TargetRef) else TargetRef(ref.store_id, ref.session_id)
            if isinstance(ref, InputRef):
                discovery_message_id = ref.message_id
            subscription = await self.transport.subscribe(SubscriptionSpec(
                self.namespace, ref.store_id, "output", "client", ref.session_id,
                discovery_message_id=discovery_message_id,
            ))
            # The raw subscription is owned by an existing OutputSubscription before
            # any await that can fail; internal cleanup never keeps it in a local only.
            output = OutputSubscription(self, ref, actor, subscription)
            self._outputs.add(output)
            await subscription.ready()
            initial = None
            if submit is not None:
                output_ref, initial = await submit()
                output.ref = output_ref
            else:
                output_ref = ref
                if read_initial:
                    if isinstance(ref, InputRef):
                        initial = await self.inputs.read_result_state(ref, actor=actor)
                        if initial.record is None and not allow_missing:
                            raise RuntimeRequestError("not_found")
                    else:
                        await self.inputs.authorize(self.inputs.db, target, actor)
                output.ref = output_ref
            output._initial_read = initial
            self._require_open(reading=True)
            self._waiters.discard(task)
            return output
        except BaseException:
            try:
                if output is not None:
                    await output.aclose()
                elif subscription is not None:
                    await subscription.aclose()
            finally:
                if output is None:
                    self._slots -= 1
                self._waiters.discard(task)
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


def _related_output(packet, input_ref):
    """The payload kind when the packet addresses ``input_ref``; None otherwise.

    A session-wide packet (no addressed input) is never another input's output.
    """
    if input_ref is not None and packet.input_ref is not None and packet.input_ref != input_ref:
        return None
    return (packet.payload or {}).get("kind")


class OutputSubscription:
    """A lossy observation stream; the durable result is always re-read from the store.

    The initial authorized read happens before this object is exposed. The first
    ``__anext__`` reuses it, so constructing a subscription never inspects twice.
    After that, only a related completion or a gap/reconnect reads the store again;
    deltas are forwarded and unrelated InputRefs are consumed without a read.
    """

    def __init__(self, client, ref, actor, subscription, initial=None):
        self.client, self.ref, self.actor, self.subscription = client, ref, actor, subscription
        self.sequence = 0
        self.closed = False
        self.initial = True
        self._initial_read = initial
        self._finished = False
        self._result = None

    async def ready(self):
        await self.subscription.ready()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        input_ref = self.ref if isinstance(self.ref, InputRef) else None
        if self.initial:
            self.initial = False
            state, self._initial_read = self._initial_read, None
            if state is not None and state.result is not None:
                return await self._completed(state.result)
            self.sequence += 1
            return OutputItem(self.ref, self.sequence, "gap", {})
        while not self.closed:
            try:
                delivery = await anext(self.subscription)
            except BaseException:
                await self.aclose()
                raise
            await delivery.ack()
            kind = _related_output(delivery.packet, input_ref)
            if kind is None:
                continue
            payload = delivery.packet.payload or {}
            if kind == "delta":
                self.sequence += 1
                return OutputItem(self.ref, self.sequence, "delta", plain(payload))
            if kind not in {"completed", "gap", "error"}:
                continue
            if input_ref is not None:
                result = await self.client._durable_result(input_ref, self.actor)
                if result is not None:
                    return await self._completed(result)
            if kind in {"gap", "error"}:
                self.sequence += 1
                return OutputItem(self.ref, self.sequence, kind, plain(payload))
        raise StopAsyncIteration

    async def _completed(self, result):
        self._result = result
        self.sequence += 1
        item = OutputItem(self.ref, self.sequence, "completed", plain(result))
        await self.aclose()
        self._finished = True
        return item

    async def wait(self, *, deadline_ms=None):
        """Consume this one InputRef until its real terminal Result is available."""
        if not isinstance(self.ref, InputRef):
            raise RuntimeRequestError("invalid_options", "wait")
        task = asyncio.current_task()
        self.client._waiters.add(task)
        loop = asyncio.get_running_loop()
        deadline_at = (None if deadline_ms is None else
                       loop.time() + max(0.0, (deadline_ms - now_ms()) / 1000))
        scope = None
        try:
            while True:
                try:
                    if deadline_at is None:
                        item = await anext(self)
                    else:
                        scope = asyncio.timeout_at(deadline_at)
                        async with scope:
                            item = await anext(self)
                except TimeoutError:
                    if scope is None or not scope.expired():
                        raise
                    if self._result is not None:
                        await self.aclose()
                        self._finished = True
                        return self._result
                    completed = await self.finish_from_store()
                    if completed is not None:
                        return self._result
                    raise WaitTimeout(str(self.ref)) from None
                except StopAsyncIteration:
                    if self.client._closed:
                        raise asyncio.CancelledError("client closed") from None
                    raise
                if item.kind == "completed":
                    if self._result is None:
                        raise RuntimeError("completed output has no durable result")
                    return self._result
        finally:
            try:
                await self.aclose()
            finally:
                self.client._waiters.discard(task)

    async def finish_from_store(self):
        """The one explicit read an outer deadline is allowed to make.

        Returns a ``completed`` item when the durable result is already terminal and
        the subscription is released; ``None`` means the input is still running, so
        the host keeps its original timeout/cancel behavior. The transport may already
        be closed by the cancellation that ended iteration; this read only needs the
        durable store and is still the one explicit read of that deadline.
        """
        if not isinstance(self.ref, InputRef):
            return None
        if self._finished:
            return None
        if self._result is not None:
            item = OutputItem(self.ref, self.sequence, "completed", plain(self._result))
            await self.aclose()
            self._finished = True
            return item
        state, self._initial_read = self._initial_read, None
        result = (state.result if state is not None and state.result is not None
                  else await self.client._durable_result(self.ref, self.actor))
        if result is None:
            return None
        self.initial = False
        self._result = result
        self.sequence += 1
        item = OutputItem(self.ref, self.sequence, "completed", plain(result))
        await self.aclose()
        self._finished = True
        return item

    async def aclose(self):
        if self.closed:
            return
        await self.subscription.aclose()
        if self.closed:
            return
        self.closed = True
        self.client._slots -= 1
        self.client._outputs.discard(self)
