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
    def __init__(self, *, store, transport, namespace="power-loop", hooks=None, hook_timeout_s=1.0,
                 wake_transport=None):
        if not namespace:
            raise ValueError("namespace must be nonempty")
        self.store, self.transport, self.namespace = store, transport, namespace
        self.hooks, self.hook_timeout_s = hooks, hook_timeout_s
        self.wake_transport = wake_transport
        self._local_worker = None
        self.inputs = RuntimeInputStore(
            store,
            hooks_provider=lambda: self.hooks,
            hook_timeout_s=hook_timeout_s,
        )
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

    async def configure_session(self, target, *, actor, descriptor=None, cancel_propagation=None):
        self._require_open()
        return await self.inputs.configure_session(
            target, actor=actor, descriptor=descriptor, cancel_propagation=cancel_propagation,
        )

    async def context_budget(self, target_or_descriptor, *, actor=None):
        self._require_open(reading=True)
        return await self.inputs.context_budget(target_or_descriptor, actor=actor)

    async def seed_context(
        self, target, messages, *, actor, overflow="truncate_oldest", max_tokens=None, summary_fn=None,
    ) -> dict:
        self._require_open()
        return await self.inputs.seed_context(
            target, messages, actor=actor, overflow=overflow,
            max_tokens=max_tokens, summary_fn=summary_fn,
        )

    async def repair(self, target, *, mode="heal_broken", transform=None, actor):
        self._require_open()
        return await self.inputs.repair(
            target, mode=mode, transform=transform, actor=actor,
        )

    async def inspect_input(self, input_ref, *, actor):
        self._require_open(reading=True)
        return await self.inputs.inspect_input(input_ref, actor=actor)

    async def send(
        self, target, body, *, message_id, actor, priority="normal",
        deadline_ms=None, wake=True, kind="user", reply_route=None,
    ) -> Receipt:
        """Accept an input: idle starts a new round, busy joins the next loop boundary.

        When a local started worker has capacity and can execute the session, takes the
        Session-Core fused accept_and_claim fast path (2-tx); otherwise transparently falls
        back to durable accept + wake notification (3-tx).
        """
        self._require_open()
        worker = self._local_worker
        can_fuse = (
            worker is not None
            and worker._started
            and bool(wake)
            and getattr(worker.runtime, "fuse_local_send", True)
            and getattr(worker.runtime.coordinator, "installs_session_fence", False)
            and worker.can_fuse_locally(target.session_id)
        )
        batch = None
        if can_fuse:
            receipt, batch, activate = await self.inputs.accept_and_claim(
                target, body, message_id=message_id, actor=actor,
                node_id=worker.runtime.node_id, boot_id=worker.runtime.boot_id,
                ttl_s=worker.runtime.session_lease_ttl_s,
                priority=priority, deadline_ms=deadline_ms, wake=wake,
                kind=kind, reply_route=reply_route, can_execute=worker._capable,
            )
        else:
            receipt, activate = await self.inputs.accept(
                target, body, message_id=message_id, actor=actor, priority=priority,
                deadline_ms=deadline_ms, wake=wake, kind=kind, reply_route=reply_route,
                _with_activation=True,
            )
        if receipt.newly_accepted:
            from power_loop.contracts.hooks import HookPoint
            from power_loop.core.execution_notifications import notify
            await notify(self.hooks, HookPoint.MESSAGE_ACCEPTED, {"receipt": receipt}, timeout_s=self.hook_timeout_s)
        if batch is not None:
            worker.dispatch_claimed(batch)
            from dataclasses import replace
            return replace(receipt, notification="published", reason=None)
        if worker is not None and priority == "urgent" and receipt.state == "accepted":
            worker.notify_urgent(target.session_id)
        if receipt.state != "accepted" or not activate:
            return receipt
        published = await self._notify(receipt.target, receipt.message_id, receipt.input_ref,
                                       ["work", "activation"])
        if self.wake_transport is not None:
            try:
                await self.wake_transport.publish_wake(receipt.target.session_id)
            except Exception:
                log.warning("wake_transport publish_wake failed for %s", receipt.target.session_id, exc_info=True)
        from dataclasses import replace
        return replace(receipt, notification="published" if published else "failed",
                       reason=None if published else "wake_pending")

    async def cancel(
        self, ref, *, actor, scope="self", reason="cancelled", cancel_by: str | None = None,
    ) -> Receipt:
        self._require_open()
        receipt, activations, completed_refs, stopped_descendants = await self.inputs.cancel(
            ref, actor=actor, scope=scope, reason=reason, cancel_by=cancel_by,
            _with_activation=True, _with_stopped=True,
        )
        target = receipt.target
        eff_cancel_by = cancel_by or ("agent" if actor.session_id is not None else "user")
        signal_target = receipt.state == "cancel_requested"
        if self._local_worker is not None:
            if signal_target:
                self._local_worker._activate(
                    target.session_id, receipt.input_ref,
                    cancelled=True, cancel_reason=reason, cancel_verified=True,
                    cancel_by=eff_cancel_by,
                )
            for d_sid in stopped_descendants:
                self._local_worker._activate(
                    d_sid, None, cancelled=True, cancel_reason=reason, cancel_verified=True,
                    cancel_by="subtree",
                )
        if self.wake_transport is not None:
            if signal_target:
                try:
                    await self.wake_transport.publish_cancel(target.session_id, cancel_by=eff_cancel_by)
                except Exception:
                    log.warning("wake_transport publish_cancel failed for %s", target.session_id, exc_info=True)
            for d_sid in stopped_descendants:
                try:
                    await self.wake_transport.publish_cancel(d_sid, cancel_by="subtree")
                except Exception:
                    log.warning("wake_transport publish_cancel failed for %s", d_sid, exc_info=True)
        # Wake the owner only when storage confirmed an active running turn was cancelled;
        # finished/voided cancels must not stop a newer turn on the same session.
        if signal_target:
            await self._notify(target, receipt.message_id or "cancel", receipt.input_ref,
                               ["activation", "work"], {"cancelled": True, "reason": reason, "verified": True})
        for d_sid in stopped_descendants:
            await self._notify(
                TargetRef(target.store_id, d_sid), "cancel", None,
                ["activation", "work"], {"cancelled": True, "reason": reason, "verified": True},
            )
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

    async def set_lifecycle(
        self, target, lifecycle, *, actor, inflight="abort", subtree="none", reason=None,
    ):
        self._require_open()
        status, stopped_map, completed_refs = await self.inputs.set_lifecycle(
            target, lifecycle, actor=actor, inflight=inflight, subtree=subtree,
            reason=reason, _with_effects=True,
        )
        eff_reason = reason or ("removed" if lifecycle == "closed" else lifecycle)
        if status.need_wake:
            if self._local_worker is not None:
                self._local_worker._activate(target.session_id)
            await self._notify(target, "lifecycle-open", None, ["work", "activation"])
            if self.wake_transport is not None:
                try:
                    await self.wake_transport.publish_wake(target.session_id)
                except Exception:
                    log.warning("wake_transport publish_wake failed for %s", target.session_id, exc_info=True)
        for tid, cby in stopped_map.items():
            if self._local_worker is not None:
                self._local_worker._activate(
                    tid, None, cancelled=True, cancel_reason=eff_reason, cancel_verified=True,
                )
            if self.wake_transport is not None:
                try:
                    await self.wake_transport.publish_cancel(tid, cancel_by=cby)
                except Exception:
                    log.warning("wake_transport publish_cancel failed for %s", tid, exc_info=True)
            await self._notify(
                TargetRef(target.store_id, tid), "lifecycle-stop", None,
                ["activation", "work"], {"cancelled": True, "reason": eff_reason, "verified": True},
            )
        for completed_ref in dict.fromkeys(completed_refs):
            await self._notify(
                TargetRef(completed_ref.store_id, completed_ref.session_id),
                completed_ref.message_id, completed_ref, ["output"],
                {"kind": "completed", "status": "cancelled",
                 "session_id": completed_ref.session_id},
            )
        return status

    async def deactivate(self, target, *, actor, stop_subtree=False, reason="removed"):
        """Thin alias for ``set_lifecycle(target, 'closed', ...)``."""
        return await self.set_lifecycle(
            target,
            "closed",
            actor=actor,
            inflight="abort",
            subtree="all" if stop_subtree else "none",
            reason=reason,
        )

    async def rebind_children(
        self, from_parent, to_parent, *, child_targets=None, replay_undelivered=True,
        rewrite_host_context=None, actor,
    ):
        self._require_open()
        result = await self.inputs.rebind_children(
            from_parent,
            to_parent,
            child_targets=child_targets,
            replay_undelivered=replay_undelivered,
            rewrite_host_context=rewrite_host_context,
            actor=actor,
        )
        if result.need_wake:
            if self._local_worker is not None:
                self._local_worker._activate(to_parent.session_id)
            await self._notify(to_parent, "rebind-replay", None, ["work", "activation"])
            if self.wake_transport is not None:
                try:
                    await self.wake_transport.publish_wake(to_parent.session_id)
                except Exception:
                    log.warning("wake_transport publish_wake failed for %s", to_parent.session_id, exc_info=True)
        return result

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
        deadline_ms=None, wake=True, kind="user", reply_route=None, on_submit_start=None,
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
                wake=wake, kind=kind, reply_route=reply_route,
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
