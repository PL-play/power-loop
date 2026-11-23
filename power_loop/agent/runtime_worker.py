"""Session-Core execution scheduling (design 217).

The worker discovers ``queued`` / running-expired sessions, claims one batch on a fixed
node (or runs a locally fused ``accept_and_claim`` batch directly), runs the ordinary agent
loop on the session/send/owner_epoch identity, and writes the result through the
accept/claim/final storage boundary with piggybacked tool-round checks and node-level
periodic lease renewal.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
from dataclasses import replace

from power_loop.contracts.events import AgentEventType
from power_loop.core.events import AgentEventBus
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.session_ownership import _CURRENT, SessionOwnership

from .runtime_store import RuntimeInputStore, now_ms
from .runtime_transport import PROTOCOL_VERSION as TRANSPORT_PROTOCOL
from .runtime_transport import SubscriptionSpec, TransportPacket
from .runtime_types import (
    InputRef,
    RequestOptions,
    RuntimeDescriptor,
    RuntimeRequestError,
    ShutdownResult,
    StoreError,
    TargetRef,
    TrustedActor,
    plain,
)

log = logging.getLogger(__name__)


class _NotifyingCancellationToken(CancellationToken):
    """CancellationToken that immediately wakes an asyncio.Event (0 DB polling)."""

    def __init__(self) -> None:
        super().__init__()
        self.event = asyncio.Event()

    def cancel(self, reason: str = "cancelled") -> None:
        super().cancel(reason)
        self.event.set()


class _SessionExecution:
    """The local context of one running send (no job identity)."""

    def __init__(self, runtime, sid, descriptor, batch, actor, parent_session_id=None):
        self.runtime = runtime
        self.store_id = runtime.store.store_id
        self.session_id = sid
        self.parent_session_id = parent_session_id
        self.node_id = runtime.node_id
        self.boot_id = runtime.boot_id
        self.descriptor = descriptor
        self.batch = batch
        self.ownership = None
        self.input_ref = batch.input_refs[0]
        self.actor = actor
        self.stop = _NotifyingCancellationToken()
        self.stop_event = self.stop.event
        self.cancel_by: str | None = batch.cancel_by
        self.task = None
        self.loop = None
        self.closing = False
        self.finalizing = False
        self.finalized = False
        self.shutdown_interrupted = False
        self._lost_notified = False
        self._lost_hook_task = None
        # The ordinary LLM boundary extends this same run-local list. None means
        # claim/factory or a non-LLM path has not established that consumption fact.
        self.consumed: list[InputRef] | None = None
        self.wake_pending = False
        self.urgent_hint = False
        self.cancel_hint = False
        self.output: asyncio.Queue = asyncio.Queue(maxsize=256)
        option_values = descriptor.config.get("child_options", descriptor.config)
        self.options = RequestOptions(**{
            k: v for k, v in option_values.items() if k in RequestOptions.__dataclass_fields__
        })
        if batch.max_rounds_override is not None:
            self.options = replace(self.options, max_rounds=batch.max_rounds_override)

    @property
    def send_index(self):
        return self.batch.send_index

    def snapshot(self):
        from power_loop.core.execution_context import ExecutionContext
        return ExecutionContext(
            self.store_id, self.session_id, self.node_id, self.boot_id,
            input_ref=self.input_ref, owner_epoch=self.batch.owner_epoch,
            actor=self.actor, options=self.options,
            cancel_token=self.stop, host_context=self.descriptor.host_context,
            send_index=self.batch.send_index,
            parent_session_id=self.parent_session_id,
        )


class RuntimeWorker:
    def __init__(self, runtime):
        self.runtime = runtime
        self.inputs = RuntimeInputStore(
            runtime.store,
            hooks_provider=lambda: self.runtime.hooks,
            hook_timeout_s=runtime.hook_timeout_s,
            on_local_claim=self._note_local_claim,
            on_local_claim_cancel=self._unnote_local_claim,
        )
        self._subscriptions, self._listeners = [], []
        self._wake_subscription = None
        self._wake_listener = None
        self._bindings = []
        self._pending = {}
        self._active = {}
        self._fusing: dict[str, int] = {}
        self._pending_claims: dict[tuple[str, int], asyncio.Task | None] = {}
        self._drivers = set()
        self._tasks = set()
        self._capacity = asyncio.Semaphore(runtime.capacity)
        self._wake = asyncio.Event()
        self._started, self._closing, self._closed = False, False, False
        self._lifecycle_lock = asyncio.Lock()
        self._overflow = False
        self._guard_monitor = None
        self.last_error = None
        # Local health: definitions this node cannot build. Never persisted as a task
        # registry; an incapable node simply leaves the session for a capable one.
        self.capability_missing = set()
        self.shutdown_result = ShutdownResult(True)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        async with self._lifecycle_lock:
            await self._start()

    async def _start(self):
        if self._closed:
            raise RuntimeRequestError("session_closed")
        if self._started:
            return
        r = self.runtime
        try:
            preflight = getattr(r.factory, "preflight", None)
            if preflight is not None:
                await preflight(r.required_profiles)
            elif r.required_profiles:
                raise RuntimeRequestError("capability_unavailable", "factory preflight is required")
            if r.process_guard is not None:
                await r.process_guard.acquire()
            for channel in ("activation", "work"):
                binding = SubscriptionSpec(r.namespace, r.store.store_id, channel, r.node_id + "/" + r.boot_id)
                sub = await r.transport.subscribe(binding)
                self._bindings.append(binding)
                self._subscriptions.append(sub)
                await sub.ready()
            # Ready precedes recovery; dispatch begins only after the bounded initial page.
            await self.recover()
            if r.timer_driver is not None:
                await r.timer_driver.start()
                if not r.timer_driver.status().ready:
                    raise RuntimeRequestError("capability_unavailable", "timer driver is not ready")
        except BaseException:
            if r.timer_driver is not None:
                await r.timer_driver.stop()
            for sub in self._subscriptions:
                await sub.aclose()
            self._subscriptions.clear()
            self._bindings.clear()
            if r.process_guard is not None:
                await r.process_guard.release()
            raise
        self._started = True
        if r.wake_transport is not None:
            async def _on_wake(sid: str, _reason: str) -> None:
                self._activate(sid)

            async def _on_cancel(sid: str, cby: str) -> None:
                self._activate(sid, cancelled=True, cancel_verified=True, cancel_by=cby)

            await r.wake_transport.subscribe(on_wake=_on_wake, on_cancel=_on_cancel)
        self._listeners = [asyncio.create_task(self._listen(s), context=contextvars.Context()) for s in self._subscriptions]
        self._dispatcher = asyncio.create_task(self._dispatch(), context=contextvars.Context())
        if self._uses_node_batch_renewal():
            if r.recovery_interval_s is not None:
                self._recovery = asyncio.create_task(
                    self._periodic_maintenance(), context=contextvars.Context(),
                )
            else:
                self._recovery = asyncio.create_task(
                    self._batch_renew_periodically(), context=contextvars.Context(),
                )
        else:
            self._recovery = (
                asyncio.create_task(self._recover_periodically(), context=contextvars.Context())
                if r.recovery_interval_s is not None else None
            )
        r.client._accepting = True
        if r.process_guard is not None:
            self._guard_monitor = asyncio.create_task(self._watch_guard(), context=contextvars.Context())

    def _active_llm_count(self) -> int:
        return sum(
            1 for ctx in self._active.values()
            if ctx is None or ctx.descriptor.kind != "workflow_driver"
        )

    def can_fuse_locally(self, sid: str) -> bool:
        from .stateful_loop import _ExecutionLoop
        return (
            not self._closing
            and sid not in self._active
            and sid not in self._pending
            and not _ExecutionLoop._session_busy(sid)
            and len(self._tasks) < 1024
            and self._active_llm_count() < self.runtime.capacity
        )

    def _begin_fuse(self, sid: str) -> None:
        self._fusing[sid] = self._fusing.get(sid, 0) + 1

    def _end_fuse(self, sid: str) -> None:
        count = self._fusing.get(sid, 0) - 1
        if count <= 0:
            self._fusing.pop(sid, None)
        else:
            self._fusing[sid] = count

    def _note_local_claim(
        self, node_id: str, boot_id: str, claims: list[tuple[str, int]],
    ) -> None:
        r = self.runtime
        if node_id != r.node_id or boot_id != r.boot_id:
            return
        task = asyncio.current_task()
        for sid, epoch in claims:
            key = (str(sid), int(epoch))
            self._pending_claims[key] = task
            if task is not None:
                def _cleanup(_t, _k=key, _owner=task):
                    if self._pending_claims.get(_k) is _owner:
                        self._pending_claims.pop(_k, None)
                task.add_done_callback(_cleanup)

    def _unnote_local_claim(
        self, node_id: str, boot_id: str, claims: list[tuple[str, int]],
    ) -> None:
        r = self.runtime
        if node_id != r.node_id or boot_id != r.boot_id:
            return
        for sid, epoch in claims:
            self._pending_claims.pop((str(sid), int(epoch)), None)

    def dispatch_claimed(self, batch) -> None:
        """Schedule a locally fused accept_and_claim or pre-claimed batch without re-claiming."""
        sid = batch.target.session_id
        self._active[sid] = None
        self._pending_claims.pop((sid, int(batch.owner_epoch)), None)
        task = asyncio.create_task(
            self._run_session(sid, preclaimed_batch=batch),
            context=contextvars.Context(),
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)

    def notify_urgent(self, sid: str) -> None:
        ctx = self._active.get(sid)
        if ctx is not None:
            ctx.urgent_hint = True

    async def _watch_guard(self):
        while True:
            await asyncio.sleep(5)
            try:
                await self.runtime.process_guard.check()
            except Exception as exc:
                self.last_error = "store_lock_lost: " + str(exc)[:500]
                await self.shutdown(timeout_s=0, release_leases=False)
                return

    def _activate(
        self, sid, input_ref=None, *, cancelled=False, cancel_reason=None,
        cancel_verified=False, cancel_by=None,
    ):
        if self._closing:
            return
        if sid in self._active:
            ctx = self._active[sid]
            if ctx is not None and cancelled:
                if cancel_by is not None:
                    ctx.cancel_by = cancel_by
                if cancel_verified:
                    if ctx.descriptor.kind == "workflow_driver":
                        self.inputs.notify_driver_cancel(
                            sid, cancel_by=ctx.cancel_by, reason=cancel_reason or "cancelled",
                        )
                    ctx.stop.cancel(cancel_reason or "cancelled")
                else:
                    ctx.cancel_hint = True
                return
            if ctx is not None and ctx.descriptor.kind == "workflow_driver" and not cancelled:
                self.inputs.notify_driver_wake(sid)
                return
            if ctx is not None and ctx.consumed is None \
                    and input_ref is not None and not cancelled:
                ctx.urgent_hint = True
                return
            if ctx is not None and ctx.consumed is not None \
                    and input_ref is not None and not cancelled \
                    and input_ref not in ctx.consumed:
                ctx.wake_pending = True
            return
        if len(self._pending) >= 1024 and sid not in self._pending:
            self._overflow = True
            return
        self._pending.setdefault(sid, now_ms())
        self._wake.set()

    def _uses_node_batch_renewal(self) -> bool:
        """True when using the default database-fence coordinator (node-level O(1) batch_renew)."""
        from power_loop.runtime.store_session_ownership import StoreSessionOwnershipCoordinator
        coord = self.runtime.coordinator
        return (
            getattr(coord, "installs_session_fence", False)
            and type(coord).renew is StoreSessionOwnershipCoordinator.renew
        )

    async def batch_renew(self):
        """Node-level 1-SQL (MySQL 2-SQL) lease renewal + cancel/close check for all active sessions."""
        r = self.runtime
        started = time.monotonic()
        rows = await self.inputs.batch_renew(
            node_id=r.node_id, boot_id=r.boot_id, ttl_s=r.session_lease_ttl_s,
        )
        by_sid = {row["session_id"]: row for row in rows}
        now_mono = started + r.session_lease_ttl_s
        for sid, ctx in list(self._active.items()):
            if (
                ctx is None
                or ctx.closing
                or ctx.finalizing
                or ctx.finalized
                or (ctx.ownership is not None and ctx.ownership.released)
            ):
                continue
            row = by_sid.get(sid)
            if (
                row is None
                or int(row["owner_epoch"]) != ctx.batch.owner_epoch
                or (ctx.ownership is not None and ctx.ownership.expires_at <= started)
            ):
                if ctx.ownership is not None:
                    ctx.ownership.lose()
                continue
            if ctx.ownership is not None:
                ctx.ownership.expires_at = now_mono
            if row["lifecycle"] == "closing" or int(row["cancel_seq"]) > ctx.batch.cancel_seq:
                cby = row.get("cancel_by") or ("removed" if row["lifecycle"] in ("closing", "closed") else "user")
                ctx.cancel_by = cby
                if ctx.descriptor.kind == "workflow_driver":
                    self.inputs.notify_driver_cancel(
                        sid, cancel_by=cby, reason=row.get("cancel_reason") or "cancelled",
                    )
                ctx.stop.cancel(row.get("cancel_reason") or "cancelled")
        # Slow path (P0-1): expire orphan running leases owned by this (node_id, boot_id)
        # that are absent from local active/fusing/pending-claim sets or have a mismatched owner_epoch.
        for row in rows:
            sid = str(row["session_id"])
            epoch = int(row["owner_epoch"])
            if sid in self._fusing:
                continue
            pending_key = (sid, epoch)
            if pending_key in self._pending_claims:
                pending_task = self._pending_claims[pending_key]
                if pending_task is None or not pending_task.done():
                    continue
                self._pending_claims.pop(pending_key, None)
            if sid in self._active:
                ctx = self._active[sid]
                if ctx is None:
                    continue
                if epoch == ctx.batch.owner_epoch:
                    if ctx.finalized or (
                        not ctx.closing
                        and not (ctx.ownership is not None and ctx.ownership.released)
                    ):
                        continue
            await self.inputs.release_orphan_lease(
                sid,
                node_id=r.node_id,
                boot_id=r.boot_id,
                owner_epoch=epoch,
            )
        return rows

    async def periodic_tick(self, *, limit: int = 32):
        """One periodic node maintenance tick: batch_renew + discover_queued + discover_expired_leases (3/3/4 SQL)."""
        r = self.runtime
        renewed = await self.batch_renew()
        queued = await self.inputs.discover_queued(limit=limit)
        expired_rows = await self.inputs.discover_expired_leases(limit=limit)
        for sid in queued:
            self._activate(sid)
        if len(queued) < limit:
            self._overflow = False
        reclaimed = []
        for row in expired_rows:
            sid = str(row["session_id"])
            if sid in self._active:
                continue
            needs_model_slot = (
                row["session_kind"] == "agent"
                and row["lifecycle"] == "open"
                and row["cancel_by"] is None
            )
            if needs_model_slot:
                if self._active_llm_count() >= r.capacity:
                    # Respect Worker capacity: queue for _dispatch() instead of bypassing concurrency!
                    self._activate(sid)
                    continue
                # Reserve the capacity slot BEFORE awaiting reclaim_running so neither _dispatch()
                # nor locally fused accept_and_claim can exceed runtime.capacity.
                self._active[sid] = None
                dispatched = False
                try:
                    batch = await self.inputs.reclaim_running(
                        TargetRef(r.store.store_id, sid),
                        expected_epoch=int(row["owner_epoch"]),
                        node_id=r.node_id, boot_id=r.boot_id,
                        ttl_s=r.session_lease_ttl_s, can_execute=self._capable,
                    )
                    if batch is None:
                        continue
                    reclaimed.append(batch)
                    if batch.reclaimed_cancelled:
                        recovered = await self.inputs.final(
                            batch, status="cancelled", final_text="",
                            error="cancelled", error_code="cancelled",
                        )
                        await self._notify_results(recovered)
                        if recovered.execution_state == "queued":
                            self._activate(sid)
                    elif not batch.reclaimed_has_assistant and "workflow_node" not in batch.descriptor.config:
                        self.dispatch_claimed(batch)
                        dispatched = True
                    else:
                        recovered = await self.inputs.final(
                            batch, status="interrupted", final_text="",
                            error="execution owner lost", error_code="interrupted",
                        )
                        await self._notify_results(recovered)
                        if recovered.execution_state == "queued":
                            self._activate(sid)
                finally:
                    if not dispatched:
                        self._active.pop(sid, None)
                        self._wake.set()
            elif row["session_kind"] == "workflow_driver":
                self._drivers.add(sid)
                self._begin_fuse(sid)
                try:
                    recovered = await self.inputs.recover(
                        TargetRef(r.store.store_id, sid),
                        node_id=r.node_id, boot_id=r.boot_id, ttl_s=r.session_lease_ttl_s,
                    )
                finally:
                    self._end_fuse(sid)
                if recovered is None:
                    continue
                reclaimed.append(recovered)
                await self._notify_results(recovered)
                if recovered.execution_state == "queued":
                    self._activate(sid)
            else:
                self._begin_fuse(sid)
                try:
                    batch = await self.inputs.reclaim_running(
                        TargetRef(r.store.store_id, sid),
                        expected_epoch=int(row["owner_epoch"]),
                        node_id=r.node_id, boot_id=r.boot_id,
                        ttl_s=r.session_lease_ttl_s, can_execute=self._capable,
                    )
                    if batch is None:
                        continue
                    reclaimed.append(batch)
                    if batch.reclaimed_cancelled:
                        recovered = await self.inputs.final(
                            batch, status="cancelled", final_text="",
                            error="cancelled", error_code="cancelled",
                        )
                    else:
                        recovered = await self.inputs.final(
                            batch, status="interrupted", final_text="",
                            error="execution owner lost", error_code="interrupted",
                        )
                finally:
                    self._end_fuse(sid)
                await self._notify_results(recovered)
                if recovered.execution_state == "queued":
                    self._activate(sid)
        return {"renewed": renewed, "queued": queued, "expired": reclaimed}

    async def _periodic_maintenance(self):
        r = self.runtime
        interval = min(r.recovery_interval_s, r.session_lease_ttl_s / 3)
        limit = min(32, r.claim_limit)
        while not self._closing:
            await asyncio.sleep(interval)
            if self._closing:
                return
            try:
                await self.periodic_tick(limit=limit)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("periodic maintenance tick failed")

    async def _batch_renew_periodically(self):
        interval = self.runtime.session_lease_ttl_s / 3
        while not self._closing:
            await asyncio.sleep(interval)
            if self._closing:
                return
            if not self._active:
                continue
            try:
                await self.batch_renew()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("periodic batch_renew failed")

    async def recover(self):
        # The ONLY discovery path: bounded scan of unfinished sessions. No job scan, no
        # pending-inbox-only discovery — a running-expired session with no pending input
        # is still found here so its old wait can be terminated (design 215 §9.3).
        page = await self.inputs.discover_unfinished(limit=self.runtime.claim_limit)
        for ref in page:
            self._activate(ref.session_id)
        if len(page) < self.runtime.claim_limit:
            self._overflow = False

    async def _recover_periodically(self):
        while True:
            await asyncio.sleep(self.runtime.recovery_interval_s)
            await self.recover()

    async def _listen(self, subscription):
        index = self._subscriptions.index(subscription)
        binding = self._bindings[index]
        while not self._closed:
            try:
                async for delivery in subscription:
                    # Hints carry no durable input. Lost hints rely on a live boundary
                    # or unfinished discovery (explicit recover if the scan is disabled).
                    packet = delivery.packet
                    payload = packet.payload or {}
                    self._activate(
                        packet.target.session_id, packet.input_ref,
                        cancelled=bool(payload.get("cancelled")),
                        cancel_reason=payload.get("reason"),
                        cancel_verified=bool(payload.get("verified")),
                    )
                    await delivery.ack()
            except Exception:
                log.exception("runtime %s subscription disconnected", binding.channel)
            if self._closing:
                return
            await subscription.aclose()
            while not self._closing:
                await asyncio.sleep(0.2)
                try:
                    subscription = await self.runtime.transport.subscribe(binding)
                    self._subscriptions[index] = subscription
                    await subscription.ready()
                    await self.recover()
                    break
                except Exception:
                    await subscription.aclose()
                    log.exception("runtime %s resubscribe failed", binding.channel)

    async def _dispatch(self):
        while not self._closing:
            await self._wake.wait()
            self._wake.clear()
            if self._overflow and len(self._pending) < 924:
                await self.recover()
            while self._pending and len(self._tasks) < 1024 and not self._closing:
                used = sum(
                    1 for ctx in self._active.values()
                    if ctx is None or ctx.descriptor.kind != "workflow_driver"
                )
                if used < self.runtime.capacity:
                    sid = next(iter(self._pending))
                else:
                    sid = next((value for value in self._pending if value in self._drivers), None)
                    if sid is None:
                        break
                self._pending.pop(sid)
                self._active[sid] = None
                task = asyncio.create_task(self._run_session(sid), context=contextvars.Context())
                self._tasks.add(task)
                task.add_done_callback(self._task_finished)
            if self._overflow and len(self._pending) < 924:
                self._wake.set()

    def _task_finished(self, task):
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("runtime execution failed", exc_info=task.exception())
        self._wake.set()

    async def _run_session(self, sid, *, preclaimed_batch=None):
        from .stateful_loop import _ExecutionLoop, _session_sync
        entry = _session_sync(sid)
        entry.users += 1
        try:
            # _dispatch enforces capacity for LLM/background sessions. Non-LLM
            # workflow drivers never take a model slot, including during recovery
            # while all their healthy leaves are still running.
            async with entry.lock:
                await self._run_session_owned(sid, preclaimed_batch=preclaimed_batch)
        finally:
            entry.users -= 1
            _ExecutionLoop._release_idle_sync(sid)

    # ── one send on a fixed node ──────────────────────────────────────────────

    async def _run_session_owned(self, sid, *, preclaimed_batch=None):
        from power_loop.contracts.errors import CancellationRequested, SessionLeaseLost
        from power_loop.contracts.hooks import HookPoint
        from power_loop.core.execution_context import _bind_execution_context
        from power_loop.core.execution_notifications import notify

        from .stateful_loop import _ExecutionLoop

        r = self.runtime
        store = r.store
        target = TargetRef(store.store_id, sid)
        ctx = None
        env = None
        env_ctx = None
        bus = None
        on_delta = None
        output_task = None
        renewal = None
        ownership = None
        ownership_token = None
        external_ownership = None
        batch = preclaimed_batch
        normal_final = None
        driver_requeue = False

        async def _flush_output():
            nonlocal output_task, on_delta
            if bus is not None and on_delta is not None:
                await bus.drain(timeout=r.hook_timeout_s)
                bus.unsubscribe(on_delta)
                on_delta = None
            if output_task is not None and ctx is not None:
                task = output_task
                output_task = None
                await ctx.output.put(None)
                await asyncio.gather(task, return_exceptions=True)

        try:
            if batch is not None:
                pass
            elif getattr(r.coordinator, "installs_session_fence", False):
                # The claim transaction is the whole entry: under the state lock it reads
                # the descriptor/identity and the real pending batch, checks capability
                # before writing the owner, and returns the merged user row it wrote.
                batch = await r.coordinator.claim(
                    self.inputs, target, node_id=r.node_id, boot_id=r.boot_id,
                    ttl_s=r.session_lease_ttl_s, can_execute=self._capable)
                if batch is None:
                    reclaimed_or_queued = await self._recover_stale(target)
                    if reclaimed_or_queued is False:
                        return
                    if reclaimed_or_queued is True:
                        batch = await r.coordinator.claim(
                            self.inputs, target, node_id=r.node_id, boot_id=r.boot_id,
                            ttl_s=r.session_lease_ttl_s, can_execute=self._capable)
                        if batch is None:
                            return
                    else:
                        # 0-assistant crash rerun: reclaim_running returned a fresh ClaimedBatch!
                        batch = reclaimed_or_queued
            else:
                # A foreign coordinator's acquire decision needs the session facts before
                # claim. Keep this single merged entry read on that topology only; the
                # database-fence topology gets the same facts inside claim.
                entry = await self.inputs.session_entry(target)
                if entry is None or entry["lifecycle"] != "open":
                    return
                descriptor = RuntimeDescriptor(**json.loads(entry["descriptor_json"]))
                if descriptor.kind == "workflow_driver":
                    self._drivers.add(sid)
                if not self._capable(descriptor):
                    return
                if self._needs_recovery(entry):
                    recovered = await self.inputs.recover(
                        target, node_id=r.node_id, boot_id=r.boot_id,
                        ttl_s=r.session_lease_ttl_s)
                    if recovered is None:
                        return
                    await self._notify_results(recovered)
                    if recovered.execution_state != "queued":
                        return
                elif entry["execution_state"] != "queued":
                    return
                external_ownership = await r.coordinator.acquire(
                    (store.store_id, sid), owner_id=r.node_id + "/" + r.boot_id,
                    ttl_s=r.session_lease_ttl_s, on_lost=lambda: None)
                if external_ownership is None:
                    return
                batch = await self.inputs.claim(
                    target, node_id=r.node_id, boot_id=r.boot_id,
                    ttl_s=r.session_lease_ttl_s, can_execute=self._capable)
                if batch is None:
                    return

            descriptor = batch.descriptor
            if descriptor.kind == "workflow_driver":
                self._drivers.add(sid)
            actor = TrustedActor(batch.tenant_id, batch.principal_id, sid)

            # The claim installed the fence; build the local handle from it. Per-ownership
            # process caches are cleared so no seq/index survives a change of owner.
            if external_ownership is None:
                ownership = SessionOwnership(
                    (store.store_id, sid), r.node_id + "/" + r.boot_id,
                    time.monotonic() + r.session_lease_ttl_s, lambda: None,
                    epoch=batch.owner_epoch,
                    node_id=r.node_id, boot_id=r.boot_id, installs_fence=True,
                )
            else:
                ownership = external_ownership
                ownership.epoch = batch.owner_epoch
                ownership.node_id, ownership.boot_id = r.node_id, r.boot_id
                ownership.installs_fence = True
            store._forget_appended(sid)
            for cache in ("_send_indexes", "_change_versions"):
                getattr(store, cache, {}).pop(sid, None)

            ctx = _SessionExecution(
                r, sid, descriptor, batch, actor, batch.parent_session_id
            )
            ctx.task = asyncio.current_task()
            ctx.ownership = ownership
            self._active[sid] = ctx

            def _on_ownership_lost() -> None:
                ctx.stop.cancel("session ownership lost")
                if not ctx._lost_notified and r.hooks is not None:
                    ctx._lost_notified = True
                    payload = {
                        "execution": ctx.snapshot(),
                        "session_id": sid,
                        "owner_epoch": ctx.batch.owner_epoch,
                    }
                    ctx._lost_hook_task = asyncio.create_task(
                        notify(r.hooks, HookPoint.EXECUTION_LOST, payload, timeout_s=r.hook_timeout_s),
                        context=contextvars.Context(),
                    )

            ownership.on_lost = _on_ownership_lost
            ownership_token = _CURRENT.set(ownership)
            renewal = (
                None if self._uses_node_batch_renewal()
                else asyncio.create_task(self._renew(sid, batch, ownership, ctx), context=contextvars.Context())
            )

            def _mark_finalized() -> None:
                ctx.finalized = True
                ctx.finalizing = False

            with _bind_execution_context(ctx.snapshot, r.host_context_fields):
                await notify(r.hooks, HookPoint.EXECUTION_ACQUIRED, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
                if batch.reclaimed_cancelled and descriptor.kind != "workflow_driver":
                    ctx.finalizing = True
                    try:
                        final_batch = await self.inputs.final(
                            batch, status="cancelled", final_text="",
                            error="cancelled", error_code="cancelled",
                            _on_committed=_mark_finalized,
                        )
                    finally:
                        if not ctx.finalized:
                            ctx.finalizing = False
                    await self._notify_results(final_batch)
                    if final_batch.execution_state == "queued":
                        self._activate(sid)
                    batch = None
                    return
                try:
                    env = await r.factory.build(descriptor, ctx.snapshot())
                except Exception as exc:
                    code = exc.code if isinstance(exc, RuntimeRequestError) else "resource_unavailable"
                    self.last_error = {"code": code, "message": str(exc)[:500],
                                       "definition_ref": descriptor.definition_ref}
                    self.capability_missing.add(descriptor.definition_ref)
                    ctx.finalizing = True
                    try:
                        await self.inputs.release_claim(batch, _on_committed=_mark_finalized)
                    finally:
                        if not ctx.finalized:
                            ctx.finalizing = False
                    batch = None
                    return
                # The worker runs each session in a fresh context; honor the factory's
                # runtime_env (workspace/home/sandbox) so file-backed tools such as
                # workflow logs/export resolve paths inside this session's execution.
                if env.runtime_env is not None:
                    from power_loop.runtime.env import runtime_env_context
                    env_ctx = runtime_env_context(env.runtime_env)
                    env_ctx.__enter__()
                if ctx.options.tools is not None and env.tool_registry is not None:
                    env.tool_registry = env.tool_registry.subset(ctx.options.tools)
                if (
                    descriptor.kind == "agent"
                    and batch.parent_session_id is not None
                    and env.tool_registry is not None
                ):
                    from power_loop.runtime.execution_roles import role_tools

                    leaf_role = (
                        "workflow_node" if "workflow_node" in descriptor.config
                        else ("subagent" if "child_spec" in descriptor.config else None)
                    )
                    if leaf_role is not None:
                        env.tool_registry = env.tool_registry.subset(
                            role_tools(env.tool_registry.names(), leaf_role, registry=env.tool_registry)
                        )
                config = self._config(env, descriptor)
                if descriptor.kind == "workflow_driver":
                    from power_loop.workflow.runner import run_driver
                    final_batch = await run_driver(self, ctx, env)
                    _mark_finalized()
                    await self._notify_results(final_batch)
                    driver_requeue = final_batch.execution_state == "queued"
                    batch = None
                    return
                if descriptor.kind == "background":
                    managed = "managed_effect" in descriptor.config
                    final_input_batch, consumed = batch, None
                    if managed:
                        from power_loop.runtime.managed_effects import run_session

                        original = tuple(ref for ref in batch.input_refs
                                         if ref.message_id == batch.start_input_message_id)
                        if not original:
                            # Unrelated sends have their own failure, not a notification
                            # about the one frozen external action.
                            final_input_batch = replace(batch, result_route=None)
                        elif len(batch.input_refs) > 1:
                            consumed = original
                    else:
                        from power_loop.runtime.background_execution import run_session
                    try:
                        result = await run_session(self, ctx, env, config)
                    except StoreError:
                        if not managed:
                            raise
                        # The checkpoint may have committed. Let the existing lease
                        # recovery settle this input without publishing a false final.
                        log.exception("managed effect checkpoint unconfirmed for %s", sid)
                        return
                    ctx.finalizing = True
                    try:
                        final_batch = await self.inputs.final(
                            final_input_batch, consumed=consumed, status=result.status, final_text=result.final_text,
                            usage=dict(result.usage or {}), structured_output=result.structured_output,
                            error=result.error, error_code=result.error_code, result_seq=None,
                            _on_committed=_mark_finalized,
                        )
                    finally:
                        if not ctx.finalized:
                            ctx.finalizing = False
                    await self._notify_results(final_batch)
                    return
                bus = env.event_bus if env.event_bus is not None else AgentEventBus()

                def on_delta(event):
                    if event.session_id != sid:
                        return
                    if ctx.output.full():
                        while not ctx.output.empty():
                            ctx.output.get_nowait()
                        ctx.output.put_nowait({"kind": "gap"})
                    ctx.output.put_nowait({
                        "kind": "delta", **event.payload,
                        "session_id": sid, "send_index": batch.send_index,
                        "owner_epoch": batch.owner_epoch,
                    })

                bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)
                bus.subscribe(AgentEventType.STREAM_THINK_DELTA, on_delta)
                output_task = asyncio.create_task(self._pump_output(ctx), context=contextvars.Context())

                loop = _ExecutionLoop(
                    llm=env.llm, store=store, config=config, tool_registry=env.tool_registry,
                    hooks=env.hooks, event_bus=bus,
                    session_cache_size=0,
                    timer_backend=r.timer_backend,
                )
                loop._runtime_worker = self
                loop._runtime_worker_context = ctx
                ctx.loop = loop

                if env.command is not None:
                    from power_loop.runtime.host_commands import run_session
                    result = await run_session(loop, ctx, env.command)
                    consumed = None
                else:
                    consumed = ctx.consumed = list(batch.input_refs)
                    result = await loop._run_session_send(
                        sid, batch=batch, ownership=ownership, config=config,
                        tool_registry=env.tool_registry, hooks=env.hooks, event_bus=bus,
                        stop_event=ctx.stop, system_prompt=config.system_prompt,
                        max_rounds=ctx.options.max_rounds, response_format=ctx.options.response_format,
                        with_history=ctx.options.history,
                        boundary=self._boundary(batch, ownership, ctx, consumed),
                    )
                status, error, error_code = self._map_result(result, ctx)
                structured = None
                if ctx.options.response_format and status == "completed":
                    from power_loop.runtime.structured import StructuredOutputError, parse_structured
                    schema = ctx.options.response_format.get("json_schema", {}).get("schema")
                    try:
                        structured = parse_structured(result.final_text, schema=schema)
                    except StructuredOutputError as exc:
                        status, error, error_code = "failed", f"structured_output: {exc}", "structured_output"
                ctx.finalizing = True
                try:
                    final_batch = await self.inputs.final(
                        batch, consumed=consumed, status=status, final_text=result.final_text,
                        usage=dict(result.usage or {}), structured_output=structured,
                        error=error, error_code=error_code, result_seq=result.result_seq,
                        assistant_meta=getattr(result, "result_meta", None),
                        next_seq=getattr(result, "next_seq", None),
                        projection=getattr(result, "pending_projection", None) if status == "completed" else None,
                        _on_committed=_mark_finalized,
                    )
                finally:
                    if not ctx.finalized:
                        ctx.finalizing = False
                if env.command is None:
                    normal_final = final_batch
                await _flush_output()
                await self._notify_results(final_batch)
        except SessionLeaseLost:
            # Another owner took over; stop immediately and trigger EXECUTION_LOST hook.
            # Hint any local output waiter to re-check durable result state without DB writes.
            if ownership is not None:
                ownership.lose()
            if ctx is not None:
                ctx.stop.cancel("session ownership lost")
                if not ctx.output.full():
                    ctx.output.put_nowait({"kind": "gap", "session_id": sid})
        except CancellationRequested:
            if batch is not None:
                await self._final_cancelled(batch, ctx, before_notify=_flush_output)
        except RuntimeRequestError as exc:
            # Domain refusal before any model/tool call (e.g. projection_unavailable) keeps its
            # code and diagnostic detail instead of becoming a generic internal_error.
            log.warning("session send stopped for %s: %s", sid, exc)
            if batch is not None:
                await self._final_failed(batch, exc.code, str(exc), ctx=ctx, before_notify=_flush_output)
        except Exception:
            log.exception("session send failed for %s", sid)
            if batch is not None:
                await self._final_failed(batch, "internal_error", "session send raised", ctx=ctx, before_notify=_flush_output)
        finally:
            try:
                if renewal is not None:
                    renewal.cancel()
                    await asyncio.gather(renewal, return_exceptions=True)
                if external_ownership is not None:
                    await r.coordinator.release(external_ownership)
            finally:
                retiring = ownership if ownership is not None else external_ownership
                if retiring is not None:
                    retiring.released = True
                if ownership_token is not None:
                    _CURRENT.reset(ownership_token)
            # Execution permission ends before post-final business maintenance.
            # Detached tasks retain the same retired handle in their own Context.
            if env_ctx is not None:
                env_ctx.__exit__(None, None, None)
            await _flush_output()
            if ctx is not None:
                ctx.closing = True
                if ctx._lost_hook_task is not None:
                    await asyncio.gather(ctx._lost_hook_task, return_exceptions=True)
                elif ownership is not None and ownership.lost and not ctx._lost_notified:
                    ctx._lost_notified = True
                    await notify(
                        r.hooks,
                        HookPoint.EXECUTION_LOST,
                        {"execution": ctx.snapshot(), "session_id": sid, "owner_epoch": ctx.batch.owner_epoch},
                        timeout_s=r.hook_timeout_s,
                    )
                await notify(r.hooks, HookPoint.EXECUTION_FINISHED, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
                ctx.loop = None
            self._active.pop(sid, None)
            if env is not None:
                await env.aclose()
            # A normal final reports its pending snapshot; real new notifications
            # bridge the interval after that snapshot and before active.pop. After
            # pop the ordinary dispatcher waits on the same session lock for aclose.
            if driver_requeue:
                self._activate(sid)
            elif normal_final is not None and ctx is not None and ctx.consumed is not None:
                if normal_final.execution_state == "queued" or ctx.wake_pending:
                    self._activate(sid)
            elif batch is not None and await self._has_pending(sid):
                self._activate(sid)

    def _capable(self, descriptor):
        """Cheap, synchronous capability decision used before any owner is written.

        The claim passes this as ``can_execute``; it must not build resources, call the
        model/tools or await external work. A descriptor this node cannot execute leaves
        the input pending for a capable node instead of acquiring and then failing it.
        """
        if descriptor.kind not in {"agent", "background", "workflow_driver"}:
            self.capability_missing.add(descriptor.definition_ref)
            return False
        try:
            capable = self._can_build(descriptor)
        except Exception as exc:
            code = exc.code if isinstance(exc, RuntimeRequestError) else "resource_unavailable"
            self.last_error = {"code": code, "message": str(exc)[:500],
                               "definition_ref": descriptor.definition_ref}
            self.capability_missing.add(descriptor.definition_ref)
            return False
        if not capable:
            self.last_error = {"code": "capability_unavailable",
                               "message": "factory rejected descriptor",
                               "definition_ref": descriptor.definition_ref}
            self.capability_missing.add(descriptor.definition_ref)
        return capable

    @staticmethod
    def _needs_recovery(entry):
        """True for a running row whose committed lease expired (no 'recovering' state)."""
        state = entry["execution_state"]
        expired = (entry["owner_lease_until_ms"] is not None
                   and int(entry["owner_lease_until_ms"]) <= int(entry["db_now"]))
        return state == "running" and expired

    async def _recover_stale(self, target):
        """Reclaim a 0-assistant crashed run for safe rerun, or terminate an interrupted run.

        Runs only after the queued claim attempt returned None, so the healthy queued
        path never pays for this read. Returns a ``ClaimedBatch`` directly for 0-assistant
        rerun, ``True`` if recovery left the session queued, or ``False`` otherwise.
        """
        entry = await self.inputs.session_entry(target)
        if entry is None or not self._needs_recovery(entry):
            return False
        descriptor = RuntimeDescriptor(**json.loads(entry["descriptor_json"]))
        if descriptor.kind == "workflow_driver":
            self._drivers.add(target.session_id)
        if not self._capable(descriptor):
            return False
        r = self.runtime
        # 0-assistant crash rerun (e.g. crashed right after fused accept_and_claim):
        # reclaim with incremented owner_epoch and execute the same send_index cleanly.
        if descriptor.kind == "agent":
            reclaimed = await self.inputs.reclaim_running(
                target, node_id=r.node_id, boot_id=r.boot_id,
                ttl_s=r.session_lease_ttl_s, can_execute=self._capable,
            )
            if reclaimed is not None:
                if reclaimed.reclaimed_cancelled:
                    recovered = await self.inputs.final(
                        reclaimed, status="cancelled", final_text="",
                        error="cancelled", error_code="cancelled",
                    )
                    await self._notify_results(recovered)
                    return recovered.execution_state == "queued"
                if (
                    entry["lifecycle"] == "open"
                    and not reclaimed.reclaimed_has_assistant
                    and "workflow_node" not in descriptor.config
                ):
                    return reclaimed
                # Already produced an assistant message or non-open lifecycle: finalize as interrupted
                recovered = await self.inputs.final(
                    reclaimed, status="interrupted", final_text="",
                    error="execution interrupted", error_code="interrupted",
                )
                await self._notify_results(recovered)
                return recovered.execution_state == "queued"
        recovered = await r.coordinator.recover(
            self.inputs, target, node_id=r.node_id, boot_id=r.boot_id,
            ttl_s=r.session_lease_ttl_s)
        if recovered is None:
            return False
        await self._notify_results(recovered)
        return recovered.execution_state == "queued"

    def _can_build(self, descriptor):
        factory = self.runtime.factory
        can_execute = getattr(factory, "can_execute", None)
        if can_execute is not None:
            return can_execute(descriptor)
        supports = getattr(factory, "supports", None)
        if supports is not None:
            return supports(descriptor)
        builders = getattr(factory, "_builders", None)
        if builders is not None:
            return descriptor.definition_ref in builders
        return True

    def _config(self, env, descriptor):
        config = env.config
        overrides = {}
        option_values = descriptor.config.get("child_options", descriptor.config)
        for field_name in ("model", "temperature", "max_tokens", "max_rounds", "max_tokens_per_run",
                           "system_prompt", "response_format"):
            value = option_values.get(field_name)
            if value is not None:
                overrides[field_name] = value
        overrides["delegation_policy"] = "single_level"
        overrides["session_lease_ttl_s"] = self.runtime.session_lease_ttl_s
        return replace(config, **overrides)

    async def _renew(self, sid, batch, ownership, ctx):
        r = self.runtime
        ttl = r.session_lease_ttl_s
        try:
            while not ctx.closing and not ctx.finalized:
                await asyncio.sleep(ttl / 3)
                if ctx.closing or ctx.finalizing or ctx.finalized:
                    continue
                async with asyncio.timeout(max(0, ownership.expires_at - time.monotonic())):
                    renewed = await r.coordinator.renew(ownership, ttl_s=ttl)
                    if renewed and not getattr(r.coordinator, "installs_session_fence", False):
                        from power_loop.runtime.store_session_ownership import StoreSessionOwnershipCoordinator
                        renewed = await StoreSessionOwnershipCoordinator(r.store).renew(ownership, ttl_s=ttl)
                if not renewed and not ctx.finalizing and not ctx.finalized:
                    ownership.lose()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("session renewal failed for %s", sid)
            if not ctx.finalizing and not ctx.finalized:
                ownership.lose()

    def _boundary(self, batch, ownership, ctx, consumed):
        target = batch.target
        # Seed with wake_inbox_seq at claim time: round 0 and piggyback-verified tool
        # rounds pay 0 control SQL; a piggyback miss or urgent hint reads/drains all pending.
        last_wake_seq = [batch.wake_inbox_seq]
        round_zero_fresh = [True]

        async def _drain_all():
            claimed = await self.inputs.claim_follow_up(batch)
            if claimed is None:
                return []
            refs, seq, merged, _version = claimed
            holding = self.runtime.store._reads_for(target.session_id)
            if holding is not None and holding.wake_inbox_seq is not None:
                last_wake_seq[0] = holding.wake_inbox_seq
            consumed.extend(refs)
            from .follow_up import render_batch_content
            return [{
                "role": "user", "content": render_batch_content(merged),
                "_persisted_seq": seq,
                "meta": {"input_refs": [plain(ref) for ref in refs]},
            }]

        async def boundary(*, final=False):
            if ctx.stop.is_cancelled():
                return []
            holding = self.runtime.store._reads_for(target.session_id)
            # Consume any diagnostic boundary row already read by a piggyback miss (0 extra SQL).
            if holding is not None and holding.piggyback_boundary is not None:
                diag = holding.piggyback_boundary
                holding.piggyback_boundary = None
                holding.piggyback_verified = False
                round_zero_fresh[0] = False
                from power_loop.runtime.store.change_signals import COLUMNS, reader
                changes = reader(self.runtime.store, target.session_id)
                if changes is not None:
                    changes.versions = tuple(diag[column] for column in COLUMNS)
                holding.background_version = int(diag["background_version"])
                holding.runtime_version = int(diag["runtime_version"])
                if (
                    diag["lifecycle"] == "closing"
                    or int(diag["cancel_seq"]) > batch.cancel_seq
                ):
                    ctx.stop.cancel(diag.get("cancel_reason") or "cancelled")
                    return []
                if (
                    not final
                    and diag["lifecycle"] == "open"
                    and int(diag["wake_inbox_seq"]) > last_wake_seq[0]
                ):
                    result_msgs = await _drain_all()
                    if not result_msgs:
                        last_wake_seq[0] = int(diag["wake_inbox_seq"])
                        holding.wake_inbox_seq = last_wake_seq[0]
                    return result_msgs
                return []

            # 0-tool / final boundary: 0 SQL unless an urgent or cancel hint arrived!
            if final:
                if not ctx.urgent_hint and not ctx.cancel_hint:
                    return []
                ctx.urgent_hint = False
                ctx.cancel_hint = False
            elif round_zero_fresh[0] and not ctx.urgent_hint and not ctx.cancel_hint:
                round_zero_fresh[0] = False
                return []
            else:
                round_zero_fresh[0] = False
                if (
                    holding is not None
                    and holding.piggyback_verified
                    and not ctx.urgent_hint
                    and not ctx.cancel_hint
                ):
                    holding.piggyback_verified = False
                    return []
                ctx.urgent_hint = False
                ctx.cancel_hint = False

            row = await self.inputs.boundary(target)
            if row is None or (
                row["execution_state"] != "running"
                or int(row["owner_epoch"]) != batch.owner_epoch
            ):
                from power_loop.contracts.errors import SessionLeaseLost
                raise SessionLeaseLost(target.session_id)
            if (
                row["lifecycle"] == "closing"
                or int(row["cancel_seq"]) > batch.cancel_seq
            ):
                ctx.stop.cancel(row["cancel_reason"] or "cancelled")
            from power_loop.runtime.store.change_signals import COLUMNS, reader
            changes = reader(self.runtime.store, target.session_id)
            if changes is not None:
                changes.versions = tuple(row[column] for column in COLUMNS)
            if holding is not None:
                holding.background_version = int(row["background_version"])
                holding.runtime_version = int(row["runtime_version"])
            if final:
                # Final boundary only checks cancel/lease; final() transitions remaining pending to queued.
                return []
            if row["lifecycle"] != "open" or int(row["wake_inbox_seq"]) <= last_wake_seq[0]:
                return []
            result_msgs = await _drain_all()
            if not result_msgs:
                last_wake_seq[0] = int(row["wake_inbox_seq"])
                if holding is not None:
                    holding.wake_inbox_seq = last_wake_seq[0]
            return result_msgs

        return boundary

    @staticmethod
    def _map_result(result, ctx):
        status = result.status
        if ctx.stop.is_cancelled():
            return "cancelled", ctx.stop.reason or "cancelled", "cancelled"
        if status == "completed":
            return "completed", None, None
        if status == "cancelled":
            return "cancelled", result.final_text or "cancelled", "cancelled"
        if status == "pending_tools":
            return "interrupted", "pending tool calls left unresolved", "interrupted"
        # degraded / timed_out / hit_round_limit / context_checkpoint / capacity_blocked / failed
        return "failed", result.final_text or status, status

    async def _final_failed(self, batch, code, message, *, ctx=None, before_notify=None):
        from power_loop.contracts.errors import SessionLeaseLost

        def _mark():
            if ctx is not None:
                ctx.finalized = True
                ctx.finalizing = False

        if ctx is not None:
            ctx.finalizing = True
        try:
            final_batch = await self.inputs.final(
                batch, status="failed", final_text="", error=message, error_code=code,
                _on_committed=_mark,
            )
            if before_notify is not None:
                await before_notify()
            await self._notify_results(final_batch)
        except SessionLeaseLost:
            if ctx is not None:
                ctx.finalizing = False
                if ctx.ownership is not None:
                    ctx.ownership.lose()
        except Exception:
            log.exception("failed to terminalize send for %s", batch.target.session_id)
        finally:
            if ctx is not None and not ctx.finalized:
                ctx.finalizing = False

    async def _final_cancelled(self, batch, ctx, *, before_notify=None):
        from power_loop.contracts.errors import SessionLeaseLost

        def _mark():
            if ctx is not None:
                ctx.finalized = True
                ctx.finalizing = False

        if ctx is not None:
            ctx.finalizing = True
        try:
            final_batch = await self.inputs.final(
                batch, status="cancelled", final_text="",
                error=(ctx.stop.reason if ctx else None) or "cancelled", error_code="cancelled",
                _on_committed=_mark,
            )
            if before_notify is not None:
                await before_notify()
            await self._notify_results(final_batch)
        except SessionLeaseLost:
            if ctx is not None:
                ctx.finalizing = False
                if ctx.ownership is not None:
                    ctx.ownership.lose()
        except Exception:
            log.exception("failed to cancel send for %s", batch.target.session_id)
        finally:
            if ctx is not None and not ctx.finalized:
                ctx.finalizing = False

    async def _notify_results(self, final_batch):
        for result in final_batch.results:
            await self.runtime.client._notify(
                result.target, result.input_ref.message_id, result.input_ref, ["output"],
                {"kind": "completed", "session_id": result.target.session_id,
                 "send_index": result.send_index},
            )
        if final_batch.result_activation is not None:
            target = final_batch.result_activation
            ref = final_batch.result_ref
            # Skip redundant local activation / wake_transport when the resident Driver on this
            # Store already received the leaf result directly in _driver_waiters (0 extra DB reads).
            if target.session_id not in self.inputs._driver_waiters:
                await self.runtime.client._notify(
                    target, ref.message_id, ref, ["work", "activation"]
                )
                if self.runtime.wake_transport is not None:
                    try:
                        await self.runtime.wake_transport.publish_wake(
                            target.session_id, reason="river",
                        )
                    except Exception:
                        log.warning(
                            "wake_transport publish_wake failed for %s",
                            target.session_id, exc_info=True,
                        )

    async def _has_pending(self, sid):
        row = await self.runtime.store._db.fetchone(
            f"SELECT 1 AS present FROM {self.inputs.q} "
            "WHERE session_id=? AND status='pending' LIMIT 1", (sid,))
        return row is not None

    async def _pump_output(self, ctx):
        while True:
            payload = await ctx.output.get()
            if payload is None:
                return
            try:
                await self.runtime.transport.publish(TransportPacket(
                    TRANSPORT_PROTOCOL, self.runtime.namespace, ctx.input_ref.message_id,
                    TargetRef(ctx.store_id, ctx.session_id), "output", ctx.input_ref, payload))
            except Exception:
                log.warning("output notification lost for %s", ctx.input_ref.message_id, exc_info=True)

    # ── shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self, *, timeout_s=30.0, cancel_running=False, release_leases=True):
        async with self._lifecycle_lock:
            return await self._shutdown(
                timeout_s=timeout_s, cancel_running=cancel_running, release_leases=release_leases,
            )

    async def _shutdown(self, *, timeout_s, cancel_running, release_leases=True):
        if self._closed:
            return self.shutdown_result
        self._closing = True
        self.runtime.client._accepting = False
        self._pending.clear()
        self._pending_claims.clear()
        if self.runtime.timer_driver is not None:
            await self.runtime.timer_driver.stop()
        if self._started:
            self._dispatcher.cancel()
            if self._recovery:
                self._recovery.cancel()
            if cancel_running:
                for ctx in tuple(self._active.values()):
                    if ctx is not None:
                        ctx.stop.cancel("worker shutdown")
            pending = set()
            if self._tasks:
                _, pending = await asyncio.wait(self._tasks, timeout=timeout_s)
            interrupted = tuple(
                InputRef(ctx.store_id, ctx.session_id, ctx.input_ref.message_id)
                for ctx in self._active.values() if ctx is not None
            )
            for ctx in self._active.values():
                if ctx is not None and ctx.task in pending:
                    ctx.shutdown_interrupted = True
            for task in pending:
                task.cancel("worker shutdown timeout")
            await asyncio.gather(*pending, return_exceptions=True)
            if pending and release_leases:
                try:
                    await self.inputs.graceful_release(
                        node_id=self.runtime.node_id, boot_id=self.runtime.boot_id,
                    )
                except Exception:
                    log.warning("graceful_release failed during shutdown", exc_info=True)
            if self._wake_subscription is not None:
                await self._wake_subscription.aclose()
            if self._wake_listener is not None:
                self._wake_listener.cancel()
                await asyncio.gather(self._wake_listener, return_exceptions=True)
            for sub in self._subscriptions:
                await sub.aclose()
            await asyncio.gather(self._dispatcher, *self._listeners,
                                 *([self._recovery] if self._recovery else []), return_exceptions=True)
            self.shutdown_result = ShutdownResult(not pending, interrupted)
        if self._guard_monitor is not None and self._guard_monitor is not asyncio.current_task():
            self._guard_monitor.cancel()
            await asyncio.gather(self._guard_monitor, return_exceptions=True)
        if self.runtime.process_guard is not None:
            await self.runtime.process_guard.release()
        self._closed = True
        return self.shutdown_result
