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
from asyncio import tasks as _asyncio_tasks
from collections import deque
from contextlib import asynccontextmanager
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


class _FiberTaskHandle(asyncio.Future):
    """Per-fiber Task-compatible handle exposed via ``asyncio.current_task()``.

    Being an ``asyncio.Future`` (not an ``asyncio.Task``), it never registers in
    ``asyncio.all_tasks()`` yet supports ``await`` / ``asyncio.gather`` /
    ``add_done_callback`` and gives each multiplexed fiber its own isolated
    ``cancel()`` / ``cancelling()`` / ``uncancel()`` state. Any ``asyncio.Timeout``
    (including external ``reschedule()`` from another task/context), ``AsyncExitStack``,
    ``asyncio.TaskGroup``, or waiter cleanup targets only the owning fiber without
    cancelling the shared worker ``asyncio.Task`` or sibling fibers.
    """

    def __init__(self, real_task: asyncio.Task, fiber: _SessionFiber, on_cancel) -> None:
        super().__init__(loop=real_task.get_loop())
        self._real_task = real_task
        self._fiber = fiber
        self._on_cancel = on_cancel
        self._cancelling = 0

    def cancel(self, msg=None) -> bool:
        if self.done():
            return False
        self._cancelling += 1
        exc = asyncio.CancelledError() if msg is None else asyncio.CancelledError(msg)
        self._on_cancel(self._fiber, exc)
        return True

    def cancelling(self) -> int:
        return self._cancelling

    def uncancel(self) -> int:
        if self._cancelling > 0:
            self._cancelling -= 1
        return self._cancelling

    def _finish(
        self,
        cancel_exc: asyncio.CancelledError | None = None,
        exc: BaseException | None = None,
    ) -> None:
        if self.done():
            return
        if cancel_exc is not None:
            super().cancel(*(cancel_exc.args if cancel_exc.args else ()))
        elif exc is not None:
            super().set_exception(exc)
            # Mark retrieved so unawaited handles never emit "Future exception was never retrieved"
            # while awaiting / calling result() / exception() still exposes ``exc``.
            super().exception()
        else:
            super().set_result(None)

    @property
    def _must_cancel(self) -> bool:
        return self._fiber.pending_exc is not None

    @property
    def _fut_waiter(self) -> asyncio.Future | None:
        return self._fiber.yielded

    @property
    def _num_cancels_requested(self) -> int:
        return self._cancelling

    @property
    def _cancel_message(self):
        exc = self._fiber.pending_exc
        return exc.args[0] if exc is not None and exc.args else None

    @property
    def _coro(self):
        return self._fiber.coro

    @property
    def _context(self):
        return self._fiber.ctx

    def get_coro(self):
        return self._fiber.coro

    def get_context(self):
        return self._fiber.ctx

    def get_name(self):
        return self._real_task.get_name()

    def set_name(self, value):
        return self._real_task.set_name(value)

    def get_stack(self, *, limit=None):
        return self._real_task.get_stack(limit=limit)

    def print_stack(self, *, limit=None, file=None):
        return self._real_task.print_stack(limit=limit, file=file)


class _SessionFiber:
    """One session coroutine multiplexed inside a RuntimeWorker._run_session Task."""

    __slots__ = (
        "sid",
        "coro",
        "ctx",
        "handle",
        "yielded",
        "is_primary",
        "done",
        "pending_exc",
        "cancel_exc",
        "exc",
    )

    def __init__(
        self,
        sid: str,
        coro,
        ctx: contextvars.Context,
        *,
        real_task: asyncio.Task | None,
        on_cancel,
        is_primary: bool,
    ) -> None:
        self.sid = sid
        self.coro = coro
        self.ctx = ctx
        self.handle: _FiberTaskHandle | None = (
            _FiberTaskHandle(real_task, self, on_cancel)
            if real_task is not None
            else None
        )
        self.yielded: asyncio.Future | None = None
        self.is_primary = is_primary
        self.done = False
        self.pending_exc: BaseException | None = None
        self.cancel_exc: asyncio.CancelledError | None = None
        self.exc: BaseException | None = None


class _TaskHandoffHost:
    """Per-worker-Task handoff channel for running awaited children in-place at the 1024 task cap."""

    __slots__ = ("queue", "accepting", "active_handoffs", "on_enqueue")

    def __init__(self, on_enqueue) -> None:
        self.queue: deque[tuple[str, object]] = deque()
        self.accepting: bool = True
        self.active_handoffs: int = 0
        self.on_enqueue = on_enqueue

    def __len__(self) -> int:
        return len(self.queue) + self.active_handoffs

    def can_accept(self) -> bool:
        return self.accepting and not self.queue and self.active_handoffs == 0

    def enqueue(self, sid: str, preclaimed_batch) -> None:
        self.queue.append((sid, preclaimed_batch))
        self.on_enqueue()


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
        self.slot_suspended = False
        self._slot_suspend_depth = 0
        self._lost_notified = False
        self._lost_hook_task = None
        # The ordinary LLM boundary extends this same run-local list. None means
        # claim/factory or a non-LLM path has not established that consumption fact.
        self.consumed: list[InputRef] | None = None
        self.wake_pending = False
        self.urgent_event = asyncio.Event()
        self.cancel_hint = False
        self.output: asyncio.Queue = asyncio.Queue(maxsize=256)
        option_values = descriptor.config.get("child_options", descriptor.config)
        self.options = RequestOptions(**{
            k: v for k, v in option_values.items() if k in RequestOptions.__dataclass_fields__
        })
        if batch.max_rounds_override is not None:
            self.options = replace(self.options, max_rounds=batch.max_rounds_override)

    @property
    def urgent_hint(self) -> bool:
        return self.urgent_event.is_set()

    @urgent_hint.setter
    def urgent_hint(self, value: bool) -> None:
        if value:
            self.urgent_event.set()
        else:
            self.urgent_event.clear()

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
        self._bindings = []
        self._pending = {}
        self._ready_awaited: dict[str, int] = {}
        self._awaited_children: dict[str, int] = {}
        self._awaited_handoffs: dict[str, asyncio.Task] = {}
        self._deferred_handoffs: dict[asyncio.Task, _TaskHandoffHost] = {}
        self._claimed_pending: dict[str, object] = {}
        self._active = {}
        self._slot_waiters: deque[tuple[_SessionExecution, asyncio.Future[bool]]] = deque()
        self._fusing: dict[str, int] = {}
        self._reserved_wakes: set[str] = set()
        self._pending_claims: dict[tuple[str, int], asyncio.Task | None] = {}
        self._closing_release_tasks: set[asyncio.Task] = set()
        self._drivers = set()
        self._tasks = set()
        self._wake = asyncio.Event()
        self._started, self._closing, self._closed = False, False, False
        self._lifecycle_lock = asyncio.Lock()
        self._overflow = False
        self._queued_cursor: str | None = None
        self._recover_cursor: tuple[str, int | None, str] | None = None
        self._guard_monitor = None
        self._guard_lost = False
        self.last_error = None
        # Local health: definitions this node cannot build. Never persisted as a task
        # registry; an incapable node simply leaves the session for a capable one.
        self.capability_missing = set()
        self.shutdown_result = ShutdownResult(True)

    _MAX_CAPABILITY_MISSING = 256

    def _note_capability_missing(self, ref) -> None:
        if ref in self.capability_missing:
            return
        if len(self.capability_missing) >= self._MAX_CAPABILITY_MISSING:
            self.capability_missing.pop()
        self.capability_missing.add(ref)

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
            if ctx is None or (ctx.descriptor.kind != "workflow_driver" and not ctx.slot_suspended)
        )

    def _suspended_llm_count(self) -> int:
        return sum(
            1 for ctx in self._active.values()
            if ctx is not None and ctx.slot_suspended
        )

    def available_llm_slots(self) -> int:
        if self._slot_waiters or self._ready_awaited or self._claimed_pending:
            return 0
        return max(0, self.runtime.capacity - self._active_llm_count())

    def _on_slot_released(self) -> None:
        """Grant freed execution slots to waiting parents (FIFO), then wake _dispatch."""
        while self._slot_waiters and self._active_llm_count() < self.runtime.capacity:
            ctx, fut = self._slot_waiters.popleft()
            if fut.done():
                continue
            if (
                self._active.get(ctx.session_id) is not ctx
                or ctx.closing
                or ctx.stop.is_cancelled()
                or ctx._slot_suspend_depth > 0
            ):
                fut.set_result(False)
                continue
            ctx.slot_suspended = False
            fut.set_result(True)
        if not self._closing and self._active_llm_count() < self.runtime.capacity:
            self._wake.set()

    async def _reacquire_slot(self, ctx: _SessionExecution, *, cancel_token=None) -> None:
        from power_loop.contracts.errors import CancellationRequested

        if not ctx.slot_suspended:
            return
        if ctx.closing or self._active.get(ctx.session_id) is not ctx:
            return
        if cancel_token is not None and cancel_token.is_cancelled():
            raise CancellationRequested(cancel_token.reason or "cancelled")
        if ctx.stop.is_cancelled():
            raise CancellationRequested(ctx.stop.reason or "cancelled")
        if not self._slot_waiters and self._active_llm_count() < self.runtime.capacity:
            ctx.slot_suspended = False
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        entry = (ctx, fut)
        self._slot_waiters.append(entry)
        stop_tasks = []
        if not ctx.stop.is_never:
            stop_tasks.append(asyncio.create_task(ctx.stop.wait()))
        if cancel_token is not None and cancel_token is not ctx.stop and not cancel_token.is_never:
            stop_tasks.append(asyncio.create_task(cancel_token.wait()))
        try:
            if stop_tasks:
                done, _ = await asyncio.wait(
                    {fut, *stop_tasks}, return_when=asyncio.FIRST_COMPLETED,
                )
                if fut not in done:
                    if not fut.done():
                        fut.cancel()
                    try:
                        self._slot_waiters.remove(entry)
                    except ValueError:
                        pass
                    reason = (
                        cancel_token.reason if cancel_token is not None and cancel_token.is_cancelled()
                        else ctx.stop.reason
                    )
                    raise CancellationRequested(reason or "cancelled")
                granted = fut.result()
            else:
                granted = await fut
        except BaseException:
            if not fut.done():
                fut.cancel()
            try:
                self._slot_waiters.remove(entry)
            except ValueError:
                pass
            if not fut.cancelled() and fut.done() and fut.exception() is None and fut.result():
                if ctx.stop.is_cancelled() or ctx.closing:
                    ctx.slot_suspended = True
                    self._on_slot_released()
            raise
        finally:
            for st in stop_tasks:
                if not st.done():
                    st.cancel()
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)

        if (cancel_token is not None and cancel_token.is_cancelled()) or ctx.stop.is_cancelled():
            if granted and not ctx.slot_suspended:
                ctx.slot_suspended = True
                self._on_slot_released()
            reason = (
                cancel_token.reason if cancel_token is not None and cancel_token.is_cancelled()
                else ctx.stop.reason
            )
            raise CancellationRequested(reason or "cancelled")

    @asynccontextmanager
    async def suspend_execution_slot(
        self, ctx: _SessionExecution | None, *, cancel_token=None, child_sids: tuple[str, ...] = (),
    ):
        """Yield this session's execution slot while waiting for child completion.

        Session ownership in ``_active`` (lease renewal, cancel delivery, single-owner
        fencing) stays intact. Awaited child sessions are queued with priority ahead of
        new top-level work so suspended parents drain instead of saturating ``_tasks``.
        At the 1024 task cap, ``_dispatch()`` hands off an awaited child directly to the
        parent's worker ``_run_session`` Task, decoupling child execution from the tool
        task so ``interrupt="abort"`` and arbitrary wrappers (e.g. ``asyncio.gather``)
        cancel the parent tool cleanly while an independent child continues on its Task.
        On normal exit of the outermost wait, the session re-acquires a worker slot
        before resuming model execution.
        """
        if (
            ctx is None
            or ctx.descriptor.kind == "workflow_driver"
            or self._active.get(ctx.session_id) is not ctx
        ):
            yield
            return
        tracked_children: list[str] = []
        registered_handoffs: list[str] = []
        parent_task = ctx.task
        for child_sid in child_sids:
            if not child_sid:
                continue
            tracked_children.append(child_sid)
            self._awaited_children[child_sid] = self._awaited_children.get(child_sid, 0) + 1
            if not self._closing and child_sid not in self._active:
                self._pending.pop(child_sid, None)
                self._ready_awaited.setdefault(child_sid, now_ms())
                if parent_task is not None and child_sid not in self._awaited_handoffs:
                    self._awaited_handoffs[child_sid] = parent_task
                    registered_handoffs.append(child_sid)
        ctx._slot_suspend_depth += 1
        if self._slot_waiters:
            kept = deque()
            for waiting_ctx, fut in self._slot_waiters:
                if waiting_ctx is ctx:
                    if not fut.done():
                        fut.set_result(False)
                else:
                    kept.append((waiting_ctx, fut))
            self._slot_waiters = kept
        if not ctx.slot_suspended:
            ctx.slot_suspended = True
            self._on_slot_released()
        elif tracked_children and not self._closing:
            self._wake.set()
        cancelled_out = False
        try:
            yield
        except asyncio.CancelledError:
            cancelled_out = True
            raise
        finally:
            for child_sid in registered_handoffs:
                if self._awaited_handoffs.get(child_sid) is parent_task:
                    self._awaited_handoffs.pop(child_sid, None)
            for child_sid in tracked_children:
                count = self._awaited_children.get(child_sid, 0) - 1
                if count <= 0:
                    self._awaited_children.pop(child_sid, None)
                    still_queued = self._ready_awaited.pop(child_sid, None) is not None
                    if still_queued and not self._closing and child_sid not in self._active:
                        self._activate(child_sid)
                else:
                    self._awaited_children[child_sid] = count
            ctx._slot_suspend_depth -= 1
            if (
                ctx._slot_suspend_depth == 0
                and ctx.slot_suspended
                and not cancelled_out
                and not ctx.closing
                and not ctx.stop.is_cancelled()
                and (cancel_token is None or not cancel_token.is_cancelled())
                and self._active.get(ctx.session_id) is ctx
            ):
                await self._reacquire_slot(ctx, cancel_token=cancel_token)

    def _reserved_fuse_slots(self) -> int:
        return sum(1 for sid in self._fusing if self._active.get(sid) is None)

    def _live_task_budget_used(self) -> int:
        return len(self._tasks) + self._reserved_fuse_slots()

    def can_fuse_locally(self, sid: str) -> bool:
        from .stateful_loop import _ExecutionLoop
        return (
            not self._closing
            and sid not in self._active
            and sid not in self._pending
            and sid not in self._claimed_pending
            and not _ExecutionLoop._session_busy(sid)
            and self._live_task_budget_used() < 1024
            and not self._slot_waiters
            and not self._ready_awaited
            and not self._claimed_pending
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

    def try_reserve_fuse(self, sid: str) -> bool:
        """Synchronously check local fuse eligibility and pre-reserve a capacity slot."""
        if not self.can_fuse_locally(sid):
            return False
        self._active[sid] = None
        self._begin_fuse(sid)
        return True

    def release_fuse_reservation(self, sid: str, *, dispatched: bool) -> None:
        """Release a fuse slot reservation; free _active[sid] and wake when not dispatched."""
        self._end_fuse(sid)
        missed_wake = sid in self._reserved_wakes
        self._reserved_wakes.discard(sid)
        if not dispatched:
            if self._active.get(sid) is None:
                self._active.pop(sid, None)
            if missed_wake and not self._closing:
                self._activate(sid)
            self._on_slot_released()

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

    def _schedule_closing_lease_release(self, batch) -> asyncio.Task | None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None
        task = asyncio.create_task(
            self._release_closing_claim(batch),
            context=contextvars.Context(),
        )
        self._closing_release_tasks.add(task)
        task.add_done_callback(self._closing_release_tasks.discard)
        return task

    async def _release_closing_claim(self, batch) -> None:
        from power_loop.contracts.errors import SessionLeaseLost
        try:
            await self.inputs.release_claim(batch)
        except SessionLeaseLost:
            pass
        except Exception:
            log.warning(
                "closing lease release failed for %s",
                batch.target.session_id, exc_info=True,
            )
            try:
                await self.inputs.release_orphan_lease(
                    batch.target.session_id,
                    node_id=self.runtime.node_id,
                    boot_id=self.runtime.boot_id,
                    owner_epoch=batch.owner_epoch,
                )
            except Exception:
                log.warning(
                    "closing orphan lease release failed for %s",
                    batch.target.session_id, exc_info=True,
                )

    async def drain_closing_releases(self) -> None:
        while self._closing_release_tasks:
            tasks = tuple(self._closing_release_tasks)
            # asyncio.wait does not cancel tasks if the caller itself is cancelled,
            # ensuring in-flight closing lease releases always finish in the DB.
            await asyncio.wait(tasks)

    def dispatch_claimed(self, batch) -> bool:
        """Schedule a locally fused accept_and_claim or pre-claimed batch without re-claiming.

        Returns False and releases the claimed lease when the worker is already closing.
        When the session does not already hold a local reservation and capacity or the
        task ceiling is currently full, the already-claimed batch is queued in
        ``_claimed_pending`` and started by ``_dispatch()`` as soon as a slot frees.
        """
        sid = batch.target.session_id
        self._pending_claims.pop((sid, int(batch.owner_epoch)), None)
        if self._closing or self._closed:
            if self._active.get(sid) is None:
                self._active.pop(sid, None)
                self._on_slot_released()
            self._schedule_closing_lease_release(batch)
            return False
        already_reserved = sid in self._active and self._active.get(sid) is None
        is_driver = batch.descriptor.kind == "workflow_driver"
        if not already_reserved and (
            self._live_task_budget_used() >= 1024
            or (
                not is_driver
                and (
                    self._slot_waiters
                    or self._ready_awaited
                    or self._active_llm_count() >= self.runtime.capacity
                )
            )
        ):
            self._pending.pop(sid, None)
            self._claimed_pending[sid] = batch
            if self._live_task_budget_used() < 1024 and (
                is_driver
                or (not self._slot_waiters and not self._ready_awaited and self._active_llm_count() < self.runtime.capacity)
            ):
                self._wake.set()
            return True
        self._active[sid] = None
        task = asyncio.create_task(
            self._run_session(sid, preclaimed_batch=batch),
            context=contextvars.Context(),
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        return True

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
                self._guard_lost = True
                await self.shutdown(timeout_s=0, release_leases=False)
                return

    def _activate(
        self, sid, input_ref=None, *, cancelled=False, cancel_reason=None,
        cancel_verified=False, cancel_by=None, is_driver=False,
    ):
        if self._closing:
            return
        if sid in self._active:
            ctx = self._active[sid]
            if ctx is None and not cancelled and sid in self._fusing:
                self._reserved_wakes.add(sid)
                return
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
        if sid in self._claimed_pending:
            if cancelled and cancel_verified:
                self._claimed_pending[sid] = replace(
                    self._claimed_pending[sid],
                    reclaimed_cancelled=True,
                    cancel_by=cancel_by or self._claimed_pending[sid].cancel_by,
                )
                self._wake.set()
            return
        if sid in self._ready_awaited:
            self._wake.set()
            return
        if len(self._pending) >= 1024 and sid not in self._pending:
            self._overflow = True
            return
        if is_driver:
            self._drivers.add(sid)
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
        owned_at_start: dict[str, int] = {}
        for sid, ctx in self._active.items():
            if ctx is not None:
                owned_at_start[sid] = int(ctx.batch.owner_epoch)
        for sid, b in self._claimed_pending.items():
            owned_at_start[sid] = max(owned_at_start.get(sid, 0), int(b.owner_epoch))
        for sid, epoch in self._pending_claims:
            owned_at_start[sid] = max(owned_at_start.get(sid, 0), int(epoch))
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
            local_epoch = int(ctx.batch.owner_epoch)
            if ctx.ownership is not None and ctx.ownership.expires_at <= started:
                ctx.ownership.lose()
                continue
            row = by_sid.get(sid)
            if row is not None:
                row_epoch = int(row["owner_epoch"])
                if row_epoch == local_epoch:
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
                elif row_epoch > local_epoch:
                    if ctx.ownership is not None:
                        ctx.ownership.lose()
                # row_epoch < local_epoch: newly claimed during scan; keep claim-time lease.
                continue
            if owned_at_start.get(sid) == local_epoch:
                if ctx.ownership is not None:
                    ctx.ownership.lose()
            # Otherwise newly claimed during scan and not covered by this round; keep claim-time lease.
        for sid, queued_batch in list(self._claimed_pending.items()):
            local_epoch = int(queued_batch.owner_epoch)
            row = by_sid.get(sid)
            if row is not None:
                row_epoch = int(row["owner_epoch"])
                if row_epoch == local_epoch:
                    updated = replace(queued_batch, claimed_at_mono=started)
                    if row["lifecycle"] == "closing" or int(row["cancel_seq"]) > queued_batch.cancel_seq:
                        cby = row.get("cancel_by") or ("removed" if row["lifecycle"] in ("closing", "closed") else "user")
                        updated = replace(updated, reclaimed_cancelled=True, cancel_by=cby)
                    self._claimed_pending[sid] = updated
                elif row_epoch > local_epoch:
                    self._claimed_pending.pop(sid, None)
                # row_epoch < local_epoch: newly claimed during scan; keep claim-time lease.
                continue
            if owned_at_start.get(sid) == local_epoch:
                self._claimed_pending.pop(sid, None)
            # Otherwise newly claimed during scan and not covered by this round; keep claim-time lease.
        # Slow path (P0-1): expire orphan running leases owned by this (node_id, boot_id)
        # that are absent from local active/fusing/pending-claim sets or have a mismatched owner_epoch.
        for row in rows:
            sid = str(row["session_id"])
            epoch = int(row["owner_epoch"])
            if sid in self._fusing:
                continue
            queued_batch = self._claimed_pending.get(sid)
            if queued_batch is not None and int(queued_batch.owner_epoch) >= epoch:
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
                if ctx.batch.owner_epoch > epoch:
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
        queued_entries = await self.inputs.discover_queued(
            limit=limit,
            after_session_id=self._queued_cursor,
            _with_kind=True,
        )
        queued = [sid for sid, _ in queued_entries]
        expired_rows = await self.inputs.discover_expired_leases(limit=limit)
        for sid, session_kind in queued_entries:
            self._activate(sid, is_driver=(session_kind == "workflow_driver"))
        if len(queued_entries) < limit:
            self._queued_cursor = None
            if len(self._pending) < 1024:
                self._overflow = False
        else:
            self._queued_cursor = queued_entries[-1][0]
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
                if (
                    self._live_task_budget_used() >= 1024
                    or self._slot_waiters
                    or self._ready_awaited
                    or self._claimed_pending
                    or self._active_llm_count() >= r.capacity
                ):
                    # Respect Worker capacity: queue for _dispatch() instead of bypassing concurrency!
                    self._activate(sid)
                    continue
                # Reserve the capacity slot BEFORE awaiting reclaim_running so neither _dispatch()
                # nor locally fused accept_and_claim can exceed runtime.capacity.
                self._active[sid] = None
                self._begin_fuse(sid)
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
                        dispatched = self.dispatch_claimed(batch)
                    else:
                        recovered = await self.inputs.final(
                            batch, status="interrupted", final_text="",
                            error="execution owner lost", error_code="interrupted",
                        )
                        await self._notify_results(recovered)
                        if recovered.execution_state == "queued":
                            self._activate(sid)
                finally:
                    self._end_fuse(sid)
                    if not dispatched:
                        self._active.pop(sid, None)
                        self._on_slot_released()
            elif row["session_kind"] == "workflow_driver":
                self._drivers.add(sid)
                self._begin_fuse(sid)
                try:
                    recovered = await self.inputs.recover(
                        TargetRef(r.store.store_id, sid),
                        node_id=r.node_id, boot_id=r.boot_id, ttl_s=r.session_lease_ttl_s,
                    )
                except BaseException:
                    if sid not in self._pending and sid not in self._active:
                        self._drivers.discard(sid)
                    raise
                finally:
                    self._end_fuse(sid)
                if recovered is None:
                    if sid not in self._pending and sid not in self._active:
                        self._drivers.discard(sid)
                    continue
                reclaimed.append(recovered)
                await self._notify_results(recovered)
                if recovered.execution_state == "queued":
                    self._activate(sid, is_driver=True)
                if sid not in self._pending and sid not in self._active:
                    self._drivers.discard(sid)
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
            if not self._active and not self._claimed_pending:
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
        page = await self.inputs.discover_unfinished(
            limit=self.runtime.claim_limit,
            after=self._recover_cursor,
            _with_kind=True,
        )
        for ref, session_kind, _cursor in page:
            self._activate(ref.session_id, is_driver=(session_kind == "workflow_driver"))
        if len(page) < self.runtime.claim_limit:
            self._recover_cursor = None
            if len(self._pending) < 1024:
                self._overflow = False
        else:
            self._recover_cursor = page[-1][2]

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
            while not self._closing:
                can_run_llm = (
                    not self._slot_waiters
                    and self._active_llm_count() < self.runtime.capacity
                )
                if self._live_task_budget_used() >= 1024:
                    # Task ceiling reached: hand off an awaited child directly to its
                    # already-allocated parent worker Task without adding to _tasks.
                    if not (can_run_llm and self._ready_awaited):
                        break
                    handoff_sid = None
                    handoff_host = None
                    for s in self._ready_awaited:
                        ptask = self._awaited_handoffs.get(s)
                        host = self._deferred_handoffs.get(ptask) if ptask is not None else None
                        if host is not None and host.can_accept():
                            handoff_sid = s
                            handoff_host = host
                            break
                    if handoff_sid is None or handoff_host is None:
                        break
                    self._ready_awaited.pop(handoff_sid, None)
                    self._pending.pop(handoff_sid, None)
                    self._awaited_handoffs.pop(handoff_sid, None)
                    batch = self._claimed_pending.pop(handoff_sid, None)
                    self._active[handoff_sid] = None
                    handoff_host.enqueue(handoff_sid, batch)
                    continue
                batch = None
                if can_run_llm and self._ready_awaited:
                    sid = next(iter(self._ready_awaited))
                    self._ready_awaited.pop(sid)
                    self._pending.pop(sid, None)
                    self._awaited_handoffs.pop(sid, None)
                    batch = self._claimed_pending.pop(sid, None)
                elif self._claimed_pending:
                    if can_run_llm:
                        sid, batch = next(iter(self._claimed_pending.items()))
                    else:
                        sid = next(
                            (s for s, b in self._claimed_pending.items() if b.descriptor.kind == "workflow_driver"),
                            None,
                        )
                        if sid is None:
                            sid = next((value for value in self._pending if value in self._drivers), None)
                            if sid is None:
                                break
                            self._pending.pop(sid)
                        else:
                            batch = self._claimed_pending[sid]
                    if batch is not None:
                        self._claimed_pending.pop(sid)
                        self._pending.pop(sid, None)
                elif self._pending:
                    if can_run_llm:
                        sid = next(iter(self._pending))
                    else:
                        sid = next((value for value in self._pending if value in self._drivers), None)
                        if sid is None:
                            break
                    self._pending.pop(sid)
                else:
                    break
                self._active[sid] = None
                task = asyncio.create_task(
                    self._run_session(sid, preclaimed_batch=batch),
                    context=contextvars.Context(),
                )
                self._tasks.add(task)
                task.add_done_callback(self._task_finished)
            if self._overflow and len(self._pending) < 924:
                self._wake.set()

    def _task_finished(self, task):
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("runtime execution failed", exc_info=task.exception())
        self._wake.set()

    async def _run_one_session(self, sid, *, preclaimed_batch=None):
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

    async def _run_session(self, sid, *, preclaimed_batch=None):
        """Run the primary session and any full-capacity handed-off children on this Task.

        Multiplexing handed-off children directly on the parent's worker ``_run_session``
        Task keeps total live session tasks <= 1024, preserves each child's single
        ``asyncio.Task`` binding (so child ``asyncio.timeout`` deadlines fire on time),
        and isolates child execution from parent tool cancellation (``interrupt="abort"``
        or ``asyncio.gather``).
        """
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        wake: list[asyncio.Future | None] = [None]

        def _signal_wake() -> None:
            w = wake[0]
            if w is not None and not w.done():
                w.set_result(None)

        def _on_fut_done(_f) -> None:
            _signal_wake()

        host = _TaskHandoffHost(_signal_wake)
        if task is not None:
            self._deferred_handoffs[task] = host

        def _cancel_fiber(fiber: _SessionFiber, exc: asyncio.CancelledError) -> None:
            if fiber.done:
                return
            fut = fiber.yielded
            if fut is not None and not fut.done():
                msg = exc.args[0] if exc.args else None
                if fut.cancel(msg):
                    _signal_wake()
                    return
            fiber.pending_exc = exc
            _signal_wake()

        fibers: list[_SessionFiber] = [
            _SessionFiber(
                sid,
                self._run_one_session(sid, preclaimed_batch=preclaimed_batch),
                contextvars.copy_context(),
                real_task=task,
                on_cancel=_cancel_fiber,
                is_primary=True,
            )
        ]

        def _step(fiber: _SessionFiber, exc: BaseException | None = None) -> None:
            fiber.yielded = None
            swapped = False
            if task is not None and fiber.handle is not None:
                _asyncio_tasks._leave_task(loop, task)
                _asyncio_tasks._enter_task(loop, fiber.handle)
                swapped = True
            try:
                if exc is not None:
                    yielded = fiber.ctx.run(fiber.coro.throw, exc)
                else:
                    yielded = fiber.ctx.run(fiber.coro.send, None)
            except StopIteration:
                fiber.done = True
                fiber.yielded = None
                return
            except asyncio.CancelledError as c_exc:
                fiber.done = True
                fiber.yielded = None
                fiber.cancel_exc = c_exc
                return
            except BaseException as err:
                fiber.done = True
                fiber.yielded = None
                fiber.exc = err
                if fiber.is_primary:
                    raise
                log.error("handed-off runtime execution failed for %s", fiber.sid, exc_info=err)
                return
            finally:
                if swapped:
                    _asyncio_tasks._leave_task(loop, fiber.handle)
                    _asyncio_tasks._enter_task(loop, task)
            if yielded is None:
                fut = loop.create_future()
                loop.call_soon(lambda _f=fut: None if _f.done() else _f.set_result(None))
                fiber.yielded = fut
                return
            if not isinstance(yielded, asyncio.Future):
                raise RuntimeError(f"unexpected yielded value from coroutine: {yielded!r}")
            yielded._asyncio_future_blocking = False
            fiber.yielded = yielded

        external_cancel: asyncio.CancelledError | None = None

        try:
            while True:
                while host.queue:
                    h_sid, h_batch = host.queue.popleft()
                    host.active_handoffs += 1
                    fibers.append(
                        _SessionFiber(
                            h_sid,
                            self._run_one_session(h_sid, preclaimed_batch=h_batch),
                            contextvars.Context(),
                            real_task=task,
                            on_cancel=_cancel_fiber,
                            is_primary=False,
                        )
                    )

                progressed = False
                for fiber in list(fibers):
                    if fiber.done:
                        continue
                    if fiber.pending_exc is not None:
                        exc_to_throw = fiber.pending_exc
                        fiber.pending_exc = None
                        fut = fiber.yielded
                        if fut is not None and not fut.done():
                            msg = exc_to_throw.args[0] if exc_to_throw.args else None
                            if fut.cancel(msg):
                                progressed = True
                                continue
                        _step(fiber, exc_to_throw)
                        progressed = True
                    elif fiber.yielded is None or fiber.yielded.done():
                        _step(fiber)
                        progressed = True

                still_alive: list[_SessionFiber] = []
                for fiber in fibers:
                    if not fiber.done:
                        still_alive.append(fiber)
                        continue
                    if fiber.handle is not None:
                        fiber.handle._finish(fiber.cancel_exc, fiber.exc)
                    if fiber.is_primary:
                        host.accepting = False
                    else:
                        host.active_handoffs -= 1
                        if not self._closing:
                            self._wake.set()
                fibers = still_alive

                if not fibers and not host.queue:
                    break
                if progressed or host.queue:
                    continue

                w = loop.create_future()
                wake[0] = w
                attached: list[asyncio.Future] = []
                for fiber in fibers:
                    fut = fiber.yielded
                    if fut is not None:
                        if fut.done():
                            _signal_wake()
                            break
                        fut.add_done_callback(_on_fut_done)
                        attached.append(fut)
                try:
                    await w
                except asyncio.CancelledError as exc:
                    # Per-fiber timeouts / TaskGroups target fiber.handle.cancel() directly,
                    # so a CancelledError on the real worker Task is always an external
                    # cancellation (e.g. worker.shutdown() or explicit task.cancel()).
                    external_cancel = exc
                    msg = exc.args[0] if exc.args else None
                    for fiber in fibers:
                        if fiber.handle is not None:
                            fiber.handle.cancel(msg)
                        else:
                            _cancel_fiber(fiber, exc)
                finally:
                    wake[0] = None
                    for fut in attached:
                        fut.remove_done_callback(_on_fut_done)

            if external_cancel is not None and (task is None or task.cancelling() > 0):
                raise external_cancel
        finally:
            host.accepting = False
            if task is not None and self._deferred_handoffs.get(task) is host:
                self._deferred_handoffs.pop(task, None)
            while host.queue:
                h_sid, h_batch = host.queue.popleft()
                if self._active.get(h_sid) is None:
                    self._active.pop(h_sid, None)
                    self._on_slot_released()
                if h_batch is not None:
                    self.dispatch_claimed(h_batch)
                elif not self._closing:
                    self._activate(h_sid)
            for fiber in fibers:
                if not fiber.done:
                    fiber.ctx.run(fiber.coro.close)
                if fiber.handle is not None:
                    fiber.handle._finish(fiber.cancel_exc, fiber.exc)

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
                claim_mono = (
                    batch.claimed_at_mono
                    if batch.claimed_at_mono is not None
                    else time.monotonic()
                )
                ownership = SessionOwnership(
                    (store.store_id, sid), r.node_id + "/" + r.boot_id,
                    claim_mono + r.session_lease_ttl_s, lambda: None,
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
            cur_task = asyncio.current_task()
            ctx.task = getattr(cur_task, "_real_task", cur_task)
            ctx.ownership = ownership
            self._active[sid] = ctx
            if descriptor.kind == "workflow_driver":
                self._on_slot_released()

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
                    self._note_capability_missing(descriptor.definition_ref)
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
                tools_filtered = ctx.options.tools is not None
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
                        tools_filtered = True
                effective_tools = (
                    (frozenset(env.tool_registry.names()) if env.tool_registry is not None else frozenset())
                    if tools_filtered
                    else None
                )
                config = self._config(env, descriptor)
                if descriptor.kind == "workflow_driver":
                    from power_loop.workflow.runner import run_driver
                    final_batch = await run_driver(self, ctx, env)
                    _mark_finalized()
                    await self._notify_results(final_batch)
                    driver_requeue = final_batch.execution_state == "queued"
                    if not driver_requeue:
                        self._drivers.discard(sid)
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
                    if final_batch.execution_state == "queued":
                        self._activate(sid)
                    batch = None
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
                    result = await run_session(loop, ctx, env.command, effective_tools=effective_tools)
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
                        effective_tools=effective_tools,
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
                        rounds=getattr(result, "rounds", None),
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
        except asyncio.CancelledError:
            if ctx is not None and ctx.shutdown_interrupted:
                batch = None
            raise
        except Exception:
            if ctx is not None and ctx.shutdown_interrupted:
                log.warning("session send interrupted by shutdown timeout for %s", sid)
                batch = None
            else:
                log.exception("session send failed for %s", sid)
                if batch is not None and not (ctx is not None and ctx.finalized):
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
            self._on_slot_released()
            if env is not None:
                await env.aclose()
            # A normal final reports its pending snapshot; real new notifications
            # bridge the interval after that snapshot and before active.pop. After
            # pop the ordinary dispatcher waits on the same session lock for aclose.
            if driver_requeue:
                self._activate(sid, is_driver=True)
            elif normal_final is not None and ctx is not None and ctx.consumed is not None:
                if normal_final.execution_state == "queued" or ctx.wake_pending:
                    self._activate(sid)
            elif batch is not None and await self._has_pending(sid):
                self._activate(sid)
            if sid not in self._pending and sid not in self._active:
                self._drivers.discard(sid)

    def _capable(self, descriptor):
        """Cheap, synchronous capability decision used before any owner is written.

        The claim passes this as ``can_execute``; it must not build resources, call the
        model/tools or await external work. A descriptor this node cannot execute leaves
        the input pending for a capable node instead of acquiring and then failing it.
        """
        if descriptor.kind not in {"agent", "background", "workflow_driver"}:
            self._note_capability_missing(descriptor.definition_ref)
            return False
        try:
            capable = self._can_build(descriptor)
        except Exception as exc:
            code = exc.code if isinstance(exc, RuntimeRequestError) else "resource_unavailable"
            self.last_error = {"code": code, "message": str(exc)[:500],
                               "definition_ref": descriptor.definition_ref}
            self._note_capability_missing(descriptor.definition_ref)
            return False
        if not capable:
            self.last_error = {"code": "capability_unavailable",
                               "message": "factory rejected descriptor",
                               "definition_ref": descriptor.definition_ref}
            self._note_capability_missing(descriptor.definition_ref)
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
            if not final and ctx.slot_suspended:
                await self._reacquire_slot(ctx)
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
                    ctx.cancel_hint = False
                    ctx.stop.cancel(diag.get("cancel_reason") or "cancelled")
                    return []
                if (
                    not final
                    and diag["lifecycle"] == "open"
                    and int(diag["wake_inbox_seq"]) > last_wake_seq[0]
                ):
                    ctx.urgent_hint = False
                    ctx.cancel_hint = False
                    result_msgs = await _drain_all()
                    if not result_msgs:
                        last_wake_seq[0] = int(diag["wake_inbox_seq"])
                        holding.wake_inbox_seq = last_wake_seq[0]
                    return result_msgs
                if not ctx.urgent_hint and not ctx.cancel_hint:
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
        self._ready_awaited.clear()
        self._awaited_handoffs.clear()
        for claimed_batch in tuple(self._claimed_pending.values()):
            self._schedule_closing_lease_release(claimed_batch)
        self._claimed_pending.clear()
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
            self._deferred_handoffs.clear()
            await self.drain_closing_releases()
            for sub in self._subscriptions:
                await sub.aclose()
            await asyncio.gather(self._dispatcher, *self._listeners,
                                 *([self._recovery] if self._recovery else []), return_exceptions=True)
            self.shutdown_result = ShutdownResult(not pending, interrupted)
        if self._guard_monitor is not None and self._guard_monitor is not asyncio.current_task():
            self._guard_monitor.cancel()
            await asyncio.gather(self._guard_monitor, return_exceptions=True)
        if self.runtime.process_guard is not None:
            try:
                await self.runtime.process_guard.release()
            except Exception:
                if not self._guard_lost:
                    raise
                log.warning("process_guard release failed after store_lock_lost", exc_info=True)
        self._closed = True
        return self.shutdown_result
