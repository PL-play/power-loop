"""Bounded timer scanning. A driver only waits for RuntimeClient acceptance."""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from dataclasses import dataclass, replace

from power_loop.agent.runtime_types import TargetRef, TrustedActor
from power_loop.contracts.event_payloads import TimerFiredPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hook_contexts import TimerFireCtx
from power_loop.contracts.hooks import HookDirective, HookPoint

from .timer_backend import TimerDriverStatus

DEFAULT_SCAN_INTERVAL_S = 2.0
DEFAULT_PRUNE_INTERVAL_S = 300.0
DEFAULT_PRUNE_OLDER_THAN_MS = 86_400_000


@dataclass
class _DeliveryAttempt:
    send_started: bool = False
    released: bool = False


_CURRENT_ATTEMPT: contextvars.ContextVar[_DeliveryAttempt | None] = contextvars.ContextVar(
    '_timer_delivery_attempt', default=None,
)


def format_timer_message(timer):
    return (f'<timer id={timer.timer_id}>\nYour scheduled wake-up has fired. '
            f'Your note to yourself was:\n{timer.note}\n</timer>')


class TimerRunner:
    def __init__(self, *, store, client, hooks=None, event_bus=None,
                 scan_interval_s=DEFAULT_SCAN_INTERVAL_S, batch_size=50, acceptance_timeout_s=10,
                 prune_interval_s=DEFAULT_PRUNE_INTERVAL_S,
                 prune_older_than_ms=DEFAULT_PRUNE_OLDER_THAN_MS):
        if (scan_interval_s <= 0 or not 1 <= batch_size <= 1000 or acceptance_timeout_s <= 0
                or prune_interval_s <= 0 or prune_older_than_ms < 0):
            raise ValueError('invalid timer driver limits')
        self.store, self.client, self.hooks, self.event_bus = store, client, hooks, event_bus
        self.scan_interval_s, self.batch_size = scan_interval_s, batch_size
        self.acceptance_timeout_s = acceptance_timeout_s
        self.prune_interval_s = prune_interval_s
        self.prune_older_than_ms = int(prune_older_than_ms)
        self._task = None
        self._last_recovery = 0.0
        self._last_prune: float | None = None
        self._status = TimerDriverStatus('store', False)

    def status(self):
        return self._status

    async def start(self):
        if self._task is None:
            self._status = replace(self._status, ready=True, last_error=None)
            self._task = asyncio.create_task(self._run(), name='power-loop-timers')

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._status = replace(self._status, ready=False)

    async def _run(self):
        while True:
            await asyncio.sleep(self.scan_interval_s)
            try:
                await self.scan_once()
            except Exception as exc:
                self._status = replace(self._status, last_error=str(exc)[:500], ready=False)

    async def _release_claim_once(self, timer, attempt: _DeliveryAttempt) -> None:
        if not attempt.released:
            attempt.released = True
            await self.store.transition_timer(
                timer.session_id, timer.timer_id,
                from_status='firing', to_status='armed',
            )

    async def scan_once(self):
        now_mono = time.monotonic()
        if now_mono - self._last_recovery >= 120:
            await self.store.recover_stale_firing_timers(
                older_than_ms=max(120000, int(self.acceptance_timeout_s*2000)), limit=self.batch_size)
            self._last_recovery = now_mono
        if self._last_prune is None or now_mono - self._last_prune >= self.prune_interval_s:
            await self.store.prune_timers(
                None, older_than_ms=self.prune_older_than_ms, limit=self.batch_size)
            self._last_prune = now_mono
        timers = await self.store.due_timers(limit=self.batch_size)
        self._status = replace(self._status, pending=len(timers),
            oldest_due_ms=timers[0].due_at if timers else None)
        count = 0
        for timer in timers:
            if not await self.store.transition_timer(timer.session_id, timer.timer_id,
                    from_status='armed', to_status='firing', expected_due_at=timer.due_at,
                    expected_fire_count=timer.fire_count):
                continue
            attempt = _DeliveryAttempt()
            tok = _CURRENT_ATTEMPT.set(attempt)
            try:
                async with asyncio.timeout(self.acceptance_timeout_s):
                    await self._deliver(timer)
                count += 1
            except asyncio.CancelledError as exc:
                if attempt.send_started:
                    # Never automatically replay an unknown accepted input. End this
                    # occurrence, allowing a recurring timer's next occurrence.
                    error = f'timer {timer.session_id}/{timer.timer_id} unknown: {type(exc).__name__}: {exc}'
                    self._status = replace(self._status, last_error=error[:500])
                    await self.store.finish_firing_timer(timer.session_id, timer.timer_id, unknown=True)
                    self._emit(timer, 'unknown')
                else:
                    # Interrupted before client.send was initiated: release claim at most once.
                    await self._release_claim_once(timer, attempt)
                raise
            except TimeoutError as exc:
                if attempt.send_started:
                    error = f'timer {timer.session_id}/{timer.timer_id} unknown: {type(exc).__name__}: {exc}'
                    self._status = replace(self._status, last_error=error[:500])
                    await self.store.finish_firing_timer(timer.session_id, timer.timer_id, unknown=True)
                    self._emit(timer, 'unknown')
                else:
                    # Timed out before client.send (e.g. TIMER_FIRE hook): keep armed for retry.
                    error = f'timer {timer.session_id}/{timer.timer_id} pre-send timeout: {type(exc).__name__}: {exc}'
                    self._status = replace(self._status, last_error=error[:500])
                    await self._release_claim_once(timer, attempt)
            except Exception as exc:
                # Never automatically replay an unknown accepted input. End this
                # occurrence, allowing a recurring timer's next occurrence.
                error = f'timer {timer.session_id}/{timer.timer_id} unknown: {type(exc).__name__}: {exc}'
                self._status = replace(self._status, last_error=error[:500])
                await self.store.finish_firing_timer(timer.session_id, timer.timer_id, unknown=True)
                self._emit(timer, 'unknown')
            finally:
                _CURRENT_ATTEMPT.reset(tok)
        self._status = replace(self._status, ready=True, last_scan_ms=int(time.time()*1000))
        return count

    async def _deliver(self, timer):
        attempt = _CURRENT_ATTEMPT.get()
        if attempt is None:
            attempt = _DeliveryAttempt()
        try:
            session = await self.store.get_session(timer.session_id)
            if session is None:
                await self.store.transition_timer(timer.session_id, timer.timer_id,
                    from_status='firing', to_status='cancelled')
                self._emit(timer, 'cancelled')
                return
            if (session.parent_session_id is not None or session.created_by_session_id is not None
                    or session.session_kind != 'agent' or session.lifecycle != 'open'):
                await self.store.transition_timer(timer.session_id, timer.timer_id,
                    from_status='firing', to_status='cancelled')
                self._emit(timer, 'forbidden')
                return
            ctx = TimerFireCtx(session_id=timer.session_id, timer_id=timer.timer_id,
                note=timer.note, due_at=timer.due_at, message=format_timer_message(timer))
            if self.hooks is not None:
                await self.hooks.run_typed_async(HookPoint.TIMER_FIRE, ctx)
            outcome = 'accepted'
            if ctx.directive == HookDirective.BREAK:
                await self.store.transition_timer(timer.session_id, timer.timer_id,
                    from_status='firing', to_status='cancelled')
                outcome = 'cancelled'
            elif ctx.directive == HookDirective.SKIP:
                await self.store.finish_firing_timer(timer.session_id, timer.timer_id)
                outcome = 'skipped'
            elif ctx.postpone_s and ctx.postpone_s > 0:
                await self.store.transition_timer(timer.session_id, timer.timer_id,
                    from_status='firing', to_status='armed', due_at=int(time.time()*1000+ctx.postpone_s*1000))
                outcome = 'postponed'
            else:
                attempt.send_started = True
                await self.client.send(TargetRef(self.store.store_id, timer.session_id), ctx.message,
                    message_id=f'timer-{timer.timer_id}-{timer.created_at}-{timer.fire_count}',
                    actor=TrustedActor(session.tenant_id,session.principal_id,None))
                await self.store.finish_firing_timer(timer.session_id, timer.timer_id)
            self._emit(timer, outcome)
        except asyncio.CancelledError:
            if not attempt.send_started:
                await self._release_claim_once(timer, attempt)
            raise

    def _emit(self, timer, outcome):
        if self.event_bus is not None:
            self.event_bus.publish(AgentEvent(type=AgentEventType.TIMER_FIRED,
                data=TimerFiredPayload(timer_id=timer.timer_id, note=timer.note,
                    due_at=timer.due_at, outcome=outcome), session_id=timer.session_id))
