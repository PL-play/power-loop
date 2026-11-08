"""Bounded timer scanning. A driver only waits for RuntimeClient acceptance."""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import replace

from power_loop.agent.runtime_types import TargetRef, TrustedActor
from power_loop.contracts.event_payloads import TimerFiredPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hook_contexts import TimerFireCtx
from power_loop.contracts.hooks import HookDirective, HookPoint

from .timer_backend import TimerDriverStatus

DEFAULT_SCAN_INTERVAL_S = 2.0


def format_timer_message(timer):
    return (f'<timer id={timer.timer_id}>\nYour scheduled wake-up has fired. '
            f'Your note to yourself was:\n{timer.note}\n</timer>')


class TimerRunner:
    def __init__(self, *, store, client, hooks=None, event_bus=None,
                 scan_interval_s=DEFAULT_SCAN_INTERVAL_S, batch_size=50, acceptance_timeout_s=10):
        if scan_interval_s <= 0 or not 1 <= batch_size <= 1000 or acceptance_timeout_s <= 0:
            raise ValueError('invalid timer driver limits')
        self.store, self.client, self.hooks, self.event_bus = store, client, hooks, event_bus
        self.scan_interval_s, self.batch_size = scan_interval_s, batch_size
        self.acceptance_timeout_s = acceptance_timeout_s
        self._task = None
        self._last_recovery = 0.0
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

    async def scan_once(self):
        if time.monotonic() - self._last_recovery >= 120:
            await self.store.recover_stale_firing_timers(
                older_than_ms=max(120000, int(self.acceptance_timeout_s*2000)), limit=self.batch_size)
            self._last_recovery = time.monotonic()
        timers = await self.store.due_timers(limit=self.batch_size)
        self._status = replace(self._status, pending=len(timers),
            oldest_due_ms=timers[0].due_at if timers else None)
        count = 0
        for timer in timers:
            if not await self.store.transition_timer(timer.session_id, timer.timer_id,
                    from_status='armed', to_status='firing', expected_due_at=timer.due_at,
                    expected_fire_count=timer.fire_count):
                continue
            try:
                async with asyncio.timeout(self.acceptance_timeout_s):
                    await self._deliver(timer)
                count += 1
            except (Exception, asyncio.CancelledError) as exc:
                # Never automatically replay an unknown accepted input. End this
                # occurrence, allowing a recurring timer's next occurrence.
                error = f'timer {timer.session_id}/{timer.timer_id} unknown: {type(exc).__name__}: {exc}'
                self._status = replace(self._status, last_error=error[:500])
                await self.store.finish_firing_timer(timer.session_id, timer.timer_id, unknown=True)
                self._emit(timer, 'unknown')
                if isinstance(exc, asyncio.CancelledError):
                    raise
        self._status = replace(self._status, ready=True, last_scan_ms=int(time.time()*1000))
        return count

    async def _deliver(self, timer):
        session = await self.store.get_session(timer.session_id)
        if session is None:
            await self.store.transition_timer(timer.session_id, timer.timer_id,
                from_status='firing', to_status='cancelled')
            self._emit(timer, 'cancelled')
            return
        metadata = session.metadata or {}
        if session.parent_session_id is not None or metadata.get('execution_role', 'main') != 'main':
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
            await self.client.append(TargetRef(self.store.store_id, timer.session_id), ctx.message,
                message_id=f'timer-{timer.timer_id}-{timer.created_at}-{timer.fire_count}',
                actor=TrustedActor(metadata['tenant_id'], metadata['principal_id'], None))
            await self.store.finish_firing_timer(timer.session_id, timer.timer_id)
        self._emit(timer, outcome)

    def _emit(self, timer, outcome):
        if self.event_bus is not None:
            self.event_bus.publish(AgentEvent(type=AgentEventType.TIMER_FIRED,
                data=TimerFiredPayload(timer_id=timer.timer_id, note=timer.note,
                    due_at=timer.due_at, outcome=outcome), session_id=timer.session_id))
