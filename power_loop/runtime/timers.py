"""Durable per-session timers: "wake this session at T with this note".

Design (one store file = one process, like everything else here):

* **The ``timers`` table is the source of truth.** A timer is data, not an
  in-memory task — it survives restarts. The in-process scan loop is only an
  accelerator over the table; resolution = ``scan_interval``.
* **Firing = a normal send.** Delivery goes through ``loop.follow_up``: an
  idle session gets a regular ``send``; a session mid-run gets the note
  injected at the next round boundary (``FollowUpQueued``). There is exactly
  one path into a conversation.
* **Recurrence is declared at creation** (``interval_s`` on the row /
  ``every_seconds`` on the tool): NULL = one-shot (``firing -> fired``);
  set = the firing re-arms at fire-time + interval (fixed-delay) until
  cancelled. Cancelling is the only way a recurring timer ends.
* **At-least-once.** A claim is a compare-and-set ``armed -> firing``; rows
  stuck in ``firing`` (process died mid-fire) are re-armed on the next
  :meth:`TimerRunner.start` / periodic recovery sweep and may deliver twice.
* **The TIMER_FIRE hook is the orchestrator's veto point.** Before delivery
  every firing runs :pyattr:`HookPoint.TIMER_FIRE` with a
  :class:`TimerFireCtx`: CONTINUE delivers, SKIP drops this firing, BREAK
  cancels the timer, ``postpone_s`` re-arms it later. No hook registered =
  deliver. Use it to dedupe after re-fires, hold off a busy system, audit.

Timers are created by the agent itself (``schedule_wakeup`` /
``cancel_wakeup`` / ``list_wakeups`` default tools) or externally via
``StatefulAgentLoop.schedule_timer`` — both write the same rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING

from power_loop.contracts.event_payloads import TimerFiredPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hook_contexts import TimerFireCtx
from power_loop.contracts.hooks import HookDirective, HookPoint
from power_loop.runtime.store.types import TimerRow

if TYPE_CHECKING:  # pragma: no cover
    from power_loop.agent.stateful_loop import StatefulAgentLoop

logger = logging.getLogger(__name__)

DEFAULT_SCAN_INTERVAL_S = 2.0
DEFAULT_STALE_FIRING_S = 120.0


def format_timer_message(timer: TimerRow) -> str:
    """The user-role text a fired timer injects into its session."""
    return (
        f"<timer id={timer.timer_id}>\n"
        f"Your scheduled wake-up has fired. Your note to yourself was:\n"
        f"{timer.note}\n"
        f"</timer>"
    )


class TimerRunner:
    """Scans the store for due timers and fires them into their sessions.

    One runner per :class:`StatefulAgentLoop` (per store/process)::

        runner = TimerRunner(loop)
        await runner.start()      # re-arms stale 'firing' rows, begins scanning
        ...
        await runner.stop()

    Not started automatically: timers only fire while a runner is running —
    callers who poll ``loop.store.due_timers()`` from their own scheduler
    don't need it at all.
    """

    def __init__(
        self,
        loop: StatefulAgentLoop,
        *,
        scan_interval_s: float = DEFAULT_SCAN_INTERVAL_S,
        stale_firing_s: float = DEFAULT_STALE_FIRING_S,
        heartbeat_interval_s: float | None = None,
    ) -> None:
        self._loop = loop
        self._scan_interval = float(scan_interval_s)
        self._stale_firing_ms = int(stale_firing_s * 1000)
        # While a claimed 'firing' row is being delivered we re-stamp it on this
        # cadence so a slow-but-live delivery stays comfortably fresh before the
        # stale cutoff. A quarter of the stale window by default (floored at 1s to
        # avoid hammering sqlite); overridable for tests / unusual configs.
        self._heartbeat_interval_s = (
            float(heartbeat_interval_s) if heartbeat_interval_s is not None
            else max(1.0, stale_firing_s / 4.0)
        )
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        store = await self._loop._ensure_store()
        recovered = await store.recover_stale_firing_timers(
            older_than_ms=self._stale_firing_ms
        )
        if recovered:
            logger.warning("timers: re-armed %d stale 'firing' row(s)", recovered)
        self._stop.clear()
        if self._task is None:
            self._task = asyncio.create_task(self._scan_loop(), name="power-loop-timers")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def scan_once(self) -> int:
        """Fire everything currently due. Returns the number fired (any
        outcome). Exposed for tests and external schedulers."""
        fired = 0
        store = await self._loop._ensure_store()
        for timer in await store.due_timers():
            try:
                await self._fire(timer)
                fired += 1
            except Exception:
                logger.exception(
                    "timers: firing %s/%d failed; postponing 30s",
                    timer.session_id, timer.timer_id,
                )
                await store.transition_timer(
                    timer.session_id, timer.timer_id,
                    from_status="firing", to_status="armed",
                    due_at=int(time.time() * 1000) + 30_000,
                )
                self._emit(timer, "error")
        return fired

    # ── internals ─────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._scan_interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                # Periodic recovery so a crashed *other* runner's rows are
                # also picked up eventually, not only at start().
                store = await self._loop._ensure_store()
                await store.recover_stale_firing_timers(
                    older_than_ms=self._stale_firing_ms
                )
                await self.scan_once()
            except Exception:
                logger.exception("timers: scan failed (continuing)")

    async def _fire(self, timer: TimerRow) -> None:
        store = await self._loop._ensure_store()
        # Claim (CAS): lost claims mean another runner took it.
        if not await store.transition_timer(
            timer.session_id, timer.timer_id, from_status="armed", to_status="firing"
        ):
            return
        # Heartbeat the 'firing' row while the (possibly slow) hook + delivery run,
        # so a live delivery longer than stale_firing_s is NOT reclaimed as stale
        # and double-fired. Cancelled in finally once the row leaves 'firing'.
        hb = asyncio.create_task(self._heartbeat_firing(timer), name="power-loop-timer-hb")
        try:
            await self._deliver_claimed(timer)
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

    async def _heartbeat_firing(self, timer: TimerRow) -> None:
        """Re-stamp the claimed 'firing' row every quarter-of-the-stale-window so
        a slow-but-live delivery stays fresh and is never re-armed by the recovery
        sweep. Stops itself if the claim is lost (row no longer 'firing')."""
        interval = self._heartbeat_interval_s
        store = await self._loop._ensure_store()
        while True:
            await asyncio.sleep(interval)
            try:
                if not await store.heartbeat_firing_timer(
                    timer.session_id, timer.timer_id
                ):
                    return  # row left 'firing' — nothing left to keep alive
            except Exception:
                logger.exception(
                    "timers: heartbeat failed for %s/%d (continuing)",
                    timer.session_id, timer.timer_id,
                )

    async def _deliver_claimed(self, timer: TimerRow) -> None:
        store = await self._loop._ensure_store()
        if await store.get_session(timer.session_id) is None:
            await store.transition_timer(
                timer.session_id, timer.timer_id,
                from_status="firing", to_status="cancelled",
            )
            self._emit(timer, "cancelled")
            return

        # ── Orchestrator veto point ──
        ctx = TimerFireCtx(
            session_id=timer.session_id,
            timer_id=timer.timer_id,
            note=timer.note,
            due_at=timer.due_at,
            message=format_timer_message(timer),
        )
        await self._loop.hooks.run_typed_async(HookPoint.TIMER_FIRE, ctx)

        if ctx.directive == HookDirective.BREAK:
            await store.transition_timer(
                timer.session_id, timer.timer_id,
                from_status="firing", to_status="cancelled",
            )
            self._emit(timer, "cancelled")
            return
        if ctx.directive == HookDirective.SKIP:
            # Skip THIS firing only: a recurring timer still re-arms for the
            # next period; a one-shot is done.
            await store.finish_firing_timer(timer.session_id, timer.timer_id)
            self._emit(timer, "skipped")
            return
        if ctx.postpone_s and ctx.postpone_s > 0:
            await store.transition_timer(
                timer.session_id, timer.timer_id,
                from_status="firing", to_status="armed",
                due_at=int(time.time() * 1000) + int(ctx.postpone_s * 1000),
            )
            self._emit(timer, "postponed")
            return

        # ── Deliver: follow_up = queued when mid-run, plain send when idle ──
        from power_loop.agent.follow_up import FollowUpQueued

        result = await self._loop.follow_up(ctx.message, timer.session_id)
        outcome = "queued" if isinstance(result, FollowUpQueued) else "delivered"
        # One-shot -> fired; recurring -> re-armed at fire-time + interval
        # (fixed-delay: periods missed while the process was down collapse).
        await store.finish_firing_timer(timer.session_id, timer.timer_id)
        self._emit(timer, outcome)
        logger.info(
            "timers: %s/%d %s (note=%r)",
            timer.session_id, timer.timer_id, outcome, timer.note[:80],
        )

    def _emit(self, timer: TimerRow, outcome: str) -> None:
        try:
            self._loop.event_bus.publish(AgentEvent(
                type=AgentEventType.TIMER_FIRED,
                data=TimerFiredPayload(
                    timer_id=timer.timer_id, note=timer.note,
                    due_at=timer.due_at, outcome=outcome,
                ),
                session_id=timer.session_id,
            ))
        except Exception:
            logger.exception("timers: event publish failed")
