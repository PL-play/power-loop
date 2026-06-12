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
import logging
import time
from typing import TYPE_CHECKING

from power_loop.contracts.event_payloads import TimerFiredPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hook_contexts import TimerFireCtx
from power_loop.contracts.hooks import HookDirective, HookPoint
from power_loop.runtime.session_store import TimerRow

if TYPE_CHECKING:  # pragma: no cover
    from power_loop.agent.stateful_loop import StatefulAgentLoop

logger = logging.getLogger("power_loop.timers")

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
    ) -> None:
        self._loop = loop
        self._scan_interval = float(scan_interval_s)
        self._stale_firing_ms = int(stale_firing_s * 1000)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        recovered = self._loop.store.recover_stale_firing_timers(
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
        for timer in self._loop.store.due_timers():
            try:
                await self._fire(timer)
                fired += 1
            except Exception:
                logger.exception(
                    "timers: firing %s/%d failed; postponing 30s",
                    timer.session_id, timer.timer_id,
                )
                self._loop.store.transition_timer(
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
                self._loop.store.recover_stale_firing_timers(
                    older_than_ms=self._stale_firing_ms
                )
                await self.scan_once()
            except Exception:
                logger.exception("timers: scan failed (continuing)")

    async def _fire(self, timer: TimerRow) -> None:
        store = self._loop.store
        # Claim (CAS): lost claims mean another runner took it.
        if not store.transition_timer(
            timer.session_id, timer.timer_id, from_status="armed", to_status="firing"
        ):
            return
        if store.get_session(timer.session_id) is None:
            store.transition_timer(
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
            store.transition_timer(
                timer.session_id, timer.timer_id,
                from_status="firing", to_status="cancelled",
            )
            self._emit(timer, "cancelled")
            return
        if ctx.directive == HookDirective.SKIP:
            # Skip THIS firing only: a recurring timer still re-arms for the
            # next period; a one-shot is done.
            store.finish_firing_timer(timer.session_id, timer.timer_id)
            self._emit(timer, "skipped")
            return
        if ctx.postpone_s and ctx.postpone_s > 0:
            store.transition_timer(
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
        store.finish_firing_timer(timer.session_id, timer.timer_id)
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
