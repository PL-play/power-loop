"""Waiting that steering can cut short (design/124 §7.3).

A tool that WAITS — for a job, a timer, a sub-task — should not keep the user waiting behind it
once they have said something new. Such tools await through :func:`wait_or_steer`: it returns as
soon as the awaited thing finishes OR a steer-mode inbox item arrives for the session the tool is
running in, whichever is first, so the tool can return early ("new message arrived — stopped
waiting") and the loop delivers the steer at the next round.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any


def _steer_event(session_id: str | None) -> asyncio.Event | None:
    if not session_id:
        return None
    from power_loop.agent.stateful_loop import _SESSION_SYNC

    entry = _SESSION_SYNC.get(session_id)
    return entry.steer if entry is not None else None


async def wait_or_steer(
    awaitable: Awaitable[Any], *, session_id: str | None = None, cancel_on_steer: bool = True,
) -> tuple[Any, bool]:
    """``(result, False)`` if ``awaitable`` finished first (its exception propagates);
    ``(None, True)`` if steering arrived first. ``session_id`` defaults to the session of the
    running tool call. With ``cancel_on_steer`` (default) the awaitable is cancelled on steer —
    right for a wait/poll; pass ``False`` to leave real work running."""
    if session_id is None:
        from power_loop.core.agent_context import get_session_id

        session_id = get_session_id()
    work = asyncio.ensure_future(awaitable)
    ev = _steer_event(session_id)
    if ev is None:
        return await work, False
    if ev.is_set():
        if cancel_on_steer:
            work.cancel()
        return None, True
    waiter = asyncio.ensure_future(ev.wait())
    try:
        done, _ = await asyncio.wait({work, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # We are being cancelled (a stop, shutdown): take the work down WITH us and let
        # it finish its own cleanup first — a model call records its (estimated) usage
        # and closes its stream in its cancel path; leaving that to run after we are gone
        # reorders events and can lose the accounting entirely.
        waiter.cancel()
        work.cancel()
        await asyncio.wait({work}, timeout=5)
        raise
    if work in done:
        waiter.cancel()
        return work.result(), False
    if cancel_on_steer:
        work.cancel()
    return None, True


__all__ = ["wait_or_steer"]
