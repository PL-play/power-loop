"""H1.7 / C6: sync publish() schedules async subscribers safely.

The task is retained (no GC mid-flight) and its error is surfaced per
`suppress_subscriber_errors` instead of being a silent "Task exception was never
retrieved" GC warning.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from power_loop import AgentEvent, AgentEventBus, AgentEventType

pytestmark = pytest.mark.unit


async def _drain(bus: AgentEventBus) -> None:
    for _ in range(20):
        if not bus._async_tasks:
            return
        await asyncio.sleep(0.01)


async def test_async_subscriber_task_is_retained_then_drained() -> None:
    """The scheduled task is held in _async_tasks (not GC-able) until it finishes,
    then removed by the done-callback."""
    bus = AgentEventBus(suppress_subscriber_errors=True)
    ran = asyncio.Event()

    async def handler(_e: AgentEvent) -> None:
        ran.set()

    bus.subscribe(AgentEventType.SESSION_STARTED, handler)
    bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED))

    assert len(bus._async_tasks) == 1  # retained immediately after publish
    await asyncio.wait_for(ran.wait(), timeout=1)
    await _drain(bus)
    assert bus._async_tasks == set()  # done-callback discarded it


async def test_async_subscriber_error_suppressed_does_not_escape() -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)

    async def boom(_e: AgentEvent) -> None:
        raise RuntimeError("async sub boom")

    bus.subscribe(AgentEventType.SESSION_STARTED, boom)
    bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED))  # must not raise
    await _drain(bus)
    assert bus._async_tasks == set()  # cleaned up, error swallowed per the flag


async def test_async_subscriber_error_is_logged_when_not_suppressed(caplog) -> None:
    """suppress=False: the error is surfaced loudly at ERROR (not a silent GC
    warning), and the exception is retrieved so it never becomes 'never retrieved'."""
    bus = AgentEventBus(suppress_subscriber_errors=False)

    async def boom(_e: AgentEvent) -> None:
        raise RuntimeError("async sub boom")

    bus.subscribe(AgentEventType.SESSION_STARTED, boom)
    with caplog.at_level(logging.ERROR, logger="power_loop.core.events"):
        bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED))
        await _drain(bus)

    assert any("async subscriber raised in sync publish()" in r.message for r in caplog.records)
    assert bus._async_tasks == set()
