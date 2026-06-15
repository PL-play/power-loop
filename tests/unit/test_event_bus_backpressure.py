"""OBS-3: opt-in background-thread dispatch for sync subscribers + overflow policy.

Default stays inline (a slow sync subscriber blocks publish — the documented contract).
``sync_dispatch="thread"`` offloads sync subscribers to a worker thread via a bounded
queue, so publish() returns immediately and a slow subscriber can't stall the loop;
overflow is handled by the configured drop policy.
"""

from __future__ import annotations

import threading
import time

import pytest

from power_loop import AgentEvent, AgentEventBus, AgentEventType

pytestmark = pytest.mark.unit


def _evt(i: int = 0) -> AgentEvent:
    return AgentEvent(type=AgentEventType.SYSTEM_LOG, payload={"i": i})


def test_inline_default_blocks_publish() -> None:
    """Baseline: an inline (default) sync subscriber runs during publish()."""
    bus = AgentEventBus()
    seen: list[int] = []
    bus.subscribe(AgentEventType.SYSTEM_LOG, lambda e: seen.append(e.payload["i"]))
    bus.publish(_evt(7))
    assert seen == [7]  # already ran, inline


def test_thread_mode_does_not_block_publish_on_slow_subscriber() -> None:
    bus = AgentEventBus(sync_dispatch="thread")
    try:
        done = threading.Event()

        def slow(_e: AgentEvent) -> None:
            time.sleep(0.3)
            done.set()

        bus.subscribe(AgentEventType.SYSTEM_LOG, slow)
        t0 = time.perf_counter()
        bus.publish(_evt())
        publish_elapsed = time.perf_counter() - t0

        assert publish_elapsed < 0.1, f"publish blocked on the slow subscriber ({publish_elapsed:.2f}s)"
        assert done.wait(timeout=2)  # it still runs, just off the publishing thread
    finally:
        bus.shutdown()


def test_thread_mode_delivers_all_events_in_order() -> None:
    bus = AgentEventBus(sync_dispatch="thread")
    seen: list[int] = []
    bus.subscribe(AgentEventType.SYSTEM_LOG, lambda e: seen.append(e.payload["i"]))
    for i in range(20):
        bus.publish(_evt(i))
    bus.shutdown()  # flushes the queue before returning
    assert seen == list(range(20))


def test_overflow_drop_newest_counts_drops() -> None:
    # tiny queue + a subscriber that blocks so the queue fills up
    bus = AgentEventBus(sync_dispatch="thread", queue_maxsize=2, on_overflow="drop_newest")
    release = threading.Event()
    started = threading.Event()

    def block_first(_e: AgentEvent) -> None:
        started.set()
        release.wait(timeout=5)

    bus.subscribe(AgentEventType.SYSTEM_LOG, block_first)
    try:
        bus.publish(_evt(0))          # taken by the worker, which now blocks
        assert started.wait(timeout=2)
        for i in range(1, 10):        # these pile into the maxsize=2 queue, then drop
            bus.publish(_evt(i))
        assert bus.dropped > 0, "expected some events to be dropped under overflow"
    finally:
        release.set()
        bus.shutdown()


def test_async_subscribers_still_scheduled_in_thread_mode() -> None:
    """Async handlers are NOT offloaded to the worker (it has no loop) — they still
    schedule on the running loop. Verified indirectly: a sync + async pair both run."""
    import asyncio

    async def _run() -> None:
        bus = AgentEventBus(sync_dispatch="thread", suppress_subscriber_errors=True)
        sync_seen: list[int] = []
        async_seen: list[int] = []

        async def ahandler(e: AgentEvent) -> None:
            async_seen.append(e.payload["i"])

        bus.subscribe(AgentEventType.SYSTEM_LOG, lambda e: sync_seen.append(e.payload["i"]))
        bus.subscribe(AgentEventType.SYSTEM_LOG, ahandler)
        bus.publish(_evt(1))
        await bus.drain(timeout=1.0)   # async task settles
        time.sleep(0.1)                # sync worker settles
        bus.shutdown()
        assert async_seen == [1]
        assert sync_seen == [1]

    asyncio.run(_run())
