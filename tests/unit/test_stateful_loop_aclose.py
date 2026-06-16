"""OPS-5: graceful async shutdown — aclose()/quiesce + async context manager.

The sync ``close()`` can race a send's background (``asyncio.to_thread``) store write and
close the connection out from under it. ``aclose()`` quiesces first: it blocks new sends,
waits for in-flight sends to release their per-session locks (by which point their
background writes have committed), drains pending async event-bus tasks, checkpoints the
WAL, and only then closes an owned store.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)

pytestmark = pytest.mark.unit


@dataclass
class _SlowLLM(LLMService):
    """Signals when a call begins, then sleeps — so a send can be held in-flight."""

    delay: float = 0.2
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        self.started.set()
        await asyncio.sleep(self.delay)
        return LLMResponse(raw_text="done")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _cfg() -> AgentLoopConfig:
    return AgentLoopConfig(max_rounds=1, compactor=None)


async def test_async_context_manager_closes_owned_store() -> None:
    loop = StatefulAgentLoop(llm=_SlowLLM(delay=0.0), db_path=":memory:", config=_cfg())
    async with loop:
        sid = await loop.new_session()
        r = await loop.send("hi", sid)
        assert r.status == "completed"
    assert loop._closing is True
    # owned store was closed → using its connection now raises
    with pytest.raises(sqlite3.ProgrammingError):
        loop.store._db._conn.execute("SELECT 1")


async def test_aclose_waits_for_inflight_send_and_rejects_new() -> None:
    """The race fix: aclose blocks new sends immediately but lets the in-flight one
    finish cleanly (no ProgrammingError), then returns."""
    store = await SessionStore.open(":memory:")
    llm = _SlowLLM(delay=0.2)
    loop = StatefulAgentLoop(llm=llm, store=store, config=_cfg())
    sid = await loop.new_session()

    send_task = asyncio.create_task(loop.send("hi", sid))
    await asyncio.wait_for(llm.started.wait(), timeout=1)  # send now holds the lock

    close_task = asyncio.create_task(loop.aclose())
    await asyncio.sleep(0.01)  # let aclose flip _closing and start waiting on the lock

    # a new send during closing is rejected at once (before the in-flight one ends)
    with pytest.raises(RuntimeError, match="closing"):
        await loop.send("again", sid)

    r = await asyncio.wait_for(send_task, timeout=2)  # in-flight send completes cleanly
    assert r.status == "completed"
    await asyncio.wait_for(close_task, timeout=2)      # aclose returns after it drained

    # store was NOT owned by the loop → aclose left it open and consistent
    assert await store.get_session(sid) is not None
    await store.close()


async def test_aclose_leaves_unowned_store_open() -> None:
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(llm=_SlowLLM(delay=0.0), store=store, config=_cfg())
    sid = await loop.new_session()
    await loop.send("hi", sid)
    await loop.aclose()
    # not owned → still usable
    assert await store.get_session(sid) is not None
    await store.close()


async def test_aclose_is_idempotent() -> None:
    loop = StatefulAgentLoop(llm=_SlowLLM(delay=0.0), db_path=":memory:", config=_cfg())
    await loop.aclose()
    await loop.aclose()  # second call must not raise
    assert loop._closing is True


async def test_send_and_follow_up_after_aclose_raise() -> None:
    loop = StatefulAgentLoop(llm=_SlowLLM(delay=0.0), db_path=":memory:", config=_cfg())
    sid = await loop.new_session()
    await loop.aclose()
    with pytest.raises(RuntimeError, match="closing"):
        await loop.send("x", sid)
    with pytest.raises(RuntimeError, match="closing"):
        await loop.follow_up("x", sid)


async def test_event_bus_drain_awaits_pending_async_subscribers() -> None:
    """drain() returns only after in-flight async subscribers (scheduled from sync
    publish) have completed — the primitive aclose relies on."""
    bus = AgentEventBus(suppress_subscriber_errors=True)
    done: list[AgentEvent] = []

    async def slow(ev: AgentEvent) -> None:
        await asyncio.sleep(0.05)
        done.append(ev)

    bus.subscribe(AgentEventType.SESSION_STARTED, slow)
    bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED))
    assert done == []  # not yet — the task is still pending right after publish

    await bus.drain(timeout=1.0)
    assert len(done) == 1
    assert bus._async_tasks == set()
