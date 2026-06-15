"""H1.9 / C8: write-path store/sink I/O is offloaded so a slow SQLite write does
not freeze the event loop (other sessions/tasks keep running)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk

pytestmark = pytest.mark.unit


@dataclass
class _FakeLLM(LLMService):
    responses: list[LLMResponse] = field(default_factory=lambda: [LLMResponse(raw_text="ok")])
    _i: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text: Callable | None = None,
                       on_chunk_think: Callable | None = None, on_stream_end: Callable | None = None) -> LLMResponse:
        r = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


async def test_blocking_store_write_does_not_freeze_event_loop() -> None:
    """A blocking append_message (simulating a contended SQLite write) is offloaded,
    so a concurrent ticker keeps advancing instead of stalling."""
    store = SessionStore.open(":memory:")
    try:
        orig_append = store.append_message

        def slow_append(*a: Any, **kw: Any):
            time.sleep(0.2)  # blocking I/O stand-in (busy_timeout in real life)
            return orig_append(*a, **kw)

        store.append_message = slow_append  # type: ignore[method-assign]

        loop_obj = StatefulAgentLoop(
            llm=_FakeLLM(), store=store,
            config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
        )
        sid = loop_obj.new_session()

        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await loop_obj.send("hi", session_id=sid)  # ≥1 offloaded slow write
        stop = True
        await t

        # With the offload the loop stays responsive during the blocking write(s);
        # without it the ticker would barely advance (~0 ticks).
        assert ticks >= 8, f"event loop appears blocked during the store write (ticks={ticks})"
    finally:
        store.close()


async def test_blocking_active_history_read_does_not_freeze_event_loop() -> None:
    """SCALE-3: the per-send load_active_messages read is offloaded too, so a large
    (slow) active-history load doesn't stall other tasks."""
    store = SessionStore.open(":memory:")
    try:
        orig_load = store.load_active_messages

        def slow_load(*a: Any, **kw: Any):
            time.sleep(0.2)  # stand-in for an O(history) read on a fat session
            return orig_load(*a, **kw)

        store.load_active_messages = slow_load  # type: ignore[method-assign]

        loop_obj = StatefulAgentLoop(
            llm=_FakeLLM(), store=store,
            config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
        )
        sid = loop_obj.new_session()

        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await loop_obj.send("hi", session_id=sid)  # the slow read is offloaded
        stop = True
        await t

        assert ticks >= 8, f"event loop appears blocked during the history read (ticks={ticks})"
    finally:
        store.close()
