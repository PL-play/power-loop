"""H1.9 / C8 / SCALE-3: the async store's SQLite backend runs every blocking
statement in a worker thread (``asyncio.to_thread``), so a slow/contended write
or a fat-session read does not freeze the event loop — other tasks keep running.

(Previously the StatefulAgentLoop offloaded the sink/read calls explicitly; with the
async store the backend owns the offload, so we inject the block at the backend's
blocking primitive and assert the loop stays responsive.)"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, SessionStore
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


async def test_blocking_store_write_does_not_freeze_event_loop(runtime_agent) -> None:
    """A blocking message INSERT (simulating a contended SQLite write) runs in the
    backend's worker thread, so a concurrent ticker keeps advancing."""
    store = await SessionStore.open(":memory:")
    try:
        db = store._db
        orig_b_exec = db._b_exec

        def slow_b_exec(sql: str, params: Any) -> int:
            if sql.lstrip().upper().startswith("INSERT INTO") and "messages" in sql:
                time.sleep(0.2)  # blocking I/O stand-in (busy_timeout in real life)
            return orig_b_exec(sql, params)

        db._b_exec = slow_b_exec  # type: ignore[method-assign]

        loop_obj = await runtime_agent(model=_FakeLLM(), store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=1, context_budget_tokens=10**9))

        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await loop_obj.send("hi")  # ≥1 thread-offloaded slow write
        stop = True
        await t

        # The backend offloads the blocking write to a thread, so the loop stays
        # responsive; without it the ticker would barely advance (~0 ticks).
        assert ticks >= 8, f"event loop appears blocked during the store write (ticks={ticks})"
    finally:
        await loop_obj.aclose()
        await store.close()


async def test_blocking_active_history_read_does_not_freeze_event_loop(runtime_agent) -> None:
    """SCALE-3: the per-send history read is offloaded too, so a large (slow)
    projection/history load doesn't stall other tasks."""
    store = await SessionStore.open(":memory:")
    try:
        loop_obj = await runtime_agent(model=_FakeLLM(), store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=1, context_budget_tokens=10**9))
        # Seed a prior send so the next send actually executes a history SELECT (cold send 1 skips history reads).
        await loop_obj.send("seed")

        db = store._db
        orig_b_fetchall = db._b_fetchall
        slow_hits = 0

        def slow_b_fetchall(sql: str, params: Any) -> Any:
            nonlocal slow_hits
            if "messages" in sql and "SELECT" in sql.upper():
                slow_hits += 1
                time.sleep(0.2)  # stand-in for a slow history/projection read on a fat session
            return orig_b_fetchall(sql, params)

        db._b_fetchall = slow_b_fetchall  # type: ignore[method-assign]

        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await loop_obj.send("hi")  # the slow read is offloaded
        stop = True
        await t

        assert slow_hits >= 1, "expected at least one offloaded history SELECT"
        assert ticks >= 8, f"event loop appears blocked during the history read (ticks={ticks})"
    finally:
        await loop_obj.aclose()
        await store.close()
