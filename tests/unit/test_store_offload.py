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

        loop_obj = StatefulAgentLoop(
            llm=_FakeLLM(), store=store,
            config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
        )
        sid = await loop_obj.new_session()

        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await loop_obj.send("hi", session_id=sid)  # ≥1 thread-offloaded slow write
        stop = True
        await t

        # The backend offloads the blocking write to a thread, so the loop stays
        # responsive; without it the ticker would barely advance (~0 ticks).
        assert ticks >= 8, f"event loop appears blocked during the store write (ticks={ticks})"
    finally:
        await store.close()


async def test_blocking_active_history_read_does_not_freeze_event_loop() -> None:
    """SCALE-3: the per-send ``load_active_messages`` read is offloaded too, so a
    large (slow) active-history load doesn't stall other tasks."""
    store = await SessionStore.open(":memory:")
    try:
        db = store._db
        orig_b_fetchall = db._b_fetchall

        def slow_b_fetchall(sql: str, params: Any) -> Any:
            if "messages" in sql and "SELECT" in sql.upper():
                time.sleep(0.2)  # stand-in for an O(history) read on a fat session
            return orig_b_fetchall(sql, params)

        db._b_fetchall = slow_b_fetchall  # type: ignore[method-assign]

        loop_obj = StatefulAgentLoop(
            llm=_FakeLLM(), store=store,
            config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
        )
        sid = await loop_obj.new_session()

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
        await store.close()
