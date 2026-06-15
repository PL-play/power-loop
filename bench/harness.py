"""Scenario runners + a deterministic FakeLLM for the SCALE-1 harness.

Three scenarios, all driving one ``StatefulAgentLoop`` over one on-disk ``SessionStore``:

* ``run_fanout``      — N concurrent sessions each doing one send (contention on the
                        single store RLock); reports sessions/sec + per-send latency.
* ``run_big_history`` — one session pre-seeded with a large active history; reports the
                        per-send cost (dominated by load_active_messages + the per-round
                        token-estimate scan), exposing O(history) growth.
* ``run_throughput``  — sustained sequential sends on one session; reports sends/sec.

Timing uses ``time.perf_counter``; latencies are reported as p50/p99 in milliseconds.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)


@dataclass
class FakeLLM(LLMService):
    """Returns a fixed answer after an optional artificial delay. No network, no tokens —
    so the harness measures store/loop overhead, and ``latency_s`` lets you model a
    provider's think-time to see how much the loop adds on top."""

    latency_s: float = 0.0
    reply: str = "ok"

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        return LLMResponse(raw_text=self.reply)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0,1]); 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _summarize_latencies(latencies_s: list[float]) -> dict[str, float]:
    ms = [v * 1000.0 for v in latencies_s]
    return {
        "count": len(ms),
        "p50_ms": round(percentile(ms, 0.50), 3),
        "p99_ms": round(percentile(ms, 0.99), 3),
        "max_ms": round(max(ms), 3) if ms else 0.0,
    }


def _new_loop(store: SessionStore, latency_s: float) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=FakeLLM(latency_s=latency_s),
        store=store,
        config=AgentLoopConfig(max_rounds=1, compactor=None),
    )


async def run_fanout(*, sessions: int, latency_s: float = 0.0) -> dict[str, Any]:
    """N concurrent sessions, one send each. Measures wall-clock throughput
    (sessions/sec) and per-send latency under single-RLock contention."""
    store = SessionStore.open(":memory:")
    try:
        loop = _new_loop(store, latency_s)
        sids = [loop.new_session() for _ in range(sessions)]
        latencies: list[float] = []

        async def _one(sid: str) -> None:
            t0 = time.perf_counter()
            await loop.send("hello", session_id=sid)
            latencies.append(time.perf_counter() - t0)

        wall_t0 = time.perf_counter()
        await asyncio.gather(*(_one(sid) for sid in sids))
        wall = time.perf_counter() - wall_t0
        return {
            "scenario": "fanout",
            "sessions": sessions,
            "latency_s": latency_s,
            "wall_s": round(wall, 4),
            "sessions_per_sec": round(sessions / wall, 1) if wall > 0 else None,
            "send_latency": _summarize_latencies(latencies),
        }
    finally:
        store.close()


async def run_big_history(
    *, history_size: int, sends: int = 5, latency_s: float = 0.0
) -> dict[str, Any]:
    """One session pre-seeded with ``history_size`` active messages, then ``sends``
    sequential sends. Per-send latency exposes the O(active-history) read + token-scan
    cost that runs each round."""
    store = SessionStore.open(":memory:")
    try:
        loop = _new_loop(store, latency_s)
        sid = loop.new_session()
        for i in range(history_size):
            store.append_message(
                sid,
                role="user" if i % 2 == 0 else "assistant",
                content="x" * 200,
                round_index=i,
            )
        latencies: list[float] = []
        for _ in range(sends):
            t0 = time.perf_counter()
            await loop.send("and another", session_id=sid)
            latencies.append(time.perf_counter() - t0)
        return {
            "scenario": "big_history",
            "history_size": history_size,
            "sends": sends,
            "latency_s": latency_s,
            "send_latency": _summarize_latencies(latencies),
        }
    finally:
        store.close()


async def run_throughput(*, total_sends: int, latency_s: float = 0.0) -> dict[str, Any]:
    """Sustained sequential sends on one session. Reports sends/sec — a steady-state
    write-path number without fan-out contention."""
    store = SessionStore.open(":memory:")
    try:
        loop = _new_loop(store, latency_s)
        sid = loop.new_session()
        latencies: list[float] = []
        wall_t0 = time.perf_counter()
        for _ in range(total_sends):
            t0 = time.perf_counter()
            await loop.send("tick", session_id=sid)
            latencies.append(time.perf_counter() - t0)
        wall = time.perf_counter() - wall_t0
        return {
            "scenario": "throughput",
            "total_sends": total_sends,
            "latency_s": latency_s,
            "wall_s": round(wall, 4),
            "sends_per_sec": round(total_sends / wall, 1) if wall > 0 else None,
            "send_latency": _summarize_latencies(latencies),
        }
    finally:
        store.close()
