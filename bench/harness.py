"""Scenario runners + a deterministic FakeLLM for the SCALE-1 harness.

Three scenarios, all driving one ``StatefulAgentLoop`` over one on-disk ``SessionStore``:

* ``run_fanout``      — N concurrent sessions each doing one send (contention on the
                        single store RLock); reports sessions/sec + per-send latency.
* ``run_big_history`` — one session prepared by completed sends, with the default
                        projection and fold. Setup cost is separate from measured sends.
* ``run_throughput``  — sustained sequential sends on one session; reports sends/sec.

Timing uses ``time.perf_counter``; latencies are reported as p50/p99 in milliseconds.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    SessionStore,
    StatefulAgentLoop,
    TrustedActor,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.fold import _SUMMARY_PROMPT

_MEASURE_MARKER = "BENCH-MEASURE-FIRST"


@dataclass
class _HistoryObservation:
    """Constant-size counters over real requests; never retain message snapshots."""

    expected_marker: str | None
    business_calls: int = 0
    fold_calls: int = 0
    max_business_rows: int = 0
    max_business_chars: int = 0
    first_measurement_seen: bool = False
    last_seed_visible: bool = False
    first_measurement_rows: int = 0
    first_measurement_chars: int = 0

    def observe(self, request: LLMRequest) -> None:
        text = "\n".join(str(m.get("content", "")) for m in request.messages)
        if text.startswith(_SUMMARY_PROMPT):
            self.fold_calls += 1
            return
        self.business_calls += 1
        self.max_business_rows = max(self.max_business_rows, len(request.messages))
        self.max_business_chars = max(self.max_business_chars, len(text))
        if not self.first_measurement_seen and _MEASURE_MARKER in text:
            self.first_measurement_seen = True
            self.first_measurement_rows = len(request.messages)
            self.first_measurement_chars = len(text)
            self.last_seed_visible = self.expected_marker is not None and self.expected_marker in text

    def verify(self) -> None:
        if self.expected_marker is not None and not (
                self.first_measurement_seen and self.last_seed_visible):
            raise RuntimeError("first measurement model request did not contain the last history seed")

    def summary(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass
class FakeLLM(LLMService):
    """Returns a fixed answer after an optional artificial delay. No network, no tokens —
    so the harness measures store/loop overhead, and ``latency_s`` lets you model a
    provider's think-time to see how much the loop adds on top."""

    latency_s: float = 0.0
    reply: str = "ok"
    observation: _HistoryObservation | None = None

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        if self.observation is not None:
            self.observation.observe(request)
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


async def _composition(store: SessionStore, latency_s: float, *, observation=None):
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(FakeLLM(latency_s=latency_s, observation=observation), None,
            config=AgentLoopConfig(max_rounds=1))
    factory.register('bench.agent@215', build)
    runtime = Runtime(store=store, factory=factory, node_id="bench", capacity=32, recovery_interval_s=None)
    await runtime.worker.start()
    return runtime, StatefulAgentLoop(runtime=runtime), RuntimeDescriptor(
        2, 'agent', 'bench.agent@215', None, 'bench.tools@215', 'bench.hooks@215', 'bench.resources@215', {}, {}), TrustedActor('bench','host',None)


async def _close(runtime, loop):
    await runtime.worker.shutdown()
    await loop.aclose()
    await runtime.client.aclose()
    await runtime.transport.aclose()


def _record_result(result, elapsed: float, latencies: list[float]) -> bool:
    if result.status != "completed":
        return False
    if result.final_text != "ok":
        raise RuntimeError(f"completed benchmark result was not ok: {result.final_text!r}")
    latencies.append(elapsed)
    return True


async def run_fanout(
    *,
    sessions: int,
    latency_s: float = 0.0,
    db_path: str = ":memory:",
) -> dict[str, Any]:
    """N concurrent sessions, one send each. Measures wall-clock throughput
    (sessions/sec) and per-send latency under the async store's single-writer model.

    (The legacy SCALE-2 read pool is gone — the async SQLite backend offloads each
    statement to a worker thread and serializes writers with one ``asyncio.Lock``.)"""
    store = await SessionStore.open(db_path, schema="tasks")
    runtime, loop, descriptor, actor = await _composition(store, latency_s)
    try:
        sids = [await loop.new_session(descriptor, actor=actor) for _ in range(sessions)]
        latencies: list[float] = []

        async def _one(sid) -> bool:
            t0 = time.perf_counter()
            result = await loop.send(sid, "hello", message_id="first", actor=actor, deadline_ms=now_ms()+120000)
            return _record_result(result, time.perf_counter() - t0, latencies)

        wall_t0 = time.perf_counter()
        outcomes = await asyncio.gather(*(_one(sid) for sid in sids))
        completed = sum(outcomes)
        wall = time.perf_counter() - wall_t0
        return {
            "scenario": "fanout",
            "sessions": sessions,
            "latency_s": latency_s,
            "wall_s": round(wall, 4),
            "completed_count": completed,
            "failed_count": sessions - completed,
            "sessions_per_sec": round(completed / wall, 1) if wall > 0 else None,
            "send_latency": _summarize_latencies(latencies),
        }
    finally:
        await _close(runtime, loop)
        await store.close()


async def run_big_history(
    *, history_size: int, sends: int = 5, latency_s: float = 0.0
) -> dict[str, Any]:
    """Prepare ``history_size`` completed sends, then measure ``sends`` new sends.

    Default projection/folding applies to both phases; not all old sends remain
    individually in the model window. setup_s includes startup, seeds and the two
    count queries, while latency/wall measure only the public measured send calls.
    """
    setup_t0 = time.perf_counter()
    store = await SessionStore.open(":memory:", schema="tasks")
    observation = _HistoryObservation(f"history-{history_size - 1}:" if history_size else None)
    runtime, loop, descriptor, actor = await _composition(store, latency_s, observation=observation)
    try:
        sid = await loop.new_session(descriptor, actor=actor)
        seed_t0 = time.perf_counter()
        seed_completed = 0
        for i in range(history_size):
            result = await loop.send(sid, f"history-{i}: " + "x" * 200,
                message_id=f"seed-{i}", actor=actor, deadline_ms=now_ms()+120000)
            if result.status != "completed" or result.final_text != "ok":
                raise RuntimeError(f"history preparation failed at seed {i}: {result.status}, {result.final_text!r}")
            seed_completed += 1
        seed_s = time.perf_counter() - seed_t0
        validation_t0 = time.perf_counter()
        audit = await store._db.fetchone(
            f"SELECT COUNT(*) AS n FROM {store.t.messages} WHERE session_id=?", (sid.session_id,))
        projection = await store._db.fetchone(
            "SELECT COUNT(*) AS projection_rows, COUNT(DISTINCT CASE WHEN kind IN ('user','project') "
            "THEN send_index END) AS projected_sends, "
            f"SUM(CASE WHEN kind='compact' THEN 1 ELSE 0 END) AS compact_rows FROM {store.t.project_messages} "
            "WHERE session_id=?", (sid.session_id,))
        if projection["projected_sends"] != seed_completed:
            raise RuntimeError("completed history sends were not all projected")
        validation_s = time.perf_counter() - validation_t0
        setup_s = time.perf_counter() - setup_t0
        setup_business_calls, setup_fold_calls = observation.business_calls, observation.fold_calls
        latencies: list[float] = []
        completed = 0
        wall_t0 = time.perf_counter()
        for index in range(sends):
            t0 = time.perf_counter()
            result = await loop.send(sid, _MEASURE_MARKER if index == 0 else "and another",
                message_id=f"send-{index}", actor=actor, deadline_ms=now_ms()+120000)
            completed += _record_result(result, time.perf_counter() - t0, latencies)
        wall = time.perf_counter() - wall_t0
        observation.verify()
        return {
            "scenario": "big_history",
            "history_size": history_size,
            "history_unit": "completed_sends",
            "seed_completed_count": seed_completed,
            "setup_s": setup_s,
            "seed_s": seed_s,
            "validation_s": validation_s,
            "setup_validation_queries": 2,
            "setup_business_model_calls": setup_business_calls,
            "setup_fold_model_calls": setup_fold_calls,
            "audit_rows_before_measurement": audit["n"],
            "projection_rows_before_measurement": projection["projection_rows"],
            "projected_sends_before_measurement": projection["projected_sends"],
            "compact_exists_before_measurement": bool(projection["compact_rows"]),
            "history_observation": observation.summary(),
            "sends": sends,
            "latency_s": latency_s,
            "send_latency": _summarize_latencies(latencies),
            "completed_count": completed,
            "failed_count": sends - completed,
            "wall_s": round(wall, 4),
            "sends_per_sec": round(completed / wall, 1) if wall > 0 else None,
        }
    finally:
        await _close(runtime, loop)
        await store.close()


async def run_throughput(*, total_sends: int, latency_s: float = 0.0) -> dict[str, Any]:
    """Sustained sequential sends on one session. Reports sends/sec — a steady-state
    write-path number without fan-out contention."""
    store = await SessionStore.open(":memory:", schema="tasks")
    runtime, loop, descriptor, actor = await _composition(store, latency_s)
    try:
        sid = await loop.new_session(descriptor, actor=actor)
        latencies: list[float] = []
        completed = 0
        wall_t0 = time.perf_counter()
        for index in range(total_sends):
            t0 = time.perf_counter()
            result = await loop.send(sid, "tick", message_id=f"send-{index}", actor=actor, deadline_ms=now_ms()+120000)
            completed += _record_result(result, time.perf_counter() - t0, latencies)
        wall = time.perf_counter() - wall_t0
        return {
            "scenario": "throughput",
            "total_sends": total_sends,
            "latency_s": latency_s,
            "wall_s": round(wall, 4),
            "completed_count": completed,
            "failed_count": total_sends - completed,
            "sends_per_sec": round(completed / wall, 1) if wall > 0 else None,
            "send_latency": _summarize_latencies(latencies),
        }
    finally:
        await _close(runtime, loop)
        await store.close()
