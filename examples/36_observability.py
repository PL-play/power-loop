"""36 · 可观测性 / Observability (0.17.0 — OBS-1/2/4)

What you learn / 你将学到
--------------------------
- 事件总线是可观测性的接入点:把 sink 订阅上去就能持久化、出指标、连 tracing。
  / The event bus is the observability seam — attach sinks for durable logs, metrics, traces.
- `attach_jsonl_sink` 把完整事件信封(`ts`/`seq`/`mono`+payload,自动脱敏)按行落盘;
  `replay(path)` 进程退出后还能按顺序读回成 `AgentEvent`。
  / The JSONL sink persists the full envelope; ``replay`` reads it back after exit.
- `attach_metrics_sink` 把事件映射成计数/直方图(这里用一个极简内存后端,出厂还有
  Prometheus/StatsD,见 `power-loop[prometheus]` / `[statsd]`)。
  / The metrics sink maps events to counters/observations (shipped Prometheus/StatsD too).

Run / 运行
----------
    python examples/36_observability.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections import Counter
from pathlib import Path

from _helpers import make_llm

from power_loop import AgentEventBus, AgentLoopConfig, StatefulAgentLoop
from power_loop.contrib.jsonl_sink import attach_jsonl_sink, replay
from power_loop.contrib.metrics_sink import attach_metrics_sink


class _CountingBackend:
    """A trivial in-process MetricsBackend so the example needs no metrics dependency
    (swap in PrometheusBackend()/StatsDBackend() for real export)."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.observations: list[tuple[str, float]] = []

    def incr(self, name, value=1, labels=None):
        self.counters[name] += value

    def observe(self, name, value, labels=None):
        self.observations.append((name, value))

    def gauge(self, name, value, labels=None):
        pass


async def main() -> str:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    tmp = Path(tempfile.mkdtemp(prefix="pl_obs_"))
    path = tmp / "events.jsonl"

    sink = attach_jsonl_sink(bus, path)          # OBS-2: durable, replayable
    metrics = _CountingBackend()
    attach_metrics_sink(bus, metrics)            # OBS-4: events → metrics

    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=64),
        event_bus=bus,
        config=AgentLoopConfig(system_prompt="Answer in one short sentence.", max_rounds=1),
    )
    sid = await loop.new_session()
    result = await loop.send("Name a primary color.", session_id=sid)
    sink.close()

    # OBS-1: events persisted with their envelope survive the run and replay in seq order.
    events = list(replay(path))
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs), "events should replay in monotonic seq order"

    summary = (
        f"answer={result.final_text!r} | persisted {len(events)} events "
        f"(seq {seqs[0]}..{seqs[-1]}) | metric rounds={metrics.counters.get('power_loop_rounds', 0)} "
        f"llm_calls={metrics.counters.get('power_loop_llm_calls', 0)}"
    )
    print(summary)
    return summary


if __name__ == "__main__":
    asyncio.run(main())
