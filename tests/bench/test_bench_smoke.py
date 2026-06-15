"""SCALE-1 smoke: the benchmark harness runs end-to-end and reports sane numbers.

Fast + deterministic (FakeLLM, latency 0, tiny grids) so it can gate in CI as a
non-blocking job. It does NOT assert absolute throughput — those numbers are
environment-sensitive and recorded out-of-band on reference hardware (SCALE-5).
"""

from __future__ import annotations

import pytest

from bench.harness import percentile, run_big_history, run_fanout, run_throughput
from bench.scenarios import run_suite

pytestmark = pytest.mark.bench


async def test_fanout_smoke_reports_sane_numbers() -> None:
    r = await run_fanout(sessions=8)
    assert r["sessions"] == 8
    assert r["sessions_per_sec"] and r["sessions_per_sec"] > 0
    lat = r["send_latency"]
    assert lat["count"] == 8
    assert lat["p99_ms"] >= lat["p50_ms"] >= 0  # monotone summary


async def test_big_history_smoke() -> None:
    r = await run_big_history(history_size=200, sends=3)
    assert r["history_size"] == 200
    assert r["send_latency"]["count"] == 3
    assert r["send_latency"]["p50_ms"] >= 0


async def test_throughput_smoke() -> None:
    r = await run_throughput(total_sends=20)
    assert r["total_sends"] == 20
    assert r["sends_per_sec"] and r["sends_per_sec"] > 0
    assert r["send_latency"]["count"] == 20


async def test_run_suite_smoke_shape() -> None:
    res = await run_suite(smoke=True)
    assert res["mode"] == "smoke"
    assert {"fanout", "big_history", "throughput"} <= set(res["scenarios"])
    assert len(res["scenarios"]["fanout"]) == 2  # SMOKE_FANOUT_SESSIONS


def test_percentile_edges() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.99) == 5.0
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([0.0, 10.0], 0.99) == pytest.approx(9.9)
