"""SCALE-1 smoke: the benchmark harness runs end-to-end and reports sane numbers.

Fast + deterministic (FakeLLM, latency 0, tiny grids) so it can gate in CI as a
non-blocking job. It does NOT assert absolute throughput — those numbers are
environment-sensitive and recorded out-of-band on reference hardware (SCALE-5).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench import harness
from bench.harness import percentile, run_big_history, run_fanout, run_throughput
from bench.scenarios import run_suite

pytestmark = pytest.mark.bench


def _evidence(name, value):
    if directory := os.environ.get("R9_EVIDENCE_DIR"):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


async def test_fanout_smoke_reports_sane_numbers() -> None:
    r = await run_fanout(sessions=8)
    assert r["sessions"] == 8
    assert r["sessions_per_sec"] and r["sessions_per_sec"] > 0
    lat = r["send_latency"]
    assert lat["count"] == 8
    assert lat["p99_ms"] >= lat["p50_ms"] >= 0  # monotone summary
    assert (r["completed_count"], r["failed_count"]) == (8, 0)
    _evidence("fanout-8", r)


async def test_big_history_smoke() -> None:
    r = await run_big_history(history_size=200, sends=3)
    assert r["history_size"] == 200
    assert r["send_latency"]["count"] == 3
    assert r["send_latency"]["p50_ms"] >= 0
    assert (r["completed_count"], r["failed_count"]) == (3, 0)
    assert r["history_unit"] == "completed_sends" and r["seed_completed_count"] == 200
    assert r["projected_sends_before_measurement"] == 200
    assert r["audit_rows_before_measurement"] == 400
    assert r["compact_exists_before_measurement"]
    assert r["setup_s"] >= r["seed_s"] + r["validation_s"] > 0
    assert r["setup_validation_queries"] == 2
    assert r["setup_business_model_calls"] == 200 and r["setup_fold_model_calls"] > 0
    observed = r["history_observation"]
    assert observed["business_calls"] == 203 and observed["fold_calls"] > 0
    assert observed["first_measurement_seen"] and observed["last_seed_visible"]
    assert observed["first_measurement_rows"] > 1 and observed["first_measurement_chars"] > 200
    assert all(not isinstance(value, (dict, list)) for value in observed.values())
    _evidence("history-200", r)


async def test_throughput_smoke() -> None:
    r = await run_throughput(total_sends=20)
    assert r["total_sends"] == 20
    assert r["sends_per_sec"] and r["sends_per_sec"] > 0
    assert r["send_latency"]["count"] == 20
    assert (r["completed_count"], r["failed_count"]) == (20, 0)
    _evidence("throughput-20", r)


async def test_run_suite_smoke_shape() -> None:
    res = await run_suite(smoke=True)
    assert res["mode"] == "smoke"
    assert {"fanout", "big_history", "throughput"} <= set(res["scenarios"])
    assert len(res["scenarios"]["fanout"]) == 2  # SMOKE_FANOUT_SESSIONS
    scenarios = res["scenarios"]
    for r in [*scenarios["fanout"], *scenarios["big_history"], scenarios["throughput"]]:
        assert r["failed_count"] == 0 and r["completed_count"] == r["send_latency"]["count"]
    assert [r["history_size"] for r in scenarios["big_history"]] == [100, 1000]
    assert scenarios["throughput"]["total_sends"] == 50
    _evidence("suite-smoke", res)


def test_percentile_edges() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.99) == 5.0
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([0.0, 10.0], 0.99) == pytest.approx(9.9)


async def test_zero_history_is_a_real_empty_case():
    r = await run_big_history(history_size=0, sends=1)
    assert r["seed_completed_count"] == r["projected_sends_before_measurement"] == 0
    assert r["audit_rows_before_measurement"] == 0
    assert not r["compact_exists_before_measurement"]
    assert r["history_observation"]["first_measurement_seen"]
    assert not r["history_observation"]["last_seed_visible"]
    assert r["completed_count"] == r["send_latency"]["count"] == 1
    _evidence("history-0", r)


_SMALL_CASES = [
    (run_fanout, {"sessions": 3}, "sessions_per_sec", 1),
    (run_throughput, {"total_sends": 3}, "sends_per_sec", 1),
    (run_big_history, {"history_size": 1, "sends": 3}, "sends_per_sec", 2),
]


@pytest.mark.parametrize("runner,kwargs,rate,failing_call", _SMALL_CASES)
async def test_real_failed_result_is_not_success(monkeypatch, runner, kwargs, rate, failing_call):
    original = harness.FakeLLM.complete
    original_send = harness.StatefulAgentLoop.send
    calls, results = 0, []

    async def fail_one(self, request, **options):
        nonlocal calls
        calls += 1
        response = await original(self, request, **options)
        if calls == failing_call:
            raise ValueError("benchmark injected model-call failure")
        return response

    async def send(self, *args, **options):
        result = await original_send(self, *args, **options)
        results.append({"message_id": result.input_ref.message_id, "status": result.status,
                        "error_code": result.error_code, "final_text": result.final_text})
        return result

    monkeypatch.setattr(harness.FakeLLM, "complete", fail_one)
    monkeypatch.setattr(harness.StatefulAgentLoop, "send", send)
    r = await runner(**kwargs)
    assert (r["completed_count"], r["failed_count"]) == (2, 1)
    assert r["send_latency"]["count"] == 2
    assert sum(result["status"] != "completed" for result in results) == 1
    assert r[rate] == pytest.approx(2 / r["wall_s"], rel=0.02, abs=0.11)
    _evidence("failed-one-" + r["scenario"], {"report": r, "actual_results": results})


@pytest.mark.parametrize("runner,kwargs,rate", [
    (run_fanout, {"sessions": 2}, "sessions_per_sec"),
    (run_throughput, {"total_sends": 2}, "sends_per_sec"),
    (run_big_history, {"history_size": 0, "sends": 2}, "sends_per_sec"),
])
async def test_all_real_model_failures_have_zero_success_latency(monkeypatch, runner, kwargs, rate):
    original = harness.FakeLLM.complete

    async def fail(self, request, **options):
        await original(self, request, **options)
        raise ValueError("benchmark all model calls fail")

    monkeypatch.setattr(harness.FakeLLM, "complete", fail)
    r = await runner(**kwargs)
    assert (r["completed_count"], r["failed_count"]) == (0, 2)
    assert r["send_latency"] == {"count": 0, "p50_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    assert r[rate] == 0
    _evidence("failed-all-" + r["scenario"], r)


async def test_missing_seed_is_rejected_from_actual_model_observation(monkeypatch):
    observation = harness._HistoryObservation("history-not-prepared:")
    monkeypatch.setattr(harness, "_HistoryObservation", lambda *args: observation)
    with pytest.raises(RuntimeError, match="did not contain the last history seed"):
        await run_big_history(history_size=1, sends=1)
    assert observation.business_calls == 2 and observation.first_measurement_seen
    assert not observation.last_seed_visible
    _evidence("history-negative-control", observation.summary())


async def test_history_preparation_rejects_noncompleted_seed(monkeypatch):
    async def fail(self, request, **options):
        raise ValueError("seed model-call failure")

    monkeypatch.setattr(harness.FakeLLM, "complete", fail)
    with pytest.raises(RuntimeError, match="history preparation failed at seed 0"):
        await run_big_history(history_size=1, sends=1)


@pytest.mark.parametrize("runner,kwargs,rate,failing_call", _SMALL_CASES)
async def test_send_exception_propagates(monkeypatch, runner, kwargs, rate, failing_call):
    error = OSError("send exception must propagate")

    async def fail(self, *args, **options):
        raise error

    monkeypatch.setattr(harness.StatefulAgentLoop, "send", fail)
    with pytest.raises(OSError) as caught:
        await runner(**kwargs)
    assert caught.value is error


async def test_completed_wrong_text_is_rejected(monkeypatch):
    from power_loop import LLMResponse

    async def wrong(self, request, **options):
        return LLMResponse(raw_text="not ok")

    monkeypatch.setattr(harness.FakeLLM, "complete", wrong)
    with pytest.raises(RuntimeError, match="completed benchmark result was not ok"):
        await run_throughput(total_sends=1)


def test_cli_real_model_failures_exit_nonzero():
    # Only this negative control uses a small suite; the normal grids stay intact.
    source = '''
import asyncio
from bench import harness, scenarios
original = harness.FakeLLM.complete
async def model(self, request, **kwargs):
    response = await original(self, request, **kwargs)
    if any("tick" in str(m.get("content", "")) for m in request.messages):
        raise ValueError("CLI negative control: model failure")
    return response
harness.FakeLLM.complete = model
async def small_suite(*, smoke=False):
    return {"mode": "smoke", "scenarios": {
        "fanout": [await harness.run_fanout(sessions=1)],
        "big_history": [await harness.run_big_history(history_size=0, sends=1)],
        "throughput": await harness.run_throughput(total_sends=2)}}
scenarios.run_suite = small_suite
raise SystemExit(scenarios.main(["--smoke"]))
'''
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["scenarios"]["throughput"]["failed_count"] == 2
    assert report["scenarios"]["throughput"]["completed_count"] == 0
    assert report["scenarios"]["throughput"]["sends_per_sec"] == 0
    _evidence("cli-negative-control", {"exit": completed.returncode, "report": report,
                                       "stdout": completed.stdout, "stderr": completed.stderr})
