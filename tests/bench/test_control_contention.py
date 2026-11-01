"""Real PostgreSQL control contention with eight independent session sends.

Run with the dedicated _test DSN. POWER_LOOP_CONTROL_BENCHMARK writes the raw
lock timings, per-send entry waits and aggregate throughput for before/after
comparison. No absolute speed gate: host load is measured, not assumed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import time
from contextvars import ContextVar
from pathlib import Path

import pytest

from bench.harness import percentile
from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.bench, pytest.mark.asyncio]
_MEASURED_SEND = ContextVar("control_benchmark_send", default=None)


class Model(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        await asyncio.sleep(.005)
        return LLMResponse(raw_text="done", token_usage=LLMTokenUsage(
            prompt_tokens=3, completion_tokens=2, total_tokens=5))

    def stream(self, request):
        raise AssertionError("The benchmark uses a nonstream model")

    async def close(self):
        pass


class Timings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.transactions = []

    def emit(self, record):
        send = _MEASURED_SEND.get()
        if send is not None:
            self.transactions.append({"send": send, "wait_ms": record.wait_ms,
                "hold_ms": record.hold_ms, "outcome": record.outcome})


def summary(values):
    return {"count": len(values), "p50": percentile(values, .50),
            "p95": percentile(values, .95), "max": max(values, default=0.0)}


@pytest.fixture(params=["postgres"])
async def pg_registry_store(request, tmp_path):
    async for value in registry_store.__wrapped__(request, tmp_path):
        yield value


async def test_eight_session_send_contention(pg_registry_store):
    store, _ = pg_registry_store
    manager = JobManager(store)
    runtimes, loops, models, sessions = [], [], [], []
    logger = logging.getLogger("power_loop.control.transactions")
    old_level, old_propagate = logger.level, logger.propagate
    timings = Timings()
    try:
        for index in range(8):
            model = Model()
            runtime = TaskRuntime(manager, tenant_id="contention", principal_id="host",
                executor_id=f"benchmark-{index}", boot_id="boot", boot_contract_hash="a" * 64)
            loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
                tool_registry=ToolRegistry(), config=AgentLoopConfig(
                    max_rounds=2, max_tokens=20, compactor=None, delegation_policy="single_level"))
            runtimes.append(runtime)
            loops.append(loop)
            models.append(model)
            sid = await loop.new_session()
            sessions.append(sid)
            assert (await loop.send("warm", sid)).status == "completed"
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(timings)
        ready = asyncio.Event()
        sends_per_session = 16
        latencies = []

        async def worker(index):
            await ready.wait()
            for step in range(sends_per_session):
                token = _MEASURED_SEND.set(f"{index}:{step}")
                try:
                    began = time.perf_counter()
                    result = await loops[index].send(f"measured {step}", sessions[index])
                    latencies.append((time.perf_counter() - began) * 1000)
                    assert result.status == "completed"
                finally:
                    _MEASURED_SEND.reset(token)

        workers = [asyncio.create_task(worker(i)) for i in range(8)]
        started = time.perf_counter()
        ready.set()
        try:
            await asyncio.gather(*workers)
        finally:
            for worker_task in workers:
                if not worker_task.done():
                    worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        elapsed = time.perf_counter() - started
        assert len(latencies) == 8 * sends_per_session
        assert all(model.calls == sends_per_session + 1 for model in models)
        assert timings.transactions and all(t["outcome"] == "committed" for t in timings.transactions)
        entry = {}
        for transaction in timings.transactions:
            entry.setdefault(transaction["send"], transaction["wait_ms"])
        assert len(entry) == len(latencies)
        report = {"engine": "postgres", "sessions": 8, "sends_per_session": sends_per_session,
            "completed": len(latencies), "elapsed_s": elapsed, "sends_per_s": len(latencies) / elapsed,
            "entry_wait_ms": summary(list(entry.values())),
            "transaction_wait_ms": summary([t["wait_ms"] for t in timings.transactions]),
            "hold_ms": summary([t["hold_ms"] for t in timings.transactions]),
            "send_latency_ms": summary(latencies), "transactions": timings.transactions,
            "entry_wait_definition": "First control transaction of each measured send; transaction entry through lock acquisition.",
            "hold_definition": "Lock acquisition through transaction commit or rollback completion.",
            "model_latency_s": .005, "python": platform.python_version(), "cpu_count": os.cpu_count()}
        target = os.environ.get("POWER_LOOP_CONTROL_BENCHMARK")
        if target:
            Path(target).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        logger.removeHandler(timings)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        for runtime in runtimes:
            await runtime.shutdown(timeout_s=1)
        for loop in loops:
            await loop.aclose(drain_timeout_s=.1)
