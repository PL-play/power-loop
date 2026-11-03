"""Evaluation: throughput / latency / control lock wait for N concurrent sessions, T tool rounds per send."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextvars import ContextVar

import pytest

from bench.harness import percentile
from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_job_registry_192 import registry_store

pytestmark = [pytest.mark.asyncio]
OUT = os.environ["EVAL_OUT"]
SEND = ContextVar("eval_send", default=None)
SENDS = int(os.environ.get("EVAL_SENDS", "8"))


@pytest.fixture(params=os.environ.get("EVAL_CONC_ENGINES", "postgres,mysql").split(","))
async def store3(request, tmp_path):
    async for value in registry_store.__wrapped__(request, tmp_path):
        yield value


class Timings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        s = SEND.get()
        if s is not None:
            self.rows.append((s, record.wait_ms, record.hold_ms))


def stat(v):
    return {"p50": round(percentile(v, .5), 2), "p95": round(percentile(v, .95), 2)}


@pytest.mark.parametrize("tools", [0, 4])
@pytest.mark.parametrize("n", [1, 4, 8, 16])
async def test_concurrency(store3, n, tools):
    store, _ = store3
    manager = JobManager(store)

    class Model(LLMService):
        def __init__(self):
            self.calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            await asyncio.sleep(.005)
            last = request.messages[-1] if getattr(request, "messages", None) else None
            role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
            pending = self.left > 0
            if pending:
                self.left -= 1
            tc = ([{"id": f"c{self.calls}", "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":"ok"}'}}] if pending else None)
            return LLMResponse(raw_text="" if tc else "done", tool_calls=tc,
                               token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

        def stream(self, request):
            raise AssertionError

        async def close(self):
            pass

    async def echo(text):
        return text
    loops, runtimes, models, sids = [], [], [], []
    logger = logging.getLogger("power_loop.control.transactions")
    timings = Timings()
    try:
        for i in range(n):
            reg = ToolRegistry()
            reg.register(ToolDefinition("echo", "Return input", input_schema={
                "type": "object", "properties": {"text": {"type": "string"}}}, pricing=Free("fixture")), echo)
            m = Model(); m.left = 0
            rt = TaskRuntime(manager, tenant_id="eval", principal_id="host", executor_id=f"c-{i}",
                             boot_id="boot", boot_contract_hash="a" * 64)
            lp = StatefulAgentLoop(llm=m, store=store, task_runtime=rt, tool_registry=reg,
                                   config=AgentLoopConfig(max_rounds=tools + 2, max_tokens=20, compactor=None,
                                                          delegation_policy="single_level"))
            loops.append(lp); runtimes.append(rt); models.append(m)
            sid = await lp.new_session(); sids.append(sid)
            assert (await lp.send("warm", sid)).status == "completed"
        logger.setLevel(logging.DEBUG); logger.propagate = False; logger.addHandler(timings)
        ready, lat = asyncio.Event(), []

        async def worker(i):
            await ready.wait()
            for step in range(SENDS):
                tok = SEND.set(f"{i}:{step}")
                try:
                    models[i].left = tools
                    t0 = time.perf_counter()
                    r = await loops[i].send(f"m{step}", sids[i])
                    lat.append((time.perf_counter() - t0) * 1000)
                    assert r.status == "completed"
                finally:
                    SEND.reset(tok)
        ws = [asyncio.create_task(worker(i)) for i in range(n)]
        t0 = time.perf_counter(); ready.set()
        await asyncio.gather(*ws)
        el = time.perf_counter() - t0
        entry = {}
        for s, w, _ in timings.rows:
            entry.setdefault(s, w)
        row = {"engine": store._db.dialect.name, "sessions": n, "tool_rounds": tools, "sends": len(lat),
               "sends_per_s": round(len(lat) / el, 2), "tool_calls_per_s": round(len(lat) * tools / el, 1),
               "latency_ms": stat(lat), "entry_wait_ms": stat(list(entry.values())),
               "tx_wait_ms": stat([w for _, w, _ in timings.rows]), "hold_ms": stat([h for *_, h in timings.rows]),
               "control_tx_per_send": round(len(timings.rows) / max(len(lat), 1), 2)}
        with open(os.path.join(OUT, "concurrency.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    finally:
        logger.removeHandler(timings)
        for rt in runtimes:
            await rt.shutdown(timeout_s=1)
        for lp in loops:
            await lp.aclose(drain_timeout_s=.1)
