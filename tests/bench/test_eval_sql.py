"""Evaluation sweep: per-send SQL by history H, model calls R, tools per round K, billing, engine."""
from __future__ import annotations

import json
import os
import time

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_send_sql_profile import Profiled, _counts

pytestmark = [pytest.mark.asyncio]
OUT = os.environ["EVAL_OUT"]


@pytest.fixture(params=os.environ.get("EVAL_ENGINES", "sqlite,postgres,mysql").split(","))
async def store3(request, tmp_path):
    async for value in registry_store.__wrapped__(request, tmp_path):
        yield value


CASES = [  # (H history sends, R model calls, K tool calls per tool round)
    (0, 1, 0), (20, 1, 0), (200, 1, 0),
    (0, 4, 1), (0, 8, 1), (0, 16, 1), (0, 32, 1), (200, 32, 1),
    (0, 4, 4), (0, 8, 4), (0, 16, 4), (0, 32, 4),
]


@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.parametrize("h,r,k", CASES)
async def test_sweep(store3, h, r, k, budgeted):
    store, _ = store3
    dialect = store._db.dialect.name
    original = store._db
    store._db = profiled = Profiled(original)
    from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
    from power_loop.runtime.response_capture import ResponseCaptureReceipt
    from tests.unit.test_model_budgets_192 import Host
    host = Host()

    async def capture(packet):
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, "fixture-original")

    async def tool_plan(context, call, *, pricing):
        return BudgetPlan(0, {"attempt_id": context["attempt_id"], "origin": context})
    host.capture_response, host.prepare_tool = capture, tool_plan
    config = BudgetConfig("send-budget", (LocalBudgetScope("shared", max_calls=100000, max_tokens=10000000,
                                                           max_inflight=64),), host)
    runtime = TaskRuntime(JobManager(store), tenant_id="eval", principal_id="host", executor_id=f"eval-{h}-{r}-{k}",
                          boot_id="boot", boot_contract_hash="a" * 64, budget_config=config if budgeted else None)
    registry = ToolRegistry()

    async def echo(text):
        return text
    registry.register(ToolDefinition("echo", "Return input", input_schema={
        "type": "object", "properties": {"text": {"type": "string"}}}, pricing=Free("fixture")), echo)

    class Model(LLMService):
        calls, target = 0, 1

        async def complete(self, request, **kwargs):
            self.calls += 1
            tools = ([{"id": f"c{self.calls}-{i}", "type": "function",
                       "function": {"name": "echo", "arguments": '{"text":"ok"}'}} for i in range(k)]
                     if self.calls < self.target and k else None)
            return LLMResponse(raw_text="done" if not tools else "", tool_calls=tools,
                               token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

        def stream(self, request):
            raise AssertionError

        async def close(self):
            pass

    model = Model()
    host.service = model
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
                             config=AgentLoopConfig(max_rounds=r + 1, max_tokens=20, compactor=None,
                                                    delegation_policy="single_level"))
    sid = await loop.new_session()
    try:
        model.target = 1
        for index in range(h):
            model.calls = 0
            assert (await loop.send(f"history {index}", sid)).status == "completed"
        model.calls, model.target = 0, r
        assert (await loop.send("warm", sid)).status == "completed"
        model.calls = 0

        # Count rows before test run
        async def count_table_rows(table_name):
            try:
                result = await store._db.fetchone(f"SELECT COUNT(*) as cnt FROM {store.table_prefix}{table_name}")
                return result["cnt"] if result else 0
            except Exception:
                return 0

        tables_to_count = ["messages", "tasks", "job_events", "session_state", "compact_rows"]
        rows_before = {table: await count_table_rows(table) for table in tables_to_count}

        profiled.records.clear(); profiled.tx_sites.clear(); profiled.transactions = 0
        started = time.monotonic()
        result = await loop.send("run", sid)
        elapsed = time.monotonic() - started
        assert result.status == "completed" and model.calls == r

        # Count rows after test run
        rows_after = {table: await count_table_rows(table) for table in tables_to_count}
        rows_delta = {table: rows_after[table] - rows_before[table] for table in tables_to_count}

        counts = _counts(profiled.records, profiled.tx_sites)
        row = {"engine": dialect, "H": h, "R": r, "K": k, "T": (r - 1) * k, "budgeted": budgeted,
               **counts, "total_sql": len(profiled.records), "all_tx": profiled.transactions,
               "rows_read": sum(x["rows"] or 0 for x in profiled.records), "elapsed_ms": elapsed * 1000,
               "new_rows": rows_delta}
        with open(os.path.join(OUT, "sweep.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    finally:
        store._db = original
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=0.1)
