"""Per-send row and byte budgets on SQLite/PostgreSQL/MySQL; limits only decrease."""
from __future__ import annotations

import json
import math
import os

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_job_registry_192 import registry_store  # noqa: F401

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
OUT = os.environ.get("ROWS_OUT")

# Baseline 3b69678, design 201 A0. Byte budgets allow the specified 2% margin.
_BASELINE = {(1, 0, False): {'execution_batches': (1, 2207),
                 'execution_checkpoints': (1, 729),
                 'job_events': (9, 9405),
                 'jobs': (1, 311),
                 'messages': (2, 221),
                 'session_stats': (0, 1)},
 (1, 0, True): {'execution_allocations': (1, 763),
                'execution_batches': (2, 8023),
                'execution_checkpoints': (1, 910),
                'job_events': (10, 13901),
                'jobs': (1, 311),
                'messages': (2, 221),
                'session_stats': (0, 1),
                'task_budget_attempts': (1, 4216),
                'task_budget_scopes': (0, 1)},
 (4, 1, False): {'execution_batches': (1, 21947),
                 'execution_checkpoints': (1, 729),
                 'job_events': (51, 44487),
                 'jobs': (1, 311),
                 'messages': (8, 1186),
                 'session_state': (0, 1),
                 'session_stats': (0, 1)},
 (4, 1, True): {'execution_allocations': (1, 766),
                'execution_batches': (2, 53857),
                'execution_checkpoints': (1, 911),
                'job_events': (58, 77651),
                'jobs': (1, 311),
                'messages': (8, 1186),
                'session_state': (0, 1),
                'session_stats': (0, 1),
                'task_budget_attempts': (7, 30712),
                'task_budget_scopes': (0, 1)},
 (32, 1, False): {'execution_batches': (1, 206676),
                  'execution_checkpoints': (1, 733),
                  'job_events': (443, 372429),
                  'jobs': (1, 312),
                  'messages': (64, 10321),
                  'session_state': (0, 1),
                  'session_stats': (0, 2)},
 (32, 1, True): {'execution_allocations': (1, 772),
                 'execution_batches': (3, 483367),
                 'execution_checkpoints': (1, 918),
                 'job_events': (506, 673287),
                 'jobs': (1, 312),
                 'messages': (64, 10321),
                 'session_state': (0, 1),
                 'session_stats': (0, 2),
                 'task_budget_attempts': (63, 278134),
                 'task_budget_scopes': (0, 1)},
 (32, 4, False): {'execution_batches': (3, 610193),
                  'execution_checkpoints': (1, 735),
                  'job_events': (1280, 994747),
                  'jobs': (1, 312),
                  'messages': (157, 25745),
                  'session_stats': (0, 2)},
 (32, 4, True): {'execution_allocations': (1, 775),
                 'execution_batches': (6, 1321005),
                 'execution_checkpoints': (1, 923),
                 'job_events': (1436, 1766371),
                 'jobs': (1, 312),
                 'messages': (157, 25745),
                 'session_stats': (0, 2),
                 'task_budget_attempts': (156, 707608)}}


async def _snapshot(db, prefix):
    d = db.dialect.name
    if d == "sqlite":
        tables = [r["name"] for r in await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")]
    elif d == "postgres" or d == "postgresql":
        tables = [r["table_name"] for r in await db.fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_schema=current_schema() AND table_type='BASE TABLE'")]
    else:
        tables = [r["TABLE_NAME"] if "TABLE_NAME" in r else r["table_name"] for r in await db.fetchall(
            "SELECT table_name AS table_name FROM information_schema.tables WHERE table_schema=DATABASE() AND table_type='BASE TABLE'")]
    tables = [t for t in tables if t.startswith(prefix)]
    out = {}
    for t in tables:
        if d == "sqlite":
            cols = [r["name"] for r in await db.fetchall(f"PRAGMA table_info({t})")]
            size = "+".join(f"COALESCE(LENGTH({c}),0)" for c in cols) or "0"
        elif d == "mysql":
            cols = [r["column_name"] for r in await db.fetchall(
                f"SELECT column_name AS column_name FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='{t}'")]
            size = "+".join(f"COALESCE(LENGTH(`{c}`),0)" for c in cols) or "0"
        else:
            cols = [r["column_name"] for r in await db.fetchall(
                f"SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='{t}'")]
            size = "+".join(f"COALESCE(OCTET_LENGTH(CAST(\"{c}\" AS TEXT)),0)" for c in cols) or "0"
        row = await db.fetchone(f"SELECT COUNT(*) AS n, COALESCE(SUM({size}),0) AS b FROM {t}")
        out[t[len(prefix):]] = (int(row["n"]), int(row["b"]))
    return out


async def _events(db, table):
    ln = "OCTET_LENGTH" if db.dialect.name.startswith("postgres") else "LENGTH"
    rows = await db.fetchall(f"SELECT event_type, action, COUNT(*) AS n, "
                             f"SUM({ln}(payload_json)) AS b FROM {table} GROUP BY event_type, action")
    return {f"{r['event_type']}/{r['action']}": (int(r["n"]), int(r["b"] or 0)) for r in rows}


@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.parametrize("r,k", [(1, 0), (4, 1), (32, 1), (32, 4)])
async def test_rows(registry_store, r, k, budgeted):  # noqa: F811
    store, _ = registry_store
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
    runtime = TaskRuntime(JobManager(store), tenant_id="eval", principal_id="host", executor_id=f"rows-{r}-{k}",
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
        model.calls, model.target = 0, r
        assert (await loop.send("warm", sid)).status == "completed"
        p = store.table_prefix
        events_table = p + "job_events"
        before, ev_before = await _snapshot(store._db, p), await _events(store._db, events_table)
        model.calls = 0
        result = await loop.send("run", sid)
        assert result.status == "completed" and model.calls == r
        after, ev_after = await _snapshot(store._db, p), await _events(store._db, events_table)
        delta = {t: (after[t][0] - before.get(t, (0, 0))[0], after[t][1] - before.get(t, (0, 0))[1])
                 for t in after if after[t] != before.get(t)}
        ev = {k2: (v[0] - ev_before.get(k2, (0, 0))[0], v[1] - ev_before.get(k2, (0, 0))[1])
              for k2, v in ev_after.items() if v != ev_before.get(k2)}
        over = {}
        limits = _BASELINE[r, k, budgeted]
        for table, (count, size) in delta.items():
            row_limit, baseline_bytes = limits.get(table, (0, 0))
            byte_limit = math.ceil(baseline_bytes * 1.02)
            if count > row_limit or size > byte_limit:
                over[table] = {"actual": (count, size), "limit": (row_limit, byte_limit)}
        if OUT:
            with open(OUT, "a") as f:
                f.write(json.dumps({"db": store._db.dialect.name, "R": r, "K": k, "T": (r - 1) * k,
                                    "budgeted": budgeted, "tables": delta, "job_events": ev}) + "\n")
        assert not over, json.dumps({"over": over, "actual": delta}, sort_keys=True)
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=0.1)
