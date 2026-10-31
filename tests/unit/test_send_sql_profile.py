"""Whole-send SQL budget, attributed by call site; writes a report when requested.

Each case asserts hard upper bounds for ordinary storage, billing and unified
control SQL (see AGENTS.md, power-loop SQL budget). Raising a bound requires
user approval. Set POWER_LOOP_SQL_PROFILE=<dir> to emit one JSON file per case.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG = os.path.join(_ROOT, "power_loop") + os.sep

_SKIP = ("store.backends", "store.db")
_BILLING = ("runtime.model_budgets", "runtime.execution_budget", "runtime.response_capture",
            "runtime.tool_billing", "runtime.budget_", "runtime.native_returns")
_ORDINARY = ("runtime.store.", "runtime.runtime_state", "agent.", "core.")
_CONTROL = ("runtime.execution_journal", "runtime.execution_state")


def _category(chain):
    chain = [c for c in chain if not c.startswith(_SKIP)]
    if any(c.startswith(_CONTROL) for c in chain):
        return "control"
    if any(c.startswith(_BILLING) for c in chain):
        return "billing"
    if any(c.startswith("runtime.") and not c.startswith(_ORDINARY) for c in chain):
        return "control"
    return "ordinary"


def _limits(dialect, tools, budgeted, elapsed_s):
    """Upper bounds per send; `tools` is the number of tool rounds.

    `sessions.updated_at` is touched at most once per second, so a long send
    may add one ordinary statement per elapsed second.
    """
    mysql = dialect == "mysql"
    return {
        "ordinary": (23 if mysql else 21) + (7 if mysql else 5) * tools + int(elapsed_s),
        "billing": 13 if budgeted else 2,
        "control": 32 + tools * 3 // 7 if budgeted else 22 + tools // 6,
        "control_tx": 3 + tools // 16 if budgeted else 2,
    }


def _counts(records, tx_sites):
    counts = {"ordinary": 0, "billing": 0, "control": 0, "control_tx": 0}
    for record in records:
        counts[_category(record["chain"])] += 1
    counts["control_tx"] = sum(_category(chain) == "control" for chain in tx_sites)
    return counts


def _sites():
    frame, chain = sys._getframe(2), []
    while frame is not None:
        path = frame.f_code.co_filename
        if path.startswith(_PKG):
            chain.append(path[len(_PKG):-3].replace(os.sep, ".") + ":" + frame.f_code.co_name)
        frame = frame.f_back
    return chain


class Profiled:
    def __init__(self, db):
        self.db, self.dialect = db, db.dialect
        self.records, self.transactions, self.tx_sites = [], 0, []
        self.tx_depth = 0

    def __getattr__(self, name):
        return getattr(self.db, name)

    def note(self, sql, rows=None):
        self.records.append({"sql": " ".join(sql.split())[:240], "chain": _sites(), "rows": rows})

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        self.tx_sites.append(_sites())
        async with self.db.transaction() as tx:
            yield _Tx(self, tx)

    async def fetchone(self, sql, params=()):
        result = await self.db.fetchone(sql, params)
        self.note(sql, 0 if result is None else 1)
        return result

    async def fetchall(self, sql, params=()):
        result = await self.db.fetchall(sql, params)
        self.note(sql, len(result))
        return result

    async def execute(self, sql, params=()):
        self.note(sql)
        return await self.db.execute(sql, params)


class _Tx:
    def __init__(self, owner, tx):
        self.owner, self.tx = owner, tx

    def __getattr__(self, name):
        return getattr(self.tx, name)

    async def execute(self, sql, params=()):
        self.owner.note(sql)
        return await self.tx.execute(sql, params)

    async def fetchone(self, sql, params=()):
        result = await self.tx.fetchone(sql, params)
        self.owner.note(sql, 0 if result is None else 1)
        return result

    async def fetchall(self, sql, params=()):
        result = await self.tx.fetchall(sql, params)
        self.owner.note(sql, len(result))
        return result


@pytest.mark.parametrize("models", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("budgeted", [False, True])
async def test_send_sql_profile(registry_store, models, budgeted):
    store, _ = registry_store
    records, transactions, tx_sites, elapsed_s = await _measure(store, models, budgeted)
    dialect = store._db.dialect.name
    target = os.environ.get("POWER_LOOP_SQL_PROFILE")
    if target:
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, f"send-{dialect}-m{models}-b{int(budgeted)}.json"), "w") as f:
            json.dump({"models": models, "tools": models - 1, "budgeted": budgeted,
                       "transactions": transactions, "tx_sites": tx_sites,
                       "records": records}, f)
    counts, limits = _counts(records, tx_sites), _limits(dialect, models - 1, budgeted, elapsed_s)
    over = {name: (counts[name], limit) for name, limit in limits.items() if counts[name] > limit}
    assert not over, f"SQL budget exceeded (actual, limit): {over}"


@pytest.mark.parametrize("budgeted", [False, True])
async def test_send_sql_independent_of_history(registry_store, budgeted):
    store, _ = registry_store
    short_records, _, short_tx, _ = await _measure(store, 3, budgeted)
    long_records, _, long_tx, _ = await _measure(store, 3, budgeted, history=20)
    short, long = _counts(short_records, short_tx), _counts(long_records, long_tx)
    for name in ("billing", "control", "control_tx"):
        assert long[name] == short[name], (name, short, long)
    assert long["ordinary"] <= short["ordinary"] + 1, (short, long)


async def _measure(store, models, budgeted, history=0):
    original = store._db
    store._db = profiled = Profiled(original)
    manager = JobManager(store)
    from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
    from power_loop.runtime.response_capture import ResponseCaptureReceipt

    from .test_model_budgets_192 import Host
    host = Host()

    async def capture(packet):
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, "fixture-original")

    async def tool_plan(context, call, *, pricing):
        return BudgetPlan(0, {"attempt_id": context["attempt_id"], "origin": context})
    host.capture_response, host.prepare_tool = capture, tool_plan
    config = BudgetConfig("send-budget", (LocalBudgetScope("shared", max_calls=10000, max_tokens=1000000,
                                                           max_inflight=64),), host)
    runtime = TaskRuntime(manager, tenant_id="profile", principal_id="host", executor_id=f"executor-h{history}",
                          boot_id="boot", boot_contract_hash="a" * 64, budget_config=config if budgeted else None)
    registry = ToolRegistry()

    async def echo(text):
        return text
    registry.register(ToolDefinition("echo", "Return input", input_schema={
        "type": "object", "properties": {"text": {"type": "string"}}}, pricing=Free("fixture")), echo)

    class Model(LLMService):
        calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            tools = ([{"id": f"call-{self.calls}", "type": "function",
                       "function": {"name": "echo", "arguments": '{"text":"ok"}'}}]
                     if self.calls < models else None)
            return LLMResponse(raw_text="done" if tools is None else "", tool_calls=tools,
                               token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

        def stream(self, request):
            raise AssertionError("nonstream fixture")

        async def close(self):
            pass

    model = Model()
    host.service = model
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
                             config=AgentLoopConfig(max_rounds=models + 1, max_tokens=20, compactor=None,
                                                    delegation_policy="single_level"))
    sid = await loop.new_session()
    try:
        for index in range(history):
            model.calls = 0
            assert (await loop.send(f"history {index}", sid)).status == "completed"
        model.calls = models - 1
        assert (await loop.send("warm", sid)).status == "completed"
        model.calls = 0
        profiled.records.clear()
        profiled.tx_sites.clear()
        profiled.transactions = 0
        started = time.monotonic()
        result = await loop.send("run", sid)
        elapsed_s = time.monotonic() - started
        records, transactions, tx_sites = list(profiled.records), profiled.transactions, list(profiled.tx_sites)
        assert result.status == "completed" and model.calls == models
        return records, transactions, tx_sites, elapsed_s
    finally:
        store._db = original
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=0.1)
