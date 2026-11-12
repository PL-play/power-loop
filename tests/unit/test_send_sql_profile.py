"""Whole-send SQL budget, attributed by call site; writes a report when requested.

Design 203 records actual costs against fixed B without a performance threshold.
Normal billing and cleanup invariants remain assertions. Set POWER_LOOP_SQL_PROFILE=<dir> to emit one JSON file per case.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager, nullcontext

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    ToolDefinition,
)
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG = os.path.join(_ROOT, "power_loop") + os.sep

_SKIP = ("store.backends", "store.db")
_BILLING = ("runtime.model_budgets", "runtime.execution_budget", "runtime.response_capture",
            "runtime.tool_billing", "runtime.budget_")
_CATEGORIES = ("message", "session_control", "legacy_job_control", "billing", "periodic")
_LEGACY_TABLES = (
    "task_store", "jobs", "job_events", "execution_checkpoints", "session_controls",
    "session_obligations", "task_context_events", "task_executors", "task_executor_events",
    "task_executor_admissions", "task_principal_admissions",
)
_MESSAGE_TABLES = (
    "messages", "project_messages", "compactions", "usage_observations", "session_stats",
    "notes", "hook_events", "context_cache", "context_attempts",
)
_SESSION_TABLES = (
    "sessions", "session_state", "follow_up_queue", "session_runtime_state", "shared_state",
    "background_tasks", "timers",
)
_LEGACY_MODULES = (
    "runtime.jobs", "runtime.lightweight_send", "runtime.subagent_resume",
    "runtime.execution_journal", "runtime.execution_membership", "runtime.finalization",
    "runtime.finalization_plan", "runtime.result_revisions", "runtime.result_proof",
    "runtime.result_consumption", "runtime.executor_admissions", "runtime.executor_lifecycle",
    "runtime.executor_registry", "runtime.session_obligations", "runtime.controller_handoff",
    "runtime.controller_recovery", "runtime.controller_retirement", "runtime.controller_task_exit",
    "runtime.control_requests", "runtime.control_runtime", "runtime.managed_claim",
    "runtime.managed_completion", "runtime.managed_delivery", "runtime.original_owner_review",
    "runtime.pending_owner", "runtime.nudges", "runtime.node_inputs", "runtime.task_control",
    "runtime.task_events", "runtime.task_inspection", "runtime.task_query", "runtime.task_sessions",
    "runtime.child_completion", "runtime.completion_windows", "runtime.result_notices",
    "runtime.operation_records", "runtime.host_send_steps", "workflow.control", "workflow.resume",
)
_LEGACY_SQL_COLUMNS = (
    "target_job_id", "expected_job_generation", "delivered_job_id", "delivered_job_generation",
    "claim_token", "claim_owner_epoch",
)
_LEGACY_RUNTIME_STORE_CALLS = (
    "agent.runtime_store:accept_input", "agent.runtime_store:claim_inputs",
    "agent.runtime_store:bind_input_job", "agent.runtime_store:resolve_job",
    "agent.runtime_store:execution_input", "agent.runtime_store:result_input",
    "agent.runtime_store:discard_cancelled_result", "agent.runtime_store:fail_start",
    "agent.runtime_store:commit_result_and_parent_ref", "agent.runtime_store:_lock_result_sessions",
    "agent.runtime_store:_commit_result", "agent.runtime_store:poll_notifications",
)


def _periodic(chain):
    return (_executor_heartbeat(chain) or any(site.startswith("runtime.timers:") for site in chain)
            or any(site.startswith("runtime.store.retention:") for site in chain))


def _category(chain, sql=""):
    chain = [c for c in chain if not c.startswith(_SKIP)]
    if _periodic(chain):
        return "periodic"
    if any(c.startswith(_BILLING) for c in chain):
        return "billing"
    normalized = sql.lower()
    if any(re.search(rf"(?<![a-z0-9]){table}(?![a-z0-9])", normalized) for table in _LEGACY_TABLES):
        return "legacy_job_control"
    if any(re.search(rf"(?<![a-z0-9]){column}(?![a-z0-9])", normalized)
           for column in _LEGACY_SQL_COLUMNS):
        return "legacy_job_control"
    if any(c.startswith(_LEGACY_MODULES) or c.startswith("runtime.child_tasks:")
           or c in _LEGACY_RUNTIME_STORE_CALLS for c in chain):
        return "legacy_job_control"
    if any(re.search(rf"(?<![a-z0-9]){table}(?![a-z0-9])", normalized) for table in _MESSAGE_TABLES):
        return "message"
    if any(re.search(rf"(?<![a-z0-9]){table}(?![a-z0-9])", normalized) for table in _SESSION_TABLES):
        return "session_control"
    if sql:
        raise AssertionError(("unclassified SQL", sql, chain))
    if any(c.startswith(("agent.", "runtime.session_ownership", "runtime.store_session_ownership"))
           for c in chain):
        return "session_control"
    if any(c.startswith(("runtime.store.", "runtime.runtime_state", "core.")) for c in chain):
        return "message"
    return "legacy_job_control"


def _counts(records, tx_sites, tx_categories=None):
    counts = {name: 0 for name in _CATEGORIES}
    counts.update({name + "_tx": 0 for name in _CATEGORIES})
    for record in records:
        counts[_category(record["chain"], record["sql"])] += 1
    for category in tx_categories or [_category(chain) for chain in tx_sites]:
        counts[category + "_tx"] += 1
    counts.update(
        ordinary=counts["message"],
        control=counts["session_control"] + counts["legacy_job_control"],
        control_tx=counts["session_control_tx"] + counts["legacy_job_control_tx"],
    )
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
    def __init__(self, db, meter=None):
        self.db, self.dialect = db, db.dialect
        self.meter = meter
        self.records, self.transactions, self.tx_sites = [], 0, []
        self.tx_categories = []
        self.tx_depth = 0

    def __getattr__(self, name):
        return getattr(self.db, name)

    def note(self, sql, rows=None):
        self.records.append({"sql": " ".join(sql.split())[:240], "chain": _sites(), "rows": rows})

    @staticmethod
    def _transaction_category(records, site):
        categories = {_category(record["chain"], record["sql"]) for record in records}
        if "legacy_job_control" in categories:
            return "legacy_job_control"
        if "billing" in categories:
            return "billing"
        if any(c.startswith(("runtime.store.store:", "runtime.runtime_state:", "core.")) for c in site):
            return "message"
        for category in ("session_control", "message", "periodic"):
            if category in categories:
                return category
        return _category(site)

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        site = _sites()
        self.tx_sites.append(site)
        start = len(self.records)
        try:
            async with self.db.transaction() as tx:
                measured = _Tx(self, tx)
                yield measured
                if measured.before is not None:
                    after = await self.meter.totals(tx)
                    self.meter.max_tx_rows = max(self.meter.max_tx_rows, after[0] - measured.before[0])
                    self.meter.max_tx_bytes = max(self.meter.max_tx_bytes, after[1] - measured.before[1])
        finally:
            self.tx_categories.append(self._transaction_category(self.records[start:], site))

    @asynccontextmanager
    async def read_snapshot(self):
        self.transactions += 1
        site = _sites()
        self.tx_sites.append(site)
        start = len(self.records)
        try:
            async with self.db.read_snapshot() as reader:
                yield _Tx(self, reader)
        finally:
            self.tx_categories.append(self._transaction_category(self.records[start:], site))

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
        async with self.meter.transaction(self.db) if self.meter else nullcontext():
            return await self.db.execute(sql, params)


class _Tx:
    def __init__(self, owner, tx):
        self.owner, self.tx = owner, tx
        self.before = None

    async def begin_meter(self, sql):
        # A PostgreSQL snapshot must set isolation before its first query.
        # Delay instrumentation until after transaction configuration.
        if self.owner.meter and self.before is None and not sql.lstrip().upper().startswith("SET TRANSACTION"):
            self.before = await self.owner.meter.totals(self.tx)

    def __getattr__(self, name):
        return getattr(self.tx, name)

    async def execute(self, sql, params=()):
        await self.begin_meter(sql)
        self.owner.note(sql)
        return await self.tx.execute(sql, params)

    async def execute_flag(self, sql, params=()):
        await self.begin_meter(sql)
        self.owner.note(sql)
        return await self.tx.execute_flag(sql, params)

    async def fetchone(self, sql, params=()):
        await self.begin_meter(sql)
        result = await self.tx.fetchone(sql, params)
        self.owner.note(sql, 0 if result is None else 1)
        return result

    async def fetchall(self, sql, params=()):
        await self.begin_meter(sql)
        result = await self.tx.fetchall(sql, params)
        self.owner.note(sql, len(result))
        return result


@pytest.mark.parametrize("models", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("tools_per_round", [1, 4])
@pytest.mark.parametrize("budgeted", [False, True])
async def test_send_sql_profile(registry_store, models, budgeted, tools_per_round, *, paid_tools=False):
    store, _ = registry_store
    records, transactions, tx_sites, elapsed_s, tx_categories = await _measure(
        store, models, budgeted, tools_per_round=tools_per_round, paid_tools=paid_tools)
    dialect = store._db.dialect.name
    target = os.environ.get("POWER_LOOP_SQL_PROFILE")
    if target:
        os.makedirs(target, exist_ok=True)
        suffix = '-paid' if paid_tools else ''
        filename = f"send-{dialect}-m{models}-k{tools_per_round}-b{int(budgeted)}{suffix}.json"
        with open(os.path.join(target, filename), "w") as f:
            json.dump({"models": models, "tools": (models - 1) * tools_per_round,
                       "tools_per_round": tools_per_round, "budgeted": budgeted, "paid_tools": paid_tools,
                       "transactions": transactions, "tx_sites": tx_sites, "elapsed_s": elapsed_s,
                       "records": records}, f)
    counts = _counts(records, tx_sites, tx_categories)
    # Heartbeats and timer scans follow wall time, not the send. A send longer
    # than one second may also refresh the session activity time once.
    foreground = [r for r in records if not _executor_heartbeat(r["chain"])
                  and not any(site.startswith("runtime.timers:") for site in r["chain"])]
    assert counts["billing"] == 0
    assert counts["control_tx"] <= 3
    # Design 215 P0-R1 / option A: three control transactions and one latest
    # inbox_version read per loop boundary. MySQL needs one LAST_INSERT_ID read
    # per inserted assistant/tool message, so each extra tool costs three SQL.
    base = 24 if dialect == "mysql" else 22
    per_round = 7 if dialect == "mysql" else 5
    per_extra_tool = 3 if dialect == "mysql" else 2
    budget = (base + per_round * (models - 1)
              + per_extra_tool * (tools_per_round - 1) * (models - 1))
    activity = [r for r in foreground if " SET updated_at=" in r["sql"]]
    assert len(activity) <= 1
    assert len(foreground) - len(activity) <= budget, (
        dialect, models, tools_per_round, len(foreground), len(activity), budget)



@pytest.mark.parametrize('models', [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize('tools_per_round', [1, 4])
async def test_paid_tool_send_sql_profile(registry_store, models, tools_per_round):
    await test_send_sql_profile(
        registry_store, models, True, tools_per_round, paid_tools=True)


@pytest.mark.parametrize("budgeted", [False, True])
async def test_send_sql_independent_of_history(registry_store, budgeted):
    store, _ = registry_store
    short_records, _, short_tx, _, short_categories = await _measure(store, 3, budgeted)
    long_records, _, long_tx, _, long_categories = await _measure(store, 3, budgeted, history=20)
    # The executor heartbeat follows elapsed wall time, not transcript length.
    # Keep it in all reported costs; compare foreground work for this invariant.
    def foreground(records, sites, categories):
        def periodic(chain):
            return _executor_heartbeat(chain) or any(site.startswith('runtime.timers:') for site in chain)
        keep = [index for index, chain in enumerate(sites) if not periodic(chain)]
        return _counts([r for r in records if not periodic(r['chain'])],
                       [sites[index] for index in keep], [categories[index] for index in keep])
    short = foreground(short_records, short_tx, short_categories)
    long = foreground(long_records, long_tx, long_categories)
    for name in ("session_control", "legacy_job_control", "billing", "control_tx"):
        assert long[name] == short[name], (name, short, long)
    assert long["message"] <= short["message"] + 1, (short, long)


def _executor_heartbeat(chain):
    return 'runtime.executor_registry:heartbeat' in chain


async def _measure(store, models, budgeted, history=0, *, tools_per_round=1, paid_tools=False):
    from tests.write_meter import WriteMeter

    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    store._db = profiled = Profiled(original, meter)
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="profile", principal_id="host", executor_id=f"executor-h{history}",
                          boot_id="boot", boot_contract_hash="a" * 64)
    registry = ToolRegistry()

    async def echo(text):
        return text
    registry.register(ToolDefinition("echo", "Return input", input_schema={
        "type": "object", "properties": {"text": {"type": "string"}}}, pricing=Free("fixture")), echo)

    class Model(LLMService):
        calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            tools = ([{"id": f"call-{self.calls}-{index}", "type": "function",
                       "function": {"name": "echo", "arguments": '{"text":"ok"}'}}
                      for index in range(tools_per_round)]
                     if self.calls < models else None)
            return LLMResponse(raw_text="done" if tools is None else "", tool_calls=tools,
                               token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

        def stream(self, request):
            raise AssertionError("nonstream fixture")

        async def close(self):
            pass

    model = Model()
    agent = await RunningAgent(store, model, registry,
        AgentLoopConfig(max_rounds=models + 1, max_tokens=20, compactor=None,
                        delegation_policy="single_level"), task_runtime=runtime).start()
    try:
        for index in range(history):
            model.calls = 0
            assert (await agent.send(f"history {index}")).status == "completed"
        model.calls = models - 1
        assert (await agent.send("warm")).status == "completed"
        model.calls = 0
        profiled.records.clear()
        profiled.tx_sites.clear()
        profiled.tx_categories.clear()
        profiled.transactions = 0
        if meter:
            await meter.reset()
        started = time.monotonic()
        result = await agent.send("run")
        elapsed_s = time.monotonic() - started
        records, transactions, tx_sites = list(profiled.records), profiled.transactions, list(profiled.tx_sites)
        tx_categories = list(profiled.tx_categories)
        assert result.status == "completed" and model.calls == models
        if target := os.environ.get("P0_HISTORY_OUT"):
            with open(target, "a") as f:
                f.write(json.dumps({"db": original.dialect.name, "models": models, "budgeted": budgeted,
                                    "history": history, "sql": _counts(records, tx_sites, tx_categories),
                                    "total_sql": len(records), "transactions": transactions,
                                    "executor_heartbeat_sql": sum(_executor_heartbeat(r['chain']) for r in records),
                                    "executor_heartbeat_tx": sum(_executor_heartbeat(chain) for chain in tx_sites)}) + "\n")
        if meter and (target := os.environ.get("P0_WRITE_OUT")):
            with open(target, "a") as f:
                f.write(json.dumps({"db": original.dialect.name, "models": models, "budgeted": budgeted,
                                    "paid_tools": paid_tools, "history": history,
                                    "writes": await meter.report()}) + "\n")
        return records, transactions, tx_sites, elapsed_s, tx_categories
    finally:
        await agent.aclose()
        await runtime.shutdown(timeout_s=1)
        store._db = original
        if meter:
            await meter.close()
