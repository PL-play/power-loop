"""Session-native whole-send SQL budget, attributed by call site (design 215 P2).

Replaces the pre-215 job-path profile. The measurement harness (Profiled / _category /
_counts) is unchanged; the scenarios now drive the accept/claim/final session worker.
B = G0-R3 frozen baseline, S = P2 start, N = this HEAD. Set P215_PROFILE_OUT to emit one
JSON record per case for the B/S/N report. legacy_job_control must be 0; billing is 0 in
the SDK (the 214 charge event lives in the agent host, P6).
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
from power_loop.agent.runtime_transport import PublishReceipt
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

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
    "runtime.completion_windows", "runtime.result_notices",
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
            or any(site.startswith("runtime.store.retention:") for site in chain)
            or any(site in ("runtime.store.store:due_timers", "runtime.store.store:recover_stale_firing_timers") for site in chain))


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
    if any(c.startswith(_LEGACY_MODULES) or c in _LEGACY_RUNTIME_STORE_CALLS for c in chain):
        return "legacy_job_control"
    if "agent.runtime_store:_db_now" in chain:
        return "session_control"
    if normalized.startswith("select last_insert_id()"):
        return "message"
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

def _executor_heartbeat(chain):
    return 'runtime.executor_registry:heartbeat' in chain


class _ProfileModel(LLMService):
    def __init__(self, models, tools_per_round):
        self.models = models
        self.tools_per_round = tools_per_round
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        tools = ([{"id": f"call-{self.calls}-{i}", "type": "function",
                   "function": {"name": "echo", "arguments": '{"text":"ok"}'}}
                  for i in range(self.tools_per_round)]
                 if self.calls < self.models else None)
        return LLMResponse(raw_text="done" if tools is None else "", tool_calls=tools,
                           token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


async def _measure(store, models, *, history=0, tools_per_round=1, backlog=1,
                   lose_notifications=False):
    from tests.write_meter import WriteMeter

    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    store._db = profiled = Profiled(original, meter)
    registry = ToolRegistry()

    async def echo(text):
        return text

    registry.register(ToolDefinition("echo", "Return input", input_schema={
        "type": "object", "properties": {"text": {"type": "string"}}}, pricing=Free("fixture")), echo)
    model = _ProfileModel(models, tools_per_round)
    agent = await RunningAgent(store, model, registry,
                               AgentLoopConfig(max_rounds=models + 1, max_tokens=20, compactor=None)).start()
    try:
        if lose_notifications:
            async def lost(packet):
                return PublishReceipt(False)
            agent.runtime.transport.publish = lost
        for index in range(history):
            model.calls = 0
            assert (await agent.send(f"history {index}")).status == "completed"
            await agent.drain()
        # Hold notifications so the measured send is claimed exactly once (one batch),
        # then drive the worker task directly — no client wait polling in the profile.
        async def held(packet):
            return PublishReceipt(False)
        agent.runtime.transport.publish = held
        # Reset counters so only the measured send is counted.
        model.calls = 0
        profiled.records.clear()
        profiled.tx_sites.clear()
        profiled.tx_categories.clear()
        profiled.transactions = 0
        if meter:
            await meter.reset()
        started = time.monotonic()
        refs = []
        for i in range(backlog):
            body = f"item {i}" if backlog > 1 else "run"
            mid = f"b{i}" if backlog > 1 else "run"
            receipt = await agent.runtime.client.send(agent.target, body, message_id=mid, actor=agent.actor)
            refs.append(receipt.input_ref)
        await agent.runtime.worker.recover()
        await agent.drain(after_input=refs[0])
        measured_ref = refs[0]
        result = await agent.runtime.client.inputs.get_result(measured_ref, actor=agent.actor)
        elapsed_s = time.monotonic() - started
        records = list(profiled.records)
        transactions = profiled.transactions
        tx_sites = list(profiled.tx_sites)
        tx_categories = list(profiled.tx_categories)
        assert result.status == "completed" and model.calls == models
        writes = await meter.report() if meter else None
        if target := os.environ.get("P215_PROFILE_OUT"):
            with open(target, "a") as f:
                f.write(json.dumps({
                    "db": original.dialect.name, "models": models, "tools_per_round": tools_per_round,
                    "history": history, "backlog": backlog, "lose_notifications": lose_notifications,
                    "total_sql": len(records), "counts": _counts(records, tx_sites, tx_categories),
                    "transactions": transactions, "writes": writes, "elapsed_s": elapsed_s,
                }) + "\n")
        return records, transactions, tx_sites, elapsed_s, tx_categories
    finally:
        await agent.aclose()
        store._db = original
        if meter:
            await meter.close()


def _per_round_ceiling(db, tools_per_round):
    base = 7 if db == "mysql" else 5
    extra = 3 if db == "mysql" else 2
    return base + (tools_per_round - 1) * extra


@pytest.mark.parametrize("models", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("tools_per_round", [1, 4])
async def test_send_sql_profile(registry_store, models, tools_per_round):
    store, _ = registry_store
    db = store._db.dialect.name
    records, transactions, tx_sites, _, tx_categories = await _measure(
        store, models, tools_per_round=tools_per_round)
    counts = _counts(records, tx_sites, tx_categories)
    assert counts["legacy_job_control"] == 0, counts
    assert counts["billing"] == 0, counts
    # Per-round growth (model + tool round incl. the boundary inbox_version read) stays
    # within the frozen session-native budget; control SQL never references job tables.
    if models >= 2:
        one, *_ = await _measure(store, 1, tools_per_round=tools_per_round)
        # Marginal cost per additional round. The time-based session activity touch
        # (UPDATE sessions SET updated_at, <=1/sec of send duration) is periodic, not a
        # per-round control cost. Timer polling is independently scheduled too; both
        # remain counted in raw total SQL / periodic, outside the per-round marginal.
        def non_periodic(recs):
            return [r for r in recs if _category(r["chain"], r["sql"]) != "periodic"
                        and not re.search(r"UPDATE \S*sessions SET updated_at", r["sql"])]
        per_round = (len(non_periodic(records)) - len(non_periodic(one))) / (models - 1)
        assert per_round <= _per_round_ceiling(db, tools_per_round), (per_round, counts)


async def test_send_sql_independent_of_history(registry_store):
    store, _ = registry_store
    cold, *_ = await _measure(store, 2, history=0)
    warm, *_ = await _measure(store, 2, history=20)
    cold_counts = _counts(cold, [], None)
    warm_counts = _counts(warm, [], None)
    assert cold_counts["legacy_job_control"] == 0 and warm_counts["legacy_job_control"] == 0
    # Control SQL is independent of how many prior sends the session has. The entry history
    # read stays ONE query; only the number of rows it returns grows with history.
    assert warm_counts["session_control"] == cold_counts["session_control"], (cold_counts, warm_counts)

    def entry_rows(records):
        return sum(r.get("rows") or 0 for r in records
                   if "load_active_messages" in " ".join(r["chain"]) and r["sql"].upper().startswith("SELECT"))

    assert entry_rows(warm) > entry_rows(cold), (entry_rows(cold), entry_rows(warm))


@pytest.mark.parametrize("backlog", [1, 20])
async def test_batch_input_sql_profile(registry_store, backlog):
    store, _ = registry_store
    records, transactions, tx_sites, _, tx_categories = await _measure(store, 1, backlog=backlog)
    counts = _counts(records, tx_sites, tx_categories)
    assert counts["legacy_job_control"] == 0
    # A 20-input backlog is ONE claim batch / one model run (asserted in _measure via
    # model.calls==1). Each input is accepted in its own transaction, then one claim/final.
    claim_tx = sum(1 for chain in tx_sites if any("runtime_store:claim" in c for c in chain))
    assert claim_tx == 1, counts
    assert counts["session_control_tx"] <= backlog + 3, counts


async def test_notification_loss_sql_profile(registry_store):
    store, _ = registry_store
    records, transactions, tx_sites, _, tx_categories = await _measure(
        store, 2, lose_notifications=True)
    counts = _counts(records, tx_sites, tx_categories)
    assert counts["legacy_job_control"] == 0
    assert counts["billing"] == 0
