"""Session-native whole-send SQL budget, attributed by call site (design 215 P2).

Replaces the pre-215 job-path profile; scenarios drive the accept/claim/final session worker.
Runtime-state INSERTs classify by their write target, including projection guards (C02-R1).
B = G0-R3 frozen baseline, S = P2 start, N = this HEAD. Set P215_PROFILE_OUT to emit one
JSON record per case for the B/S/N report. legacy_job_control must be 0; billing is 0 in
the SDK (the 214 charge event lives in the agent host, P6).
"""
from __future__ import annotations

import asyncio
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
    "sessions", "follow_up_queue", "session_runtime_state", "shared_state",
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
    "agent.runtime_store:_commit_result",
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
    if "runtime.store.process_guard:acquire" in chain and normalized in {
        "select pg_try_advisory_lock($1)", "select get_lock(%s,0)",
    }:
        return "session_control"
    if "runtime.store.process_guard:check" in chain and normalized == "select 1":
        return "periodic" if "agent.runtime_worker:_watch_guard" in chain else "session_control"
    if "agent.runtime_store:poll_notifications" in chain and re.match(r"\s*select\b", normalized):
        if re.search(r"\bfrom\s+(?:[a-z0-9_]+_)?follow_up_queue\b", normalized):
            return "periodic"
        if ("agent.runtime_store:discover_unfinished" in chain
                and re.search(r"\bfrom\s+(?:[a-z0-9_]+_)?sessions\b", normalized)):
            return "periodic"
    # append_message touches activity at most once per second. Metadata's initial
    # write also touches activity, so cold/warm sends can differ by this timed write.
    if ("runtime.store.store:append_message" in chain
            and re.match(r"update \S*sessions set updated_at", normalized)):
        return "periodic"
    if "agent.runtime_store:_db_now" in chain:
        return "session_control"
    # Accept's post-lock session/input/capacity/time join is a control operation,
    # including the input data it reads.
    if "agent.runtime_store:_accept_read" in chain:
        return "session_control"
    # Claim's post-lock batch read also joins the session identity. Attribute that
    # control read by call site; UPDATE/INSERT still classify by their own tables.
    if normalized.startswith("select q.id as id") and any(
        site in chain for site in ("agent.runtime_store:claim", "agent.runtime_store:claim_follow_up")
    ):
        return "session_control"
    if "agent.runtime_store:session_entry" in chain:
        return "session_control"
    # The merged result read spans sessions, the addressed input and its result
    # message. Attribute that control read by its live caller; the separate
    # continuation result message read below is still an ordinary message read.
    if ("agent.runtime_store:_result_state" in chain
            and "agent.runtime_store:_terminal_result" not in chain):
        return "session_control"
    if normalized.startswith("select last_insert_id()"):
        return "message"
    # A runtime-state INSERT may probe projections in NOT EXISTS. Classify the write
    # target before inspecting referenced message tables; window reads stay message SQL.
    if re.match(r"insert\s+into\s+[a-z0-9_]*session_runtime_state\s*\(", normalized):
        return "session_control"
    if any(re.search(rf"(?<![a-z0-9]){table}(?![a-z0-9])", normalized) for table in _MESSAGE_TABLES):
        return "message"
    if any(re.search(rf"(?<![a-z0-9]){table}(?![a-z0-9])", normalized) for table in _SESSION_TABLES):
        return "session_control"
    if sql:
        raise AssertionError(("unclassified SQL", sql, chain))
    if any(c.startswith(("agent.", "workflow.", "runtime.session_ownership", "runtime.store_session_ownership"))
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
    if tx_categories is None:
        tx_categories = [_category(chain) for chain in tx_sites]
    for category in tx_categories:
        if category is None:
            counts["unfinished_transactions"] = counts.get("unfinished_transactions", 0) + 1
        else:
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
        self.records.append({"sql": " ".join(sql.split()), "chain": _sites(), "rows": rows})

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

    def _reserve_transaction(self, site):
        self.transactions += 1
        index = len(self.tx_sites)
        self.tx_sites.append(site)
        # The index follows entry order even when exits overlap. A legacy window
        # reset may clear only tx_sites, so discard categories from that old window.
        self.tx_categories[index:] = [None]
        return index

    def _finish_transaction(self, index, site, measured):
        # Reset can reuse the index and even the same call chain in a new window.
        if index < len(self.tx_sites) and index < len(self.tx_categories) and self.tx_sites[index] is site:
            self.tx_categories[index] = self._transaction_category(
                measured.records if measured is not None else (), site,
            )

    @asynccontextmanager
    async def transaction(self):
        site = _sites()
        index = self._reserve_transaction(site)
        measured = None
        try:
            async with self.db.transaction() as tx:
                measured = _Tx(self, tx)
                yield measured
                if measured.before is not None:
                    after = await self.meter.totals(tx)
                    self.meter.max_tx_rows = max(self.meter.max_tx_rows, after[0] - measured.before[0])
                    self.meter.max_tx_bytes = max(self.meter.max_tx_bytes, after[1] - measured.before[1])
        finally:
            self._finish_transaction(index, site, measured)

    @asynccontextmanager
    async def read_snapshot(self):
        site = _sites()
        index = self._reserve_transaction(site)
        measured = None
        try:
            async with self.db.read_snapshot() as reader:
                measured = _Tx(self, reader)
                yield measured
        finally:
            self._finish_transaction(index, site, measured)

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
        self.records = []

    async def begin_meter(self, sql):
        # A PostgreSQL snapshot must set isolation before its first query.
        # Delay instrumentation until after transaction configuration.
        if self.owner.meter and self.before is None and not sql.lstrip().upper().startswith("SET TRANSACTION"):
            self.before = await self.owner.meter.totals(self.tx)

    def __getattr__(self, name):
        return getattr(self.tx, name)

    def note(self, sql, rows=None):
        self.owner.note(sql, rows)
        self.records.append(self.owner.records[-1])

    async def execute(self, sql, params=()):
        await self.begin_meter(sql)
        self.note(sql)
        return await self.tx.execute(sql, params)

    async def execute_packed(self, sql, params=()):
        await self.begin_meter(sql)
        self.note(sql)
        return await self.tx.execute_packed(sql, params)

    async def fetchone(self, sql, params=()):
        await self.begin_meter(sql)
        result = await self.tx.fetchone(sql, params)
        self.note(sql, 0 if result is None else 1)
        return result

    async def fetchall(self, sql, params=()):
        await self.begin_meter(sql)
        result = await self.tx.fetchall(sql, params)
        self.note(sql, len(result))
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
                   lose_notifications=False, fused=True):
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
    use_fused = bool(fused) and backlog == 1 and not lose_notifications
    agent = await RunningAgent(store, model, registry,
                               AgentLoopConfig(max_rounds=models + 1, max_tokens=20, context_budget_tokens=10**9),
                               fuse_local_send=use_fused).start()
    try:
        if lose_notifications:
            async def lost(packet):
                return PublishReceipt(False)
            agent.runtime.transport.publish = lost
        for index in range(history):
            model.calls = 0
            assert (await agent.send(f"history {index}")).status == "completed"
            await agent.drain()
        # Hold notifications on standard 3-tx/backlog runs so the measured send is
        # claimed in one deterministic batch without client wait polling.
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
        if not use_fused:
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
                    "fused": use_fused,
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
    # Design 217 Session-Core: +0 control SQL per tool round across SQLite/PG/MySQL;
    # business SQL is 1 assistant INSERT + tools_per_round tool INSERTs (last tool piggybacks boundary).
    _ = db
    return 1 + tools_per_round


async def test_frozen_fixed_send_budget(registry_store):
    """Design 217 M1c tightened fixed budget: 3-tx standard <= 10/10/11 (worker-only 8/8/9)."""
    store, _ = registry_store
    records, _, sites, _, _ = await _measure(store, 1, fused=False)
    fixed = sum(_category(record["chain"], record["sql"]) != "periodic" for record in records)
    maximum = 11 if store._db.dialect.name == "mysql" else 10
    control_transactions = sum(any(site in {
        "agent.runtime_store:accept", "agent.runtime_store:accept_and_claim",
        "agent.runtime_store:claim", "agent.runtime_store:final",
    } for site in chain) for chain in sites)
    assert control_transactions <= 3, control_transactions
    assert fixed <= maximum, {"fixed_sql": fixed, "frozen_maximum": maximum}


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
    if models >= 2:
        one, *_ = await _measure(store, 1, tools_per_round=tools_per_round)
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
    assert warm_counts["session_control"] == cold_counts["session_control"], (cold_counts, warm_counts)

    def entry_rows(records):
        return sum(r.get("rows") or 0 for r in records
                   if "load_projection_window" in " ".join(r["chain"])
                   and (r["sql"].upper().startswith("SELECT")
                        or r["sql"].upper().startswith("WITH COMPACT")))

    assert entry_rows(warm) > entry_rows(cold), (entry_rows(cold), entry_rows(warm))


@pytest.mark.parametrize("backlog", [1, 20])
async def test_batch_input_sql_profile(registry_store, backlog):
    store, _ = registry_store
    records, transactions, tx_sites, _, tx_categories = await _measure(
        store, 1, backlog=backlog, fused=False)
    counts = _counts(records, tx_sites, tx_categories)
    assert counts["legacy_job_control"] == 0
    claim_tx = sum(1 for chain in tx_sites if any("runtime_store:claim" in c for c in chain))
    assert claim_tx == 1, counts
    assert counts["session_control_tx"] <= backlog + 3, counts


async def test_notification_loss_sql_profile(registry_store):
    store, _ = registry_store
    records, transactions, tx_sites, _, tx_categories = await _measure(
        store, 2, lose_notifications=True, fused=False)
    counts = _counts(records, tx_sites, tx_categories)
    assert counts["legacy_job_control"] == 0
    assert counts["billing"] == 0


def _split_m1a_core_vs_m1b(records):
    """Separate M1a core control SQL from M1b projection_fold and test harness reads."""
    core_ctrl, m1b_ctrl, harness_ctrl, business = [], [], [], []
    for rec in records:
        cat = _category(rec["chain"], rec["sql"])
        chain_str = " ".join(rec["chain"])
        if cat == "message":
            business.append(rec)
        elif cat == "session_control":
            if "projection_fold:" in chain_str:
                m1b_ctrl.append(rec)
            elif "get_result" in chain_str or "read_result_state" in chain_str or "discover_unfinished" in chain_str:
                harness_ctrl.append(rec)
            else:
                core_ctrl.append(rec)
    return core_ctrl, m1b_ctrl, harness_ctrl, business


async def test_m1a_fused_2tx_sql_profile(registry_store):
    """Design 217 M1b fused 2-tx: core control SQL == 3/3/4, m1b_ctrl == 0, 0 KV SQL, 2 total txs, 3 biz SQL."""
    store, _ = registry_store
    db = store._db.dialect.name
    records, transactions, tx_sites, _, tx_categories = await _measure(store, 1, tools_per_round=0, fused=True)
    counts = _counts(records, tx_sites, tx_categories)
    core_ctrl, m1b_ctrl, _, business = _split_m1a_core_vs_m1b(records)
    core_txs = [
        chain for chain in tx_sites
        if any(s in {"agent.runtime_store:accept_and_claim", "agent.runtime_store:final"} for s in chain)
    ]
    kv_sql = [r for r in records if store.t.session_runtime_state in r["sql"]]
    proj_inserts = [r for r in business if f"INSERT INTO {store.t.project_messages}" in r["sql"]]
    assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
    assert len(kv_sql) == 0, kv_sql
    assert len(m1b_ctrl) == 0, m1b_ctrl
    assert transactions == 2 and len(core_txs) == 2, (transactions, tx_sites)
    expected_core = 4 if db == "mysql" else 3
    assert len(core_ctrl) == expected_core, {"core_ctrl": core_ctrl, "m1b_ctrl": m1b_ctrl, "business": business}
    assert len(business) == 3, business
    assert len(proj_inserts) == 1, proj_inserts


async def test_m1a_standard_3tx_sql_profile(registry_store):
    """Design 217 M1b standard 3-tx: core control SQL == 5/5/6, m1b_ctrl == 0, 0 KV SQL, 3 total txs, 3 biz SQL."""
    store, _ = registry_store
    db = store._db.dialect.name
    records, transactions, tx_sites, _, tx_categories = await _measure(store, 1, tools_per_round=0, fused=False)
    counts = _counts(records, tx_sites, tx_categories)
    core_ctrl, m1b_ctrl, _, business = _split_m1a_core_vs_m1b(records)
    core_txs = [
        chain for chain in tx_sites
        if any(s in {"agent.runtime_store:accept", "agent.runtime_store:claim", "agent.runtime_store:final"} for s in chain)
    ]
    kv_sql = [r for r in records if store.t.session_runtime_state in r["sql"]]
    assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
    assert len(kv_sql) == 0, kv_sql
    assert len(m1b_ctrl) == 0, m1b_ctrl
    assert transactions == 3 and len(core_txs) == 3, (transactions, tx_sites)
    expected_core = 6 if db == "mysql" else 5
    assert len(core_ctrl) == expected_core, {"core_ctrl": core_ctrl, "m1b_ctrl": m1b_ctrl, "business": business}
    assert len(business) == 3, business


async def test_m1b_warm_session_projection_read_plus_one_business_sql(registry_store):
    """Design 217 M1b warm session (history=20) adds +1 business SQL (load_projections) and +0 control/KV SQL."""
    store, _ = registry_store
    cold_recs, cold_txs, cold_sites, _, cold_cats = await _measure(store, 1, tools_per_round=0, history=0, fused=True)
    warm_recs, warm_txs, warm_sites, _, warm_cats = await _measure(store, 1, tools_per_round=0, history=20, fused=True)
    cold_core, cold_m1b, _, cold_biz = _split_m1a_core_vs_m1b(cold_recs)
    warm_core, warm_m1b, _, warm_biz = _split_m1a_core_vs_m1b(warm_recs)
    assert cold_txs == 2 and warm_txs == 2
    assert len(cold_m1b) == 0 and len(warm_m1b) == 0
    assert len(warm_core) == len(cold_core)
    assert len(warm_biz) - len(cold_biz) == 1, (cold_biz, warm_biz)
    assert sum(store.t.session_runtime_state in r["sql"] for r in warm_recs) == 0


@pytest.mark.parametrize("tools_per_round", [1, 4])
async def test_m1a_tool_round_zero_control_sql(registry_store, tools_per_round):
    """Design 217 M1a tool rounds add +0 control SQL per round across all 3 DBs."""
    store, _ = registry_store
    r1, _, tx1, _, tc1 = await _measure(store, 1, tools_per_round=tools_per_round, fused=True)
    r4, _, tx4, _, tc4 = await _measure(store, 4, tools_per_round=tools_per_round, fused=True)
    c1 = _counts(r1, tx1, tc1)
    c4 = _counts(r4, tx4, tc4)
    assert c4["session_control"] - c1["session_control"] == 0, (c1, c4)
    assert (c4["message"] - c1["message"]) == 3 * (1 + tools_per_round), (c1, c4)


async def test_m1a_periodic_node_sql_profile(registry_store):
    """Design 217 M1a node periodic tick: batch_renew + discover_queued(32) + sweep_expired_leases(32) == 3/3/4."""
    store, _ = registry_store
    db = store._db.dialect.name
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    model = _ProfileModel(1, 0)
    agent = await RunningAgent(store, model, ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        profiled.records.clear()
        profiled.tx_sites.clear()
        profiled.transactions = 0
        await agent.runtime.worker.periodic_tick(limit=32)
        expected = 4 if db == "mysql" else 3
        assert len(profiled.records) == expected, [r["sql"] for r in profiled.records]
    finally:
        await agent.aclose()
        store._db = original


# ── Design 217 M2: Genuine Cross-Process 5-Leaf Workflow SQL Ledgers ──────────


class _ShardedWakeBus:
    """Signal-only cross-node WakeTransport (carries ONLY (session_id, reason/cancel_by), 0 payload!).

    Routes Main + Driver signals to Node A and Leaf signals to Node B (matching sharded/routed
    pub-sub without sharing SessionStore, RuntimeInputStore, or in-memory result queues).
    """

    def __init__(self):
        self.node_a_sids: set[str] = {"000_main"}
        self._a_wake = None
        self._a_cancel = None
        self._b_wake = None
        self._b_cancel = None

    def endpoint(self, role: str):
        bus = self

        class _Ep:
            async def subscribe(self, *, on_wake, on_cancel=None) -> None:
                if role == "a":
                    bus._a_wake, bus._a_cancel = on_wake, on_cancel
                else:
                    bus._b_wake, bus._b_cancel = on_wake, on_cancel

            async def publish_wake(self, session_id: str, *, reason: str = "accept") -> None:
                # submit() runs on Node A for the workflow_driver session before leaves exist
                if role == "a" and reason == "accept" and len(bus.node_a_sids) == 1:
                    bus.node_a_sids.add(session_id)
                cb = bus._a_wake if session_id in bus.node_a_sids else bus._b_wake
                if cb is not None:
                    await cb(session_id, reason)

            async def publish_cancel(self, session_id: str, *, cancel_by: str = "user") -> None:
                cb = bus._a_cancel if session_id in bus.node_a_sids else bus._b_cancel
                if cb is not None:
                    await cb(session_id, cancel_by)

            async def poll_once(self, *, limit: int = 32) -> list[str]:
                return []

            async def aclose(self) -> None:
                return None

        return _Ep()
class _SharedLedgerProfiled(Profiled):
    """Wraps a second independent SessionStore._db so both processes append to ONE SQL ledger."""

    def __init__(self, db, master: Profiled):
        super().__init__(db, master.meter)
        self.master = master
        self.records = master.records
        self.tx_sites = master.tx_sites
        self.tx_categories = master.tx_categories

    def _reserve_transaction(self, site):
        self.master.transactions += 1
        self.transactions = self.master.transactions
        index = len(self.tx_sites)
        self.tx_sites.append(site)
        self.tx_categories[index:] = [None]
        return index


@asynccontextmanager
async def _cross_process_workflow_cluster(store_a, connect, model_a, model_b, *, leaf_capacity=5):
    """Two independent SessionStore connections + two independent Runtimes (Node A: Main+Driver, Node B: Leaves)."""
    from power_loop import (
        DatabasePollingTransport,
        ExecutionEnvironment,
        ProjectedRepresentation,
        RegistryRuntimeFactory,
        Runtime,
    )
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import RuntimeDescriptor, TrustedActor
    from power_loop.workflow import register_workflow_tool

    store_b = await connect()
    orig_a, orig_b = store_a._db, store_b._db
    master = Profiled(orig_a)
    peer = _SharedLedgerProfiled(orig_b, master)
    wake_bus = _ShardedWakeBus()

    reg_a = ToolRegistry()
    register_workflow_tool(reg_a)
    reg_b = ToolRegistry()

    cfg_a = AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation())
    cfg_b = AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation())

    fac_a = RegistryRuntimeFactory()
    fac_b = RegistryRuntimeFactory()

    def can_exec_a(desc):
        if desc.kind == "workflow_driver":
            return True
        return desc.kind == "agent" and "workflow_node" not in (desc.config or {})

    def can_exec_b(desc):
        return desc.kind == "agent" and "workflow_node" in (desc.config or {})

    async def build_a(desc, ctx):
        return ExecutionEnvironment(model_a, reg_a, cfg_a)

    async def build_b(desc, ctx):
        return ExecutionEnvironment(model_b, reg_b, cfg_b)

    fac_a.register("m2.agent@215", build_a, can_execute=can_exec_a)
    fac_b.register("m2.agent@215", build_b, can_execute=can_exec_b)

    # Separate DatabasePollingTransport per node (interval=3600s so 0 background poll SQL);
    # cross-node wakes/cancels travel strictly over signal-only WakeTransport.
    tp_a = DatabasePollingTransport(RuntimeInputStore(store_a), interval_s=3600.0)
    tp_b = DatabasePollingTransport(RuntimeInputStore(store_b), interval_s=3600.0)
    rt_a = Runtime(
        store=store_a, factory=fac_a, node_id="node-a", boot_id="boot-a",
        capacity=2, transport=tp_a, wake_transport=wake_bus.endpoint("a"),
        session_lease_ttl_s=300.0, recovery_interval_s=3600.0,
        shared_execution=True, timers_enabled=False,
    )
    rt_b = Runtime(
        store=store_b, factory=fac_b, node_id="node-b", boot_id="boot-b",
        capacity=leaf_capacity, transport=tp_b, wake_transport=wake_bus.endpoint("b"),
        session_lease_ttl_s=300.0, recovery_interval_s=3600.0,
        shared_execution=True, timers_enabled=False,
    )
    await rt_a.worker.start()
    await rt_b.worker.start()

    actor = TrustedActor("tenant-m2", "user-m2", None)
    desc = RuntimeDescriptor(
        2, "agent", "m2.agent@215", None, "m2.tools@215", "m2.hooks@215", "m2.res@215", {}, {},
    )
    # Create Main session with '000_main' < 'ses_...' so Main < Driver < Leaf IDs (verifies 0 final_lock_asc!).
    main_target = await rt_a.client.create_session(desc, actor=actor, options={"session_id": "000_main"})

    # Attach profiler AFTER startup & session seed so we measure exact workflow + leaf + parent ingestion SQL!
    store_a._db = master
    store_b._db = peer
    master.records.clear()
    master.tx_sites.clear()
    master.tx_categories.clear()
    master.transactions = 0
    try:
        yield rt_a, rt_b, main_target, actor, desc, master
    finally:
        store_a._db = orig_a
        store_b._db = orig_b
        await rt_b.worker.shutdown(timeout_s=5, cancel_running=True)
        await rt_b.client.aclose()
        await tp_b.aclose()
        await rt_a.worker.shutdown(timeout_s=5, cancel_running=True)
        await rt_a.client.aclose()
        await tp_a.aclose()


async def test_m2_cross_process_5_leaf_workflow_sql_profile(registry_store):
    """Real 2-store / 2-worker cross-process 5-leaf parallel Workflow SQL ledger across all 3 DBs."""
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef, result_message_id

    store_a, connect = registry_store
    db = store_a._db.dialect.name
    leaf_started = [asyncio.Event() for _ in range(5)]
    release_leaves = asyncio.Event()

    class _MainModel(LLMService):
        def __init__(self):
            self.calls = 0

        async def complete(self, req, **kwargs):
            self.calls += 1
            return LLMResponse(
                raw_text="main handled workflow",
                token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        async def close(self):
            pass

    class _LeafModel(LLMService):
        def __init__(self):
            self.calls = 0

        async def complete(self, req, **kwargs):
            idx = int(req.system_prompt.splitlines()[0].split("-")[1])
            self.calls += 1
            leaf_started[idx].set()
            await release_leaves.wait()
            return LLMResponse(
                raw_text=f"leaf-{idx}-ok",
                token_usage=LLMTokenUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            )

        async def close(self):
            pass

    model_a, model_b = _MainModel(), _LeafModel()
    branches = [
        {
            "type": "agent", "id": f"L{i}", "input": f"work-{i}",
            "spec": {"name": f"L{i}", "system_prompt": f"LEAF-{i}", "tools": []},
        }
        for i in range(5)
    ]
    spec = {
        "name": "five-leaf-xproc",
        "root": {"type": "parallel", "max_concurrency": 5, "branches": branches},
    }

    async with _cross_process_workflow_cluster(store_a, connect, model_a, model_b) as (
        rt_a, rt_b, main_target, actor, desc, master,
    ):
        # Verify Node A and Node B do NOT share local waiter maps!
        assert rt_a.worker.inputs is not rt_b.worker.inputs
        assert rt_a.worker.inputs._driver_waiters is not rt_b.worker.inputs._driver_waiters

        handle = await rt_a.client.workflow(
            main_target, spec, message_id="wf-5leaf", actor=actor, descriptor=desc,
        )
        await asyncio.wait_for(asyncio.gather(*(ev.wait() for ev in leaf_started)), 10)
        # Release all 5 remote leaves together so they commit River results on Node B and wake Driver on Node A
        release_leaves.set()
        wf_res = await handle.wait(deadline_ms=now_ms() + 15000)
        assert wf_res.status == "completed" and len(wf_res.results) == 5
        parent_ref = InputRef(
            store_a.store_id, main_target.session_id,
            result_message_id(handle.target, handle.input_ref, 1, "workflow_result"),
        )
        main_res = await rt_a.client.wait(parent_ref, actor=actor, deadline_ms=now_ms() + 15000)
        assert main_res.status == "completed" and main_res.final_text == "main handled workflow"
        # Wait for both workers to settle all tasks
        async with asyncio.timeout(5):
            while rt_a.worker._active or rt_b.worker._active:
                await asyncio.sleep(0.01)

        records = [r for r in master.records if _category(r["chain"], r["sql"]) != "periodic"]
        counts = _counts(records, master.tx_sites, master.tx_categories)
        assert counts["legacy_job_control"] == 0, counts
        assert counts["billing"] == 0, counts
        assert counts.get("unfinished_transactions", 0) == 0, counts
        assert model_b.calls == 5 and model_a.calls == 1
        # Verify 0 ascending parent+child FOR UPDATE lock SQL (deep-first River!)
        assert not any("_lock_result_sessions" in c for r in records for c in r["chain"])
        # Verify start_input_message_id settled with non-null result_json
        start_info = await rt_a.worker.inputs.db.fetchone(
            f"SELECT terminal_status,result_json,completed_at FROM {store_a.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (handle.target.session_id, handle.input_ref.message_id),
        )
        assert start_info["terminal_status"] == "completed"
        assert start_info["result_json"] is not None and start_info["completed_at"] is not None

        # Exact cross-process 5-leaf Workflow + Main ingestion turn:
        # 18 explicit txs = 14 (M0 baseline: 1 submit + 1 driver claim + 1 create_many + 5*(leaf claim + leaf final)
        #                     + 1 driver final) + 2 (workflow node/run journal txs: put_many + _consume_results)
        #                     + 2 (Main ingestion turn: claim + final)
        assert master.transactions == 18, (db, master.transactions, master.tx_sites)
        ctrl_limit = 63 if db == "mysql" else 55
        total_limit = 78 if db == "mysql" else 70
        assert counts["session_control"] <= ctrl_limit, (db, counts, [r["sql"] for r in records])
        assert len(records) <= total_limit, (db, len(records), counts)
        if out := os.environ.get("M2_WORKFLOW_LEDGER_OUT"):
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "case": "5_leaf_xproc_normal",
                    "db": db,
                    "transactions": master.transactions,
                    "session_control": counts["session_control"],
                    "message": counts["message"],
                    "total": len(records),
                }) + "\n")


async def test_m2_cross_process_5_leaf_mid_stop_one_leaf_sql_profile(registry_store):
    """Cross-process 5-leaf Workflow where user stops 1 leaf mid-run (scope='self', cancel_by='user')."""
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef, TargetRef, result_message_id

    store_a, connect = registry_store
    db = store_a._db.dialect.name
    leaf_started = [asyncio.Event() for _ in range(5)]
    release_leaf_0 = asyncio.Event()
    release_others = asyncio.Event()

    class _MainModel(LLMService):
        async def complete(self, req, **kwargs):
            return LLMResponse(raw_text="main handled partial workflow")

        async def close(self):
            pass

    class _LeafModel(LLMService):
        async def complete(self, req, **kwargs):
            idx = int(req.system_prompt.splitlines()[0].split("-")[1])
            leaf_started[idx].set()
            if idx == 0:
                await release_leaf_0.wait()
            else:
                await release_others.wait()
            return LLMResponse(raw_text=f"leaf-{idx}-ok")

        async def close(self):
            pass

    branches = [
        {
            "type": "agent", "id": f"L{i}", "input": f"work-{i}",
            "spec": {"name": f"L{i}", "system_prompt": f"LEAF-{i}", "tools": []},
        }
        for i in range(5)
    ]
    spec = {
        "name": "five-leaf-mid-stop",
        "root": {"type": "parallel", "max_concurrency": 5, "on_error": "continue", "branches": branches},
    }

    async with _cross_process_workflow_cluster(store_a, connect, _MainModel(), _LeafModel()) as (
        rt_a, rt_b, main_target, actor, desc, master,
    ):
        handle = await rt_a.client.workflow(
            main_target, spec, message_id="wf-midstop", actor=actor, descriptor=desc,
        )
        await asyncio.wait_for(asyncio.gather(*(ev.wait() for ev in leaf_started)), 10)
        children = await rt_a.client.list_children(handle.target, actor=actor)
        assert len(children.items) == 5
        l0_row = await rt_a.worker.inputs.db.fetchone(
            f"SELECT session_id FROM {store_a.t.sessions} "
            "WHERE parent_session_id=? AND descriptor_json LIKE '%\"node_id\":\"L0\"%'",
            (handle.target.session_id,),
        )
        l0_target = TargetRef(store_a.store_id, l0_row["session_id"])
        stop_rcpt = await rt_a.client.cancel(l0_target, actor=actor, scope="self", reason="user stopped L0")
        assert stop_rcpt.state == "cancel_requested"
        release_leaf_0.set()
        release_others.set()
        wf_res = await handle.wait(deadline_ms=now_ms() + 15000)
        assert wf_res.results["L0"].status == "cancelled"
        assert wf_res.results["L0"].cancel_by == "user"
        for i in range(1, 5):
            assert wf_res.results[f"L{i}"].status == "completed"
        parent_ref = InputRef(
            store_a.store_id, main_target.session_id,
            result_message_id(handle.target, handle.input_ref, 1, "workflow_result"),
        )
        await rt_a.client.wait(parent_ref, actor=actor, deadline_ms=now_ms() + 15000)
        async with asyncio.timeout(5):
            while rt_a.worker._active or rt_b.worker._active:
                await asyncio.sleep(0.01)

        records = [r for r in master.records if _category(r["chain"], r["sql"]) != "periodic"]
        counts = _counts(records, master.tx_sites, master.tx_categories)
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        assert counts.get("unfinished_transactions", 0) == 0
        notice = await rt_a.worker.inputs.db.fetchone(
            f"SELECT wake,kind FROM {store_a.t.follow_up_queue} "
            "WHERE session_id=? AND kind='system_notice'",
            (main_target.session_id,),
        )
        assert notice is not None and int(notice["wake"]) == 0
        if out := os.environ.get("M2_WORKFLOW_LEDGER_OUT"):
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "case": "5_leaf_xproc_mid_stop_1_leaf",
                    "db": db,
                    "transactions": master.transactions,
                    "session_control": counts["session_control"],
                    "message": counts["message"],
                    "total": len(records),
                }) + "\n")


async def test_m2_cross_process_5_leaf_subtree_stop_and_halt_sql_profile(registry_store):
    """Cross-process stop(driver, scope='subtree') (0 dead-wait, wake=1 to Main) + halt (wake=0)."""
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef, result_message_id

    store_a, connect = registry_store
    db = store_a._db.dialect.name
    leaf_started = [asyncio.Event() for _ in range(5)]
    release_leaves = asyncio.Event()

    class _MainModel(LLMService):
        async def complete(self, req, **kwargs):
            return LLMResponse(raw_text="main handled cancelled workflow")

        async def close(self):
            pass

    class _LeafModel(LLMService):
        async def complete(self, req, **kwargs):
            idx = int(req.system_prompt.splitlines()[0].split("-")[1])
            leaf_started[idx].set()
            await release_leaves.wait()
            return LLMResponse(raw_text=f"leaf-{idx}-late")

        async def close(self):
            pass

    branches = [
        {
            "type": "agent", "id": f"L{i}", "input": f"work-{i}",
            "spec": {"name": f"L{i}", "system_prompt": f"LEAF-{i}", "tools": []},
        }
        for i in range(5)
    ]
    spec = {
        "name": "five-leaf-subtree-stop",
        "root": {"type": "parallel", "max_concurrency": 5, "branches": branches},
    }

    async with _cross_process_workflow_cluster(store_a, connect, _MainModel(), _LeafModel()) as (
        rt_a, rt_b, main_target, actor, desc, master,
    ):
        handle = await rt_a.client.workflow(
            main_target, spec, message_id="wf-substop", actor=actor, descriptor=desc,
        )
        await asyncio.wait_for(asyncio.gather(*(ev.wait() for ev in leaf_started)), 10)
        # Do NOT release leaves yet! Verify stop(driver, scope='subtree') finishes with 0 dead-wait!
        rcpt = await rt_a.client.cancel(
            handle.target, actor=actor, scope="subtree", reason="abort workflow subtree",
        )
        assert rcpt.state == "cancel_requested"
        wf_res = await handle.wait(deadline_ms=now_ms() + 5000)
        assert wf_res.status == "cancelled" and wf_res.cancel_by == "user"
        parent_ref = InputRef(
            store_a.store_id, main_target.session_id,
            result_message_id(handle.target, handle.input_ref, 1, "workflow_result"),
        )
        main_res = await rt_a.client.wait(parent_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert main_res.status == "completed"

        # Now let all 5 remote leaves finish and confirm 0 ghost re-wake of finished Driver!
        release_leaves.set()
        async with asyncio.timeout(10):
            while rt_a.worker._active or rt_b.worker._active:
                await asyncio.sleep(0.01)

        drv_row = await rt_a.worker.inputs.db.fetchone(
            f"SELECT execution_state,send_index,wake_pending_count FROM {store_a.t.sessions} "
            "WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert dict(drv_row) == {
            "execution_state": "idle", "send_index": 1, "wake_pending_count": 0,
        }
        late_rows = await rt_a.worker.inputs.db.fetchall(
            f"SELECT wake,content_json FROM {store_a.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_node_result'",
            (handle.target.session_id,),
        )
        assert len(late_rows) == 5
        assert all(int(r["wake"]) == 0 for r in late_rows)
        assert all(json.loads(r["content_json"])["cancel_by"] == "subtree" for r in late_rows)

        records = [r for r in master.records if _category(r["chain"], r["sql"]) != "periodic"]
        counts = _counts(records, master.tx_sites, master.tx_categories)
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        assert counts.get("unfinished_transactions", 0) == 0
        if out := os.environ.get("M2_WORKFLOW_LEDGER_OUT"):
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "case": "5_leaf_xproc_stop_driver_subtree",
                    "db": db,
                    "transactions": master.transactions,
                    "session_control": counts["session_control"],
                    "message": counts["message"],
                    "total": len(records),
                }) + "\n")


async def test_m2_cross_process_5_leaf_halt_sql_profile(registry_store):
    """Cross-process 5-leaf on_error='halt': failing leaf L0 halts 4 slow siblings (cancel_by='subtree', wake=0)."""
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef, result_message_id

    store_a, connect = registry_store
    db = store_a._db.dialect.name
    siblings_started = [asyncio.Event() for _ in range(4)]
    release_siblings = asyncio.Event()

    class _MainModel(LLMService):
        async def complete(self, req, **kwargs):
            return LLMResponse(raw_text="main handled halted workflow")

        async def close(self):
            pass

    class _LeafModel(LLMService):
        async def complete(self, req, **kwargs):
            idx = int(req.system_prompt.splitlines()[0].split("-")[1])
            if idx == 0:
                await asyncio.gather(*(ev.wait() for ev in siblings_started))
                raise RuntimeError("L0 deliberate halt failure")
            siblings_started[idx - 1].set()
            await release_siblings.wait()
            return LLMResponse(raw_text=f"leaf-{idx}-late")

        async def close(self):
            pass

    branches = [
        {
            "type": "agent", "id": f"L{i}", "input": f"work-{i}",
            "spec": {"name": f"L{i}", "system_prompt": f"LEAF-{i}", "tools": []},
        }
        for i in range(5)
    ]
    spec = {
        "name": "five-leaf-real-halt",
        "root": {"type": "parallel", "on_error": "halt", "max_concurrency": 5, "branches": branches},
    }

    async with _cross_process_workflow_cluster(store_a, connect, _MainModel(), _LeafModel()) as (
        rt_a, rt_b, main_target, actor, desc, master,
    ):
        handle = await rt_a.client.workflow(
            main_target, spec, message_id="wf-real-halt", actor=actor, descriptor=desc,
        )
        # Failing leaf L0 triggers real on_error='halt' while L1..L4 are still blocked (0 dead-wait!)
        wf_res = await handle.wait(deadline_ms=now_ms() + 10000)
        assert wf_res.status == "failed"
        parent_ref = InputRef(
            store_a.store_id, main_target.session_id,
            result_message_id(handle.target, handle.input_ref, 1, "workflow_result"),
        )
        main_res = await rt_a.client.wait(parent_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert main_res.status == "completed"

        # Now release slow siblings L1..L4 and confirm cancel_by='subtree', wake=0, 0 ghost re-wake!
        release_siblings.set()
        async with asyncio.timeout(10):
            while rt_a.worker._active or rt_b.worker._active:
                await asyncio.sleep(0.01)

        drv_row = await rt_a.worker.inputs.db.fetchone(
            f"SELECT execution_state,send_index,wake_pending_count FROM {store_a.t.sessions} "
            "WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert dict(drv_row) == {
            "execution_state": "idle", "send_index": 1, "wake_pending_count": 0,
        }
        late_rows = await rt_a.worker.inputs.db.fetchall(
            f"SELECT wake,content_json FROM {store_a.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_node_result' AND status='pending'",
            (handle.target.session_id,),
        )
        assert len(late_rows) == 4
        assert all(int(r["wake"]) == 0 for r in late_rows)
        assert all(json.loads(r["content_json"])["cancel_by"] == "subtree" for r in late_rows)

        records = [r for r in master.records if _category(r["chain"], r["sql"]) != "periodic"]
        counts = _counts(records, master.tx_sites, master.tx_categories)
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        assert counts.get("unfinished_transactions", 0) == 0
        if out := os.environ.get("M2_WORKFLOW_LEDGER_OUT"):
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "case": "5_leaf_xproc_real_halt",
                    "db": db,
                    "transactions": master.transactions,
                    "session_control": counts["session_control"],
                    "message": counts["message"],
                    "total": len(records),
                }) + "\n")
