"""Real provider: a task-managed send folds its history through its own model-call booking.

Each fold call is one prepared attempt with purpose "fold", reserved and settled once by the
host budget, made on the configured summary model; the compact holds the real summary.
"""
from __future__ import annotations

import json
from collections import Counter

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime.budget_contracts import (
    BudgetAck,
    BudgetConfig,
    BudgetGrant,
    BudgetPlan,
    BudgetSubmission,
    LocalBudgetScope,
)
from power_loop.runtime.fold import AgenticFold, LLMSummaryFold
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._llm import make_llm
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "managed_workflow"]

pytestmark = pytest.mark.asyncio


class RecordingHost:
    """Host budget that records which service and purpose each reservation was for."""

    def __init__(self):
        self.prepared, self.grants, self.applied = [], {}, Counter()

    async def prepare(self, context, logical_request, *, service, purpose=None):
        self.prepared.append((context["attempt_id"], service, purpose))
        return BudgetPlan(200000, {"attempt_id": context["attempt_id"], "origin": context})

    async def reserve(self, plan):
        key = plan.request["attempt_id"]
        return self.grants.setdefault(key, BudgetGrant(key, plan.request_hash, 1, "reserved", {"original": key}))

    async def lookup(self, plan):
        return self.grants.get(plan.request["attempt_id"])

    async def prepare_usage(self, context, grant, observation):
        return BudgetSubmission(context["attempt_id"], 1, {"attempt_id": context["attempt_id"], "evidence": observation})

    async def apply_usage(self, submission):
        self.applied[submission.attempt_id] += 1
        return BudgetAck(submission.attempt_id, submission.revision, submission.request_hash, {"recorded": True})

    async def prepare_release(self, context, grant, proof):
        return BudgetSubmission(context["attempt_id"], 1, {"attempt_id": context["attempt_id"], **proof})

    async def release(self, submission):
        return BudgetAck(submission.attempt_id, submission.revision, submission.request_hash, {"released": True})


@pytest.mark.parametrize("variant", ["llm_summary", "summary_model", "agentic"])
async def test_real_task_send_fold_is_booked_once_as_fold(tmp_path, variant):
    store = await open_store(str(tmp_path / "fold.db"), table_prefix="real_fold_", schema="tasks")
    host = RecordingHost()
    runtime = TaskRuntime(JobManager(store), tenant_id="real-fold", principal_id="host", executor_id="executor",
                          boot_id="boot", lease_ms=300000, boot_contract_hash="a" * 64,
                          budget_config=BudgetConfig("real-fold-host", (LocalBudgetScope("all", max_inflight=2),),
                                                     host, max_pending=4, timeout_s=30))
    main = make_llm(max_tokens=2048, temperature=0.0)
    summary = make_llm(max_tokens=2048, temperature=0.0) if variant == "summary_model" else None
    fold = (AgenticFold(keep_last_sends=1, summary_max_tokens=1024, max_rounds=3) if variant == "agentic"
            else LLMSummaryFold(keep_last_sends=1, summary_max_tokens=1024, summary_llm=summary))
    loop = StatefulAgentLoop(llm=main, store=store, task_runtime=runtime, config=AgentLoopConfig(
        delegation_policy="single_level", system_prompt="You are terse. Reply with one short sentence.",
        max_rounds=3, max_tokens=2048, context_budget_tokens=40, model=main._cfg.model,
        representation=ProjectedRepresentation(), fold_strategy=fold))
    try:
        sid = await loop.new_session()
        for i in range(1, 5):
            result = await loop.send(f"Note the word apple{i} and reply OK.", session_id=sid)
            assert result.status == "completed", result.status
        compact = await store.latest_project_compact(sid)
        assert compact is not None and (compact.content or {}).get("summary", "").strip()
        rows = await runtime.manager.db.fetchall(
            f"SELECT payload_json FROM {runtime.manager.events} WHERE event_type='model_attempt_prepared'")
        prepared = [json.loads(r["payload_json"]) for r in rows]
        folds = [p["attempt_id"] for p in prepared if p.get("purpose") == "fold"]
        assert folds and all(p["session_id"] == sid for p in prepared)
        booked = {attempt: (service, purpose) for attempt, service, purpose in host.prepared}
        assert set(booked) == {p["attempt_id"] for p in prepared}
        assert {booked[a][1] for a in folds} == {"fold"}
        assert all(purpose is None for a, (_, purpose) in booked.items() if a not in folds)
        assert {booked[a][0] for a in folds} == {summary or main}
        assert set(host.applied) == set(booked) and set(host.applied.values()) == {1}
    finally:
        await store.close()


def _real_positions(monkeypatch):
    from tests.unit import test_fold_calls_positions_192 as positions
    real = make_llm(max_tokens=1024, temperature=0.0)
    calls = {"n": 0}
    original = real.complete

    async def counted(request, **kwargs):
        calls["n"] += 1
        return await original(request, **kwargs)

    real.complete = counted
    monkeypatch.setattr(positions, "Summarizer", lambda: _Counted(real, calls))
    return positions


class _Counted:
    """Real summary model whose call count the position tests compare with booked attempts."""

    def __init__(self, real, calls):
        self._real, self._calls = real, calls

    def __getattr__(self, name):
        return getattr(self._real, name)

    @property
    def calls(self):
        return self._calls["n"]


async def test_real_summary_model_folds_subagent_repair_send(registry_store, monkeypatch):
    positions = _real_positions(monkeypatch)
    await positions.test_subagent_schema_repair_send_folds_under_the_child_session(registry_store)


async def test_real_summary_model_folds_workflow_node_second_send(managed_workflow, monkeypatch):
    positions = _real_positions(monkeypatch)
    await positions.test_workflow_node_second_send_folds_under_the_node_session(managed_workflow)
