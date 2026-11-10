"""Real provider: a task-managed send folds its history as a fold model call.

The compact holds the real summary.
"""
from __future__ import annotations

import json

import pytest

from power_loop import AgentLoopConfig
from power_loop.runtime.fold import AgenticFold, LLMSummaryFold
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._llm import make_llm
from tests.unit.test_job_registry_192 import registry_store

__all__ = ['registry_store']

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("variant", ["llm_summary", "summary_model", "agentic"])
async def test_real_task_send_fold_is_booked_once_as_fold(tmp_path, variant, runtime_agent):
    store = await open_store(str(tmp_path / "fold.db"), table_prefix="real_fold_", schema="tasks")
    runtime = TaskRuntime(JobManager(store), tenant_id="real-fold", principal_id="host", executor_id="executor",
                          boot_id="boot", lease_ms=300000, boot_contract_hash="a" * 64)
    main = make_llm(max_tokens=2048, temperature=0.0)
    summary = make_llm(max_tokens=2048, temperature=0.0) if variant == "summary_model" else None
    fold = (AgenticFold(keep_last_sends=1, summary_max_tokens=1024, max_rounds=3) if variant == "agentic"
            else LLMSummaryFold(keep_last_sends=1, summary_max_tokens=1024, summary_llm=summary))
    loop = await runtime_agent(model=main, store=store, task_runtime=runtime, config=AgentLoopConfig(
        delegation_policy="single_level", system_prompt="You are terse. Reply with one short sentence.",
        max_rounds=3, max_tokens=2048, context_budget_tokens=40, model=main._cfg.model,
        representation=ProjectedRepresentation(), fold_strategy=fold))
    try:
        sid = loop.target.session_id
        for i in range(1, 5):
            result = await loop.send(f"Note the word apple{i} and reply OK.")
            assert result.status == "completed", result.status
        compact = await store.latest_project_compact(sid)
        assert compact is not None and (compact.content or {}).get("summary", "").strip()
        rows = await runtime.manager.db.fetchall(
            f"SELECT payload_json FROM {runtime.manager.events} WHERE event_type='model_attempt_prepared'")
        prepared = [json.loads(r["payload_json"]) for r in rows]
        folds = [p["attempt_id"] for p in prepared if p.get("purpose") == "fold"]
        assert folds and all(p["session_id"] == sid for p in prepared)
    finally:
        await loop.aclose()
        if summary is not None:
            await summary.close()
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
