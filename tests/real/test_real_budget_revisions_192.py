"""A real original model response survives late evidence and ACK-loss recovery."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime import model_budgets
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, BudgetUsageEvidence, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.unit.test_budget_revisions_192 import RevisionHost

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


class RealRevisionHost(RevisionHost):
    async def prepare(self, context, logical_request, *, service):
        assert service is self.service
        return BudgetPlan(16384, {"attempt_id": context["attempt_id"], "origin": context})


async def test_real_original_response_is_not_replayed_when_late_receipt_ack_is_lost(tmp_path, record_property):
    path = str(tmp_path / "late.db")
    store = await SessionStore.open(path, schema="tasks")
    provider, host = BoundedProvider(limit=1), RealRevisionHost()
    host.service = provider
    config = BudgetConfig("real-original", (LocalBudgetScope("scope", max_calls=1),), host)
    def runtime(opened, name):
        return TaskRuntime(JobManager(opened), tenant_id="test", principal_id="host",
            executor_id=name, boot_id=name, boot_contract_hash="a" * 64, budget_config=config)
    original = runtime(store, "original")
    opened, controls = [], []
    try:
        async with asyncio.timeout(120):
            loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=original,
                config=AgentLoopConfig(delegation_policy="single_level", max_rounds=1, max_tokens=1024))
            sid = await loop.new_session()
            result = await loop.send("Reply only with the number 42.", sid)
            assert "42" in result.final_text and provider.calls == 1
            await original.shutdown(timeout_s=5)
            [row] = await store._db.fetchall(f"SELECT attempt_id,job_id FROM {store.table_prefix}task_budget_attempts")
            old = await model_budgets.get_budget(original.manager, original.actor(sid), row["attempt_id"])
            messages = await store.load_all_messages(sid)
            counts = await store._db.fetchall(f"SELECT * FROM {store.table_prefix}task_budget_scopes")
            # A trusted test collector confirms the captured original token facts.
            # No supplier invoice or final monetary price is invented here.
            evidence = BudgetUsageEvidence("confirmation", "test-collector", "captured-original-response",
                "actual", old["observation"]["usage"], details={"confirmation": "captured response verified"})
            host.lost_revision = 2
            for i in range(2):
                current_store = await SessionStore.open(path, schema="compatible")
                opened.append(current_store)
                control = runtime(current_store, "control-" + str(i))
                controls.append(control)
                await control.stage_budget_usage_revision(sid, **row, expected_revision=1, evidence=evidence)
                if i == 0:
                    with pytest.raises(OSError, match="ACK lost"):
                        await control.reconcile_budget_usage(sid, **row)
                    await control.shutdown(timeout_s=5)
                    host.lost_revision = None
                else:
                    ack = await control.reconcile_budget_usage(sid, **row)
                    assert ack["revision"] == 2 and ack["usage_acknowledged"]
                assert control._lifecycle._start_task is None
            assert len(host.revisions) == 2 and host.reserve_calls == provider.calls == 1
            assert await store.load_all_messages(sid) == messages
            assert await store._db.fetchall(f"SELECT * FROM {store.table_prefix}task_budget_scopes") == counts
    finally:
        provider.report(record_property)
        await original.shutdown(timeout_s=5)
        for control in controls:
            await control.shutdown(timeout_s=5)
        for current_store in opened:
            await current_store.close()
        await provider.close()
        await store.close()
