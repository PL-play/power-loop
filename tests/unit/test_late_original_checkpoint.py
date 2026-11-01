"""A late original result survives process death without replacing financial facts."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from power_loop.runtime import model_attempts, model_budgets
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_budget_initial_evidence_192 import InitialHost, assert_counts, initial_case
from .test_budget_reconcile_192 import reconcile, reopened
from .test_budget_revisions_192 import evidence, stage
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import start

__all__ = ["registry_store", "initial_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def case(store, *, cold=False):
    host = InitialHost()
    scopes = tuple(LocalBudgetScope(key, max_calls=2, max_tokens=200, max_inflight=1, binding=key)
                   for key in ("authority", "root", "owner"))
    config = BudgetConfig("host-budget-v1", scopes, host, max_pending=1)
    runtime = TaskRuntime(JobManager(store), tenant_id="budget-tenant", principal_id="host",
        executor_id="reconciler" if cold else "executor", boot_id="new-boot" if cold else "boot",
        boot_contract_hash="a" * 64, budget_config=config)
    return SimpleNamespace(store=store, host=host, config=config, runtime=runtime, manager=runtime.manager)


@pytest.mark.parametrize("returned", [True, False])
async def test_independent_controller_cannot_make_original_late_return_overwrite_its_charge(initial_case, returned):
    from power_loop.runtime.control_runtime import flush_binding, retained_owner

    c = initial_case
    attempt = await start(c)
    controller = await reopened(c)
    try:
        await stage(c, attempt, evidence(12), expected=0, runtime=controller)
        await reconcile(controller, c, attempt)
        # The original process still has its old RAM head: the other manager
        # cannot evict this process's map when it commits a correction.
        assert retained_owner(attempt.binding)
        before = await model_budgets.get_budget(controller.manager, c.actor, attempt.attempt_id)
        await model_attempts.observe(attempt, usage={"total_tokens": 7}, estimated=not returned,
                                     error=None if returned else OSError("original outcome unknown"))
        original = await model_attempts._read(c.manager, c.manager.db, "finished", attempt.attempt_id)
        assert original[1]["usage"] == {"total_tokens": 7}
        assert original[1]["status"] == ("returned" if returned else "unknown")
        await flush_binding(attempt.binding, close=True)
        first = await reconcile(controller, c, attempt)
        assert await reconcile(controller, c, attempt) == first
        after = await model_budgets.get_budget(controller.manager, c.actor, attempt.attempt_id)
        assert after["submission"] == before["submission"] and after["ack"] == before["ack"]
        assert after["observation"] == original[1]
        await assert_counts(c, tokens=12, inflight=0 if returned else 1)
        assert c.host.reserve_calls == c.host.apply_calls == len(c.host.revisions) == 1
        assert controller._lifecycle._start_task is None
    finally:
        await controller.shutdown(timeout_s=1)


@pytest.mark.parametrize("fault", ["rollback", "ack_lost"])
async def test_late_original_commit_retry_keeps_one_response_and_one_charge(initial_case, monkeypatch, fault):
    c = initial_case
    attempt = await start(c)
    controller = await reopened(c)
    try:
        await stage(c, attempt, evidence(12), expected=0, runtime=controller)
        await reconcile(controller, c, attempt)
        transaction = c.manager.db.transaction

        @asynccontextmanager
        async def fail_commit():
            async with transaction() as tx:
                yield tx
                if fault == "rollback":
                    raise OSError("late original rolled back")
            raise OSError("late original committed, ACK lost")

        with monkeypatch.context() as patch:
            patch.setattr(c.manager.db, "transaction", fail_commit)
            with pytest.raises(OSError, match="late original"):
                await model_attempts.finish(attempt, status="returned", usage={"total_tokens": 7}, estimated=False)
        recorded = await model_attempts._read(c.manager, c.manager.db, "finished", attempt.attempt_id, required=False)
        assert (recorded is not None) == (fault == "ack_lost")
        value = await model_attempts.finish(attempt, status="returned", usage={"total_tokens": 7}, estimated=False)
        assert await model_attempts.finish(attempt, status="returned", usage={"total_tokens": 7}, estimated=False) == value
        await model_budgets.observe(attempt, value)
        first = await reconcile(controller, c, attempt)
        assert await reconcile(controller, c, attempt) == first
        records = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_finished'")
        assert len(records) == 1
        await assert_counts(c, tokens=12, inflight=0)
        assert c.host.reserve_calls == c.host.apply_calls == len(c.host.revisions) == 1
    finally:
        await controller.shutdown(timeout_s=1)


@pytest.mark.parametrize("point", ["after_original", "during_projection"])
async def test_process_death_preserves_late_return_and_original_settled_charge(registry_store, tmp_path, point):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "late.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._late_original_crash_worker",
        dsn, store.table_prefix, str(evidence), point, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == (94 if point == "after_original" else 95), output.decode()
    saved = json.loads(evidence.read_text())
    c = case(await connect(), cold=True)
    actor = c.runtime.actor(saved["sid"])
    try:
        original = await model_attempts._read(c.manager, c.manager.db, "finished", saved["attempt_id"])
        assert original[1]["status"] == "returned" and original[1]["usage"] == {"total_tokens": 7}
        before = await model_budgets.get_budget(c.manager, actor, saved["attempt_id"])
        assert before["observation"] is None and before["ack"] == saved["ack"]
        counters = await c.manager.db.fetchall(f"SELECT counters_json FROM {c.manager.prefix}task_budget_scopes")
        assert len(counters) == 3
        assert all(json.loads(row["counters_json"])["inflight"] == 1 for row in counters)
        first = await c.runtime.reconcile_budget_usage(saved["sid"], job_id=saved["job_id"], attempt_id=saved["attempt_id"])
        repeated = await c.runtime.reconcile_budget_usage(saved["sid"], job_id=saved["job_id"], attempt_id=saved["attempt_id"])
        assert first == repeated and first["usage_acknowledged"] and not first["execution_authority"]
        after = await model_budgets.get_budget(c.manager, actor, saved["attempt_id"])
        assert after["observation"]["usage"]["total_tokens"] == 7
        assert after["submission"] == saved["submission"] and after["ack"] == saved["ack"]
        counters = await c.manager.db.fetchall(f"SELECT counters_json FROM {c.manager.prefix}task_budget_scopes")
        assert all(json.loads(row["counters_json"]) == dict(used_calls=1, held_calls=0,
            used_tokens=12, held_tokens=0, inflight=0) for row in counters)
        assert saved["reserve_calls"] == saved["apply_calls"] == 1
        assert c.host.reserve_calls == c.host.apply_calls == 0 and not c.host.revisions
        assert c.runtime._lifecycle._start_task is None
    finally:
        await c.runtime.shutdown(timeout_s=1)
