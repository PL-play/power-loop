"""Original usage recovery requires no live lease, boot, or model object."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, counters, observed, start

__all__ = ["registry_store", "budget_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def reopened(c, **kwargs):
    store = await c.connect()
    return TaskRuntime(JobManager(store), tenant_id="budget-tenant", principal_id="host",
        executor_id="reconciler", boot_id="new-boot", boot_contract_hash="a" * 64,
        budget_config=kwargs.get("config", c.config))


async def reconcile(runtime, c, a):
    return await runtime.reconcile_budget_usage(c.sid, job_id=c.binding.lease.job_id, attempt_id=a.attempt_id)


@pytest.mark.parametrize("unknown", [False, True])
async def test_new_boot_can_replay_after_cancel_without_reopening_or_double_counting(budget_case, unknown):
    c = budget_case
    c.host.apply_lost_ack = True
    a = await start(c)
    original = await observed(c, a, error=OSError("response uncertain") if unknown else None)
    assert original["pending"]
    c.host.apply_lost_ack = False
    await c.manager.request_cancel(c.actor, c.binding.lease.job_id, system_key="stop", reason="test")
    before_job = await c.manager.get_job(c.actor, c.binding.lease.job_id)
    before_counters = await counters(c)
    runtime = await reopened(c)
    results = await asyncio.gather(reconcile(runtime, c, a), reconcile(runtime, c, a))
    assert results[0] == results[1] == await reconcile(runtime, c, a)
    assert results[0]["usage_acknowledged"] and not results[0]["execution_authority"]
    assert await c.manager.get_job(c.actor, c.binding.lease.job_id) == before_job
    assert await counters(c) == before_counters
    assert c.host.reserve_calls == 1 and len(c.host.applied) == 1 and not c.host.released
    assert runtime._lifecycle._start_task is None  # Reconciliation never activates a boot.
    saved = await budgets.get_budget(runtime.manager, c.actor, a.attempt_id)
    assert saved["submission"] == original["submission"] and not saved["pending"]
    assert saved["observation"]["status"] == ("unknown" if unknown else "returned")


async def test_original_model_observation_projects_after_budget_projection_failed(budget_case, monkeypatch):
    c = budget_case
    a = await start(c)
    original_observe = budgets.observe
    async def unavailable(*args):
        raise OSError("projection unavailable after actual response persisted")
    monkeypatch.setattr(budgets, "observe", unavailable)
    await observed(c, a, tokens=12)
    before = await counters(c)
    assert before["used_calls"] == 0 and before["held_calls"] == 1
    monkeypatch.setattr(budgets, "observe", original_observe)
    r1, r2 = await reopened(c), await reopened(c)
    original_time = (await attempts._read(c.manager, c.manager.db, "finished", a.attempt_id))[0]["created_at"]
    r1.manager.now = r2.manager.now = lambda: original_time + 60000
    await asyncio.gather(reconcile(r1, c, a), reconcile(r2, c, a))
    assert (await counters(c))["used_calls"] == 1 and (await counters(c))["used_tokens"] == 12
    assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))["finished_at_ms"] == original_time
    assert len(c.host.applied) == c.host.reserve_calls == 1


@pytest.mark.parametrize("fault", ["session", "job", "config", "prepared", "dispatched", "finished", "tracking"])
async def test_foreign_or_missing_evidence_never_calls_provider(budget_case, fault):
    c = budget_case
    c.host.apply_lost_ack = True
    a = await start(c)
    await observed(c, a)
    before = (c.host.apply_calls, c.host.reserve_calls)
    config = replace(c.config, provider_key="foreign") if fault == "config" else c.config
    runtime = await reopened(c, config=config)
    if fault in ("prepared", "dispatched", "finished", "tracking"):
        identity = a.prepared["scope"] if fault == "tracking" else a.attempt_id
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?",
            (attempts.event_id(c.manager, fault, identity),))
    with pytest.raises(JobError):
        await runtime.reconcile_budget_usage("foreign" if fault == "session" else c.sid,
            job_id="missing" if fault == "job" else c.binding.lease.job_id, attempt_id=a.attempt_id)
    assert (c.host.apply_calls, c.host.reserve_calls) == before


async def test_dispatched_without_observation_cannot_be_forged_or_released(budget_case):
    c = budget_case
    a = await start(c)
    runtime = await reopened(c)
    before = await counters(c)
    with pytest.raises(JobError, match="model_attempt_evidence_missing"):
        await reconcile(runtime, c, a)
    assert not c.host.applied and not c.host.released and await counters(c) == before


async def test_ack_commit_failure_replays_same_request_without_reapplying_local_usage(budget_case, monkeypatch):
    c = budget_case
    c.host.apply_lost_ack = True
    a = await start(c)
    original = await observed(c, a)
    c.host.apply_lost_ack = False
    runtime = await reopened(c)
    save = budgets._save
    async def broken(manager, tx, state, job, **kwargs):
        await save(manager, tx, state, job, **kwargs)
        if state["ack"]:
            raise OSError("rollback local ACK")
    monkeypatch.setattr(budgets, "_save", broken)
    with pytest.raises(OSError):
        await reconcile(runtime, c, a)
    pending = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert pending["pending"] and pending["submission"] == original["submission"]
    monkeypatch.setattr(budgets, "_save", save)
    await reconcile(runtime, c, a)
    assert len(c.host.applied) == 1 and (await counters(c))["used_calls"] == 1


async def test_disabled_policy_still_accepts_original_usage_without_reenabling(budget_case):
    c = budget_case
    c.host.apply_lost_ack = True
    a = await start(c)
    await observed(c, a, tokens=101)
    c.host.apply_lost_ack = False
    runtime = await reopened(c)
    await reconcile(runtime, c, a)
    [row] = await c.manager.db.fetchall(f"SELECT enabled FROM {c.manager.prefix}task_budget_policies")
    assert row["enabled"] == 0 and (await counters(c))["used_tokens"] == 101


@pytest.mark.parametrize('fault', [None, 'provider', 'scope', 'marker', 'disabled'])
async def test_recovery_configuration_uses_original_manifest_without_writing(budget_case, fault):
    c = budget_case
    a = await start(c)
    if fault == 'marker':
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?",
                                  ('mb_policy_' + budgets.policy_key(c.runtime)[3:],))
    if fault == 'disabled':
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_budget_policies SET enabled=0")
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    kwargs = dict(provider_key='different' if fault == 'provider' else c.config.provider_key,
                  provider=c.host)
    actor = replace(c.actor, session_id='other') if fault == 'scope' else c.actor
    if fault in ('provider', 'scope', 'marker'):
        with pytest.raises(JobError):
            await budgets.recovery_config(c.manager, actor, c.binding.lease.job_id, a.attempt_id, **kwargs)
    else:
        config = await budgets.recovery_config(c.manager, actor, c.binding.lease.job_id, a.attempt_id, **kwargs)
        assert config.manifest() == c.config.manifest() and config.provider is c.host
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id") == before
    assert c.host.reserve_calls == 1 and c.host.apply_calls == 0
