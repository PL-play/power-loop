"""Original usage recovery requires no live lease, boot, or model object."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, counters, start
from .test_model_budgets_192 import observed as observe_original

__all__ = ["registry_store", "budget_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def reopened(c, **kwargs):
    store = await c.connect()
    return TaskRuntime(JobManager(store), tenant_id="budget-tenant", principal_id="host",
        executor_id="reconciler", boot_id="new-boot", boot_contract_hash="a" * 64,
        budget_config=kwargs.get("config", c.config))


async def reconcile(runtime, c, a):
    return await runtime.reconcile_budget_usage(c.sid, job_id=c.binding.lease.job_id, attempt_id=a.attempt_id)


async def observed(c, attempt, **kwargs):
    from power_loop.runtime.control_runtime import flush_binding
    value = await observe_original(c, attempt, **kwargs)
    await flush_binding(attempt.binding, close=True)
    return value


async def test_cold_reconciler_cannot_overtake_a_live_owner(budget_case):
    c = budget_case
    c.host.apply_lost_ack = True
    attempt = await start(c)
    await observe_original(c, attempt)
    c.host.apply_lost_ack = False
    fresh = await reopened(c)
    before = c.host.apply_calls
    with pytest.raises(JobError, match="execution_reconciliation_requires_quiescence"):
        await reconcile(fresh, c, attempt)
    assert c.host.apply_calls == before
    from power_loop.runtime.control_runtime import flush_binding
    await flush_binding(attempt.binding, close=True)
    assert (await reconcile(fresh, c, attempt))["usage_acknowledged"]


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


@pytest.mark.parametrize("fault", ["session", "job", "config"])
async def test_foreign_or_missing_evidence_never_calls_provider(budget_case, fault):
    c = budget_case
    c.host.apply_lost_ack = True
    a = await start(c)
    await observed(c, a)
    before = (c.host.apply_calls, c.host.reserve_calls)
    config = replace(c.config, provider_key="foreign") if fault == "config" else c.config
    runtime = await reopened(c, config=config)
    with pytest.raises(JobError):
        await runtime.reconcile_budget_usage("foreign" if fault == "session" else c.sid,
            job_id="missing" if fault == "job" else c.binding.lease.job_id, attempt_id=a.attempt_id)
    assert (c.host.apply_calls, c.host.reserve_calls) == before


async def test_closed_dispatch_without_observation_releases_unknown_usage(budget_case):
    c = budget_case
    a = await start(c)
    from power_loop.runtime.control_runtime import flush_binding
    await flush_binding(a.binding, close=True)
    runtime = await reopened(c)
    result = await reconcile(runtime, c, a)
    assert result['released'] and not result['execution_authority']
    assert not c.host.applied and len(c.host.released) == 1
    assert all(v == 0 for v in (await counters(c)).values())


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


@pytest.mark.parametrize('fault', [None, 'provider', 'scope', 'disabled'])
async def test_recovery_configuration_uses_original_manifest_without_writing(budget_case, fault):
    c = budget_case
    a = await start(c)
    from power_loop.runtime.control_runtime import flush_binding
    await flush_binding(a.binding)
    if fault == 'disabled':
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_budget_policies SET enabled=0")
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    kwargs = dict(provider_key='different' if fault == 'provider' else c.config.provider_key,
                  provider=c.host)
    actor = replace(c.actor, session_id='other') if fault == 'scope' else c.actor
    if fault in ('provider', 'scope'):
        with pytest.raises(JobError):
            await budgets.recovery_config(c.manager, actor, c.binding.lease.job_id, a.attempt_id, **kwargs)
    else:
        config = await budgets.recovery_config(c.manager, actor, c.binding.lease.job_id, a.attempt_id, **kwargs)
        assert config.manifest() == c.config.manifest() and config.provider is c.host
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id") == before
    assert c.host.reserve_calls == 1 and c.host.apply_calls == 0
