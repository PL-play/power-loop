"""Finalization removes transient execution data only with a settled closure."""
import pytest

from power_loop.runtime import finalization
from power_loop.runtime.jobs import JobActor, JobError, JobManager

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, observed, start
from .test_send_sql_profile import _measure

__all__ = ['registry_store', 'budget_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def transient_rows(manager):
    rows = {name: await manager.db.fetchall(f'SELECT * FROM {manager.prefix}{name} ORDER BY job_id')
            for name in ('execution_checkpoints', 'execution_allocations', 'task_budget_attempts')}
    rows['events'] = await manager.db.fetchall(f"SELECT * FROM {manager.events} "
        "WHERE event_type IN ('execution_scope','execution_membership_seal') ORDER BY event_id")
    return rows


async def test_paid_send_deletes_temporary_rows_but_retains_cold_usage_and_closure(registry_store):
    store, connect = registry_store
    await _measure(store, 2, True)
    cold = JobManager(await connect())
    assert not any((await transient_rows(cold)).values())
    jobs = await cold.db.fetchall(f'SELECT * FROM {cold.jobs} ORDER BY created_at,job_id')
    assert len(jobs) == 2 and all(row['supervision_status'] == 'closed' for row in jobs)
    for row in jobs:
        actor = JobActor(row['tenant_id'], row['principal_id'], row['session_id'])
        membership = await cold.get_execution_membership(actor, row['job_id'], 1)
        assert membership['sealed']
        usage = await cold.execution_usage(actor, row['job_id'], generation=1)
        assert usage == membership['model']['payload']['evidence']['usage']
        assert (await cold.finalize_supervision(actor, row['job_id'], generation=1))['event_type'] == 'supervision_closed'


async def test_unacknowledged_fee_prevents_finalization_and_is_retained(budget_case):
    c = budget_case
    c.host.apply_lost_ack = True
    attempt = await start(c)
    await observed(c, attempt)
    await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={'exit': 'completed'})
    before = await transient_rows(c.manager)
    assert before['task_budget_attempts'][0]['pending']
    with pytest.raises(JobError, match='budget_reconciliation_required'):
        await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)
    assert await transient_rows(c.manager) == before
    assert (await c.manager.get_job(c.actor, c.binding.lease.job_id)).supervision_status == 'open'


async def test_failed_finalization_rolls_back_deletions_with_the_closure(budget_case, monkeypatch):
    c = budget_case
    attempt = await start(c)
    await observed(c, attempt)
    await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={'exit': 'completed'})
    before = await transient_rows(c.manager)
    assert all(before.values())
    publish = finalization._publish

    async def fail_after_delete(*args, **kwargs):
        await publish(*args, **kwargs)
        raise OSError('commit failed after cleanup')

    with monkeypatch.context() as patch:
        patch.setattr(finalization, '_publish', fail_after_delete)
        with pytest.raises(OSError, match='commit failed'):
            await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)
    assert await transient_rows(c.manager) == before
    assert (await c.manager.get_job(c.actor, c.binding.lease.job_id)).supervision_status == 'open'
    await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)
    assert not any((await transient_rows(c.manager)).values())
