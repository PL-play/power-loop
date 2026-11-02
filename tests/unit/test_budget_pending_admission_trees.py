"""New admission counts pending fees without reopening another task's journal."""
from __future__ import annotations

import pytest

from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.control_runtime import flush_binding
from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, new_send, observed, start

__all__ = ['budget_case', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('budget_case', [{'max_pending':1}, {'max_pending':64}], indirect=True)
async def test_another_roots_pending_usage_is_counted_without_reading_its_originals(budget_case, monkeypatch):
    c = budget_case
    first = c.binding
    attempt = await start(c)
    c.host.apply_lost_ack = True
    saved = await observed(c, attempt)
    assert saved['pending'] and saved['ack'] is None
    await flush_binding(first, close=True)
    before = await c.manager.db.fetchone(f'SELECT * FROM {c.manager.prefix}task_budget_attempts WHERE attempt_id=?',
        (attempt.attempt_id,))
    originals = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} WHERE job_id=? ORDER BY event_id',
        (first.lease.job_id,))
    read = budgets._read

    async def own_attempt_only(manager, reader, identity):
        if identity == attempt.attempt_id:
            assert manager._control_intents.get() is None, 'new admission reopened another roots pending attempt inside its writer'
        return await read(manager, reader, identity)

    with monkeypatch.context() as patch:
        patch.setattr(budgets, '_read', own_attempt_only)
        if c.config.max_pending == 1:
            with pytest.raises(JobError, match='budget_reconciliation_required'):
                await new_send(c)
        else:
            await new_send(c)
            assert c.binding.lease.job_id != first.lease.job_id
            await flush_binding(c.binding, close=True)
    assert await c.manager.db.fetchone(f'SELECT * FROM {c.manager.prefix}task_budget_attempts WHERE attempt_id=?',
        (attempt.attempt_id,)) == before
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} WHERE job_id=? ORDER BY event_id',
        (first.lease.job_id,)) == originals
    assert c.host.reserve_calls == 1 and c.host.apply_calls == 1 and len(c.host.applied) == 1
    inspected = await budgets.get_budget(c.manager, first.actor, attempt.attempt_id)
    assert inspected['pending'] and inspected['ack'] is None and inspected['submission'] == saved['submission']
