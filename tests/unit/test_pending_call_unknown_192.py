"""Unknown effects remain blocked until original query-only evidence resolves them."""
import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.managed_delivery import enqueue_runtime
from power_loop.runtime.managed_effects import reconcile
from power_loop.runtime.pending_calls import retire_pending_runtime
from power_loop.runtime.representation import ProjectedRepresentation

from .test_job_registry_192 import registry_store
from .test_managed_delivery_192 import consume_queued, controller
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('cancelled', [False, True])
async def test_unknown_pending_needs_original_query_even_after_actual_callback_exit(registry_store, cancelled):
    c = await scenario(registry_store, budgeted=True, pending=True, raw_receipt_failure=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    source = await c.manager.get_job(c.actor, row['parent_job_id'])
    pending = (await c.store.get_state(c.sid)).pending
    assert pending is not None and source.status == 'model_closed'
    loop, reviewer = await controller(c)
    loop.config.representation = ProjectedRepresentation()
    try:
        if cancelled:
            await c.manager.request_cancel(c.actor, source.job_id, system_key='stop-unknown', reason='Stop original work')
        with pytest.raises(JobError):
            await enqueue_runtime(c.runtime, c.actor, row['job_id'])
        with pytest.raises(JobError):
            await retire_pending_runtime(c.runtime, c.actor, source.job_id)
        assert (await c.store.get_state(c.sid)).pending == pending
        assert reviewer.calls == c.state.lookups == 0
        c.state.resolved = True
        await reconcile(c.runtime, c.actor, row['job_id'], c.adapter)
        await enqueue_runtime(c.runtime, c.actor, row['job_id'])
        retired = await retire_pending_runtime(c.runtime, c.actor, source.job_id)
        assert not retired['acknowledgement_created'] and (await c.store.get_state(c.sid)).pending is None
        result = await consume_queued(loop)
        assert result.status == 'completed' and reviewer.calls == 1
        assert c.state.lookups == c.state.prepared == c.state.dispatched == c.model.calls == 1
        assert c.host.reserve_calls == 2
        assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'handled'
        assert (await c.manager.get_job(c.actor, source.job_id)).supervision_status == 'closed'
        assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    finally:
        await loop.aclose()
