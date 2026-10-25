"""A suppressed negative remains canonical while separate review obligations are visible."""
import pytest

from power_loop.runtime import task_query
from power_loop.runtime.pending_calls import retire_pending
from power_loop.runtime.tool_call_plans import reconcile_pending

from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario
from .test_tool_call_plans_192 import controller

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_closed_unstarted_intent_is_not_a_permanent_active_job(registry_store):
    c = await scenario(registry_store, claim_failure=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    listing = await task_query.list_jobs(c.manager, c.actor, root_job_id=row['parent_job_id'], status='active')
    assert listing['matched_count'] == 0
    details = await task_query.get_job(c.manager, c.actor, row['job_id'])
    assert details['node']['processing_obligation']['state'] == 'not_required'
    assert details['node']['active_reason'] == []


@pytest.mark.parametrize('corrupt', [None, 'notice_missing', 'negative_proof_missing'])
async def test_recovered_negative_review_is_visible_until_real_main_processing_and_queries_never_ack(registry_store, corrupt):
    c = await scenario(registry_store, raw_receipt_failure=True, call_count=2, budgeted=True)
    jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool' ORDER BY generation DESC")
    negative, source_id = jobs[1], jobs[0]['parent_job_id']
    loop, reviewer = await controller(c)
    try:
        await reconcile_pending(loop, c.actor, source_id)
        await retire_pending(loop, c.actor, source_id)
        before = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
        details = await task_query.get_job(c.manager, c.actor, negative['job_id'])
        assert details['node']['processing_obligation']['state'] == 'pending'
        assert details['node']['active_reason'] == ['managed_review_pending']
        assert details['node']['processing_obligation']['notice_event_id']
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == before
        assert reviewer.calls == 0
        await loop.send('Read both original outcomes without performing actions.', c.sid)
        listing = await task_query.list_jobs(c.manager, c.actor, root_job_id=source_id, status='active')
        assert listing['matched_count'] == 0
        details = await task_query.get_job(c.manager, c.actor, negative['job_id'])
        assert details['node']['processing_obligation']['state'] == 'handled' and details['node']['active_reason'] == []
        assert details['node']['result_status'] == 'suppressed'
        if corrupt is not None:
            event_type = 'managed_effect_delivery' if corrupt == 'notice_missing' else 'managed_effect_unstarted'
            await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE job_id=? AND event_type=?', (negative['job_id'], event_type))
            details = await task_query.get_job(c.manager, c.actor, negative['job_id'])
            assert details['node']['processing_obligation']['state'] == 'unknown'
            assert details['node']['active_reason'] == ['managed_review_unknown']
        assert reviewer.calls == 1 and c.state.dispatched == 1
    finally:
        await loop.aclose(drain_timeout_s=.1)
