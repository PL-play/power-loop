"""No callback replay or next model when an automatic negative write fails."""
import json

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.pending_calls import retire_pending
from power_loop.runtime.tool_call_plans import reconcile_pending

from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario
from .test_tool_call_plans_192 import controller

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('stage', ['result', 'negative', 'closure', 'commit', 'commit_ack'])
async def test_automatic_negative_is_atomic_and_recovered_without_effect_or_extra_authority(registry_store, stage):
    c = await scenario(registry_store, budgeted=True, claim_failure=True, unstarted_failure=stage)
    assert c.fault.fired
    assert c.result.status == 'interrupted'
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    assert c.model.calls == c.host.reserve_calls == 1
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert row['generation'] == 0
    terminal = await c.manager.db.fetchall(f"SELECT event_type FROM {c.manager.events} WHERE job_id=? AND event_type IN ('result','managed_effect_unstarted','supervision_closed')", (row['job_id'],))
    assert len(terminal) == (3 if stage == 'commit_ack' else 0)
    assert row['supervision_status'] == ('closed' if stage == 'commit_ack' else 'open')
    assert row['resource_state'] == ('stopped' if stage == 'commit_ack' else 'not_started')
    raw = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    [error] = [r for r in raw if r['role'] == 'tool']
    assert 'test_claim_not_started' in error['content']
    assert (await c.store.get_state(c.sid)).pending['tool_call_ids'] == ['original-call']
    loop, reviewer = await controller(c)
    try:
        staged = await reconcile_pending(loop, c.actor, row['parent_job_id'])
        assert len(staged['notices']) == 1 and reviewer.calls == 0
        assert not staged['acknowledgement_created'] and not staged['billing_settled']
        await retire_pending(loop, c.actor, row['parent_job_id'])
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, row['parent_job_id'], generation=1)
        result = await loop.send('Review the original negative result. Do not execute the effect.', c.sid)
        assert result.status == 'completed' and reviewer.calls == 1
        current = await c.manager.get_job(c.actor, row['job_id'])
        assert current.result_status == 'suppressed' and current.supervision_status == 'closed'
        assert current.result['managed_effect_result']['resolution'] == 'not_dispatched'
        assert (await c.manager.get_job(c.actor, row['parent_job_id'])).supervision_status == 'closed'
        assert c.model.calls == 1 and c.host.reserve_calls == 2
        assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
        for original in raw:
            assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (original['id'],)) == original
        [canonical] = await c.manager.db.fetchall(f"SELECT payload_json,delivery_status FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (row['job_id'],))
        assert canonical['delivery_status'] == 'suppressed'
        assert json.loads(canonical['payload_json'])['unstarted'] is True
    finally:
        await loop.aclose(drain_timeout_s=.1)
