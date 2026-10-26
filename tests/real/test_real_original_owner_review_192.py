"""Real model requests remain under the original child owner and its quota."""
import asyncio

import pytest

from power_loop.runtime import delegation
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_original_owner_review_192 import original_owner_case, wait_owned

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm,pytest.mark.asyncio]


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
@pytest.mark.parametrize('allowed_calls',[2,1])
async def test_real_original_owner_never_releases_output_before_actual_review_or_gets_extra_quota(registry_store,kind,allowed_calls,record_property):
    provider = BoundedProvider(limit=7)
    c = await original_owner_case(registry_store,kind,allowed_calls=allowed_calls,provider=provider)
    initial = set(delegation._TASKS)
    try:
        result = await c.loop.send('Use launch_owned exactly once to launch the requested original child task. Then reply QUEUED.',c.state.root)
        assert result.status == 'completed'
        await asyncio.wait_for(c.state.original.wait(),90)
        if allowed_calls==2:
            await asyncio.wait_for(c.state.entered.wait(),90)
            assert not c.state.notifications and c.state.counts['AFTER']==0 and not c.state.effects
            source = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
            assert source.status=='model_closed' and source.result_status=='pending_delivery'
            owner = await c.manager.get_job(c.runtime.actor(c.state.root),source.control_owner_job_id)
            assert owner.status=='running' and owner.result_status=='none'
        c.state.release.set()
        await wait_owned(c,kind,initial)
        source = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root),source.control_owner_job_id)
        assert source.execution_role==kind and source.result['aborted_call_batch']['review_owner']['owner']['job_id']==owner.job_id
        assert not c.state.effects
        assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'",(source.session_id,))
        if allowed_calls==2:
            assert c.state.counts['BUILD']==2 and c.state.counts['AFTER']==(1 if kind=='workflow_node' else 0)
            assert source.result_status=='handled' and source.supervision_status=='closed' and owner.outcome=='failed'
            assert c.state.review_requests and (await c.store.get_state(source.session_id)).pending is None
            await c.loop.flush_follow_ups(c.state.root,tools=[],max_rounds=1)
            owner = await c.manager.get_job(c.runtime.actor(c.state.root),owner.job_id)
            assert owner.supervision_status=='closed'
        else:
            assert c.state.counts['BUILD']==1 and not c.state.review_requests and c.state.counts['AFTER']==0
            assert source.result_status=='pending_delivery' and source.supervision_status=='open'
            assert owner.result_status=='none' and owner.supervision_status=='open'
            if kind=='subagent':
                assert not c.state.notifications
    finally:
        c.state.release.set()
        provider.report(record_property)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)
        await provider.close()
