"""Current timer authority is main-only, including orphaned child sessions."""

import pytest

from power_loop import RuntimeRequestError, StoreTimerBackend, TimerContext, TrustedActor

from .test_job_registry_192 import registry_store
from .test_runtime_messaging_203 import fixture

__all__ = ['registry_store']


@pytest.mark.parametrize('role', ['subagent','workflow_node','workflow'])
@pytest.mark.parametrize('orphan', [False,True])
async def test_child_timer_cannot_be_created(registry_store,role,orphan):
    store, _ = registry_store
    _, parent, actor = await fixture(store)
    child = await store.create_session(parent_session_id=None if orphan else parent.session_id,
        metadata={'tenant_id':actor.tenant_id,'principal_id':actor.principal_id,'execution_role':role})
    backend = StoreTimerBackend(store)
    source = TimerContext(store.store_id, child, actor)
    with pytest.raises(RuntimeRequestError,match='only main'):
        await backend.create(source,delay_s=10,note='child',interval_s=60)
    assert await store.list_timers(child) == []
    assert await store.list_timers(parent.session_id) == []


async def test_child_actor_cannot_schedule_for_parent(registry_store):
    store, _ = registry_store
    _, parent, actor = await fixture(store)
    child = await store.create_session(parent_session_id=parent.session_id,metadata={
        'tenant_id':actor.tenant_id,'principal_id':actor.principal_id,'execution_role':'subagent'})
    source = TimerContext(store.store_id,parent.session_id,TrustedActor(actor.tenant_id,actor.principal_id,child))
    with pytest.raises(RuntimeRequestError,match='forbidden'):
        await StoreTimerBackend(store).create(source,delay_s=10,note='not your parent')
    assert await store.list_timers(parent.session_id) == []
