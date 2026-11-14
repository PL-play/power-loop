"""Current timer authority is main-only, including orphaned child sessions."""

from dataclasses import replace

import pytest

from power_loop import RuntimeRequestError, StoreTimerBackend, TimerContext, TrustedActor

from .session_native_fixture import DESCRIPTOR, fixture


@pytest.mark.parametrize('role', ['agent','workflow_driver','background'])
@pytest.mark.parametrize('orphan', [False,True])
async def test_child_timer_cannot_be_created(registry_store,role,orphan):
    store, _ = registry_store
    _, parent, actor = await fixture(store)
    child = await store.create_session(replace(DESCRIPTOR, kind=role), actor=actor,
        parent_session_id=None if orphan else parent.session_id, created_by_session_id=parent.session_id)
    backend = StoreTimerBackend(store)
    source = TimerContext(store.store_id, child, actor)
    with pytest.raises(RuntimeRequestError,match='only main'):
        await backend.create(source,delay_s=10,note='child',interval_s=60)
    assert await store.list_timers(child) == []
    assert await store.list_timers(parent.session_id) == []


async def test_child_actor_cannot_schedule_for_parent(registry_store):
    store, _ = registry_store
    _, parent, actor = await fixture(store)
    child = await store.create_session(DESCRIPTOR, actor=actor,
        parent_session_id=parent.session_id, created_by_session_id=parent.session_id)
    source = TimerContext(store.store_id,parent.session_id,TrustedActor(actor.tenant_id,actor.principal_id,child))
    with pytest.raises(RuntimeRequestError,match='forbidden'):
        await StoreTimerBackend(store).create(source,delay_s=10,note='not your parent')
    assert await store.list_timers(parent.session_id) == []
