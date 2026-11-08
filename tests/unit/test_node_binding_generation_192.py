"""Real node identity and scope, without the retired binding/closure proof events."""

import pytest

from power_loop import JobRef, RuntimeRequestError, TargetRef, TrustedActor
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime import node_inputs
from power_loop.runtime.jobs import JobManager
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = pytest.mark.asyncio


async def test_current_node_binding_is_fixed_and_visible_after_worker_restart(registry_store):
    store, _ = registry_store
    first = await RunningAgent(store,Model()).start()
    second = None
    try:
        handle = await first.runtime.client.workflow(first.target,{'name':'fixed','root':leaf('BUILD')},
                                                    message_id='workflow',actor=first.actor)
        assert (await handle.wait(deadline_ms=now_ms()+10000)).status == 'completed'
        await first.drain()
        row, = await store._db.fetchall(f"SELECT * FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        manager = JobManager(store)
        descriptor = await node_inputs.descriptor(manager,manager.db,row,row['session_id'])
        assert descriptor['session_id'] == row['session_id'] and descriptor['generation'] == 1
        assert await node_inputs.is_closed(manager,manager.db,row,row['session_id'])
        target = first.target
        await first.aclose()
        second = await RunningAgent(store,Model()).start(target=target)
        ref = JobRef(store.store_id,row['job_id'],row['generation'])
        known = await second.runtime.client.wait(ref,actor=second.actor,deadline_ms=now_ms()+10000)
        assert known.status == 'completed' and known.job_ref == ref
        with pytest.raises(RuntimeRequestError,match='forbidden'):
            await second.runtime.client.inspect(ref,actor=TrustedActor('other','host',None))
        with pytest.raises(RuntimeRequestError,match='child sessions require resume'):
            await second.runtime.client.send(TargetRef(store.store_id,row['session_id']), 'new root',
                                             message_id='bypass',actor=second.actor)
        assert (await store._db.fetchone(f"SELECT generation,session_id FROM {store.table_prefix}jobs WHERE job_id=?",
                                        (row['job_id'],))) == {'generation':1,'session_id':row['session_id']}
        assert await node_inputs.descriptor(manager,manager.db,row,row['session_id']) == descriptor
    finally:
        if second:
            await second.aclose()
        await first.aclose()


@pytest.mark.parametrize('generation',[-1,True,1.0,'2'])
async def test_invalid_generation_is_rejected_at_public_reference(generation):
    with pytest.raises(RuntimeRequestError):
        JobRef('store','job',generation)
