"""Independent clients enqueue corrections using current immutable message IDs."""
import asyncio

import pytest

from power_loop import JobRef, RuntimeClient, RuntimeRequestError
from power_loop.agent.runtime_store import now_ms

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import managed, receipt

__all__ = ['registry_store','managed']
pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('managed',[{'capacity':2}],indirect=True)
async def test_independent_clients_nudge_same_original_execution_and_reopen_query(managed):
    store,_,_,agent,model,_,connect = managed
    await agent.send('delegate')
    await asyncio.wait_for(model.entered.wait(),5)
    child = await receipt(managed)
    ref = JobRef(**child['job_ref'])
    original = await agent.runtime.client.inspect(ref,actor=agent.actor)
    peer = await connect()
    client = RuntimeClient(store=peer,transport=agent.runtime.transport,namespace=agent.runtime.namespace)
    bodies = ['first correction','  second correction\n    indentation preserved\n']
    try:
        replies = await asyncio.gather(*(port.nudge(original.job_ref,body,actor=agent.actor,message_id=f'correction-{i}')
            for i,(port,body) in enumerate(zip((agent.runtime.client,client),bodies))))
        repeat = await client.nudge(original.job_ref,bodies[0],actor=agent.actor,message_id='correction-0')
        assert repeat.input_ref == replies[0].input_ref and not repeat.newly_accepted
        with pytest.raises(RuntimeRequestError,match='conflict'):
            await client.nudge(original.job_ref,'different',actor=agent.actor,message_id='correction-0')
        for reply,body in zip(replies,bodies):
            record = await client.inputs.input(client.inputs.db,reply.input_ref)
            import json
            assert json.loads(record.row['content']) == body
        await client.aclose()
        client = RuntimeClient(store=peer,transport=agent.runtime.transport,namespace=agent.runtime.namespace)
        assert (await client.inspect(ref,actor=agent.actor)).job_ref == original.job_ref
        model.release.set()
        result = await client.wait(ref,actor=agent.actor,deadline_ms=now_ms()+10000)
        assert result.status == 'completed' and result.job_ref == original.job_ref
        await agent.drain()
        text = '\n'.join(m.content or '' for m in await store.load_all_messages(child['session_id']))
        assert all(body.strip() in text for body in bodies), text
    finally:
        model.release.set()
        await client.aclose()
