"""Worker capacity, queued cancellation and waiting through the current client."""
import asyncio
import json

from power_loop import AgentLoopConfig, JobRef, ToolDefinition, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ['registry_store']


async def test_queue_releases_capacity_on_failure_and_cancelled_work_never_invokes(registry_store, tmp_path):
    store, _ = registry_store
    gates = [asyncio.Event(), asyncio.Event()]
    started, active, peak = [], 0, 0
    async def read(i):
        nonlocal active, peak
        started.append(i)
        active += 1
        peak = max(peak, active)
        try:
            await gates[i].wait()
            if i == 0:
                raise ValueError('test failure releases slot')
            return 'image ready'
        finally:
            active -= 1
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    registry.register(ToolDefinition('image', 'local read', input_schema={
        'type':'object','properties':{'i':{'type':'integer'}},'required':['i']},
        async_capable=True, pricing=Free('test')), read)
    calls = [_tool_resp(str(i),'background_run',json.dumps({'action':'tool','tool':'image','args':{'i':i}}))
             for i in range(3)]
    model = _Scripted(calls)
    agent = await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None),capacity=1).start()
    waiter = None
    try:
        assert (await agent.send('start three readers')).status == 'completed'
        messages = await store.load_all_messages(agent.target.session_id)
        refs = [JobRef(**json.loads(next(m.content for m in messages if m.role=='tool' and m.tool_call_id==str(i)))['job_ref'])
                for i in range(3)]
        async with asyncio.timeout(5):
            while not started:
                await asyncio.sleep(.01)
        assert started == [0] and peak == 1
        await agent.runtime.client.cancel(refs[2],actor=agent.actor)
        waiter = asyncio.create_task(agent.runtime.client.wait(refs[1],actor=agent.actor,deadline_ms=now_ms()+10000))
        await asyncio.sleep(.02)
        assert not waiter.done()  # A queued task is included in wait.
        gates[0].set()
        first = await agent.runtime.client.wait(refs[0],actor=agent.actor,deadline_ms=now_ms()+10000)
        assert first.status == 'failed'
        async with asyncio.timeout(5):
            while len(started)<2:
                await asyncio.sleep(.01)
        assert started == [0,1] and peak == 1
        gates[1].set()
        result = await waiter
        assert result.status == 'completed' and 'image ready' in result.final_text
        cancelled = await agent.runtime.client.wait(refs[2],actor=agent.actor,deadline_ms=now_ms()+10000)
        assert cancelled.status == 'cancelled' and 2 not in started
        await agent.drain()
        assert not agent.runtime.worker._tasks and not agent.runtime.worker._pending
        assert not agent.runtime.client._waiters
    finally:
        for gate in gates:
            gate.set()
        if waiter and not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter,return_exceptions=True)
        await agent.aclose()
