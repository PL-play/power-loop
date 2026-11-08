"""Finished runtime executions release loops and locks while store caches stay bounded."""
import asyncio
import gc
import weakref

import pytest

from power_loop import AgentLoopConfig, AgentSpec, ToolDefinition, ToolRegistry
from power_loop.agent.stateful_loop import _SESSION_SYNC
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from tests.unit.test_subagent import _Scripted

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_sessions_and_children_release_execution_memory(runtime_agent, record_property):
    loops = []
    registry = ToolRegistry()
    spec = AgentSpec(name='child', system_prompt='Finish.', max_rounds=1)
    async def delegate():
        loop = get_current_loop()
        loops.append(weakref.ref(loop))
        return await run_agent_spec(spec, 'Finish.', parent_loop=loop)
    registry.register(ToolDefinition(name='delegate', description='delegate', pricing=Free('test_fixture')), delegate)
    class Model(_Scripted):
        async def complete(self, request, **kwargs):
            loops.append(weakref.ref(get_current_loop()))
            return await super().complete(request, **kwargs)
    agent = await runtime_agent(model=Model(), registry=registry, command=('delegate', {}),
                                config=AgentLoopConfig(compactor=None))
    live_counts = []
    for batch in range(3):
        for i in range(40):
            agent.target = await agent.loop.new_session(agent.descriptor, actor=agent.actor)
            result = await agent.send('Finish.', message_id=f'{batch}-{i}')
            assert result.status == 'completed'
            await agent.drain()
            assert agent.target.session_id not in _SESSION_SYNC
        await asyncio.sleep(0)
        gc.collect()
        live_counts.append(sum(ref() is not None for ref in loops))
        assert not agent.runtime.worker._tasks and not agent.runtime.worker._active
        assert len(agent.runtime.store._session_holdings) <= 4096
        assert len(agent.runtime.store._appended_refs) <= 4096
    record_property('live_execution_loops', live_counts)
    assert not any(ref() is not None for ref in loops[:-80]), live_counts
    await agent.aclose()
    await asyncio.sleep(0)
    gc.collect()
    assert not any(ref() is not None for ref in loops), live_counts


async def test_cancelled_waiter_does_not_split_the_session_lock(tmp_path):
    from power_loop import SessionStore
    from tests.unit.test_loop_resume_cache import _cache_loop
    store = await SessionStore.open(str(tmp_path/'locks.db'))
    loop = _cache_loop(store)
    async def waiter():
        async with loop._session_guard('memory-race'):
            raise AssertionError('cancelled waiter must not enter')
    try:
        async with loop._session_guard('memory-race'):
            waiting = asyncio.create_task(waiter())
            await asyncio.sleep(0)
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            assert loop._session_busy('memory-race')
        assert 'memory-race' not in _SESSION_SYNC and not loop._sync_sids
    finally:
        await loop.aclose()
        await store.close()
