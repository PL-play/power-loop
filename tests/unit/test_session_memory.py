"""Long-lived loops release idle sessions; active and failed caches stay bounded."""
import asyncio
import gc
import tracemalloc

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop
from power_loop.agent.stateful_loop import _SESSION_SYNC

from .test_subagent_events import _run_child, _Scripted

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_thousand_sessions_and_children_release_memory(record_property):
    store = await SessionStore.open(':memory:')
    loop = StatefulAgentLoop(llm=_Scripted(), store=store,
        config=AgentLoopConfig(max_rounds=1, compactor=None))
    spec = AgentSpec(name='child', system_prompt='Finish.', max_rounds=1)
    async def batch(n):
        for _ in range(n):
            sid = await loop.new_session()
            await loop.send('Finish.', sid)
            assert (await _run_child(loop, sid, spec, 'Finish.'))['status'] == 'completed'
            assert sid not in _SESSION_SYNC and not loop._sync_sids
        gc.collect()
    tracemalloc.start()
    try:
        # Fill the existing bounded message/session caches before measuring.
        await batch(2048)
        samples = [tracemalloc.get_traced_memory()[0]]
        for _ in range(2):
            await batch(500)
            samples.append(tracemalloc.get_traced_memory()[0])
        record_property('retained_bytes', samples)
        assert max(samples) - min(samples) < 256_000, samples
    finally:
        tracemalloc.stop()
        await loop.aclose()
        await store.close()


async def test_cancelled_waiter_does_not_split_the_session_lock():
    loop = StatefulAgentLoop(llm=_Scripted())
    async def waiter():
        async with loop._session_guard('memory-race'):
            raise AssertionError('cancelled waiter must not enter')
    async with loop._session_guard('memory-race'):
        waiting = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        assert loop._session_busy('memory-race')
    assert 'memory-race' not in _SESSION_SYNC and not loop._sync_sids
    await loop.aclose()
