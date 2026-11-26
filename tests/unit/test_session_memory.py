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
from tests.runtime_fixture import ScriptedModel as _Scripted

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
                                config=AgentLoopConfig(context_budget_tokens=10**9))
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


async def test_cancelled_waiter_does_not_split_the_session_lock(tmp_path, monkeypatch):
    from power_loop import RegistryRuntimeFactory, Runtime, SessionStore
    from power_loop.agent.stateful_loop import _ExecutionLoop

    sid = 'memory-race'
    store = await SessionStore.open(str(tmp_path / 'locks.db'))
    runtime = Runtime(
        store=store,
        factory=RegistryRuntimeFactory(),
        node_id='lock-node',
        boot_id='lock-boot',
        recovery_interval_s=None,
        timers_enabled=False,
    )
    worker = runtime.worker
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()
    later_entered = asyncio.Event()
    release_later = asyncio.Event()
    entered: list[str] = []
    active_inside = 0
    peak_inside = 0

    async def fake_run_session_owned(target_sid, *, preclaimed_batch=None):
        nonlocal active_inside, peak_inside
        assert target_sid == sid
        tag = str(preclaimed_batch)
        active_inside += 1
        peak_inside = max(peak_inside, active_inside)
        entered.append(tag)
        try:
            if tag == 'holder':
                holder_entered.set()
                await release_holder.wait()
            elif tag == 'later':
                later_entered.set()
                await release_later.wait()
        finally:
            active_inside -= 1

    monkeypatch.setattr(worker, '_run_session_owned', fake_run_session_owned)
    holder = waiter = later = None
    try:
        holder = asyncio.create_task(worker._run_one_session(sid, preclaimed_batch='holder'))
        await asyncio.wait_for(holder_entered.wait(), 5)

        waiter = asyncio.create_task(worker._run_one_session(sid, preclaimed_batch='waiter'))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert 'waiter' not in entered

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert 'waiter' not in entered
        assert _ExecutionLoop._session_busy(sid) and sid in _SESSION_SYNC

        # Cancelling the waiter must not split or release the lock while the holder is active.
        later = asyncio.create_task(worker._run_one_session(sid, preclaimed_batch='later'))
        await asyncio.sleep(0.02)
        assert not later_entered.is_set()
        assert entered == ['holder'] and peak_inside == 1

        release_holder.set()
        await asyncio.wait_for(holder, 5)

        await asyncio.wait_for(later_entered.wait(), 5)
        release_later.set()
        await asyncio.wait_for(later, 5)

        assert entered == ['holder', 'later']
        assert peak_inside == 1
        assert sid not in _SESSION_SYNC
    finally:
        release_holder.set()
        release_later.set()
        for task in (holder, waiter, later):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (holder, waiter, later) if t is not None), return_exceptions=True)
        await runtime.client.aclose()
        await runtime.transport.aclose()
        await store.close()
