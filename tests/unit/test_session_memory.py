"""Long-lived loops release idle sessions; active and failed caches stay bounded."""
import asyncio
import gc
import tracemalloc
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop
from power_loop.agent.stateful_loop import _SESSION_SYNC
from power_loop.runtime.controller_retirement import RetirementCache
from power_loop.runtime.controller_task_exit import TaskExitCache
from power_loop.runtime.jobs import JobError
from power_loop.workflow.runner import remember_handle

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


async def test_failed_controller_records_are_bounded_and_new_workflow_handles_are_retained():
    @dataclass
    class Lease:
        job_id: str
    task = asyncio.current_task()
    runtime = SimpleNamespace(_managed_completions=SimpleNamespace(check=lambda: None),
        _adapter_tasks={task}, _local_executions={}, _workflow_handles={},
        track_task=lambda *args, **kwargs: None)
    runtime._controller_retirements = retirement = RetirementCache(runtime)
    runtime._controller_task_exits = exits = TaskExitCache(runtime)
    # Failed writes occupy the same slots as ordinary exits, with no extra queue.
    for i in range(1000):
        binding = SimpleNamespace(runtime=runtime, actor='actor', lease=Lease(str(i)),
            stop_renewal=asyncio.Event())
        binding.stop_renewal.set()
        local = dict(binding=binding, task=task)
        runtime._local_executions[id(local)] = local
        retirement.capture(binding, local).failed = True
        runtime._local_executions.clear()
        handle = SimpleNamespace(operation_id=f'op-{i}', run_id=f'run-{i}',
            job_id=str(i), task=task, cancel_token=None)
        remember_handle(runtime, handle)
    assert len(retirement.pending) == len(exits.slots) == 1000
    binding.lease = Lease('overflow')
    runtime._local_executions[id(local)] = local
    with pytest.raises(JobError, match='controller_retirement_capacity_reached'):
        retirement.capture(binding, local)
    retirement.pending.clear()
    with pytest.raises(JobError, match='controller_task_exit_capacity_reached'):
        retirement.capture(binding, local)
    assert len(exits.slots) == 1000
    assert len(runtime._workflow_handles) == 256
    assert runtime._workflow_handles['run-999'] is handle
    assert 'run-0' not in runtime._workflow_handles
    exits.slots.clear()
