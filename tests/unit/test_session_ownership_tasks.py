"""Loss of session ownership stops actual detached child and workflow tasks."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, MemorySessionOwnershipCoordinator, StatefulAgentLoop

from .test_executor_lifecycle_192 import host
from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import managed
from .test_timers import _Scripted
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ['registry_store', 'managed', 'managed_workflow', 'host']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def lose_and_resume(store, manager, runtime, loop, sid):
    [lease] = list(loop.session_coordinator._leases.values())
    before = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE outcome IS NULL "
        "AND status<>'model_closed'")
    assert before
    lease.lose()
    async with asyncio.timeout(5):
        while runtime._adapter_tasks:
            await asyncio.gather(*tuple(runtime._adapter_tasks), return_exceptions=True)
    assert not runtime._local_executions and not runtime._model_executions
    assert not runtime._controller_retirements.pending and not runtime._controller_task_exits.slots
    runtime.check_local_admission()  # Losing one session must not poison this runtime.
    assert (await loop.send('continue after loss', sid)).status == 'completed'
    for row in before:
        job = await manager.get_job(runtime.actor(sid), row['job_id'])
        assert job.outcome == 'interrupted'
    assert not loop.session_coordinator._leases


async def test_loss_stops_detached_subagent_and_allows_a_new_send(managed):
    store, manager, runtime, loop, model, sid, _ = managed
    loop.session_coordinator = MemorySessionOwnershipCoordinator()
    await loop.send('spawn', sid)
    await asyncio.wait_for(model.entered.wait(), 3)
    await lose_and_resume(store, manager, runtime, loop, sid)
    assert model.child_calls == 1


async def test_loss_stops_workflow_and_its_node_then_allows_a_new_send(managed_workflow):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    loop.session_coordinator = MemorySessionOwnershipCoordinator()
    await loop.send('workflow', sid)
    await asyncio.wait_for(model.entered['BUILD'].wait(), 3)
    await lose_and_resume(store, manager, runtime, loop, sid)
    assert state['handles'][0].task.done() and model.counts['REVIEW'] == 0


async def test_executor_heartbeat_outlives_the_send_that_started_it(host):
    from power_loop.runtime import executor_registry

    store, manager, runtime = host
    coordinator = MemorySessionOwnershipCoordinator()
    loop = StatefulAgentLoop(llm=_Scripted(), store=store, task_runtime=runtime,
        session_coordinator=coordinator, config=AgentLoopConfig(compactor=None, delegation_policy='single_level'))
    try:
        sid = await loop.new_session()
        await loop.send('start executor', sid)
        assert not coordinator._leases
        first = await executor_registry.get(manager, runtime._lifecycle.identity)
        async with asyncio.timeout(5):
            while True:
                current = await executor_registry.get(manager, runtime._lifecycle.identity)
                if current['lease_until_ms'] > first['lease_until_ms']:
                    break
                await asyncio.sleep(.02)
        assert current['admission_open'] and current['lease_until_ms'] > first['lease_until_ms']
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_cached_foreground_history_does_not_hide_an_interrupted_child(managed):
    store, manager, runtime, loop, model, sid, _ = managed
    loop.session_coordinator = MemorySessionOwnershipCoordinator()
    await loop.send('spawn', sid)
    await asyncio.wait_for(model.entered.wait(), 3)
    replacement = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
        session_coordinator=loop.session_coordinator, config=loop.config)
    try:
        # The foreground transcript is unchanged while its detached child runs.
        await replacement.prewarm(sid)
        await lose_and_resume(store, manager, runtime, replacement, sid)
    finally:
        await replacement.aclose(drain_timeout_s=.1)
