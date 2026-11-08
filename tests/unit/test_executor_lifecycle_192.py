"""Current worker startup/shutdown and the reusable task-budget fixture."""
import asyncio

import pytest

from power_loop import RuntimeRequestError
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_stateful_loop import _Scripted

__all__=['registry_store']
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


@pytest.fixture
async def host(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(
        manager,
        tenant_id="lifecycle",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        lease_ms=3000,
        boot_contract_hash="a" * 64,
    )
    try:
        yield store, manager, runtime
    finally:
        await runtime.shutdown(timeout_s=0.5, grace_s=0.05)


async def test_concurrent_worker_start_has_one_live_subscription_set(registry_store):
    store,_=registry_store
    agent=RunningAgent(store,_Scripted())
    try:
        await asyncio.gather(*(agent.runtime.worker.start() for _ in range(6)))
        assert agent.runtime.status()['accepting']
        assert len(agent.runtime.transport._subscriptions)==3
        await agent.start()
        assert (await agent.send('once')).status=='completed'
    finally:
        await agent.aclose()
    assert not agent.runtime.transport._subscriptions


async def test_shutdown_is_idempotent_and_new_input_does_not_reach_provider(registry_store):
    store,_=registry_store
    model=_Scripted()
    agent=await RunningAgent(store,model).start()
    first=await agent.send('original')
    results=await asyncio.gather(*(agent.runtime.worker.shutdown(timeout_s=2) for _ in range(3)))
    assert all(r.drained for r in results)
    assert not agent.runtime.status()['accepting']
    with pytest.raises(RuntimeRequestError,match='session_closed'):
        await agent.runtime.client.send(agent.target,'refused',message_id='refused',actor=agent.actor)
    assert len(model.calls)==1
    from power_loop.agent.runtime_store import now_ms
    assert await agent.runtime.client.wait(first.input_ref,actor=agent.actor,deadline_ms=now_ms()+1000)==first
    await agent.aclose()


async def test_shutdown_during_start_finishes_closed_and_releases_store(registry_store):
    store,_=registry_store
    agent=RunningAgent(store,_Scripted())
    entered,release=asyncio.Event(),asyncio.Event()
    original=agent.runtime.factory.preflight
    async def preflight(profiles):
        entered.set()
        await release.wait()
        await original(profiles)
    agent.runtime.factory.preflight=preflight
    starting=asyncio.create_task(agent.runtime.worker.start())
    await entered.wait()
    closing=asyncio.create_task(agent.runtime.worker.shutdown(timeout_s=2))
    release.set()
    await asyncio.gather(starting,closing)
    assert not agent.runtime.status()['accepting'] and not agent.runtime.transport._subscriptions
    with pytest.raises(RuntimeRequestError,match='session_closed'):
        await agent.runtime.worker.start()
    await agent.aclose()
    replacement=await RunningAgent(store,_Scripted()).start()
    try:
        assert (await replacement.send('new worker')).status=='completed'
    finally:
        await replacement.aclose()
