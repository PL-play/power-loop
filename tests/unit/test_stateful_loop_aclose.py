"""Facade shutdown releases its references; worker shutdown owns execution."""
import asyncio
from types import SimpleNamespace

import pytest

from power_loop import AgentEvent, AgentEventBus, AgentEventType, RuntimeClient, StatefulAgentLoop
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError
from tests.model_fixture import _Gate, _wait_calls, store
from tests.runtime_fixture import RunningAgent

__all__ = ['store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_closing_one_facade_does_not_cancel_execution_or_other_client(store):
    model = _Gate([])
    release = model.gate(0)
    agent = await RunningAgent(store, model).start()
    other = StatefulAgentLoop(runtime=SimpleNamespace(store=store, client=RuntimeClient(store=store, transport=agent.runtime.transport)))
    try:
        run = asyncio.create_task(agent.loop.send(agent.target, 'hello', message_id='one',
            actor=agent.actor, deadline_ms=now_ms()+10000))
        await _wait_calls(model, 1)
        ctx = agent.runtime.worker._active[agent.target.session_id]
        await agent.loop.aclose()
        assert (await asyncio.gather(run, return_exceptions=True))[0].__class__ is asyncio.CancelledError
        assert not ctx.stop.is_cancelled()
        release.set()
        result = await other.client.wait(ctx.input_ref, actor=agent.actor, deadline_ms=now_ms()+5000)
        assert result.status == 'completed'
        assert await store.get_session(agent.target.session_id)
        await agent.loop.aclose()
        with pytest.raises(RuntimeRequestError, match='session_closed'):
            await agent.loop.send(agent.target, 'no', message_id='closed', actor=agent.actor)
    finally:
        release.set()
        await other.aclose()
        await agent.aclose()


async def test_worker_shutdown_drains_then_closes_subscriptions(store):
    model = _Gate([])
    release = model.gate(0)
    agent = await RunningAgent(store, model).start()
    receipt = await agent.runtime.client.send(agent.target, 'running', message_id='one', actor=agent.actor)
    await _wait_calls(model, 1)
    shutdown = asyncio.create_task(agent.runtime.worker.shutdown(timeout_s=5))
    await asyncio.sleep(.01)
    assert not shutdown.done() and len(agent.runtime.transport._subscriptions) == 2
    release.set()
    result = await shutdown
    assert result.drained
    assert not agent.runtime.transport._subscriptions and not agent.runtime.worker._active
    # A separate authorized client can still read after the worker has stopped.
    assert (await agent.loop.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+5000)).status == 'completed'
    await agent.aclose()


async def test_shutdown_cancellation_is_persisted_before_releasing_worker(store):
    model = _Gate([])
    model.gate(0)
    agent = await RunningAgent(store, model).start()
    receipt = await agent.runtime.client.send(agent.target, 'running', message_id='one', actor=agent.actor)
    await _wait_calls(model, 1)
    result = await agent.runtime.worker.shutdown(timeout_s=3, cancel_running=True)
    assert result.drained
    result = await agent.loop.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+5000)
    assert result.status == 'cancelled'
    state = await store._db.fetchone(f'SELECT execution_state FROM {store.t.session_state} WHERE session_id=?',
                                    (agent.target.session_id,))
    assert state['execution_state'] == 'idle'
    await agent.aclose()


async def test_event_bus_drain_awaits_pending_async_subscribers():
    bus = AgentEventBus(suppress_subscriber_errors=True)
    done = []
    async def subscriber(event):
        await asyncio.sleep(.01)
        done.append(event)
    bus.subscribe(AgentEventType.SESSION_STARTED, subscriber)
    bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED))
    await bus.drain(timeout=1)
    assert len(done) == 1 and not bus._async_tasks


async def test_worker_shutdown_timeout_reports_interrupted_and_releases_slots(store):
    model = _Gate([])
    model.gate(0)
    agent = await RunningAgent(store, model).start()
    receipt = await agent.runtime.client.send(agent.target, 'running', message_id='one', actor=agent.actor)
    await _wait_calls(model, 1)
    result = await asyncio.wait_for(agent.runtime.worker.shutdown(timeout_s=.01), 5)
    assert not result.drained and len(result.interrupted_inputs) == 1
    status = await agent.loop.client.inspect(receipt.input_ref, actor=agent.actor)
    assert status.terminal_status is None
    assert status.execution_state == 'running'
    state = await store._db.fetchone(f'SELECT owner_token FROM {store.t.session_state} WHERE session_id=?', (agent.target.session_id,))
    assert state['owner_token'] is not None  # expired ownership is recovered by another worker
    assert not agent.runtime.worker._active and not agent.runtime.worker._tasks
    await agent.aclose()
