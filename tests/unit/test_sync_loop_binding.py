"""A synchronous host owns one asyncio runner; SDK clients and pools stay on it."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, SessionStore
from tests.runtime_fixture import RunningAgent
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_subagent import _Scripted

__all__ = ['registry_store']
pytestmark = pytest.mark.unit


def test_synchronous_host_reuses_one_runner_for_start_requests_and_close(tmp_path):
    with asyncio.Runner() as runner:
        async def start():
            store = await SessionStore.open(str(tmp_path/'sync-host.db'))
            return await RunningAgent(store, _Scripted(), config=AgentLoopConfig(compactor=None)).start()
        agent = runner.run(start())
        try:
            for body in ('one', 'two', 'three'):
                result = runner.run(agent.send(body))
                assert result.status == 'completed'
            rows = runner.run(agent.runtime.store.load_active_messages(agent.target.session_id))
            assert [row.content for row in rows if row.role == 'user'] == ['one', 'two', 'three']
        finally:
            runner.run(agent.aclose())
            runner.run(agent.runtime.store.close())


async def test_repeated_runtime_calls_keep_pooled_store_on_owning_event_loop(registry_store):
    store, _ = registry_store
    calls = []
    class Model(_Scripted):
        async def complete(self, request, **kwargs):
            calls.append(asyncio.get_running_loop())
            return await super().complete(request, **kwargs)
    agent = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()
    try:
        for i in range(3):
            agent.target = await agent.loop.new_session(agent.descriptor, actor=agent.actor)
            assert (await agent.send(str(i))).status == 'completed'
        assert len(calls) == 3 and all(loop is asyncio.get_running_loop() for loop in calls)
    finally:
        await agent.aclose()
