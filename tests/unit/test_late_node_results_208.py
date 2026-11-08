"""Late descendant output is retained for explicit node resume, without replay."""
import asyncio
import json
from contextlib import nullcontext

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_late_child_is_retained_and_consumed_only_by_explicit_node_resume(registry_store, *, late_measure=None):
    store, _ = registry_store
    child_started, child_release = asyncio.Event(), asyncio.Event()
    class Model(LLMService):
        node_calls = 0
        child_calls = 0
        async def complete(self, request, **kwargs):
            prompt = request.system_prompt or ''
            if prompt.startswith('CHILD'):
                self.child_calls += 1
                child_started.set()
                await child_release.wait()
                return LLMResponse(raw_text='late child result')
            if prompt.startswith('NODE'):
                self.node_calls += 1
                if self.node_calls == 1:
                    return LLMResponse(raw_text='', tool_calls=[{'id':'delegate','type':'function','function':{
                        'name':'spawn_agent','arguments':json.dumps({'task':'work','system_prompt':'CHILD','tools':[]})}}])
                if self.node_calls == 3:
                    assert 'late child result' in str(request.messages)
                return LLMResponse(raw_text='node done')
            return LLMResponse(raw_text='workflow acknowledged')
        async def close(self):
            pass
    model, registry = Model(), ToolRegistry()
    register_spawn_agent(registry)
    agent = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None), capacity=2).start()
    try:
        workflow = await agent.runtime.client.workflow(agent.target,
            {'name':'late result','root':{'type':'agent','id':'node','input':'work',
             'spec':{'name':'node','system_prompt':'NODE','tools':['spawn_agent'],'max_rounds':1}}},
            message_id='workflow', actor=agent.actor)
        assert (await workflow.wait(deadline_ms=now_ms()+15000)).status == 'failed'
        await asyncio.wait_for(child_started.wait(), 5)
        async with late_measure(store) if late_measure is not None else nullcontext():
            child_release.set()
            await agent.drain()
        assert model.node_calls == 2 and model.child_calls == 1  # tool round plus forced summary
        [node] = await store._db.fetchall(f"SELECT session_id,generation FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        [child] = await store._db.fetchall(f"SELECT job_id,result_status FROM {store.table_prefix}jobs WHERE kind='subagent'")
        assert node['generation'] == 1 and child['result_status'] == 'available'
        [notice] = await store._db.fetchall(f"SELECT status,delivered_seq FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='task_done'", (node['session_id'],))
        assert notice['status'] == 'delivered' and notice['delivered_seq'] is not None
        assert any('late child result' in str(row.content) for row in await store.load_all_messages(node['session_id']))
        resumed = await workflow.resume(generation=1, node_id='node', message='Read the saved child result.', message_id='resume')
        assert (await resumed.wait(deadline_ms=now_ms()+15000)).status == 'completed'
        await agent.drain()
        assert model.node_calls == 3 and model.child_calls == 1
        current = await store._db.fetchone(f"SELECT result_status FROM {store.table_prefix}jobs WHERE job_id=?", (child['job_id'],))
        assert current['result_status'] == 'handled'
    finally:
        child_release.set()
        await agent.aclose()
