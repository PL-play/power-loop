"""Independent children remain queryable and controllable after the parent ends."""
import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, JobRef, LLMResponse, LLMService, LLMTokenUsage, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.core.agent_context import get_session_id
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('control', ['cancel', 'nudge', 'foreign'])
async def test_two_children_keep_separate_control_inputs_results_and_usage(registry_store, runtime_agent, control):
    store, _ = registry_store
    entered = {key: asyncio.Event() for key in 'AB'}
    release = {key: asyncio.Event() for key in 'AB'}
    calls = {'MAIN':0, 'A':0, 'B':0}
    class Model(LLMService):
        root = None
        async def complete(self, request, **kwargs):
            role = 'MAIN' if get_session_id()==self.root else ('B' if 'ROLE-B' in (request.system_prompt or '') else 'A')
            calls[role] += 1
            if role=='MAIN':
                tools = ([{'id':'spawn-'+key, 'type':'function', 'function':{'name':'spawn_agent', 'arguments':json.dumps({
                    'name':key,'task':'work '+key,'system_prompt':'ROLE-'+key,'tools':[],'max_rounds':3})}} for key in 'AB']
                    if calls[role]==1 else None)
                return LLMResponse(raw_text='' if tools else 'parent finished', tool_calls=tools,
                    token_usage=LLMTokenUsage(prompt_tokens=7,completion_tokens=3,total_tokens=10))
            entered[role].set()
            await release[role].wait()
            return LLMResponse(raw_text='RESULT-'+role, token_usage=LLMTokenUsage(prompt_tokens=7,completion_tokens=3,total_tokens=10))
        async def close(self):
            pass
    model, registry = Model(), ToolRegistry()
    register_spawn_agent(registry)
    agent = await runtime_agent(store=store, model=model, registry=registry, capacity=2,
        config=AgentLoopConfig(compactor=None, max_rounds=1))
    model.root = agent.target.session_id
    try:
        parent = await agent.send('Start the independent children')
        assert parent.status == 'interrupted' and parent.usage['total_tokens'] >= 10
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 10)
        rows = await store.load_all_messages(model.root)
        accepted = {key:json.loads(next(row.content for row in rows if row.tool_call_id=='spawn-'+key)) for key in 'AB'}
        refs = {key:JobRef(**item['job_ref']) for key,item in accepted.items()}
        current = {key:(await agent.runtime.client.inspect(ref,actor=agent.actor)).job_ref for key,ref in refs.items()}
        client = agent.runtime.client
        if control=='cancel':
            await client.cancel(current['A'], actor=agent.actor, reason='stop only A')
        elif control=='nudge':
            await client.nudge(current['A'], 'ONLY-A-NOTE', message_id='note-A', actor=agent.actor)
        else:
            with pytest.raises(RuntimeRequestError,match='forbidden'):
                await client.nudge(current['B'], 'FORBIDDEN',message_id='foreign',
                    actor=replace(agent.actor,session_id=accepted['A']['session_id']))
        release['A'].set()
        release['B'].set()
        results = {key:await client.wait(ref,actor=agent.actor,deadline_ms=now_ms()+10000) for key,ref in refs.items()}
        await agent.drain()
        assert results['B'].status=='completed' and results['B'].final_text=='RESULT-B'
        assert results['B'].usage['total_tokens']==10 and calls['B']==1
        assert results['A'].status==('cancelled' if control=='cancel' else 'completed')
        a_rows = await store.load_all_messages(accepted['A']['session_id'])
        b_rows = await store.load_all_messages(accepted['B']['session_id'])
        if control=='nudge':
            assert any('ONLY-A-NOTE' in str(row.content) for row in a_rows)
        assert not any('ONLY-A-NOTE' in str(row.content) or 'FORBIDDEN' in str(row.content) for row in b_rows)
        await client.cancel(current['B'], actor=agent.actor, reason='already finished')
        assert await client.wait(refs['B'], actor=agent.actor, deadline_ms=now_ms()+1000)==results['B']
        assert await client.wait(parent.input_ref, actor=agent.actor, deadline_ms=now_ms()+1000)==parent
        assert len([row for row in rows if row.tool_call_id and row.tool_call_id.startswith('spawn-')])==2
    finally:
        for event in release.values():
            event.set()
        await agent.aclose()
