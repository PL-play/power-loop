"""A workflow node retains its child's late real result for explicit continuation."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


@pytest.mark.parametrize('child_ending',['text','pass_then_text'])
async def test_real_nested_child_continues_after_node_exit_without_replacement(runtime_agent,record_property,child_ending):
    child_started,release=asyncio.Event(),asyncio.Event()
    class Provider(BoundedProvider):
        async def complete(self,request,**kwargs):
            if (request.system_prompt or '').startswith('CHILD'):
                child_started.set()
                await release.wait()
            return await super().complete(request,**kwargs)
    provider,registry,accepted=Provider(limit=16),ToolRegistry(),[]
    register_spawn_agent(registry)
    register_workflow_tool(registry,ports=WorkflowToolPorts(on_accepted=lambda value:accepted.append(value)))
    agent=await runtime_agent(model=provider,registry=registry,capacity=2,
        config=AgentLoopConfig(compactor=None,max_tokens=2048,max_rounds=4))
    child_prompt='CHILD. Compute 19+23 and return only 42 in assistant text. Do not use tools.'
    if child_ending=='pass_then_text':
        child_prompt='CHILD. The exact code is 42. First call pass_turn once without text. When asked for a final answer, return 42 without tools.'
    spec={'name':'nested','root':{'type':'sequence','steps':[
        {'type':'agent','id':'calculate','input':'Delegate the calculation once.',
         'spec':{'name':'calculate','tools':['spawn_agent'],'max_rounds':1,'system_prompt':
             'On the first task call spawn_agent exactly once with tools=[], max_rounds=4, task="Compute 19+23", system_prompt='+json.dumps(child_prompt)+
             '. When the original child result later appears, return its number in assistant text. Never launch replacement work.'}},
        {'type':'agent','id':'review','inputs_from':['calculate'],'input':'Read and verify the provided answer.',
         'spec':{'name':'review','tools':[],'max_rounds':2,'system_prompt':'Reply only with the verified number in assistant text.'}}]}}
    store=agent.runtime.store
    try:
        first=await agent.send('Call workflow action=create exactly once with spec='+json.dumps(spec)+'. Then reply WAITING. Later report any outcome without more tools.')
        assert first.status=='completed' and len(accepted)==1
        await asyncio.wait_for(child_started.wait(),10)
        handle=accepted[0].handle
        assert (await handle.wait(deadline_ms=now_ms()+120000)).status=='failed'
        release.set()
        await agent.drain()
        [child]=await store._db.fetchall(f"SELECT job_id,session_id FROM {store.table_prefix}jobs WHERE kind='subagent'")
        original_child=await store.load_all_messages(child['session_id'])
        assert any(r.role=='assistant' and '42' in (r.content or '') for r in original_child)
        resumed=await handle.resume(generation=1,node_id='calculate',message='Read the original child result and return its answer.',message_id='continue-original')
        result=await resumed.wait(deadline_ms=now_ms()+120000)
        assert result.status=='completed' and all('42' in result.results[n].text for n in ('calculate','review'))
        await agent.drain()
        assert await store.load_all_messages(child['session_id'])==original_child
        assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {store.table_prefix}jobs WHERE kind='subagent'"))['n']==1
    finally:
        release.set()
        provider.report(record_property)
