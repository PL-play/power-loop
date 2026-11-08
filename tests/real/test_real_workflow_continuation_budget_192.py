"""Workflow continuations count every real node request in the current budget."""
import pytest

from power_loop import AgentLoopConfig, ToolDefinition, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.pricing import Free

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_real_continuation_budget_equals_actual_reported_requests(runtime_agent,record_property):
    node_responses=[]
    class Provider(BoundedProvider):
        async def complete(self,request,**kwargs):
            response=await super().complete(request,**kwargs)
            if get_session_id()!=self.root_sid:
                node_responses.append(response)
            return response
    provider,registry=Provider(limit=8),ToolRegistry()
    touched=[]
    async def touch():
        touched.append(True)
        return 'done; the result code is 42'
    registry.register(ToolDefinition('touch','Read the result code once.',input_schema={'type':'object','properties':{}},pricing=Free('fixture')),touch)
    agent=await runtime_agent(model=provider,registry=registry,config=AgentLoopConfig(compactor=None,max_tokens=2048))
    provider.root_sid=agent.target.session_id
    spec={'name':'real-continuation-budget','budget':{'max_tokens':50000},'root':{
        'type':'agent','id':'calculate','input':'Read the result code with touch and report it.',
        'spec':{'name':'calculate','tools':['touch'],'max_rounds':1,'system_prompt':
            'First call touch exactly once. After its recorded result, return exactly 42 without tools. If asked to finalize, return 42.'},
        'continuation':{'max_continuations':1,'extra_rounds':2,'gate':'always'}}}
    try:
        handle=await agent.runtime.client.workflow(agent.target,spec,message_id='original',actor=agent.actor)
        result=await handle.wait(deadline_ms=now_ms()+120000)
        assert result.status=='completed' and '42' in result.results['calculate'].text
        actual=sum(r.token_usage.total_tokens for r in node_responses)
        assert len(touched)==1 and len(node_responses)>=2
        assert result.usage['total_tokens']==result.results['calculate'].usage['total_tokens']==actual>0
        record_property('actual_node_tokens',actual)
        assert (await agent.runtime.store._db.fetchone(f"SELECT COUNT(*) AS n FROM {agent.runtime.store.table_prefix}jobs WHERE kind='workflow_node'"))['n']==1
    finally:
        provider.report(record_property)
