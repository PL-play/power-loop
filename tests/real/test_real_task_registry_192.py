"""Real providers use current input/job references and persisted tool receipts."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, JobRef, ToolDefinition, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free
from power_loop.tools.spawn_agent import register_spawn_agent

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_tool_call_and_receipt_have_separate_recallable_message_ids(runtime_agent, record_property):
    provider, registry = BoundedProvider(limit=4), ToolRegistry()
    async def add(a,b):
        return json.dumps({'answer':a+b})
    registry.register(ToolDefinition('local_add','Add two integers locally.',input_schema={
        'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']},pricing=Free('fixture')),add)
    agent=await runtime_agent(model=provider,registry=registry,config=AgentLoopConfig(compactor=None,max_rounds=3,max_tokens=2048))
    store=agent.runtime.store
    try:
        result=await agent.send('Call local_add exactly once for 19+23, then reply with its answer.')
        assert result.status=='completed' and '42' in result.final_text
        rows=await store.load_all_messages(agent.target.session_id)
        [receipt]=[r for r in rows if r.role=='tool' and r.name=='local_add']
        [call]=[r for r in rows if any(t['id']==receipt.tool_call_id for t in r.tool_calls or [])]
        call_ref=await store.message_ref(agent.target.session_id,call.seq)
        receipt_ref=await store.message_ref(agent.target.session_id,receipt.seq)
        assert call_ref.id!=receipt_ref.id and call_ref.role=='assistant' and receipt_ref.role=='tool'
        assert (await store.message_by_id(agent.target.session_id,receipt_ref.id)).content==receipt.content
        assert await agent.runtime.client.wait(result.input_ref,actor=agent.actor,deadline_ms=now_ms()+1000)==result
    finally:
        provider.report(record_property)


@pytest.mark.parametrize('ending',['natural','pass_turn'])
async def test_real_parent_and_child_have_original_recallable_result(runtime_agent,record_property,ending):
    provider,registry=BoundedProvider(limit=8),ToolRegistry()
    register_spawn_agent(registry)
    agent=await runtime_agent(model=provider,registry=registry,config=AgentLoopConfig(compactor=None,
        max_rounds=4,max_tokens=2048,delegation_policy='single_level'))
    provider.root_sid,provider.hold_children=agent.target.session_id,True
    try:
        finish='call pass_turn with reason waiting for child' if ending=='pass_turn' else 'reply WAITING without tools'
        first=await agent.send("Call spawn_agent exactly once with tools=[], max_rounds=2, task='Compute 19+23', "
            "system_prompt='Return the arithmetic answer as plain assistant text. Do not call pass_turn.'. Then "+finish+
            '. When its result arrives later report the answer without starting another task.')
        assert first.status=='completed' and first.passed==(ending=='pass_turn')
        await asyncio.wait_for(provider.child_entered.wait(),5)
        store=agent.runtime.store
        [tool]=[r for r in await store.load_all_messages(agent.target.session_id) if r.role=='tool' and r.name=='spawn_agent']
        accepted=json.loads(tool.content)
        ref=JobRef(**accepted['job_ref'])
        assert not (await agent.runtime.client.inspect(ref,actor=agent.actor)).result_available
        provider.release_children.set()
        child=await agent.runtime.client.wait(ref,actor=agent.actor,deadline_ms=now_ms()+90000)
        assert child.status=='completed' and '42' in child.final_text and child.usage['total_tokens']>0
        await agent.drain()
        assert (await store.inbox_counts(agent.target.session_id))['pending']==0
        assert any(r.role=='assistant' and '42' in (r.content or '') for r in await store.load_all_messages(agent.target.session_id))
        assert await agent.runtime.client.wait(first.input_ref,actor=agent.actor,deadline_ms=now_ms()+1000)==first
        assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {store.table_prefix}jobs WHERE kind='subagent'"))['n']==1
    finally:
        provider.release_children.set()
        provider.report(record_property)
