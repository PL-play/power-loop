"""Real model steers the original active child through the public runtime."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, InputRef, JobRef
from power_loop.agent.runtime_store import now_ms
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_nudge_is_durable_and_processed_in_the_original_active_child(runtime_agent,record_property):
    provider=BoundedProvider(limit=12)
    registry=ToolRegistry()
    register_spawn_agent(registry)
    agent=await runtime_agent(model=provider,registry=registry,capacity=2,
        config=AgentLoopConfig(max_rounds=4,max_tokens=2048,compactor=None,
            system_prompt='Follow requested tool sequence exactly. Do not create replacement tasks. When a child result arrives, report its number without tools or pass_turn.'))
    parent=agent.target.session_id
    provider.root_sid,provider.hold_children=parent,True
    store=agent.runtime.store
    try:
        async with asyncio.timeout(180):
            first=await agent.send("Call spawn_agent create once with tools=[], max_rounds=4, system_prompt='Compute the latest arithmetic request. Follow new corrections. Reply only with the final number, no pass_turn.', task='Compute 19+23'. Then reply WAITING without more tools.")
            assert first.status=='completed'
            await asyncio.wait_for(provider.child_entered.wait(),10)
            [created]=[json.loads(m.content) for m in await store.load_all_messages(parent) if m.role=='tool' and m.name=='spawn_agent']
            second=await agent.send("Use spawn_agent action=nudge once, sub_session='"+created['session_id']+"', message='Correction: compute 20+23 instead. Return only the resulting number.'. Then reply WAITING without more tools. Do not create another child.")
            assert second.status=='completed'
            rows=[m for m in await store.load_all_messages(parent) if m.role=='tool' and m.name=='spawn_agent']
            assert len(rows)==2
            nudged=json.loads(rows[-1].content)
            assert nudged['target']['job_id']==created['job_id'] and nudged['input_ref']['session_id']==created['session_id']
            queued=await agent.runtime.client.inputs.input(agent.runtime.client.inputs.db,InputRef(**nudged['input_ref']))
            assert json.loads(queued.row['content'])=='Correction: compute 20+23 instead. Return only the resulting number.'
            provider.release_children.set()
            result=await agent.runtime.client.wait(JobRef(**created['job_ref']),actor=agent.actor,deadline_ms=now_ms()+90000)
            await agent.drain()
            assert result.status=='completed' and result.final_text.strip()=='43' and result.job_ref.generation==1
            assert result.usage['total_tokens']>0
            assert len(await store.list_children(parent))==1
            assert any(m.role=='assistant' and '43' in (m.content or '') for m in await store.load_all_messages(parent))
    finally:
        provider.release_children.set()
        await agent.aclose()
        provider.report(record_property)
