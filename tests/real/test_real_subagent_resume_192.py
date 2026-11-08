"""A real finished child resumes with the same session and a distinct result generation."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, InputRef, JobRef
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_finished_subagent_resumes_original_session_without_overwriting_result(runtime_agent,record_property):
    provider=BoundedProvider(limit=14)
    registry=ToolRegistry()
    register_spawn_agent(registry)
    agent=await runtime_agent(model=provider,registry=registry,
        config=AgentLoopConfig(max_rounds=4,max_tokens=2048,compactor=None,
            system_prompt='Follow requested tool sequence exactly. An acceptance receipt is not a result. Never create replacement tasks. When a child result arrives report its number without tools or pass_turn.'))
    parent=agent.target.session_id
    provider.root_sid=parent
    store=agent.runtime.store
    try:
        async with asyncio.timeout(180):
            await agent.send("Call spawn_agent create once with tools=[], max_rounds=4, system_prompt='Compute the latest arithmetic request. Follow new corrections. Reply only with the final number, no pass_turn.', task='Compute 19+23'. Then reply WAITING without more tools.")
            [created]=[json.loads(m.content) for m in await store.load_all_messages(parent) if m.role=='tool' and m.name=='spawn_agent']
            client=agent.runtime.client
            first=await client.wait(JobRef(**created['job_ref']),actor=agent.actor,deadline_ms=now_ms()+90000)
            await agent.drain()
            assert first.final_text.strip()=='42' and first.usage['total_tokens']>0
            await agent.send("Use spawn_agent action=nudge once, sub_session='"+created['session_id']+"', message='New request: compute 20+23 instead. Return only the number.'. Then reply WAITING without more tools; do not create another child.")
            rows=[m for m in await store.load_all_messages(parent) if m.role=='tool' and m.name=='spawn_agent']
            assert len(rows)==2
            resumed=json.loads(rows[-1].content)
            assert resumed['target']['job_id']==created['job_id'] and resumed['input_ref']['session_id']==created['session_id']
            current=await client.wait(InputRef(**resumed['input_ref']),actor=agent.actor,deadline_ms=now_ms()+90000)
            await agent.drain()
            assert current.status=='completed' and current.final_text.strip()=='43'
            assert current.job_ref.job_id==first.job_ref.job_id and current.job_ref.generation==2
            assert await client.wait(first.job_ref,actor=agent.actor,deadline_ms=now_ms()+1000)==first
            assert current.usage['total_tokens']>0 and len(await store.list_children(parent))==1
            manager=JobManager(store)
            actor=JobActor(agent.actor.tenant_id,agent.actor.principal_id,parent)
            lifetime=await manager.execution_usage(actor,created['job_id'])
            assert lifetime['total_tokens']==first.usage['total_tokens']+current.usage['total_tokens']
    finally:
        await agent.aclose()
        provider.report(record_property)
