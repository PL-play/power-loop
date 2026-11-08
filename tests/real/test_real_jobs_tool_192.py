"""A real model reads task state and requests cancellation through the jobs tool."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, JobRef, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.tools.spawn_agent import register_spawn_agent

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_jobs_backend_valueerror_stays_private_and_is_not_an_empty_task_list(runtime_agent,record_property,monkeypatch):
    from power_loop.runtime import task_query
    provider=BoundedProvider(limit=4)
    registry=create_default_tool_registry(include=['jobs'])
    agent=await runtime_agent(model=provider,registry=registry,
        config=AgentLoopConfig(max_rounds=3,max_tokens=2048,compactor=None,
            system_prompt='Call jobs action=list exactly once. If it fails, explain that the directory is unavailable. Never retry, infer an empty list, or call pass_turn.'))
    async def unavailable(*args,**kwargs):
        raise ValueError('PRIVATE-CONNECTION-SECRET')
    monkeypatch.setattr(task_query,'list_jobs',unavailable)
    try:
        async with asyncio.timeout(90):
            result=await agent.send('Inspect the task directory once with jobs action=list, then report the actual outcome as assistant text.')
            assert result.status=='completed' and result.final_text.strip()
            rows=await agent.runtime.store.load_all_messages(agent.target.session_id)
            [reply]=[m for m in rows if m.role=='tool' and m.name=='jobs']
            assert json.loads(reply.content)=={'version':1,'error':{'code':'jobs_backend_unavailable'}}
            assert 'PRIVATE-CONNECTION-SECRET' not in str([(m.content,m.tool_calls) for m in rows])
    finally:
        await agent.aclose()
        provider.report(record_property)


async def test_real_jobs_list_get_stop_preserves_original_task_and_request_identity(runtime_agent,record_property):
    provider=BoundedProvider(limit=12)
    registry=create_default_tool_registry(include=['jobs'])
    register_spawn_agent(registry)
    agent=await runtime_agent(model=provider,registry=registry,capacity=2,
        config=AgentLoopConfig(max_rounds=6,max_tokens=2048,compactor=None,
            system_prompt='Follow requested tool sequence exactly. Create only once. A cancellation receipt is not proof of resource stop or billing settlement. Never create replacement work.'))
    sid=agent.target.session_id
    provider.root_sid,provider.hold_children=sid,True
    try:
        async with asyncio.timeout(180):
            await agent.send("Call spawn_agent create once, task='Compute 19+23', tools=[], max_rounds=3. Then reply WAITING without more tools or pass_turn.")
            await asyncio.wait_for(provider.child_entered.wait(),10)
            [created]=[json.loads(m.content) for m in await agent.runtime.store.load_all_messages(sid) if m.role=='tool' and m.name=='spawn_agent']
            await agent.send("Call jobs in this exact order: action=list, kinds=['subagent']; action=get, job_id='"+created['job_id']+"'; action=stop with that same job_id and reason='user stop'. Then acknowledge in assistant text. Do not poll or create work.")
            rows=[m for m in await agent.runtime.store.load_all_messages(sid) if m.role=='tool' and m.name=='jobs']
            assert len(rows)==3
            listed,detail,stopped=[json.loads(m.content) for m in rows]
            assert listed['matched_count']==1 and detail['node']['job_id']==created['job_id']
            assert stopped['status']=='accepted' and stopped['receipt']['job_ref']['job_id']==created['job_id']
            assert all(m.tool_call_id for m in rows)
            result=await agent.runtime.client.wait(JobRef(**created['job_ref']),actor=agent.actor,deadline_ms=now_ms()+15000)
            await agent.drain()
            assert result.status=='cancelled' and len(await agent.runtime.store.list_children(sid))==1
    finally:
        provider.release_children.set()
        await agent.aclose()
        provider.report(record_property)
