"""Append and background results retain projected history on the current runtime."""
import asyncio
import json

from power_loop import AgentLoopConfig, JobRef, LLMResponse, ProjectedRepresentation, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free
from power_loop.tools import create_default_tool_registry
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

from .test_background_runtime_203 import BackgroundParent
from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ['registry_store']


async def test_idle_append_is_a_new_projected_send(registry_store):
    store,_=registry_store
    registry=ToolRegistry()
    registry.register(ToolDefinition('echo','echo',pricing=Free('test')),lambda **kw:'a')
    model=_Scripted([_tool_resp('echo','echo','{}'),LLMResponse(raw_text='done1'),LLMResponse(raw_text='done2')])
    agent=await RunningAgent(store,model,registry,AgentLoopConfig(representation=ProjectedRepresentation(),compactor=None)).start()
    try:
        await agent.send('first')
        assert {(r.send_index,r.kind) for r in await store.load_project_messages(agent.target.session_id)}=={(1,'user'),(1,'project')}
        accepted=await agent.runtime.client.send(agent.target,'second',message_id='append',actor=agent.actor)
        result=await agent.runtime.client.wait(accepted.input_ref,actor=agent.actor,deadline_ms=now_ms()+10000)
        assert result.status=='completed'
        await agent.drain(after_input=accepted.input_ref)
        assert {(r.send_index,r.kind) for r in await store.load_project_messages(agent.target.session_id)}=={(1,'user'),(1,'project'),(2,'user'),(2,'project')}
    finally:
        await agent.aclose()


async def test_background_completion_is_delivered_once_and_remains_queryable(registry_store,tmp_path):
    store,_=registry_store
    release=asyncio.Event()
    async def slow():
        await release.wait()
        return 'BUILD OK'
    registry=create_default_tool_registry(workspace_dir=tmp_path)
    registry.register(ToolDefinition('slow','slow',async_capable=True,pricing=Free('test')),slow)
    agent=await RunningAgent(store,BackgroundParent(),registry,
        AgentLoopConfig(representation=ProjectedRepresentation(),compactor=None)).start()
    try:
        await agent.send('background build')
        rows=await store.load_all_messages(agent.target.session_id)
        receipt=json.loads(next(r.content for r in rows if r.role=='tool'))
        ref=JobRef(**receipt['job_ref'])
        release.set()
        original=await agent.runtime.client.wait(ref,actor=agent.actor,deadline_ms=now_ms()+10000)
        assert original.final_text=='BUILD OK'
        await agent.drain()
        await agent.send('next turn')
        rows=await store.load_all_messages(agent.target.session_id)
        notices=[r for r in rows if '[background_tool_results]' in str(r.content)]
        assert len(notices)==1 and 'BUILD OK' in notices[0].content
        # Status panels may query repeatedly without reinjecting a result.
        assert await agent.runtime.client.wait(ref,actor=agent.actor,deadline_ms=now_ms()+10000)==original
        assert len([r for r in await store.load_all_messages(agent.target.session_id)
                    if '[background_tool_results]' in str(r.content)])==1
    finally:
        release.set()
        await agent.aclose()
