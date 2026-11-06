"""Late nudges and output repair stay in the original child execution."""
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_subagent_runtime_203 import ParentAndChild

__all__=['registry_store']
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('mode',['repair','nudge_at_finish'])
async def test_child_finishes_original_work_before_delivering(registry_store,monkeypatch,mode):
    store,_=registry_store
    model,registry=ParentAndChild(),ToolRegistry()
    register_spawn_agent(registry)
    original=model.complete
    child_requests=[]
    async def complete(request,**kwargs):
        if request.system_prompt.startswith('CHILD'):
            child_requests.append(request)
            return LLMResponse(raw_text=('not JSON' if len(child_requests)==1 else '{"value":43}') if mode=='repair'
                               else ('old result' if len(child_requests)==1 else 'new direction applied'))
        response=await original(request,**kwargs)
        if mode=='repair' and response.tool_calls:
            args=json.loads(response.tool_calls[0]['function']['arguments'])
            args['output_schema']={'type':'object','properties':{'value':{'type':'integer'}},'required':['value']}
            response.tool_calls[0]['function']['arguments']=json.dumps(args)
        return response
    model.complete=complete
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None)).start()
    if mode=='nudge_at_finish':
        from power_loop.runtime import child_completion
        finish=child_completion.continue_before_finish
        injected=False
        async def nudge_first(pipeline,text,index):
            nonlocal injected
            if pipeline.system_prompt.startswith('CHILD') and not injected:
                injected=True
                ctx=next(c for c in runner.runtime.worker._active.values() if c and c.session_id==pipeline.session_id)
                await runner.runtime.client.nudge(ctx.job_ref,'new direction',message_id='late',actor=runner.actor)
            return await finish(pipeline,text,index)
        monkeypatch.setattr(child_completion,'continue_before_finish',nudge_first)
    try:
        await runner.send('delegate')
        rows=await store.load_all_messages(runner.target.session_id)
        receipt=json.loads(next(r.content for r in rows if r.tool_call_id=='spawn'))
        result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status=='completed' and result.job_ref.generation==1
        assert len(child_requests)==2
        if mode=='repair':
            assert result.structured_output=={'value':43}
            assert not child_requests[-1].tools
        else:
            assert 'new direction' in str(child_requests[-1].messages)
            assert result.final_text=='new direction applied'
        await runner.drain()
    finally:
        await runner.aclose()


async def test_child_collects_its_background_result_before_final_answer(registry_store,tmp_path):
    import asyncio

    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.pricing import Free
    from power_loop.tools import create_default_tool_registry
    store,_=registry_store
    model=ParentAndChild()
    registry=create_default_tool_registry(workspace_dir=str(tmp_path))
    register_spawn_agent(registry)
    gate,premature=asyncio.Event(),asyncio.Event()
    async def slow():
        await gate.wait()
        return 'original evidence'
    registry.register(ToolDefinition('slow','collect evidence',async_capable=True,pricing=Free('local')),slow)
    original=model.complete
    child_calls=0
    async def complete(request,**kwargs):
        nonlocal child_calls
        if request.system_prompt.startswith('CHILD'):
            child_calls+=1
            if child_calls==1:
                return LLMResponse(raw_text='',tool_calls=[{'id':'background','type':'function','function':{
                    'name':'background_run','arguments':json.dumps({'action':'tool','tool':'slow','args':{}})}}])
            if child_calls==2:
                premature.set()
                return LLMResponse(raw_text='premature answer')
            assert 'original evidence' in str(request.messages)
            return LLMResponse(raw_text='final answer from original evidence')
        response=await original(request,**kwargs)
        if response.tool_calls:
            args=json.loads(response.tool_calls[0]['function']['arguments'])
            args['tools']=['background_run','slow']
            response.tool_calls[0]['function']['arguments']=json.dumps(args)
        return response
    model.complete=complete
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None)).start()
    try:
        await runner.send('delegate')
        rows=await store.load_all_messages(runner.target.session_id)
        receipt=json.loads(next(r.content for r in rows if r.tool_call_id=='spawn'))
        ref=JobRef(**receipt['job_ref'])
        await asyncio.wait_for(premature.wait(),5)
        assert not (await runner.runtime.client.inspect(ref,actor=runner.actor)).result_available
        gate.set()
        result=await runner.runtime.client.wait(ref,actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status=='completed' and result.final_text=='final answer from original evidence'
        await runner.drain()
        jobs=await store._db.fetchall(f"SELECT result_status,supervision_status FROM {store.table_prefix}jobs WHERE kind='background_tool'")
        assert jobs==[{'result_status':'handled','supervision_status':'closed'}]
    finally:
        gate.set()
        await runner.aclose()
