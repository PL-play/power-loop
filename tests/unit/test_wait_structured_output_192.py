"""A child resolves input and accepted nudges before publishing structured output."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, JobRef, LLMResponse, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
SCHEMA = {'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer']}


@pytest.mark.parametrize('ending',['valid','repair','cancel'])
async def test_original_wait_is_resolved_before_final_schema_repair(registry_store,ending):
    store,_ = registry_store
    class Model(_Scripted):
        sent = False
        child_calls = []
        async def complete(self,request,**kwargs):
            if request.system_prompt.startswith('CHILD'):
                self.child_calls.append(request)
                if len(self.child_calls)==1:
                    return _tool_resp('question','request_user_input','{"prompt":"Which number?"}')
                if len(self.child_calls)==2 and ending=='repair':
                    return LLMResponse(raw_text='The number is 42.')
                return LLMResponse(raw_text='{"answer":42}')
            if not self.sent:
                self.sent=True
                return _tool_resp('spawn','spawn_agent',json.dumps({'task':'answer','system_prompt':'CHILD',
                    'tools':['request_user_input'],'output_schema':SCHEMA,'max_rounds':5}))
            return LLMResponse(raw_text='parent done')
    model=Model()
    model.child_calls=[]
    registry=create_default_tool_registry(include=['request_user_input'],bind=False)
    register_spawn_agent(registry)
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None)).start()
    try:
        await runner.send('delegate')
        rows=await store.load_all_messages(runner.target.session_id)
        receipt=json.loads(next(row.content for row in rows if row.tool_call_id=='spawn'))
        ref=JobRef(**receipt['job_ref'])
        async with asyncio.timeout(10):
            while True:
                state=await store.get_state(receipt['session_id'])
                if state.pending and state.pending.get('runtime_wait'):
                    break
                await asyncio.sleep(.02)
        status=await runner.runtime.client.inspect(ref,actor=runner.actor)
        assert status.state=='waiting_input' and not status.result_available and len(model.child_calls)==1
        if ending=='cancel':
            await runner.runtime.client.cancel(status.job_ref,actor=runner.actor)
        else:
            question=state.pending['pending_interactions'][0]
            await runner.runtime.client.answer(status.job_ref,interaction_id=question['interaction_id'],
                wait_round=state.pending['round_index'],value='42',message_id='answer',actor=runner.actor)
        result=await runner.runtime.client.wait(ref,actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status==('cancelled' if ending=='cancel' else 'completed')
        assert len(model.child_calls)=={'valid':2,'repair':3,'cancel':1}[ending]
        assert result.structured_output==(None if ending=='cancel' else {'answer':42})
        if ending=='repair':
            assert not model.child_calls[-1].tools
        assert len(await store.list_children(runner.target.session_id))==1
        await runner.drain()
    finally:
        await runner.aclose()


@pytest.mark.parametrize('status',['failed','cancelled','hit_round_limit','degraded','timed_out'])
async def test_schema_shaped_negative_result_does_not_become_success(status):
    from types import SimpleNamespace

    from power_loop import RequestOptions
    from power_loop.agent.runtime_worker import _InputStart
    context=SimpleNamespace(options=RequestOptions(response_format={'type':'json_schema','json_schema':{'name':'Answer','schema':SCHEMA}}),
        descriptor=SimpleNamespace(config={'child_spec':{}}))
    start=_InputStart(None,context,None,1)
    result=SimpleNamespace(status=status,final_text='{"answer":42}')
    start.completed(result)
    assert result.status==status and start.structured_output is None
