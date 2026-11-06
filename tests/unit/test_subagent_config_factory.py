"""Child config isolation through the public runtime and persisted admission."""
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, AgentSpec, LLMResponse, LLMService, SessionStore
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent


@pytest.fixture
async def store():
    value = await SessionStore.open(':memory:', schema='tasks')
    yield value
    await value.close()


class _Capturing(LLMService):
    def __init__(self):
        self.seen=[]
    async def complete(self, request, **kwargs):
        self.seen.append(request)
        if len(self.seen)==1:
            return LLMResponse(raw_text='',tool_calls=[{'id':'delegate','type':'function',
                'function':{'name':'delegate','arguments':'{}'}}])
        return LLMResponse(raw_text='child out' if request.system_prompt.startswith('p') else 'parent done')
    async def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['rewrite','default','bad_type','exception','system_prompt'])
async def test_child_factory_is_applied_before_admission_and_on_worker(store,mode):
    seen=[]
    def factory(spec,default):
        seen.append((spec.name,default))
        if mode=='bad_type':
            return {'not':'config'}
        if mode=='exception':
            raise RuntimeError('host factory bug')
        if mode=='system_prompt':
            return replace(default,system_prompt=default.system_prompt+' HOST NOTE')
        return replace(default,temperature=.9,max_tokens=1234)
    model, registry=_Capturing(),ToolRegistry()
    config=AgentLoopConfig(compactor=None,max_rounds=3,
        subagent_config_factory=None if mode=='default' else factory)
    async def delegate():
        return await run_agent_spec(AgentSpec(name='kid',system_prompt='p',max_tokens=500,temperature=.2),
                                    'task',parent_loop=get_current_loop())
    registry.register(ToolDefinition(name='delegate',description='delegate',pricing=Free('composite')),
        FunctionRuntimeTool(delegate,ToolRuntimeContract('sdk.subagent',background_allowed=False,allowed_roles=('main',)),
                            policy=lambda arguments: ToolDispatchPolicy(native_admitted=True)))
    runner=await RunningAgent(store,model,registry,config).start()
    try:
        result=await runner.send('delegate')
        assert result.status=='completed'
        children=await store._db.fetchall(f"SELECT session_id FROM {store.table_prefix}jobs WHERE kind='subagent'")
        requests=[r for r in model.seen if r.system_prompt.startswith('p')]
        if mode in {'bad_type','exception'}:
            assert not children and not requests
            rows=await store.load_all_messages(runner.target.session_id)
            text=next(r.content for r in rows if r.tool_call_id=='delegate')
            assert ('subagent_config_factory must return' if mode=='bad_type' else 'host factory bug') in text
        else:
            assert len(children)==1 and len(requests)==1
            request=requests[0]
            assert request.max_tokens==(1234 if mode=='rewrite' else 500)
            assert request.temperature==(.9 if mode=='rewrite' else .2)
            if mode=='system_prompt':
                assert 'HOST NOTE' in request.system_prompt
            if seen:
                assert seen[0][0]=='kid' and seen[0][1].max_tokens==500
        assert config.max_rounds==3 and config.temperature==0
    finally:
        await runner.aclose()
