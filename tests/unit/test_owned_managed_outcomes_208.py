"""Owned finite effects keep original authority, budget and query-only outcomes."""
import json
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, ToolDefinition, ToolRegistry
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.managed_delivery import enqueue_runtime
from power_loop.runtime.managed_effects import (
    FunctionManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
    reconcile,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow.facade import register_workflow_tool
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_managed_delivery_192 import consume_notice
from .test_model_budgets_192 import Host

__all__=['registry_store']


def call(name,args,identity):
    return LLMResponse(raw_text='',tool_calls=[{'id':identity,'type':'function',
        'function':{'name':name,'arguments':json.dumps(args)}}])
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('kind',['subagent','workflow_node'])
@pytest.mark.parametrize('outcome',['known','unknown','missing_receipt','partial','exhausted'])
async def test_owned_effect_original_result_is_visible_without_replaying_action(registry_store,kind,outcome,*,provider=None):
    store,_=registry_store
    state=SimpleNamespace(main_calls=0,child_calls=0,prepared=0,executed=0,queries=0,resolved=False,raw_failed=False,raw_count=0)
    manager,host=JobManager(store),Host()
    if provider is not None:
        from tests.real.test_real_budget_tree_192 import RealBudgetHost
        host=RealBudgetHost()
    class Model(LLMService):
        async def complete(self,request,**kwargs):
            if request.system_prompt.startswith('CHILD'):
                state.child_calls+=1
                if state.child_calls==1:
                    response=call('original_effect',{},'effect')
                    if outcome == 'partial':
                        response.tool_calls += call('original_effect',{},'effect-2').tool_calls
                else:
                    response=LLMResponse(raw_text='Original child result')
            else:
                state.main_calls+=1
                if state.main_calls==1:
                    spec=dict(name='child',system_prompt='CHILD: Call original_effect exactly once, then state its result. Never retry an unknown action.',tools=['original_effect'],max_rounds=3)
                    if kind=='subagent':
                        response=call('spawn_agent',dict(task='act once',**spec),'spawn')
                    else:
                        response=call('workflow',dict(action='create',spec={'name':'original','root':{
                            'type':'agent','id':'act','input':'act once','spec':spec}}),'start')
                else:
                    response=LLMResponse(raw_text='Acknowledged original outcome')
            if provider is not None and request.system_prompt.startswith("CHILD"):
                return await provider.complete(request,**kwargs)
            response.token_usage=LLMTokenUsage(prompt_tokens=5,completion_tokens=5,total_tokens=10)
            return response
        async def close(self):pass
    model=Model()
    host.service=model
    runtime=TaskRuntime(manager,tenant_id='owned',principal_id='host',executor_id='original',boot_id='boot',boot_contract_hash='c'*64,
        budget_config=BudgetConfig('owned-v1',(LocalBudgetScope('shared',max_calls=12),
            LocalBudgetScope('child',max_calls=1 if outcome=='exhausted' else 2,binding='owner',owner_kinds=('subagent','workflow_node'))),host))
    async def prepare(context,arguments):
        assert arguments=={}
        state.prepared+=1
        return ManagedToolPreparation({'call_id':context.tool_call_id})
    async def execute(context,reference):
        state.executed+=1
        if outcome=='unknown':
            raise TimeoutError('external result unknown')
        return ManagedToolResult('Original effect result',reference)
    async def query(context,reference):
        state.queries+=1
        return ManagedToolResult('Original effect result',reference) if state.resolved else None
    adapter=FunctionManagedTool(ManagedToolContract('example.owned208','d'*64),prepare,execute,query)
    definition=ToolDefinition('original_effect','Perform original action once',input_schema={'type':'object'},pricing=Free('fixture'))
    registry=ToolRegistry()
    registry.register(definition,adapter.bind(runtime,definition))
    register_spawn_agent(registry)
    register_workflow_tool(registry)
    append=store.append_message
    async def original_receipt(sid,**options):
        if options.get('name') == 'original_effect' and options.get('role') == 'tool':
            state.raw_count += 1
            if (outcome == 'missing_receipt' or outcome == 'partial' and state.raw_count == 2) and not state.raw_failed:
                state.raw_failed = True
                raise OSError('original receipt write unavailable')
        return await append(sid,**options)
    store.append_message=original_receipt
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None,max_rounds=4,model="test-model" if provider is None else None,max_tokens=20 if provider is None else 2048),task_runtime=runtime,capacity=1).start()
    actor=runtime.actor(runner.target.session_id)
    try:
        initial=await runner.send('Launch exactly one original task')
        assert initial.status=='completed',initial
        try:
            await runner.drain()
        except OSError as exc:
            assert outcome in {'missing_receipt','partial'} and str(exc)=='original receipt write unavailable'
        assert state.raw_failed==(outcome in {'missing_receipt','partial'})
        effects=await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool' ORDER BY job_id")
        expected_calls = 2 if outcome == 'partial' else 1
        assert len(effects) == state.executed == state.prepared == expected_calls
        effect = next((row for row in effects if row['result_status'] != 'handled'), effects[0])
        original=await store.load_all_messages(effect['session_id'])
        if outcome == 'partial':
            replies = [row for row in original if row.role == 'tool']
            assert len(replies) == 1 and replies[0].tool_call_id == 'effect'
            assert sum(row['result_status'] == 'handled' for row in effects) == 1
        if outcome=='unknown':
            state.resolved=True
            await reconcile(runtime,actor,effect['job_id'],adapter)
        elif outcome in {'known','exhausted'}:
            assert effect['result_status']=='handled'
            if outcome=='exhausted':
                assert state.child_calls==1
                [owner]=await manager.db.fetchall(f"SELECT outcome FROM {manager.jobs} WHERE kind=?",(kind,))
                assert owner['outcome']=='failed'
            assert host.reserve_calls==len(host.applied) and state.queries==0
            return
        notice=await enqueue_runtime(runtime,actor,effect['job_id'])
        assert not notice['execution_authority'] and not notice['billing_settled']
        result=await consume_notice(runner,notice['notification_id'])
        assert result.status=='completed'
        assert (await manager.get_job(actor,effect['job_id'])).result_status=='handled'
        assert await store.load_all_messages(effect['session_id'])==original
        assert state.executed==state.prepared==expected_calls and state.queries==(1 if outcome=='unknown' else 0)
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.jobs} WHERE kind=?",(kind,)))['n']==1
        assert host.reserve_calls==len(host.applied) and host.reserve_calls>0
    finally:
        await runner.aclose()
