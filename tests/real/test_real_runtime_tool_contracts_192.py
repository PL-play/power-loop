"""A real model invokes named host control tools through their declared contract."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, HookPoint, ToolDefinition
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.runtime.tool_contracts import RuntimeTool, ToolDispatchPolicy, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('name',['read_probe','task_snapshot'])
async def test_real_host_control_base_needs_no_builtin_tool_name(tmp_path,record_property,runtime_agent,name):
    from power_loop import SessionStore
    store=await SessionStore.open(str(tmp_path/'host-contract.db'))
    provider=BoundedProvider(limit=2)
    runtime=TaskRuntime(JobManager(store),tenant_id='host-contract',principal_id='host',executor_id='worker',boot_id='boot',
        boot_contract_hash='a'*64)
    class Snapshot(RuntimeTool):
        calls=0
        @property
        def runtime_contract(self):
            return ToolRuntimeContract('example.original-snapshot',allowed_roles=('main',))
        def dispatch_policy(self,arguments,*,runtime=None):
            return ToolDispatchPolicy(native_admitted=True,control_only=True)
        async def __call__(self,**arguments):
            self.calls+=1
            assert arguments=={}
            execution=_SEND.get()
            original=await runtime.manager.get_job(execution.actor,execution.lease.job_id)
            assert original.status=='running'
            return 'SNAPSHOT_OK'
    implementation,registry=Snapshot(),ToolRegistry()
    definition=ToolDefinition(name,'Read the original task snapshot once, without arguments.',
        input_schema={'type':'object','properties':{}},pricing=Free('fixture'))
    registry.register(definition,implementation)
    agent=await runtime_agent(model=provider,store=store,registry=registry,task_runtime=runtime,
        config=AgentLoopConfig(max_rounds=3,max_tokens=2048,compactor=None))
    async def schema(ctx):
        ctx.tools=[definition.to_openai_tool()] if provider.calls==0 else []
    agent.hooks.register(HookPoint.LLM_BEFORE,schema)
    try:
        async with asyncio.timeout(150):
            result=await agent.send('Call '+name+' exactly once with no arguments. After SNAPSHOT_OK reply only DONE. Never call other tools.')
            assert result.status=='completed' and result.final_text.strip()=='DONE'
            assert implementation.calls==1
            assert provider.calls==2
            assert len(await runtime.manager.db.fetchall(f'SELECT job_id FROM {runtime.manager.jobs}'))==1
    finally:
        await agent.aclose()
        provider.report(record_property)
        record_property('snapshot_queries',implementation.calls)
        await store.close()
