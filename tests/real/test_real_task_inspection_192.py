"""A real main model reads a persisted alert and queries the original unknown job."""
import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    ExecutionEnvironment,
    HookPoint,
    InputRef,
    RegistryRuntimeFactory,
    Runtime,
    ToolRegistry,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_inspection import CONSTRAINTS, inspect_tasks
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.jobs import register_jobs_tools
from tests.unit.test_task_inspection_192 import POLICY, registry_runtime, registry_store, runtime, troubled

from ._bounded import BoundedProvider
from ._live_fixture import BorrowedProvider

__all__=['registry_store','registry_runtime','runtime']
pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_real_main_reads_original_alert_without_replaying_unknown_job(runtime,record_property):
    store,manager,actor,root,lease,connect,now=runtime
    target=await troubled(runtime)
    reader=JobManager(await connect(),now_ms=manager.now)
    [key]=(await inspect_tasks(reader,actor,root.job.job_id,policy=POLICY))['notification_ids']
    before=await manager.get_job(actor,target.job_id)
    provider,registry=BoundedProvider(limit=5),ToolRegistry()
    register_jobs_tools(registry)
    managed=TaskRuntime(manager,tenant_id=actor.tenant_id,principal_id=actor.principal_id,executor_id='alert-main',boot_id='boot')
    seen=[]
    hooks=AgentHooks()
    async def shape(ctx):
        seen.extend(m.get('content','') for m in ctx.messages if isinstance(m.get('content'),str))
        if any(m.get('role')=='tool' and target.job_id in m.get('content','') for m in ctx.messages):
            ctx.tools=[]
            ctx.system_prompt+=' The original task inspection returned. Reply with a short explanation now.'
    hooks.register(HookPoint.LLM_BEFORE,shape,name='test.actual-alert')
    factory=RegistryRuntimeFactory()
    async def build(descriptor,context):
        return ExecutionEnvironment(BorrowedProvider(provider),registry,AgentLoopConfig(compactor=None,max_rounds=4,max_tokens=2048,
            system_prompt='When a task alert arrives call jobs action=get with its affected job_id exactly once. Then explain the unknown state. Never create, stop or replace work.'),hooks=hooks,task_runtime=managed)
    factory.register('inspection@1',build)
    factory.register('inspection@2',build)
    worker=Runtime(store=store,factory=factory,node_id='inspector',timers_enabled=False,recovery_interval_s=None)
    trusted=TrustedActor(actor.tenant_id,actor.principal_id,None)
    try:
        await worker.worker.start()
        result=await worker.client.wait(InputRef(store.store_id,actor.session_id,key),actor=trusted,deadline_ms=now_ms()+120000)
        assert result.status=='completed' and result.final_text and provider.calls>=2
        rows=await store.load_all_messages(actor.session_id)
        tools=[r for r in rows if r.role=='tool' and r.name=='jobs']
        assert len(tools)==1 and target.job_id in tools[0].content
        assert any(CONSTRAINTS in text and target.job_id in text for text in seen)
        assert await manager.get_job(actor,target.job_id)==before
    finally:
        await worker.worker.shutdown(timeout_s=3,cancel_running=True)
        await worker.client.aclose()
        await worker.transport.aclose()
        provider.report(record_property)
        await provider.close()
