"""A trusted root stop blocks future work under that origin and retains the original usage."""
import asyncio
from contextlib import ExitStack

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    SessionStore,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime import root_control
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from ._bounded import BoundedProvider
from ._live_fixture import BorrowedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]
NAMESPACE='deeptalk.agent-run.v1'
PAYLOAD={'origin_run_id':101}


@pytest.mark.parametrize('inflight',[False,True])
async def test_real_context_stop_never_renews_execution_and_retains_original_usage(tmp_path,record_property,inflight,monkeypatch):
    store=await SessionStore.open(str(tmp_path/'root-stop.db'))
    provider=BoundedProvider(limit=1)
    manager=JobManager(store)
    task_runtime=TaskRuntime(manager,tenant_id='real-context',principal_id='host',executor_id='worker',boot_id='boot')
    factory=RegistryRuntimeFactory()
    class Resources:
        def __init__(self,scope):
            self.scope=scope
        async def aclose(self):
            self.scope.close()
    async def build(descriptor,context):
        scope=ExitStack()
        scope.enter_context(task_runtime.root_context(context.session_id,namespace=NAMESPACE,payload=PAYLOAD))
        return ExecutionEnvironment(BorrowedProvider(provider),None,AgentLoopConfig(max_rounds=2,max_tokens=2048,compactor=None),
            task_runtime=task_runtime,resources=Resources(scope))
    factory.register('root-real@1',build)
    runtime=Runtime(store=store,factory=factory,node_id='root-real',timers_enabled=False)
    actor=TrustedActor('real-context','host',None)
    descriptor=RuntimeDescriptor(1,'agent','root-real@1',None,'builtin.empty@1','builtin.none@1','builtin.none@1',{}, {})
    entered=asyncio.Event()
    complete=provider.complete
    async def started(request,**kwargs):
        work=asyncio.create_task(complete(request,**kwargs))
        while provider.calls==0:
            await asyncio.sleep(0)
        entered.set()
        return await work
    monkeypatch.setattr(provider,'complete',started)
    try:
        await runtime.worker.start()
        target=await runtime.open_session(descriptor,actor=actor)
        client=runtime.client
        async with asyncio.timeout(90):
            original=await client.send(target,'Reply exactly 42 without tools or pass_turn.',message_id='original',actor=actor)
            await entered.wait()
            if not inflight:
                first=await client.wait(original.input_ref,actor=actor,deadline_ms=now_ms()+60000)
                assert first.status=='completed' and first.final_text.strip()=='42'
                assert first.usage['total_tokens']>0
            stopped=await root_control.cancel(JobManager(store),task_runtime.actor(target.session_id),namespace=NAMESPACE,
                payload=PAYLOAD,system_key='original-stop',reason='user stop')
            assert len(stopped['roots'])==1
            assert not stopped['resource_stop_confirmed'] and not stopped['billing_settled'] and not stopped['execution_authority']
            if inflight:
                await client.cancel(original.input_ref,actor=actor,scope='subtree',reason='user stop')
            saved=await client.wait(original.input_ref,actor=actor,deadline_ms=now_ms()+10000)
            assert saved.status==('cancelled' if inflight else 'completed')
            if not inflight:
                assert saved==first
            rejected=await client.send(target,'Must not call the provider.',message_id='rejected',actor=actor)
            result=await client.wait(rejected.input_ref,actor=actor,deadline_ms=now_ms()+10000)
            assert result.status=='cancelled' and 'root_context_cancelled' in result.error
            assert provider.calls==1
            record_property('provider_server_receipt_confirmed',not inflight)
    finally:
        await runtime.worker.shutdown(timeout_s=3,cancel_running=True)
        await runtime.client.aclose()
        await runtime.transport.aclose()
        provider.report(record_property)
        await provider.close()
        await store.close()
