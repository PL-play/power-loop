"""Host fee provenance follows the accepted InputRef across worker reconstruction."""
import json
import time
from contextlib import ExitStack

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    ToolRegistry,
)
from power_loop.agent.runtime_types import RuntimeDescriptor, TrustedActor
from power_loop.runtime import root_context
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_root_provenance_uses_original_input_after_new_run_and_reopen(registry_store):
    store, connect = registry_store
    observed = []

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            binding = _SEND.get()
            value = await root_context.read(binding.runtime.manager, binding.actor,
                                             binding.lease.job_id, namespace='deeptalk.test')
            observed.append(value)
            return LLMResponse(raw_text='done')

        def stream(self, request):
            raise NotImplementedError

        async def close(self):
            pass

    class Resources:
        def __init__(self, scope):
            self.scope = scope

        async def aclose(self):
            self.scope.close()

    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        task_runtime = TaskRuntime(JobManager(store), tenant_id='tenant', principal_id='principal',
                                   executor_id=context.node_id, boot_id=context.boot_id)
        scope = ExitStack()
        scope.enter_context(task_runtime.root_context(context.session_id,
            namespace='deeptalk.test', payload={'origin_run_id': descriptor.config['run_id']}))
        return ExecutionEnvironment(Model(), ToolRegistry(), AgentLoopConfig(max_rounds=1),
                                    resources=Resources(scope), task_runtime=task_runtime)

    factory.register('host@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='worker', boot_id='boot')
    actor = TrustedActor('tenant', 'principal', None)
    try:
        await runtime.worker.start()
        target = None
        for run_id in (101, 202):
            descriptor = RuntimeDescriptor(1, 'agent', 'host@1', None, 'builtin.empty@1',
                'builtin.none@1', 'builtin.none@1', {'run_id': run_id}, {})
            target = target or await runtime.open_session(descriptor, actor=actor)
            receipt = await runtime.client.send(target, 'work', message_id=f'run-{run_id}',
                                                actor=actor, descriptor=descriptor)
            result = await runtime.client.wait(receipt.input_ref, actor=actor,
                                                deadline_ms=int(time.time() * 1000) + 10_000)
            assert result.status == 'completed', result.error
        assert [r['payload']['origin_run_id'] for r in observed] == [101, 202]
        assert observed[0]['root_job_id'] != observed[1]['root_job_id']
        reopened = await connect()
        manager = JobManager(reopened)
        from power_loop.runtime.jobs import JobActor
        reread = await root_context.read(manager, JobActor('tenant', 'principal', target.session_id),
                                         observed[0]['root_job_id'], namespace='deeptalk.test')
        assert reread == observed[0]
        assert reread['execution_authority'] is False
        row = await manager.db.fetchone(f'SELECT payload_json FROM {manager.events} '
            "WHERE event_id=?", (reread['creation_event_id'],))
        assert json.loads(row['payload_json'])['request']['input_ref']['message_id'] == 'run-101'
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()
