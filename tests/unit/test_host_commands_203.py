"""HTTP-style commands are inputs, never forged model tool calls."""
import json

import pytest

from power_loop import AgentLoopConfig, ExecutionEnvironment, LLMResponse, LLMService, Runtime, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef, RuntimeDescriptor, RuntimeRequestError, TrustedActor
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import AgentSpec, run_agent_spec
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_worker_command_and_inline_child_without_assistant_call(registry_store):
    store, _ = registry_store
    executed, model_calls = [], []

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            assert request.system_prompt.startswith('CHILD')
            model_calls.append(request.model)
            return LLMResponse(raw_text='42')

        async def close(self):
            pass

    async def command():
        executed.append(True)
        loop = get_current_loop()
        result = await run_agent_spec(AgentSpec(name='helper', system_prompt='CHILD', tools=[], max_rounds=2),
            'compute', parent_loop=loop)
        ref = JobRef(**result['job_ref'])
        client = loop._runtime_worker.runtime.client
        for overrides in ({'tools': ['host_helper']}, {'max_rounds': 3}, {'model': 'other'}):
            with pytest.raises(RuntimeRequestError):
                await client.resume(ref, actor=actor, message_id='bad', options=overrides)
        receipt = await client.resume(ref, actor=actor, message_id='repair', body='Return the same answer',
            options={'tools': [], 'max_rounds': 1})
        fixed = await client.wait(receipt.input_ref, actor=actor, deadline_ms=now_ms()+10000)
        assert fixed.job_ref.generation == 2
        assert (await client.wait(ref, actor=actor, deadline_ms=now_ms()+1000)).final_text == '42'
        from power_loop.runtime.host_commands import consume_result
        await consume_result(loop, fixed)
        return {'answer': fixed.final_text}

    class Factory:
        async def build(self, descriptor, execution_context):
            registry = ToolRegistry()
            registry.register(ToolDefinition('host_helper', 'Registered host action', pricing=Free('composite')), command)
            return ExecutionEnvironment(Model(), registry, AgentLoopConfig(model='inherited-model'),
                command=None if 'child_spec' in descriptor.config else ('host_helper', {}))

    descriptor = RuntimeDescriptor(1, 'agent', 'host@1', None, 'host.tools@1',
        'builtin.none@1', 'builtin.none@1', {'allowed_models': ['inherited-model']}, {})
    runtime = Runtime(store=store, factory=Factory(), node_id='worker', capacity=1, recovery_interval_s=None)
    actor = TrustedActor('tenant', 'principal', None)
    try:
        await runtime.worker.start()
        target = await runtime.open_session(descriptor, actor=actor)
        first = await runtime.client.send(target, {'content': 'explicit helper request'},
            actor=actor, message_id='http-request-1')
        result = await runtime.client.wait(first.input_ref, actor=actor, deadline_ms=now_ms()+10000)
        assert result.status == 'completed', result
        assert json.loads(result.final_text) == {'answer': '42'}
        retry = await runtime.client.send(target, {'content': 'explicit helper request'},
            actor=actor, message_id='http-request-1')
        assert not retry.newly_accepted
        await runtime.worker.shutdown()
        assert executed == [True] and model_calls == ['inherited-model', 'inherited-model']
        messages = await store.load_all_messages(target.session_id)
        assert all(not message.tool_calls for message in messages)
        jobs = await store._db.fetchall(f"SELECT kind,result_status,supervision_status FROM {runtime.worker.inputs.jobs}")
        child = next(job for job in jobs if job['kind'] == 'subagent')
        assert child['result_status'] == 'handled'
        assert child['supervision_status'] == 'closed'
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()
