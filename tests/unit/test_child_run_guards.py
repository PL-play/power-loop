"""The factory gives each parent/child fresh hooks and releases its resources.

Inline child guards were retired with inline execution. Their normal purpose,
host state isolation and cleanup on failure, belongs to ExecutionEnvironment.
"""
import json
from contextvars import ContextVar

import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    AgentSpec,
    ExecutionEnvironment,
    HookPoint,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    ToolDefinition,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('failed', [False, True])
async def test_factory_isolates_child_hook_state_and_closes_original_resources(registry_store, failed):
    store, _ = registry_store
    host_state = ContextVar('host_state', default=None)
    states, closed, observed = {}, [], []
    factory = RegistryRuntimeFactory()
    registry = ToolRegistry()

    async def delegate():
        parent = host_state.get()
        child = await run_agent_spec(AgentSpec(name='child', system_prompt='child'),
                                     'work', parent_loop=get_current_loop())
        assert host_state.get() is parent
        assert parent['calls'] == 0
        return child

    registry.register(ToolDefinition('delegate', 'Run child', {'type': 'object', 'properties': {}},
                                    pricing=Free('fixture')), delegate)

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            state = host_state.get()
            assert state['calls'] == 1
            observed.append(state['session'])
            if failed:
                raise OSError('supplier unavailable')
            return LLMResponse(raw_text='child answer')
        async def close(self):
            pass

    async def build(descriptor, context):
        state = {'session': context.input_ref.session_id, 'calls': 0}
        states[context.input_ref] = state
        host_state.set(state)
        hooks = AgentHooks()
        async def before(ctx):
            assert host_state.get() is state
            state['calls'] += 1
        hooks.register(HookPoint.LLM_BEFORE, before)
        class Resources:
            async def aclose(self):
                closed.append(context.input_ref)
        return ExecutionEnvironment(Model(), registry, AgentLoopConfig(compactor=None), hooks=hooks,
            resources=Resources(), command=None if 'child_spec' in descriptor.config else ('delegate', {}))

    factory.register('guard-replacement@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='factory-isolation', capacity=1,
                      timers_enabled=False, recovery_interval_s=None)
    actor = TrustedActor('guards', 'host', None)
    descriptor = RuntimeDescriptor(1, 'agent', 'guard-replacement@1', None,
        'builtin.empty@1', 'builtin.none@1', 'builtin.none@1', {}, {})
    try:
        await runtime.worker.start()
        target = await runtime.open_session(descriptor, actor=actor)
        receipt = await runtime.client.send(target, 'delegate', actor=actor, message_id='first')
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=now_ms()+15000)
        assert result.status == 'completed', result
        child = json.loads(result.final_text)
        assert child['status'] == ('failed' if failed else 'completed'), child
        if not failed:
            assert child['final_text'] == 'child answer'
        assert observed == [child['session_id']]
        assert host_state.get() is None
    finally:
        await runtime.worker.shutdown(timeout_s=2)
        await runtime.client.aclose()
        await runtime.transport.aclose()
    assert set(closed) == set(states) and len(closed) == len(states)
    assert {state['session'] for state in states.values()} >= {target.session_id, child['session_id']}
