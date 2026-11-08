"""Role tool filtering and public control authorization on actual Runtime jobs."""
import json
from dataclasses import replace

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    LLMResponse,
    LLMService,
    TargetRef,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def role_case(registry_store, runtime_agent):
    store, _ = registry_store
    seen, registry = [], ToolRegistry()
    class Model(LLMService):
        async def complete(self, request, **kwargs):
            seen.append(request)
            return LLMResponse(raw_text='child finished')
        async def close(self):
            pass
    for name, roles in [('workflow',('main',)),('spawn_agent',('main','workflow_node')),
                        ('schedule_wakeup',('main',)),('local_read',('main','workflow_node','subagent'))]:
        registry.register(ToolDefinition(name, name, pricing=Free('test')),
            FunctionRuntimeTool(lambda: 'unused', ToolRuntimeContract('test.'+name, allowed_roles=roles)))
    async def command():
        return await run_agent_spec(AgentSpec(name='audit', system_prompt='Audit',
            tools=['workflow','spawn_agent','schedule_wakeup','local_read'], lifecycle='linked', max_rounds=2),
            'Review the material.', parent_loop=get_tool_runtime_context(required=True).loop)
    registry.register(ToolDefinition('delegate','Host delegation',pricing=Free('composite')),command)
    agent = await runtime_agent(store=store, model=Model(), registry=registry, command=('delegate',{}),
        config=AgentLoopConfig(compactor=None, delegation_policy='single_level'))
    result = await agent.send('Audit')
    assert result.status == 'completed', result
    child = json.loads(result.final_text)
    assert child['status'] == 'completed'
    return agent, child, seen


async def test_child_factory_filters_orchestration_and_timer_capabilities(role_case):
    agent, child, seen = role_case
    offered = {tool['function']['name'] for tool in seen[0].tools or []}
    assert offered == {'local_read', 'pass_turn'}
    session = await agent.runtime.store.get_session(child['session_id'])
    assert session.metadata['execution_role'] == 'subagent'
    assert session.parent_session_id == agent.target.session_id


async def test_direct_new_turn_cannot_promote_child_to_a_new_root(role_case):
    agent, child, seen = role_case
    before = len(seen)
    target = TargetRef(agent.target.store_id, child['session_id'])
    with pytest.raises(RuntimeRequestError, match='invalid_options'):
        await agent.runtime.client.send(target, 'Start a new root', message_id='forbidden-root', actor=agent.actor)
    assert len(seen) == before


@pytest.mark.parametrize('operation', ['send','append'])
async def test_child_scoped_actor_cannot_address_its_parent(role_case, operation):
    agent, child, seen = role_case
    before = len(seen)
    actor = replace(agent.actor, session_id=child['session_id'])
    with pytest.raises(RuntimeRequestError, match='forbidden'):
        await getattr(agent.runtime.client, operation)(agent.target, 'Redirect parent', message_id='foreign', actor=actor)
    assert len(seen) == before


async def test_invalid_declared_delegation_policy_is_rejected():
    with pytest.raises(ValueError, match='delegation_policy'):
        AgentLoopConfig(delegation_policy='typo')
