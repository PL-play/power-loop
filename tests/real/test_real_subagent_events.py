"""A live child emits lifecycle events through its execution factory's bus."""
import json

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    AgentSpec,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec

from ._llm import make_llm

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_subagent_emits_lifecycle_events(runtime_agent):
    bus, seen, registry = AgentEventBus(), [], ToolRegistry()
    bus.subscribe(None, lambda event: seen.append(event.type) if event.source == 'subagent' else None)
    async def delegate():
        return await run_agent_spec(AgentSpec(name='child', system_prompt='Answer in one short sentence.'),
            'Name the capital of France.', parent_loop=get_tool_runtime_context(required=True).loop)
    registry.register(ToolDefinition('delegate', 'Host child command', pricing=Free('composite')), delegate)
    agent = await runtime_agent(model=make_llm(max_tokens=256, temperature=0), registry=registry,
        event_bus=bus, command=('delegate', {}),
        config=AgentLoopConfig(system_prompt='parent', max_rounds=2, max_tokens=256, temperature=0, compactor=None))
    result = await agent.send('Run child')
    child = json.loads(result.final_text)
    assert result.status == child['status'] == 'completed'
    assert AgentEventType.SUBAGENT_TASK_START in seen
    assert AgentEventType.SUBAGENT_COMPLETED in seen
