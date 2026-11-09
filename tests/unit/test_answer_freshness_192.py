"""A pre-correction answer cannot satisfy a newer user input after empty pass."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store
from .test_pass_turn_builtin import _call, _Script

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_empty_pass_cannot_recover_text_from_before_the_latest_input(registry_store, runtime_agent):
    store, _ = registry_store
    model = _Script(responses=[_call('old', 'correct', {}, text='42'),
        _call('pass', 'pass_turn', {'reason':'Computed 43'})])
    registry = ToolRegistry()
    async def correct():
        await agent.runtime.client.send(agent.target, 'Correction: return 43.', message_id='correction', actor=agent.actor)
        return 'Correction queued.'
    registry.register(ToolDefinition(name='correct', description='Queue correction',
        input_schema={'type':'object'}, pricing=Free('test_fixture')), correct)
    agent = await runtime_agent(model=model, store=store, registry=registry,
        config=AgentLoopConfig(max_rounds=3, compactor=None))
    result = await agent.send('Return 42, then apply any correction.')
    assert result.status == 'completed' and model.calls == 2
    assert result.final_text == ''
    rows = await store.load_all_messages(agent.target.session_id)
    assert any(row.role == 'assistant' and row.content == '42' for row in rows)
    assert any(row.name == 'follow_up' and '43' in row.content for row in rows)
