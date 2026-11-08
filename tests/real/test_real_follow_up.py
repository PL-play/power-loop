"""Live provider coverage for idle wakeup and busy input append."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, ToolDefinition, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free

from ._llm import make_llm
from .judge import assert_passes

pytestmark = pytest.mark.asyncio


async def test_idle_follow_up_steers_the_reply(runtime_agent):
    agent = await runtime_agent(model=make_llm(max_tokens=256, temperature=0),
        config=AgentLoopConfig(system_prompt='Answer briefly in English.',
            max_rounds=1, max_tokens=256, temperature=0, compactor=None))
    await agent.send('Name a primary color.')
    steer = "Forget the previous color. Reply with exactly one word — the color 'teal'."
    receipt = await agent.runtime.client.append(agent.target, steer, message_id='idle-followup', actor=agent.actor)
    result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+60000)
    assert result.status == 'completed'
    await assert_passes(question=steer, answer=result.final_text,
                        rubric="The reply contains the word 'teal' (case-insensitive).")


async def test_inflight_follow_up_is_queued_merged_and_steers(runtime_agent):
    entered, release = asyncio.Event(), asyncio.Event()
    registry = ToolRegistry()
    async def wait_tool():
        entered.set()
        await release.wait()
        return 'waited'
    registry.register(ToolDefinition('wait', 'Wait briefly before continuing.',
        input_schema={'type':'object', 'properties':{}}, pricing=Free('test')), wait_tool)
    agent = await runtime_agent(model=make_llm(max_tokens=384, temperature=0), registry=registry,
        config=AgentLoopConfig(system_prompt='First call the wait tool exactly once, then answer the user.',
            max_rounds=4, max_tokens=384, temperature=0, compactor=None))
    first = await agent.runtime.client.send(agent.target, 'Name a fruit.', message_id='fruit', actor=agent.actor)
    try:
        await asyncio.wait_for(entered.wait(), 60)
        extra = await agent.runtime.client.append(agent.target, "Make sure the fruit you name is specifically 'mango'.",
            message_id='fruit-followup', actor=agent.actor)
        assert extra.newly_accepted
        release.set()
        result = await agent.runtime.client.wait(first.input_ref, actor=agent.actor, deadline_ms=now_ms()+60000)
        assert result.status == 'completed'
        rows = await agent.runtime.store.load_active_messages(agent.target.session_id)
        assert any(row.role == 'user' and 'mango' in str(row.content) for row in rows)
        await assert_passes(question='Name a fruit, steered to mango.', answer=result.final_text,
            rubric="The reply names the fruit 'mango' (case-insensitive), not another fruit.")
    finally:
        release.set()
