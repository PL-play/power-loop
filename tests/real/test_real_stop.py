"""Real-provider cancellation and valid next-input history through RuntimeClient."""
import asyncio
import time

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig
from power_loop.agent.runtime_store import now_ms

from ._llm import make_llm

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_stop_mid_generation_is_fast_and_leaves_a_valid_transcript(runtime_agent):
    bus, chars = AgentEventBus(), [0]
    def count(event):
        chars[0] += len((event.payload or {}).get('text', ''))
    bus.subscribe(AgentEventType.STREAM_DELTA, count)
    bus.subscribe(AgentEventType.STREAM_THINK_DELTA, count)
    agent = await runtime_agent(model=make_llm(max_tokens=6000, temperature=.7), event_bus=bus,
        config=AgentLoopConfig(system_prompt='你是作家。', max_rounds=3, max_tokens=6000, context_budget_tokens=10**9))
    client = agent.runtime.client
    receipt = await client.send(agent.target, '写一篇至少 5000 字的小说。', message_id='long', actor=agent.actor)
    async with asyncio.timeout(60):
        while chars[0] < 200:
            await asyncio.sleep(.01)
    started = time.monotonic()
    await client.cancel(receipt.input_ref, actor=agent.actor, reason='user stop')
    result = await client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+30000)
    assert result.status == 'cancelled' and time.monotonic()-started < 3
    next_result = await agent.send('好了，不写了。只回一个字：好')
    assert next_result.status == 'completed'
    assert (await agent.runtime.store.get_state(agent.target.session_id)).pending is None


