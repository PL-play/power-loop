"""Real-provider cancellation and valid next-input history through RuntimeClient."""
import asyncio
import time

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, JobRef, ToolDefinition, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free
from power_loop.tools.spawn_agent import register_spawn_agent

from ._llm import make_llm

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_stop_mid_generation_is_fast_and_leaves_a_valid_transcript(runtime_agent):
    bus, chars = AgentEventBus(), [0]
    def count(event):
        chars[0] += len((event.payload or {}).get('text', ''))
    bus.subscribe(AgentEventType.STREAM_DELTA, count)
    bus.subscribe(AgentEventType.STREAM_THINK_DELTA, count)
    agent = await runtime_agent(model=make_llm(max_tokens=6000, temperature=.7), event_bus=bus,
        config=AgentLoopConfig(system_prompt='你是作家。', max_rounds=3, max_tokens=6000, compactor=None))
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


async def test_real_subtree_stop_reaches_original_running_subagent(runtime_agent):
    started, registry = asyncio.Event(), ToolRegistry()
    async def slow_lookup(topic='', **kwargs):
        started.set()
        await asyncio.sleep(30)
        return f'{topic}: done'
    registry.register(ToolDefinition('slow_lookup', 'Look something up slowly.',
        input_schema={'type':'object','properties':{'topic':{'type':'string'}},'required':['topic']},
        interrupt='abort', pricing=Free('test_fixture')), slow_lookup)
    register_spawn_agent(registry)
    agent = await runtime_agent(model=make_llm(max_tokens=2048, temperature=0), registry=registry,
        config=AgentLoopConfig(system_prompt='Follow the requested native tool call exactly.',
            max_rounds=4, max_tokens=2048, compactor=None))
    client = agent.runtime.client
    receipt = await client.send(agent.target,
        'Call spawn_agent once with task="Call slow_lookup for solar panels.", tools=["slow_lookup"], '
        'system_prompt="Always call slow_lookup first then answer.". Wait for its result.',
        message_id='delegate', actor=agent.actor)
    await asyncio.wait_for(started.wait(), 120)
    [child] = await agent.runtime.store._db.fetchall(
        f"SELECT job_id,generation FROM {agent.runtime.store.table_prefix}jobs WHERE kind='subagent'")
    child_ref = JobRef(agent.target.store_id, child['job_id'], child['generation'])
    start = time.monotonic()
    await client.cancel(receipt.input_ref, actor=agent.actor, scope='subtree', reason='user stop')
    child_result = await client.wait(child_ref, actor=agent.actor, deadline_ms=now_ms()+60000)
    assert child_result.status == 'cancelled' and time.monotonic()-start < 12
    await agent.drain()
    next_result = await agent.send('Never mind. Do not use tools. Reply with just: OK')
    assert next_result.status == 'completed' and 'ok' in next_result.final_text.lower()
