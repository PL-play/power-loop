"""Optional live-model timers through the current backend and RuntimeClient."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, InputRef, TimerContext, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms

from ._llm import make_llm

MARKER = 'TIMER_PROOF_4242'
pytestmark = pytest.mark.asyncio


def _registry():
    return create_default_tool_registry(include=['schedule_wakeup', 'current_time'])


async def test_real_agent_schedules_and_acts_on_wakeup(runtime_agent):
    agent = await runtime_agent(model=make_llm(), registry=_registry(), config=AgentLoopConfig(
        system_prompt='被要求稍后做事时用 schedule_wakeup 排唤醒，note 写清楚届时任务。被 <timer> 唤醒时执行 note。', max_rounds=6))
    await agent.runtime.timer_driver.stop()
    result = await agent.send(f'请在 6 秒后唤醒自己，届时原样说出暗号 {MARKER}。排好后简短确认。')
    assert result.status == 'completed'
    store = agent.runtime.store
    timers = await store.list_timers(agent.target.session_id)
    assert len(timers) == 1 and timers[0].interval_s is None
    timer = timers[0]
    await asyncio.sleep(max(0, (timer.due_at-now_ms())/1000))
    assert await agent.runtime.timer_driver.scan_once() == 1
    ref = InputRef(store.store_id, agent.target.session_id, f'timer-{timer.timer_id}-{timer.created_at}-0')
    result = await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+120000)
    assert result.status == 'completed' and MARKER in result.final_text
    await agent.drain()


async def test_real_recurring_timer_fires_repeatedly_until_cancelled(runtime_agent):
    agent = await runtime_agent(model=make_llm(), registry=_registry(), config=AgentLoopConfig(
        system_prompt='被 <timer> 唤醒时回复一行心跳确认。', max_rounds=3))
    await agent.runtime.timer_driver.stop()
    store = agent.runtime.store
    source = TimerContext(store.store_id, agent.target.session_id, agent.actor)
    timer = await agent.runtime.timer_backend.create(source, delay_s=0, note='心跳打卡', interval_s=2)
    for fire_count in range(2):
        row = await store.get_timer(agent.target.session_id, timer.timer_id)
        await asyncio.sleep(max(0, (row.due_at-now_ms())/1000))
        assert await agent.runtime.timer_driver.scan_once() == 1
        ref = InputRef(store.store_id, agent.target.session_id, f'timer-{timer.timer_id}-{timer.created_at}-{fire_count}')
        result = await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+120000)
        assert result.status == 'completed' and result.final_text.strip()
        await agent.drain()
    assert await agent.runtime.timer_backend.cancel(source, timer.timer_id)
    row = await store.get_timer(agent.target.session_id, timer.timer_id)
    assert row.fire_count == 2 and row.status == 'cancelled'
    assert await agent.runtime.timer_driver.scan_once() == 0
