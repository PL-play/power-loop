"""Real-LLM timers: the model schedules its own wake-up, the TimerRunner
fires it, and the model acts on the note. Plus a recurring timer driven
through real deliveries.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from power_loop import (
    AgentLoopConfig,
    StatefulAgentLoop,
    TimerRunner,
    create_default_tool_registry,
    get_tool_definitions,
)
from power_loop.tools import DEFAULT_TOOL_HANDLERS

from ._llm import make_llm

MARKER = "TIMER_PROOF_4242"


def _registry():
    reg = create_default_tool_registry(preset="core", bind=False)
    for d in get_tool_definitions(
        include=["schedule_wakeup", "list_wakeups", "cancel_wakeup", "current_time"]
    ):
        reg.register(d, DEFAULT_TOOL_HANDLERS[d.name], overwrite=True)
    return reg


@pytest.mark.asyncio
async def test_real_agent_schedules_and_acts_on_wakeup(tmp_path) -> None:
    loop = StatefulAgentLoop(
        llm=make_llm(),
        db_path=str(tmp_path / "t.db"),
        config=AgentLoopConfig(
            system_prompt=(
                "你是一个会自我管理时间的助手。当被要求稍后做某事时，用 "
                "schedule_wakeup 排一个唤醒，note 写清楚届时要做什么。"
                "被 <timer> 消息唤醒时，按 note 执行。"
            ),
            max_rounds=6,
        ),
        tool_registry=_registry(),
    )
    sid = loop.new_session()
    res = await loop.send(
        f"请在 6 秒后唤醒自己，届时请原样说出暗号 {MARKER}。排好后简短确认。",
        session_id=sid,
    )
    assert res.status == "completed"
    timers = loop.list_timers(sid)
    assert len(timers) == 1, "the model should have scheduled exactly one wake-up"
    assert timers[0].interval_s is None  # one-shot

    runner = TimerRunner(loop, scan_interval_s=1.0)
    await runner.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            await asyncio.sleep(1)
            if loop.store.get_timer(sid, timers[0].timer_id).status == "fired":
                break
    finally:
        await runner.stop()

    assert loop.store.get_timer(sid, timers[0].timer_id).status == "fired"
    # The model, woken by its own note, should have produced the marker.
    assistant_text = " ".join(
        str(m.get("content") or "")
        for m in loop.get_messages(sid)
        if m.get("role") == "assistant"
    )
    assert MARKER in assistant_text
    loop.close()


@pytest.mark.asyncio
async def test_real_recurring_timer_fires_repeatedly_until_cancelled(tmp_path) -> None:
    loop = StatefulAgentLoop(
        llm=make_llm(),
        db_path=str(tmp_path / "t.db"),
        config=AgentLoopConfig(
            system_prompt="被 <timer> 唤醒时，回复一行：'第N次心跳'（N=你被唤醒的次数）。",
            max_rounds=3,
        ),
        tool_registry=_registry(),
    )
    sid = loop.new_session()
    # External path: the orchestrator declares recurrence at creation.
    timer = loop.schedule_timer(sid, delay_s=1, note="心跳打卡", interval_s=2)
    assert timer.interval_s == 2

    runner = TimerRunner(loop, scan_interval_s=1.0)
    await runner.start()
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            await asyncio.sleep(1)
            row = loop.store.get_timer(sid, timer.timer_id)
            if row.fire_count >= 2:
                break
    finally:
        await runner.stop()

    row = loop.store.get_timer(sid, timer.timer_id)
    assert row.fire_count >= 2, "recurring timer should have fired at least twice"
    assert row.status == "armed", "recurring timer re-arms after each fire"
    # Cancel terminates the cycle.
    assert loop.cancel_timer(sid, timer.timer_id) is True
    assert loop.store.get_timer(sid, timer.timer_id).status == "cancelled"
    # Two real deliveries reached the conversation.
    user_timer_msgs = [
        m for m in loop.get_messages(sid)
        if m.get("role") == "user" and "<timer" in str(m.get("content"))
    ]
    assert len(user_timer_msgs) >= 2
    loop.close()
