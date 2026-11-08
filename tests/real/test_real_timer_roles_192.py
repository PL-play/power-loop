"""A real finite child cannot use a host wrapper to escape timer ownership."""

from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    AgentSpec,
    HookPoint,
    SessionStore,
    ToolDefinition,
    get_tool_definitions,
)
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.tools import DEFAULT_TOOL_HANDLERS
from power_loop.tools.default_tools import run_schedule_wakeup
from power_loop.tools.registry import ToolRegistry

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_child_wrapper_timer_is_rejected_without_creating_or_scheduling_replacement(
    tmp_path, record_property, runtime_agent
):
    store = await SessionStore.open(str(tmp_path / "finite-timer.db"))
    provider = BoundedProvider(limit=4)
    registry, attempted = ToolRegistry(), []

    async def reminder_probe():
        attempted.append(True)
        return await run_schedule_wakeup(action="schedule", delay_seconds=10, note="original wait")

    registry.register(
        ToolDefinition(name="reminder_probe", description="Try to schedule the requested reminder.",
                       input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")),
        reminder_probe,
    )
    [reminder] = get_tool_definitions(include=["schedule_wakeup"])
    registry.register(reminder, DEFAULT_TOOL_HANDLERS["schedule_wakeup"])
    hooks, offered = AgentHooks(), []
    hooks.register(HookPoint.LLM_BEFORE, lambda ctx: offered.append([tool['function']['name'] for tool in ctx.tools]))
    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="timer-probe", lifecycle="linked", max_rounds=2, max_tokens=2048,
                tools=["reminder_probe", "schedule_wakeup"],
                system_prompt="Call reminder_probe exactly once, without arguments. Then read its tool result. "
                    "If it reports execution_role_forbids_timer, reply BLOCKED in final assistant text. "
                    "Never retry, schedule another reminder, spawn work, or use pass_turn."),
            "Try the reminder once and report its result.", parent_loop=get_current_loop())
    registry.register(ToolDefinition('delegate', 'Run finite child', input_schema={'type':'object','properties':{}}, pricing=Free('fixture')), delegate)
    loop = await runtime_agent(model=provider, store=store, registry=registry, hooks=hooks, command=('delegate', {}),
        config=AgentLoopConfig(max_rounds=2, max_tokens=2048))
    parent = loop.target.session_id
    try:
        async with asyncio.timeout(120):
            completed = await loop.send('try finite child')
            assert completed.status == 'completed', completed
            result = json.loads(completed.final_text)
            sid = result["session_id"]
            messages = await store.load_all_messages(sid)
            record_property("original_messages", json.dumps([
                {"role": m.role, "content": m.content, "tool_calls": m.tool_calls} for m in messages
            ]))
            assert attempted == [True]
            receipts = [m for m in messages if m.role == "tool" and m.name == "reminder_probe"]
            assert len(receipts) == 1 and "execution_role_forbids_timer" in receipts[0].content
            assert offered and all("reminder_probe" in names and "schedule_wakeup" not in names for names in offered)
            assert registry.has("schedule_wakeup")
            assert "BLOCKED" in result["final_text"]
            assert await store.list_timers(sid) == await store.list_timers(parent) == []
            assert len(await store.list_children(parent)) == 1
    finally:
        await loop.aclose()
        provider.report(record_property)
        await provider.close()
        await store.close()
