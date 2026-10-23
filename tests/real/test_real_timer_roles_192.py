"""A real finite child cannot use a host wrapper to escape timer ownership."""

from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop, ToolDefinition, get_tool_definitions
from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime.spec import run_agent_spec
from power_loop.tools import DEFAULT_TOOL_HANDLERS
from power_loop.tools.default_tools import run_schedule_wakeup
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_child_wrapper_timer_is_rejected_without_creating_or_scheduling_replacement(
    tmp_path, record_property
):
    store = await SessionStore.open(str(tmp_path / "finite-timer.db"))
    provider = BoundedProvider(limit=4)
    registry, attempted = ToolRegistry(), []

    async def reminder_probe():
        attempted.append(True)
        return await run_schedule_wakeup(action="schedule", delay_seconds=10, note="original wait")

    registry.register(
        ToolDefinition(name="reminder_probe", description="Try to schedule the requested reminder.",
                       input_schema={"type": "object", "properties": {}}),
        reminder_probe,
    )
    [reminder] = get_tool_definitions(include=["schedule_wakeup"])
    registry.register(reminder, DEFAULT_TOOL_HANDLERS["schedule_wakeup"])
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=2048))
    parent = await loop.new_session()
    token = set_session_id(parent)
    try:
        async with asyncio.timeout(120):
            result = await run_agent_spec(
                AgentSpec(name="timer-probe", lifecycle="linked", max_rounds=2, max_tokens=2048,
                    tools=["reminder_probe", "schedule_wakeup"],
                    system_prompt="Call reminder_probe exactly once, without arguments. Then read its tool result. "
                        "If it reports execution_role_forbids_timer, reply BLOCKED in final assistant text. "
                        "Never retry, schedule another reminder, spawn work, or use pass_turn."),
                "Try the reminder once and report its result.", parent_loop=loop,
            )
            sid = result["session_id"]
            messages = await store.load_all_messages(sid)
            record_property("original_messages", json.dumps([
                {"role": m.role, "content": m.content, "tool_calls": m.tool_calls} for m in messages
            ]))
            assert attempted == [True]
            receipts = [m for m in messages if m.role == "tool" and m.name == "reminder_probe"]
            assert len(receipts) == 1 and "execution_role_forbids_timer" in receipts[0].content
            contract = await store.get_runtime_state(sid, "workflow_execution_contract")
            assert contract["tools"] == ["reminder_probe"] and registry.has("schedule_wakeup")
            assert "BLOCKED" in result["final_text"]
            assert await store.list_timers(sid) == await store.list_timers(parent) == []
            assert len(await store.list_children(parent)) == 1
    finally:
        reset_session_id(token)
        provider.report(record_property)
        await provider.close()
        await store.close()
