"""Actual round-limit continuation reports every provider request exactly once."""

from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import SharedBudget, create_workflow

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_continuation_budget_equals_actual_reported_requests(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "continued.db"))
    provider = BoundedProvider(limit=6)
    tools = ToolRegistry()
    touched = []

    async def touch():
        touched.append(True)
        return "done; the result code is 42"

    tools.register(
        ToolDefinition(
            name="touch", description="Read the result code once.", input_schema={"type": "object", "properties": {}}
        , pricing=Free("test_fixture")),
        touch,
    )
    loop = StatefulAgentLoop(
        llm=provider, store=store, tool_registry=tools, config=AgentLoopConfig(max_tokens=2048, max_rounds=2)
    )
    pool = SharedBudget(50000)
    pool.commit({"total_tokens": 13})
    spec = {
        "name": "real-continuation-budget",
        "root": {
            "type": "agent",
            "id": "calculate",
            "input": "Read the result code once with touch, then report it.",
            "spec": {
                "name": "calculate",
                "tools": ["touch"],
                "max_rounds": 1,
                "system_prompt": "On the first task turn call touch exactly once with empty arguments. "
                "When touch has a recorded result anywhere in the conversation, do not call it again; "
                "return exactly 42 in assistant text. If asked to continue or finalize, return 42 without tools.",
            },
            "continuation": {"max_continuations": 1, "extra_rounds": 2, "gate": "always"},
        },
    }
    try:
        async with asyncio.timeout(150):
            result = await create_workflow(spec, parent_loop=loop, allowed_tools=["touch"], budget=pool).run()
            actual = sum(int(getattr(response.token_usage, "total_tokens", 0) or 0) for response in provider.responses)
            record_property("actual_reported_tokens", actual)
            record_property("shared_pool_spent", pool.spent)
            record_property("aggregate_usage", result.usage.get("total_tokens", 0))
            assert result.status == "completed" and result.final is not None and "42" in result.final.text, result
            assert len(touched) == 1
            assert 3 <= provider.calls <= 6
            assert pool.spent == 13 + actual, (pool.spent, actual, result.usage)
            assert result.usage["total_tokens"] == actual
            assert result.final.usage["total_tokens"] == actual
    finally:
        provider.report(record_property)
        await provider.close()
        await store.close()
