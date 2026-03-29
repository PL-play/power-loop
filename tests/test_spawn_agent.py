"""Unit tests for spawn_agent tool — no real LLM needed."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.runner import AgentRunner
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import (
    DEFAULT_MAX_DEPTH,
    SPAWN_AGENT_DEFINITION,
    _make_spawn_handler,
    _spawn_depth,
    register_spawn_agent,
    run_spawn_agent,
)


# ---------------------------------------------------------------------------
# 1. ToolDefinition is well-formed
# ---------------------------------------------------------------------------

def test_spawn_agent_definition():
    d = SPAWN_AGENT_DEFINITION
    assert d.name == "spawn_agent"
    assert "task" in d.required_params
    tool = d.to_openai_tool()
    assert tool["function"]["name"] == "spawn_agent"
    props = tool["function"]["parameters"]["properties"]
    assert "task" in props
    assert "preset" in props
    assert "max_rounds" in props


# ---------------------------------------------------------------------------
# 2. Depth guard rejects when at max depth
# ---------------------------------------------------------------------------

def test_depth_guard_rejects():
    async def _run():
        _spawn_depth.set(DEFAULT_MAX_DEPTH)
        try:
            result = await run_spawn_agent(task="do something", _llm=object(), _max_depth=DEFAULT_MAX_DEPTH)
            assert "rejected" in result.lower() or "max nesting depth" in result.lower()
        finally:
            _spawn_depth.set(0)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. No LLM returns error
# ---------------------------------------------------------------------------

def test_no_llm_returns_error():
    async def _run():
        result = await run_spawn_agent(task="test", _llm=None)
        assert "not configured" in result.lower()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. register_spawn_agent adds to registry
# ---------------------------------------------------------------------------

def test_register_spawn_agent():
    registry = ToolRegistry()
    mock_llm = object()
    register_spawn_agent(registry, mock_llm)
    assert registry.has("spawn_agent")
    defn = registry.get("spawn_agent")
    assert defn is not None
    assert defn.definition.name == "spawn_agent"


# ---------------------------------------------------------------------------
# 5. Handler closure calls run_spawn_agent
# ---------------------------------------------------------------------------

def test_handler_closure():
    """The handler created by _make_spawn_handler should be async and callable."""
    handler = _make_spawn_handler(None, max_depth=1)
    assert asyncio.iscoroutinefunction(handler)

    async def _run():
        # With _llm=None it should return error immediately
        result = await handler(task="hello")
        assert "not configured" in result.lower()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Event bubbling captures SUBAGENT_TASK_START
# ---------------------------------------------------------------------------

def test_event_bubbling_subagent_start():
    """Verify parent bus receives SUBAGENT_TASK_START when spawn is attempted."""
    captured: list[AgentEvent] = []

    bus = AgentEventBus()
    bus.subscribe(AgentEventType.SUBAGENT_TASK_START, lambda e: captured.append(e))

    async def _run():
        runner = AgentRunner(event_bus=bus, hooks=AgentHooks())
        async with runner.session_async(session_id="parent-1"):
            # This will fail (no real LLM) but should still fire the event
            # before the error.  We need a fake LLM that makes the
            # AgentLoop.run() work minimally.
            from llm_client.interface import LLMResponse, LLMService, LLMTokenUsage

            class FakeLLM(LLMService):
                async def complete(self, request, **kw):
                    return LLMResponse(
                        content_text="done",
                        raw_text="done",
                        token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5),
                    )

            result = await run_spawn_agent(
                task="list files",
                preset="explore",
                _llm=FakeLLM(),
                _max_depth=1,
                _bubble_events=True,
            )
            # Sub-agent should have completed
            assert "done" in result.lower() or len(result) > 0

    asyncio.run(_run())

    # The parent bus should have received SUBAGENT_TASK_START
    start_events = [e for e in captured if e.type == AgentEventType.SUBAGENT_TASK_START]
    assert len(start_events) == 1
    assert start_events[0].payload["task"] == "list files"
    assert start_events[0].payload["depth"] == 1


# ---------------------------------------------------------------------------
# 7. Depth increments and decrements correctly
# ---------------------------------------------------------------------------

def test_depth_increments():
    """After spawn completes, depth should be back to 0."""
    async def _run():
        assert _spawn_depth.get() == 0

        from llm_client.interface import LLMResponse, LLMService, LLMTokenUsage

        class FakeLLM(LLMService):
            async def complete(self, request, **kw):
                # Verify depth inside sub-agent context
                return LLMResponse(
                    content_text=f"depth={_spawn_depth.get()}",
                    raw_text=f"depth={_spawn_depth.get()}",
                    token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5),
                )

        bus = AgentEventBus()
        runner = AgentRunner(event_bus=bus, hooks=AgentHooks())
        async with runner.session_async(session_id="test"):
            result = await run_spawn_agent(task="check depth", _llm=FakeLLM(), _max_depth=3)
            # Inside the sub-agent, depth should be 1
            assert "depth=1" in result

        # After return, depth should be back to 0
        assert _spawn_depth.get() == 0

    asyncio.run(_run())


if __name__ == "__main__":
    test_spawn_agent_definition()
    test_depth_guard_rejects()
    test_no_llm_returns_error()
    test_register_spawn_agent()
    test_handler_closure()
    test_event_bubbling_subagent_start()
    test_depth_increments()
    print("All spawn_agent tests passed!")
