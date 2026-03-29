"""Tests for AgentPipeline and @phase decorator."""
from __future__ import annotations

import asyncio
from typing import Any

from llm_client.interface import LLMResponse, LLMService, LLMTokenUsage

from power_loop.agent.types import AgentLoopConfig
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.phase import PhaseContext, PhaseResult, phase
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.runner import AgentRunner
from power_loop.core.state import ContextManager


# ── Fake LLM ──

class FakeLLM(LLMService):
    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or ["Hello!"])
        self._idx = 0

    async def complete(self, request, **kw):
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        on_delta = kw.get("on_chunk_delta_text")
        if on_delta:
            on_delta(text)
        return LLMResponse(
            raw_text=text,
            content_text=text,
            token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5),
        )


class FakeLLMWithTools(LLMService):
    """Returns a tool call on first call, then plain text on second."""
    def __init__(self):
        self._call = 0

    async def complete(self, request, **kw):
        self._call += 1
        if self._call == 1:
            resp = LLMResponse(
                raw_text="I'll read the file",
                content_text="I'll read the file",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "mock_tool", "arguments": '{"x": 1}'},
                }],
                token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5),
            )
            return resp
        return LLMResponse(
            raw_text="Done!",
            content_text="Done!",
            token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5),
        )


# ── Helper to create a pipeline ──

def _make_pipeline(llm=None, hooks=None, bus=None, tool_registry=None, config=None):
    return AgentPipeline(
        llm=llm or FakeLLM(),
        config=config or AgentLoopConfig(max_rounds=5),
        tool_registry=tool_registry,
        hooks=hooks or AgentHooks(),
        bus=bus or AgentEventBus(suppress_subscriber_errors=True),
        ctx=ContextManager(role="main"),
        session_id="test-session",
    )


# ---------------------------------------------------------------------------
# 1. Basic pipeline run — no tools, completes in 1 round
# ---------------------------------------------------------------------------

def test_basic_pipeline_completes():
    async def _run():
        bus = AgentEventBus()
        events: list[AgentEvent] = []
        bus.subscribe(None, lambda e: events.append(e))

        runner = AgentRunner(event_bus=bus, hooks=AgentHooks())
        async with runner.session_async(session_id="s1"):
            from power_loop.core.agent import agent_loop_async
            result = await agent_loop_async(
                llm=FakeLLM(["Simple answer"]),
                config=AgentLoopConfig(max_rounds=5),
                tool_registry=None,
                messages=[{"role": "user", "content": "Hi"}],
                session_id="s1",
            )

        assert result.status == "completed"
        assert "Simple answer" in result.final_text
        assert result.rounds == 1

        # Should have SESSION_STARTED, ROUND_STARTED, STREAM events, ROUND_COMPLETED, SESSION_ENDED
        event_types = [e.type for e in events]
        assert AgentEventType.SESSION_STARTED in event_types
        assert AgentEventType.ROUND_STARTED in event_types
        assert AgentEventType.SESSION_ENDED in event_types

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. Pipeline with tools — executes mock tool
# ---------------------------------------------------------------------------

def test_pipeline_with_tools():
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="mock_tool", description="test", required_params=("x",),
                       input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}),
        lambda x: f"result={x}",
    )

    async def _run():
        bus = AgentEventBus()
        events: list[AgentEvent] = []
        bus.subscribe(None, lambda e: events.append(e))

        runner = AgentRunner(event_bus=bus, hooks=AgentHooks())
        async with runner.session_async(session_id="s2"):
            from power_loop.core.agent import agent_loop_async
            result = await agent_loop_async(
                llm=FakeLLMWithTools(),
                config=AgentLoopConfig(max_rounds=5),
                tool_registry=registry,
                messages=[{"role": "user", "content": "Read file"}],
                session_id="s2",
            )

        assert result.status == "completed"
        assert result.rounds == 2  # round 1: tool call, round 2: final text
        assert "Done!" in result.final_text

        # Check tool events
        event_types = [e.type for e in events]
        assert AgentEventType.TOOL_CALL_STARTED in event_types
        assert AgentEventType.TOOL_CALL_COMPLETED in event_types

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. ROUND_START BREAK directive stops the loop
# ---------------------------------------------------------------------------

def test_round_start_break():
    """ROUND_START BREAK at round 0 should prevent any LLM call."""
    hooks = AgentHooks()

    def budget_guard(ctx: HookContext) -> HookResult:
        ctx.values["reason"] = "budget_exceeded"
        return HookResult(context=ctx, directive=HookDirective.BREAK)

    hooks.register(HookPoint.ROUND_START, budget_guard)

    class NeverCalledLLM(LLMService):
        async def complete(self, request, **kw):
            raise AssertionError("LLM should not be called when ROUND_START breaks")

    async def _run():
        runner = AgentRunner(event_bus=AgentEventBus(), hooks=hooks)
        async with runner.session_async(session_id="s3"):
            from power_loop.core.agent import agent_loop_async
            result = await agent_loop_async(
                llm=NeverCalledLLM(),
                config=AgentLoopConfig(max_rounds=10),
                tool_registry=None,
                messages=[{"role": "user", "content": "Count"}],
                session_id="s3",
            )

        assert result.status == "completed"
        assert result.rounds == 0

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. LLM_BEFORE SHORT_CIRCUIT returns cached response
# ---------------------------------------------------------------------------

def test_llm_before_short_circuit():
    hooks = AgentHooks()

    def cache_hook(ctx: HookContext) -> HookResult:
        # Always return a cached response
        ctx.values["output"] = LLMResponse(
            raw_text="cached!",
            content_text="cached!",
            token_usage=LLMTokenUsage(prompt_tokens=0, completion_tokens=0),
        )
        return HookResult(context=ctx, directive=HookDirective.SHORT_CIRCUIT)

    hooks.register(HookPoint.LLM_BEFORE, cache_hook)

    async def _run():
        # Use a LLM that should never be called
        class NeverCalledLLM(LLMService):
            async def complete(self, request, **kw):
                raise AssertionError("LLM should not be called")

        runner = AgentRunner(event_bus=AgentEventBus(), hooks=hooks)
        async with runner.session_async(session_id="s4"):
            from power_loop.core.agent import agent_loop_async
            result = await agent_loop_async(
                llm=NeverCalledLLM(),
                config=AgentLoopConfig(max_rounds=5),
                tool_registry=None,
                messages=[{"role": "user", "content": "test"}],
                session_id="s4",
            )

        assert result.status == "completed"
        assert "cached!" in result.final_text

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. TOOL_BEFORE SKIP skips tool execution
# ---------------------------------------------------------------------------

def test_tool_before_skip():
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="dangerous", description="test", required_params=("cmd",),
                       input_schema={"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}),
        lambda cmd: "SHOULD NOT RUN",
    )

    hooks = AgentHooks()

    def block_dangerous(ctx: HookContext) -> HookResult:
        if ctx.values.get("tool_name") == "dangerous":
            ctx.values["output"] = "[blocked by policy]"
            return HookResult(context=ctx, directive=HookDirective.SKIP)
        return HookResult(context=ctx)

    hooks.register(HookPoint.TOOL_BEFORE, block_dangerous)

    class ToolCallLLM(LLMService):
        def __init__(self):
            self._call = 0
        async def complete(self, request, **kw):
            self._call += 1
            if self._call == 1:
                return LLMResponse(
                    raw_text="running dangerous",
                    tool_calls=[{"id": "c1", "type": "function",
                                 "function": {"name": "dangerous", "arguments": '{"cmd": "rm -rf /"}'}}],
                    token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5),
                )
            return LLMResponse(raw_text="ok", token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5))

    async def _run():
        runner = AgentRunner(event_bus=AgentEventBus(), hooks=hooks)
        async with runner.session_async(session_id="s5"):
            from power_loop.core.agent import agent_loop_async
            result = await agent_loop_async(
                llm=ToolCallLLM(),
                config=AgentLoopConfig(max_rounds=5),
                tool_registry=registry,
                messages=[{"role": "user", "content": "delete everything"}],
                session_id="s5",
            )

        assert result.status == "completed"
        # The tool result in history should be the blocked message
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "[blocked by policy]" in tool_msgs[0]["content"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. MESSAGE_APPEND hook can modify messages
# ---------------------------------------------------------------------------

def test_message_append_hook():
    hooks = AgentHooks()

    def redactor(ctx: HookContext) -> HookContext:
        msg = ctx.values.get("message", {})
        if msg.get("role") == "assistant":
            msg = dict(msg)
            msg["content"] = msg.get("content", "").replace("secret", "***")
            ctx.values["message"] = msg
        return ctx

    hooks.register(HookPoint.MESSAGE_APPEND, redactor)

    class SecretLLM(LLMService):
        async def complete(self, request, **kw):
            return LLMResponse(
                raw_text="The secret is 42",
                token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5),
            )

    async def _run():
        runner = AgentRunner(event_bus=AgentEventBus(), hooks=hooks)
        async with runner.session_async(session_id="s6"):
            from power_loop.core.agent import agent_loop_async
            result = await agent_loop_async(
                llm=SecretLLM(),
                config=AgentLoopConfig(max_rounds=2),
                tool_registry=None,
                messages=[{"role": "user", "content": "tell me"}],
                session_id="s6",
            )

        # The assistant message in history should have "secret" redacted
        assistant_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert "***" in assistant_msgs[0]["content"]
        assert "secret" not in assistant_msgs[0]["content"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 7. @phase decorator unit test
# ---------------------------------------------------------------------------

def test_phase_decorator_standalone():
    """Test @phase decorator in isolation with a mock pipeline."""

    class MockPipeline:
        def __init__(self):
            self.hooks = AgentHooks()
            self.bus = AgentEventBus(suppress_subscriber_errors=True)

        @phase(before=HookPoint.LLM_BEFORE, after=HookPoint.LLM_AFTER)
        async def my_phase(self, ctx: PhaseContext) -> str:
            return f"computed_{ctx.values.get('input', 'none')}"

    async def _run():
        p = MockPipeline()
        result = await p.my_phase(PhaseContext(values={"input": "42"}, round_index=0))
        assert isinstance(result, PhaseResult)
        assert result.output == "computed_42"
        assert result.directive == HookDirective.CONTINUE

    asyncio.run(_run())


def test_phase_decorator_before_skip():
    """Before hook returns SKIP — method body should not execute."""

    class MockPipeline:
        def __init__(self):
            self.hooks = AgentHooks()
            self.bus = AgentEventBus()

        @phase(before=HookPoint.TOOL_BEFORE)
        async def my_tool(self, ctx: PhaseContext) -> str:
            raise AssertionError("Should not be called")

    hooks = AgentHooks()
    hooks.register(HookPoint.TOOL_BEFORE, lambda ctx: HookResult(context=ctx, directive=HookDirective.SKIP))

    async def _run():
        p = MockPipeline()
        p.hooks = hooks
        result = await p.my_tool(PhaseContext(values={"tool_name": "x"}))
        assert result.should_skip

    asyncio.run(_run())


def test_phase_decorator_error_hook():
    """Error hook can swallow exception and provide fallback output."""

    class MockPipeline:
        def __init__(self):
            self.hooks = AgentHooks()
            self.bus = AgentEventBus()

        @phase(error=HookPoint.TOOL_ERROR)
        async def failing_phase(self, ctx: PhaseContext) -> str:
            raise RuntimeError("boom")

    hooks = AgentHooks()
    def error_handler(ctx: HookContext) -> HookResult:
        ctx.values["output"] = "fallback"
        return HookResult(context=ctx, directive=HookDirective.SKIP)

    hooks.register(HookPoint.TOOL_ERROR, error_handler)

    async def _run():
        p = MockPipeline()
        p.hooks = hooks
        result = await p.failing_phase(PhaseContext(values={}))
        assert result.output == "fallback"

    asyncio.run(_run())


if __name__ == "__main__":
    test_basic_pipeline_completes()
    test_pipeline_with_tools()
    test_round_start_break()
    test_llm_before_short_circuit()
    test_tool_before_skip()
    test_message_append_hook()
    test_phase_decorator_standalone()
    test_phase_decorator_before_skip()
    test_phase_decorator_error_hook()
    print("All pipeline tests passed!")
