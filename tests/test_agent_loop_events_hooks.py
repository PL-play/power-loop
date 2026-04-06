from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from llm_client.interface import LLMResponse

from power_loop.agent.loop import AgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.contracts.hooks import HookContext, HookPoint
from power_loop.contracts.hook_contexts import (
    BaseHookCtx,
    LlmAfterCtx,
    LlmBeforeCtx,
    RoundEndCtx,
    RoundStartCtx,
    SessionEndCtx,
    SessionStartCtx,
    ToolAfterCtx,
    ToolBeforeCtx,
    ToolsBatchAfterCtx,
    ToolsBatchBeforeCtx,
)
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.contracts.events import AgentEventType
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry


class _FakeLLM:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def complete(self, request: Any, **kwargs: Any) -> LLMResponse:
        if self.mode == "plain":
            return LLMResponse(raw_text="hello from llm")

        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo_tool", "arguments": "{\"text\": \"hi\"}"},
        }

        if self.mode == "tool":
            return LLMResponse(raw_text="I will call a tool", tool_calls=[tool_call])

        if self.mode == "tool_exec":
            if request.messages and request.messages[-1].get("role") == "tool":
                return LLMResponse(raw_text="tool processed")
            return LLMResponse(raw_text="I will call a tool", tool_calls=[tool_call])

        raise AssertionError(f"unknown mode: {self.mode}")

    async def stream(self, request: Any):
        raise NotImplementedError

    async def close(self) -> None:
        return None


def test_plain_completion() -> None:
    loop = AgentLoop(_FakeLLM("plain"), AgentLoopConfig(max_rounds=2))
    result = loop.run_sync([{"role": "user", "content": "hi"}])
    assert result.status == "completed"
    assert result.final_text == "hello from llm"


def test_pending_tools_when_no_registry() -> None:
    loop = AgentLoop(_FakeLLM("tool"), AgentLoopConfig(max_rounds=2))
    result = loop.run_sync([{"role": "user", "content": "use tool"}])
    assert result.status == "pending_tools"
    assert len(result.pending_tool_calls) == 1


def test_events_and_hooks_for_tool_execution() -> None:
    events: list[AgentEventType] = []

    bus = AgentEventBus(suppress_subscriber_errors=True)

    def _on_event(event: Any) -> None:
        events.append(event.type)

    bus.subscribe(None, _on_event)

    hooks = AgentHooks()
    hook_calls: list[str] = []
    tool_before: list[dict[str, Any]] = []
    tool_after: list[tuple[str, str]] = []

    def _mk(name: str):
        def _handler(ctx: BaseHookCtx) -> None:
            hook_calls.append(name)

        return _handler

    def _tool_before(ctx: ToolBeforeCtx) -> None:
        hook_calls.append("tool.before")
        tool_before.append(dict(ctx.tool_args))

    def _tool_after(ctx: ToolAfterCtx) -> None:
        hook_calls.append("tool.after")
        tool_after.append((ctx.tool_name, ctx.output))

    hooks.register(HookPoint.SESSION_START, _mk("session.start"))
    hooks.register(HookPoint.ROUND_START, _mk("round.start"))
    hooks.register(HookPoint.LLM_BEFORE, _mk("llm.before"))
    hooks.register(HookPoint.LLM_AFTER, _mk("llm.after"))
    hooks.register(HookPoint.TOOLS_BATCH_BEFORE, _mk("tools.batch.before"))
    hooks.register(HookPoint.TOOLS_BATCH_AFTER, _mk("tools.batch.after"))
    hooks.register(HookPoint.TOOL_BEFORE, _tool_before)
    hooks.register(HookPoint.TOOL_AFTER, _tool_after)
    hooks.register(HookPoint.ROUND_END, _mk("round.end"))
    hooks.register(HookPoint.SESSION_END, _mk("session.end"))

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo_tool",
            description="Echo tool",
            required_params=("text",),
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        lambda text: f"echo:{text}",
    )

    loop = AgentLoop(
        _FakeLLM("tool_exec"),
        AgentLoopConfig(max_rounds=2),
        tool_registry=registry,
        event_bus=bus,
        hooks=hooks,
    )

    result = loop.run_sync([{"role": "user", "content": "use tool"}])
    assert result.status == "completed"
    assert result.final_text == "tool processed"

    assert AgentEventType.TOOL_CALL_STARTED in events
    assert AgentEventType.TOOL_CALL_COMPLETED in events

    assert tool_before and tool_before[0].get("text") == "hi"
    assert tool_after and "echo:hi" in tool_after[0][1]

    # Ensure the hook order is plausible for the executed tool.
    assert hook_calls.index("tool.before") < hook_calls.index("tool.after")

