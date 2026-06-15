from __future__ import annotations

from power_loop import (
    AgentEvent,
    AgentEventType,
    AgentMessage,
    HookContext,
    HookPoint,
    SystemPromptContext,
    ToolCall,
    ToolDefinition,
    build_agent_system_prompt,
    build_explore_subagent_system_prompt,
    build_subagent_system_prompt,
    validate_tool_args,
)
from power_loop.contracts.messages import AssistantMessage, ToolResultMessage, UserMessage


def main() -> None:
    user = UserMessage("hi")
    tool_call = ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})
    assistant = AssistantMessage("I will read file", tool_calls=[tool_call])
    tool = ToolResultMessage("file content", tool_call_id="call_1", name="read_file")

    assert isinstance(user, AgentMessage)
    assert assistant.tool_calls[0].name == "read_file"
    assert tool.role == "tool"

    event = AgentEvent(type=AgentEventType.ROUND_STARTED, payload={"round": 1})
    assert event.type == AgentEventType.ROUND_STARTED
    assert AgentEventType.STREAM_THINK_DELTA.value == "stream_think_delta"
    assert AgentEventType.SUBAGENT_LIMIT.value == "subagent_limit"

    hook_ctx = HookContext(values={"round": 1})
    assert hook_ctx.values["round"] == 1
    assert HookPoint.LLM_BEFORE.value == "llm.before"
    assert HookPoint.TOOLS_BATCH_BEFORE.value == "tools.batch.before"

    prompt_main = build_agent_system_prompt(
        SystemPromptContext(model="test-model", workspace_dir="/tmp/workspace"),
        extra="Keep answers concise.",
    )
    prompt_sub = build_subagent_system_prompt(
        SystemPromptContext(model="test-model", workspace_dir="/tmp/workspace"),
    )
    prompt_explore = build_explore_subagent_system_prompt(
        SystemPromptContext(model="test-model", workspace_dir="/tmp/workspace"),
    )
    assert "test-model" in prompt_main
    assert "Keep answers concise." in prompt_main
    assert "/tmp/workspace" in prompt_main
    assert "coding subagent" in prompt_sub
    assert "/tmp/workspace" in prompt_sub
    assert "read-only" in prompt_explore
    assert "/tmp/workspace" in prompt_explore

    tool_def = ToolDefinition(
        name="read_file",
        description="Read file content",
        required_params=("path",),
    )
    as_tool = tool_def.to_openai_tool()
    assert as_tool["function"]["name"] == "read_file"
    assert validate_tool_args("read_file", {"path": "README.md"}) is None
    assert validate_tool_args("read_file", {}) is not None

    print("smoke_contract_definitions passed")


def test_agent_event_has_ts_and_monotonic_seq() -> None:
    """H4.2: every event auto-stamps a wall-clock ``ts`` and a process-wide
    monotonic ``seq`` so streams are timestampable and totally orderable."""
    e1 = AgentEvent(type=AgentEventType.ROUND_STARTED, payload={"round": 1})
    e2 = AgentEvent(type=AgentEventType.ROUND_COMPLETED, payload={"round": 1})
    assert e1.ts > 0 and e2.ts > 0
    assert isinstance(e1.seq, int) and e2.seq > e1.seq  # strictly increasing order
    # seq/ts are excluded from equality (envelope metadata, not identity)
    assert AgentEvent(type=AgentEventType.SESSION_ENDED) == AgentEvent(type=AgentEventType.SESSION_ENDED)
    # explicit values are preserved (a deserializer can replay original ordering)
    replayed = AgentEvent(type=AgentEventType.SESSION_ENDED, ts=123.5, seq=7)
    assert replayed.ts == 123.5 and replayed.seq == 7


if __name__ == "__main__":
    main()
