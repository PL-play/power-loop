from __future__ import annotations

from power_loop import (
    AgentEvent,
    AgentEventType,
    AgentMessage,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT,
    DEFAULT_SUBAGENT_SYSTEM_PROMPT,
    HookContext,
    HookPoint,
    SystemPromptContext,
    ToolDefinition,
    ToolCall,
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
    assert DEFAULT_AGENT_SYSTEM_PROMPT.strip() in prompt_main
    assert DEFAULT_SUBAGENT_SYSTEM_PROMPT.strip() in prompt_sub
    assert DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT.strip() in prompt_explore

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


if __name__ == "__main__":
    main()
