from __future__ import annotations

from dataclasses import dataclass

from llm_client.interface import LLMResponse
from power_loop.agent.loop import AgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry


@dataclass
class _FakeLLM:
    mode: str = "plain"

    async def complete(self, request, **kwargs):
        if self.mode == "tool_exec":
            if request.messages and request.messages[-1].get("role") == "tool":
                return LLMResponse(raw_text="tool processed")
            return LLMResponse(
                raw_text="calling tool",
                tool_calls=[
                    {
                        "id": "call_echo_1",
                        "type": "function",
                        "function": {"name": "echo_tool", "arguments": "{\"text\": \"hello\"}"},
                    }
                ],
            )
        if self.mode == "tool":
            return LLMResponse(
                raw_text="I will call a tool",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"},
                    }
                ],
            )
        return LLMResponse(raw_text="hello from llm")

    async def stream(self, request):
        raise NotImplementedError

    async def close(self):
        return None


def _test_plain_completion() -> None:
    loop = AgentLoop(_FakeLLM(mode="plain"), AgentLoopConfig(max_rounds=3))
    result = loop.run_sync([{"role": "user", "content": "hi"}])
    assert result.status == "completed"
    assert result.final_text == "hello from llm"


def _test_pending_tools() -> None:
    loop = AgentLoop(_FakeLLM(mode="tool"), AgentLoopConfig(max_rounds=3))
    result = loop.run_sync([{"role": "user", "content": "use tool"}])
    assert result.status == "pending_tools"
    assert len(result.pending_tool_calls) == 1


def _test_tool_execution_chain() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo_tool",
            description="Echo input",
            required_params=("text",),
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        lambda text: f"echo:{text}",
    )
    loop = AgentLoop(_FakeLLM(mode="tool_exec"), AgentLoopConfig(max_rounds=3), tool_registry=registry)
    result = loop.run_sync([{"role": "user", "content": "use tool and continue"}])
    assert result.status == "completed"
    assert result.final_text == "tool processed"
    assert any(m.get("role") == "tool" and m.get("content") == "echo:hello" for m in result.messages)


if __name__ == "__main__":
    _test_plain_completion()
    _test_pending_tools()
    _test_tool_execution_chain()
    print("smoke_agent_loop_v1 passed")
