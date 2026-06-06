"""Unit tests for the native Anthropic Messages API transport."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from llm_client.anthropic_factory import AnthropicMessagesLLMService
from llm_client.interface import AnthropicChatConfig, LLMRequest


class _FakeMessages:
    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response: Any):
        self.messages = _FakeMessages(response)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _service(response: Any) -> tuple[AnthropicMessagesLLMService, _FakeClient]:
    svc = AnthropicMessagesLLMService(
        AnthropicChatConfig(
            base_url="https://anthropic.example",
            api_key="sk-test",
            model="claude-test",
            max_tokens=256,
            temperature=0.2,
        )
    )
    client = _FakeClient(response)
    svc._client = cast(Any, client)
    return svc, client


def test_dashscope_anthropic_omits_forced_tool_choice() -> None:
    svc = AnthropicMessagesLLMService(
        AnthropicChatConfig(
            base_url="https://dashscope.aliyuncs.com/apps/anthropic",
            api_key="sk-test",
            model="deepseek-v4-flash",
        )
    )
    assert svc._tool_choice({"type": "function", "function": {"name": "lookup"}}) is None


@pytest.mark.asyncio
async def test_complete_maps_text_response_and_usage() -> None:
    response = SimpleNamespace(
        content=[{"type": "text", "text": "hello"}],
        usage={"input_tokens": 3, "output_tokens": 4},
    )
    svc, client = _service(response)

    result = await svc.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))

    assert result.raw_text == "hello"
    assert result.token_usage.prompt_tokens == 3
    assert result.token_usage.completion_tokens == 4
    assert result.token_usage.total_tokens == 7
    assert client.messages.calls[0]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


@pytest.mark.asyncio
async def test_openai_tools_and_tool_results_are_converted() -> None:
    response = SimpleNamespace(
        content=[
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "teal"}},
        ],
        usage={"input_tokens": 1, "output_tokens": 2},
    )
    svc, client = _service(response)

    result = await svc.complete(
        LLMRequest(
            system_prompt="Be useful.",
            messages=[
                {"role": "user", "content": "find it"},
                {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "teal"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "found"},
                {"role": "user", "content": "summarize"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup a value.",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    },
                }
            ],
        )
    )

    call = client.messages.calls[0]
    assert call["system"] == "Be useful."
    assert call["tools"] == [
        {
            "name": "lookup",
            "description": "Lookup a value.",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]
    assert call["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {"q": "teal"},
    }
    assert call["messages"][2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "found"},
            {"type": "text", "text": "summarize"},
        ],
    }
    assert result.get_tool_calls() == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q": "teal"}'},
        }
    ]
