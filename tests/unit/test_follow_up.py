"""Unit tests for follow-up merge helpers and queue draining."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import pytest

from power_loop import AgentLoopConfig, FollowUpQueued, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.agent.follow_up import (
    FOLLOW_UP_MESSAGE_NAME,
    format_follow_up_user_message,
    merge_follow_up_inputs,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry


def test_format_follow_up_user_message_wraps_body() -> None:
    msg = format_follow_up_user_message("  focus on feelings  ")
    assert msg["role"] == "user"
    assert msg["name"] == FOLLOW_UP_MESSAGE_NAME
    assert msg["content"] == "<follow_up>\nfocus on feelings\n</follow_up>"


def test_merge_follow_up_inputs_empty_returns_none() -> None:
    assert merge_follow_up_inputs([]) is None
    assert merge_follow_up_inputs(["", "   "]) is None


def test_merge_follow_up_inputs_joins_multiple_strings() -> None:
    merged = merge_follow_up_inputs(["first", "second"])
    assert merged is not None
    assert merged["name"] == FOLLOW_UP_MESSAGE_NAME
    assert "first\n\nsecond" in merged["content"]


def test_merge_follow_up_inputs_accepts_loop_messages() -> None:
    merged = merge_follow_up_inputs(
        [
            {"role": "user", "content": "alpha"},
            format_follow_up_user_message("beta"),
        ]
    )
    assert merged is not None
    assert "alpha" in merged["content"]
    assert "beta" in merged["content"]


class _GateLLM(LLMService):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []
        self._idx = 0
        self.release_first = asyncio.Event()

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        self.calls.append(list(request.messages))
        if self._idx == 0:
            await self.release_first.wait()
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        response = self.responses[self._idx]
        self._idx += 1
        return response

    async def close(self) -> None:
        return None


def _tool_resp(call_id: str, name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    )


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="Echo text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            required_params=("text",),
        ),
        lambda **kw: str(kw.get("text") or ""),
    )
    return reg


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


@pytest.mark.asyncio
async def test_multiple_follow_ups_merge_at_next_round(store: SessionStore) -> None:
    llm = _GateLLM(
        responses=[
            _tool_resp("c1", "echo", '{"text":"step1"}'),
            LLMResponse(raw_text="merged steering applied"),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        tool_registry=_echo_registry(),
    )
    sid = loop.new_session()
    send_task = asyncio.create_task(loop.send("start", sid))

    for _ in range(200):
        if loop._lock_for(sid).locked():
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("expected session lock during send")

    first = await loop.follow_up("focus on feelings", sid)
    second = await loop.follow_up("keep it short", sid)
    assert isinstance(first, FollowUpQueued)
    assert isinstance(second, FollowUpQueued)
    assert first.queue_depth == 1
    assert second.queue_depth == 2

    llm.release_first.set()
    result = await send_task
    assert result.status == "completed"

    rows = store.load_active_messages(sid)
    follow_rows = [r for r in rows if r.name == FOLLOW_UP_MESSAGE_NAME]
    assert len(follow_rows) == 1
    assert "focus on feelings" in follow_rows[0].content
    assert "keep it short" in follow_rows[0].content

    second_request = llm.calls[1]
    follow_contents = [
        str(m.get("content") or "")
        for m in second_request
        if m.get("name") == FOLLOW_UP_MESSAGE_NAME
        or "<follow_up>" in str(m.get("content") or "")
    ]
    assert any("focus on feelings" in c and "keep it short" in c for c in follow_contents)
