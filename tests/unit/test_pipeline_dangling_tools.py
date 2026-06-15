"""Regression: early tool-loop exits must not leave dangling tool_calls.

When the per-tool loop stops early — a TOOL_AFTER hook returning BREAK, or a
``request_user_input`` batched before later tool_calls — the un-executed
tool_calls used to be left without ``tool`` responses, producing a
protocol-invalid sequence the provider rejects (HTTP 400) and a session wrongly
left 'pending'. They are now resolved with a synthetic skipped response.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from power_loop import (
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    create_default_tool_registry,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.hooks import HookDirective, HookPoint
from power_loop.core.hooks import AgentHooks
from power_loop.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _i: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        if self._i >= len(self.responses):
            return LLMResponse(raw_text="done", content_text="done")
        r = self.responses[self._i]
        self._i += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _multi_tool(*calls: tuple[str, str, str]) -> LLMResponse:
    return LLMResponse(raw_text="", tool_calls=[
        {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
        for cid, name, args in calls
    ])


def _echo_def() -> ToolDefinition:
    return ToolDefinition(name="echo", description="echo",
                          input_schema={"type": "object", "properties": {}})


def _assert_no_dangling(messages: list[dict]) -> None:
    requested: set = set()
    responded: set = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            requested |= {tc["id"] for tc in m["tool_calls"]}
        elif m.get("role") == "tool":
            responded.add(m.get("tool_call_id"))
    assert requested <= responded, f"dangling tool_calls: {requested - responded}"


async def test_tool_after_break_resolves_remaining_tool_calls() -> None:
    hooks = AgentHooks()
    hooks.register(HookPoint.TOOL_AFTER, lambda ctx: HookDirective.BREAK)  # break after 1st tool
    reg = ToolRegistry()
    reg.register(_echo_def(), lambda **k: "echoed")
    llm = _Scripted(responses=[
        _multi_tool(("c1", "echo", "{}"), ("c2", "echo", "{}")),
        LLMResponse(raw_text="done", content_text="done"),
    ])
    loop = StatefulAgentLoop(
        llm=llm, store=SessionStore.open(":memory:"), tool_registry=reg, hooks=hooks,
        config=AgentLoopConfig(system_prompt="o", max_rounds=4, compactor=None),
    )
    sid = loop.new_session()
    result = await loop.send("go", session_id=sid)
    assert result.status == "completed"
    _assert_no_dangling(loop.get_messages(sid))  # c2 must have a (skipped) response
    # the session must NOT be left pending → a second send works (no SessionPendingError)
    r2 = await loop.send("again", session_id=sid)
    assert r2.status in ("completed", "hit_round_limit")


async def test_request_user_input_batched_resolves_later_tool_calls() -> None:
    reg = create_default_tool_registry(include=["request_user_input"])
    reg.register(_echo_def(), lambda **k: "echoed")
    llm = _Scripted(responses=[
        _multi_tool(
            ("c1", "echo", "{}"),
            ("c2", "request_user_input",
             '{"kind":"confirm","prompt":"ok?","options":[{"id":"yes","label":"Yes"}]}'),
            ("c3", "echo", "{}"),
        ),
        LLMResponse(raw_text="resumed done", content_text="resumed done"),
    ])
    loop = StatefulAgentLoop(
        llm=llm, store=SessionStore.open(":memory:"), tool_registry=reg,
        config=AgentLoopConfig(system_prompt="o", max_rounds=4, compactor=None),
    )
    sid = loop.new_session()
    result = await loop.send("go", session_id=sid)
    assert result.status == "waiting_for_input"
    # c1 executed, c3 resolved as skipped (not dangling); c2 is the interaction.
    responded = {m.get("tool_call_id") for m in loop.get_messages(sid) if m.get("role") == "tool"}
    assert "c1" in responded and "c3" in responded

    interaction_id = result.pending_interactions[0]["interaction_id"]
    resumed = await loop.submit_input(sid, interaction_id, {"choice": "yes"})
    assert resumed.status == "completed"
    _assert_no_dangling(loop.get_messages(sid))  # full valid sequence after resume
