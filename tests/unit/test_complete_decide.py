"""COMPLETE_DECIDE hook: same-send injection at a send's terminal boundaries.

Covers: natural-completion injection (loop continues in the same send with a
durable user message), round-limit injection (budget extends past max_rounds),
handler self-bounding via fire_count, and the no-hook baseline staying intact.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from power_loop import (
    AgentLoopConfig,
    CompleteDecideCtx,
    HookDirective,
    HookPoint,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
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


def _tool_call(cid: str, name: str) -> LLMResponse:
    return LLMResponse(raw_text="", tool_calls=[
        {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}])


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="echo", description="e",
                 input_schema={"type": "object", "properties": {}}), lambda **k: "ok")
    return reg


def _finalize_once(prompt: str, extra_rounds: int = 4):
    """A COMPLETE_DECIDE handler that injects once, then lets the send end."""
    def handler(ctx: CompleteDecideCtx):
        if ctx.fire_count >= 1:
            return None
        ctx.inject = prompt
        ctx.extra_rounds = extra_rounds
        ctx.directive = HookDirective.SHORT_CIRCUIT
        return None
    return handler


async def test_natural_completion_injects_and_continues_same_send() -> None:
    # round 0: plain text (would complete) → hook injects → round 1 runs → completes.
    llm = _Scripted(responses=[
        LLMResponse(raw_text="first answer", content_text="first answer"),
        LLMResponse(raw_text="finalized", content_text="finalized"),
    ])
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt="o", max_rounds=10, compactor=None),
    )
    loop.hooks.register(HookPoint.COMPLETE_DECIDE, _finalize_once("[system] wrap up now"))
    sid = await loop.new_session()
    res = await loop.send("go", session_id=sid)

    assert res.status == "completed"
    assert res.final_text == "finalized"
    msgs = await loop.get_messages(sid)
    contents = [(m["role"], m.get("content") or "") for m in msgs]
    # The injected prompt is a DURABLE user message in the same send,
    # between the two assistant turns.
    idx = [i for i, (r, c) in enumerate(contents) if c == "[system] wrap up now"]
    assert len(idx) == 1 and contents[idx[0]][0] == "user"
    first = next(i for i, (r, c) in enumerate(contents) if c == "first answer")
    last = next(i for i, (r, c) in enumerate(contents) if c == "finalized")
    assert first < idx[0] < last


async def test_round_limit_injects_and_extends_budget() -> None:
    # max_rounds=1; round 0 calls a tool → budget exhausted → hook injects →
    # round 1 runs and completes with text (no forced wrap-up call).
    llm = _Scripted(responses=[
        _tool_call("c1", "echo"),
        LLMResponse(raw_text="made it past the limit", content_text="made it past the limit"),
    ])
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt="o", max_rounds=1, compactor=None),
    )
    loop.hooks.register(HookPoint.COMPLETE_DECIDE, _finalize_once("[system] budget gone — wrap up", 2))
    sid = await loop.new_session()
    res = await loop.send("go", session_id=sid)

    assert res.status == "completed"
    assert res.final_text == "made it past the limit"
    msgs = await loop.get_messages(sid)
    texts = [(m.get("content") or "") for m in msgs]
    assert "[system] budget gone — wrap up" in texts
    # The stock "You have reached the maximum" prompt must NOT appear.
    assert not any("reached the maximum" in t for t in texts)


async def test_handler_bounds_itself_with_fire_count() -> None:
    # A handler that fires every time it is allowed to (fire_count < 2):
    # two injections, then the send ends normally.
    fires: list[int] = []

    def handler(ctx: CompleteDecideCtx):
        fires.append(ctx.fire_count)
        if ctx.fire_count >= 2:
            return None
        ctx.inject = f"again #{ctx.fire_count}"
        ctx.directive = HookDirective.SHORT_CIRCUIT
        return None

    llm = _Scripted(responses=[
        LLMResponse(raw_text="a", content_text="a"),
        LLMResponse(raw_text="b", content_text="b"),
        LLMResponse(raw_text="c", content_text="c"),
    ])
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt="o", max_rounds=10, compactor=None),
    )
    loop.hooks.register(HookPoint.COMPLETE_DECIDE, handler)
    sid = await loop.new_session()
    res = await loop.send("go", session_id=sid)

    assert res.status == "completed"
    assert res.final_text == "c"
    assert fires == [0, 1, 2]  # consulted three times, injected twice


async def test_no_hook_baseline_unchanged() -> None:
    # Without a COMPLETE_DECIDE handler: natural completion and round-limit
    # behave exactly as before.
    llm = _Scripted(responses=[LLMResponse(raw_text="plain", content_text="plain")])
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt="o", max_rounds=3, compactor=None),
    )
    sid = await loop.new_session()
    res = await loop.send("go", session_id=sid)
    assert res.status == "completed" and res.final_text == "plain"

    llm2 = _Scripted(responses=[_tool_call("c1", "echo"),
                                LLMResponse(raw_text="summary", content_text="summary")])
    store2 = await SessionStore.open(":memory:")
    loop2 = StatefulAgentLoop(
        llm=llm2, store=store2, tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt="o", max_rounds=1, compactor=None),
    )
    sid2 = await loop2.new_session()
    res2 = await loop2.send("go", session_id=sid2)
    assert res2.status == "hit_round_limit"
    assert "summary" in (res2.final_text or "")


async def test_empty_inject_or_wrong_directive_ends_normally() -> None:
    # SHORT_CIRCUIT with empty inject, or CONTINUE with inject set → no continuation.
    def empty_inject(ctx: CompleteDecideCtx):
        ctx.directive = HookDirective.SHORT_CIRCUIT
        ctx.inject = "   "
        return None

    llm = _Scripted(responses=[LLMResponse(raw_text="plain", content_text="plain")])
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt="o", max_rounds=5, compactor=None),
    )
    loop.hooks.register(HookPoint.COMPLETE_DECIDE, empty_inject)
    sid = await loop.new_session()
    res = await loop.send("go", session_id=sid)
    assert res.status == "completed" and res.final_text == "plain"
    msgs = await loop.get_messages(sid)
    assert sum(1 for m in msgs if m["role"] == "assistant") == 1
