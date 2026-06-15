"""Regression: the round-limit summary call goes through call_llm (so it gets
retry/cancel/stream events), and every LLM call emits a STREAM_COMPLETED even on
failure (no dangling STREAM_STARTED)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    LLMRetryPolicy,
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


async def test_round_limit_final_call_routes_through_call_llm() -> None:
    bus = AgentEventBus()
    events: list = []
    bus.subscribe(None, events.append)
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="echo", description="e",
                 input_schema={"type": "object", "properties": {}}), lambda **k: "ok")
    # round 0 makes a tool call (resolves); max_rounds=1 → the summary call fires.
    llm = _Scripted(responses=[_tool_call("c1", "echo"),
                               LLMResponse(raw_text="summary", content_text="summary")])
    loop = StatefulAgentLoop(
        llm=llm, store=SessionStore.open(":memory:"), tool_registry=reg, event_bus=bus,
        config=AgentLoopConfig(system_prompt="o", max_rounds=1, compactor=None),
    )
    res = await loop.send("go", session_id=loop.new_session())
    assert res.status == "hit_round_limit"
    starts = sum(1 for e in events if e.type == AgentEventType.STREAM_STARTED)
    completes = sum(1 for e in events if e.type == AgentEventType.STREAM_COMPLETED)
    # round-0 call AND the final summary call both route through call_llm → 2 each.
    # (Before the fix the summary used a raw llm.complete and emitted neither → 1.)
    assert starts == 2 and completes == 2


@dataclass
class _FailLLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        raise RuntimeError("boom")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


async def test_call_llm_emits_stream_completed_on_failure() -> None:
    bus = AgentEventBus()
    events: list = []
    bus.subscribe(None, events.append)
    loop = StatefulAgentLoop(
        llm=_FailLLM(), store=SessionStore.open(":memory:"), event_bus=bus,
        config=AgentLoopConfig(
            system_prompt="o", max_rounds=2, compactor=None,
            retry_policy=LLMRetryPolicy(max_attempts=2, backoff_initial=0.001, backoff_max=0.002),
        ),
    )
    res = await loop.send("go", session_id=loop.new_session())
    assert res.status == "degraded"
    starts = sum(1 for e in events if e.type == AgentEventType.STREAM_STARTED)
    completes = sum(1 for e in events if e.type == AgentEventType.STREAM_COMPLETED)
    # the call failed (retry exhausted) but the stream still got a terminal event
    assert starts >= 1 and completes >= 1
