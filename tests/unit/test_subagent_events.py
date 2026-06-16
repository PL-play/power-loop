"""Sub-agent lifecycle events (SUBAGENT_* + AgentEvent.source).

``run_agent_spec`` publishes four lifecycle events on the *parent* bus so a host
can observe delegated work without instrumenting the child loop:

* ``SUBAGENT_TASK_START`` — on every spawn (task / preset / sub_session_id / depth)
* ``SUBAGENT_TEXT``       — only when the child produced an answer (status completed)
* ``SUBAGENT_LIMIT``      — only on the round-limit failure mode
* ``SUBAGENT_COMPLETED``  — terminal marker, always last

All four carry ``source="subagent"`` and are attributed to the parent session.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.spec import SUBAGENT_EVENT_SOURCE, run_agent_spec
from power_loop.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _idx: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done", content_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


def _collect(loop: StatefulAgentLoop) -> tuple[list[AgentEvent], Callable[[AgentEventType], list[AgentEvent]]]:
    events: list[AgentEvent] = []
    loop.event_bus.subscribe(None, events.append)
    return events, lambda t: [e for e in events if e.type == t]


async def _run_child(parent_loop: StatefulAgentLoop, parent_sid: str, spec: AgentSpec,
                     user_input: str) -> dict[str, Any]:
    tok_loop = set_current_loop(parent_loop)
    tok_sid = set_session_id(parent_sid)
    try:
        return await run_agent_spec(spec, user_input, parent_loop=parent_loop)
    finally:
        reset_current_loop(tok_loop)
        reset_session_id(tok_sid)


async def test_completed_subagent_emits_start_text_completed(store: SessionStore) -> None:
    parent_loop = StatefulAgentLoop(
        llm=_Scripted(responses=[LLMResponse(raw_text="parent", content_text="parent")]),
        store=store, config=AgentLoopConfig(max_rounds=1),
    )
    parent_sid = (await parent_loop.send("hi", session_id=await parent_loop.new_session())).session_id
    events, of_type = _collect(parent_loop)

    parent_loop.llm = _Scripted(responses=[LLMResponse(raw_text="child out", content_text="child out")])
    result = await _run_child(parent_loop, parent_sid, AgentSpec(name="c", system_prompt="P"), "do work")
    assert result["status"] == "completed"

    starts = of_type(AgentEventType.SUBAGENT_TASK_START)
    texts = of_type(AgentEventType.SUBAGENT_TEXT)
    dones = of_type(AgentEventType.SUBAGENT_COMPLETED)
    assert len(starts) == 1 and len(texts) == 1 and len(dones) == 1
    assert not of_type(AgentEventType.SUBAGENT_LIMIT)  # only on round-limit

    # source + attribution
    for e in starts + texts + dones:
        assert e.source == SUBAGENT_EVENT_SOURCE
        assert e.session_id == parent_sid
    # payload fields wired
    assert starts[0].data.task == "do work"
    sub_sid = starts[0].data.sub_session_id
    assert sub_sid and texts[0].data.sub_session_id == sub_sid == dones[0].data.sub_session_id
    assert texts[0].data.final_text == "child out"
    assert dones[0].data.status == "completed"

    # ordering: START before COMPLETED
    types_in_order = [e.type for e in events
                      if e.type.value.startswith("subagent") and e.source == SUBAGENT_EVENT_SOURCE]
    assert types_in_order[0] == AgentEventType.SUBAGENT_TASK_START
    assert types_in_order[-1] == AgentEventType.SUBAGENT_COMPLETED


async def test_round_limited_subagent_emits_limit_not_text(store: SessionStore) -> None:
    parent_loop = StatefulAgentLoop(
        llm=_Scripted(responses=[LLMResponse(raw_text="parent", content_text="parent")]),
        store=store, config=AgentLoopConfig(max_rounds=1),
    )
    parent_sid = (await parent_loop.send("hi", session_id=await parent_loop.new_session())).session_id
    _, of_type = _collect(parent_loop)

    # Child calls a resolvable tool every round so rounds are exhausted cleanly
    # (rather than ending mid tool-call as pending_tools) → hit_round_limit.
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="echo", description="echo", input_schema={"type": "object", "properties": {}}),
        lambda **kw: "echoed",
    )
    parent_loop.tool_registry = reg

    def _toolish() -> LLMResponse:
        return LLMResponse(raw_text="", tool_calls=[{
            "id": "tc", "type": "function",
            "function": {"name": "echo", "arguments": json.dumps({})}}])

    parent_loop.llm = _Scripted(responses=[_toolish(), _toolish(), _toolish()])
    result = await _run_child(parent_loop, parent_sid,
                              AgentSpec(name="hopeless", system_prompt="P", max_rounds=2), "go")
    assert result["status"] == "hit_round_limit"

    assert len(of_type(AgentEventType.SUBAGENT_TASK_START)) == 1
    assert len(of_type(AgentEventType.SUBAGENT_LIMIT)) == 1
    assert not of_type(AgentEventType.SUBAGENT_TEXT)  # no answer was produced
    assert len(of_type(AgentEventType.SUBAGENT_COMPLETED)) == 1
    limit = of_type(AgentEventType.SUBAGENT_LIMIT)[0]
    assert limit.source == SUBAGENT_EVENT_SOURCE
    assert limit.data.max_rounds == 2
