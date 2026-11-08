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
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.events import AgentEventType
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
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


async def _run_child(runtime_agent, store, model, spec, *, bus=None, registry=None):
    from power_loop.core.agent_context import get_current_loop
    reg = registry or ToolRegistry()
    async def delegate():
        return await run_agent_spec(spec, "do work", parent_loop=get_current_loop())
    reg.register(ToolDefinition(name="delegate", description="delegate",
        input_schema={"type":"object", "properties":{}}, pricing=Free("test_fixture")), delegate)
    agent = await runtime_agent(model=model, store=store, registry=reg, event_bus=bus,
        config=AgentLoopConfig(compactor=None), command=("delegate", {}))
    result = await agent.send("go")
    assert result.status == "completed", result
    await agent.drain()
    return agent, json.loads(result.final_text)


@pytest.mark.parametrize("limited", [False, True])
async def test_child_lifecycle_events_keep_parent_attribution_and_order(store, runtime_agent, limited):
    from power_loop import AgentEventBus
    bus = AgentEventBus()
    events = []
    bus.subscribe(None, events.append)
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="echo", description="echo",
        input_schema={"type":"object", "properties":{}}, pricing=Free("test_fixture")), lambda **kw: "echoed")
    responses = [LLMResponse(raw_text="child out")]
    if limited:
        responses = [LLMResponse(raw_text="", tool_calls=[{"id":f"tc{i}", "type":"function",
            "function":{"name":"echo", "arguments":"{}"}}]) for i in range(2)]
    agent, result = await _run_child(runtime_agent, store, _Scripted(responses=responses),
        AgentSpec(name="c", system_prompt="P", max_rounds=2), bus=bus, registry=reg)
    assert result["status"] == ("interrupted" if limited else "completed")
    child_events = [e for e in events if e.type.value.startswith("subagent")]
    expected = [AgentEventType.SUBAGENT_TASK_START,
        AgentEventType.SUBAGENT_LIMIT if limited else AgentEventType.SUBAGENT_TEXT,
        AgentEventType.SUBAGENT_COMPLETED]
    assert [e.type for e in child_events] == expected
    assert all(e.source == SUBAGENT_EVENT_SOURCE and e.session_id == agent.target.session_id for e in child_events)
    assert all(e.data.sub_session_id == result["session_id"] for e in child_events)
    assert child_events[0].data.task == "do work" and child_events[0].data.depth == 1
    if limited:
        assert child_events[1].data.max_rounds == 2
    else:
        assert child_events[1].data.final_text == "child out"
        assert child_events[-1].data.status == "completed"
