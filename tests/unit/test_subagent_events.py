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
        config=AgentLoopConfig(context_budget_tokens=10**9), command=("delegate", {}))
    result = await agent.send("go")
    assert result.status == "completed", result
    await agent.drain()
    return agent, json.loads(result.final_text)


@pytest.mark.parametrize("limited", [False, True])
async def test_child_lifecycle_events_keep_parent_attribution_and_order(store, runtime_agent, limited):
    from power_loop import AgentEventBus
    bus = AgentEventBus()
    events = []
    timeline: list[str] = []

    def _record(event):
        events.append(event)
        if event.type.value.startswith("subagent"):
            timeline.append(event.type.value)

    bus.subscribe(None, _record)
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="echo", description="echo",
        input_schema={"type":"object", "properties":{}}, pricing=Free("test_fixture")), lambda **kw: "echoed")
    responses = [LLMResponse(raw_text="child out")]
    if limited:
        responses = [LLMResponse(raw_text="", tool_calls=[{"id":f"tc{i}", "type":"function",
            "function":{"name":"echo", "arguments":"{}"}}]) for i in range(2)]

    class _TracingScripted(_Scripted):
        async def complete(self, request, **kwargs):
            timeline.append("child_llm")
            return await super().complete(request, **kwargs)

    agent, result = await _run_child(runtime_agent, store, _TracingScripted(responses=responses),
        AgentSpec(name="c", system_prompt="P", max_rounds=2), bus=bus, registry=reg)
    assert result["status"] == ("failed" if limited else "completed")
    if limited:
        assert result["error_code"] == "hit_round_limit"
    else:
        assert result["error_code"] is None
    child_events = [e for e in events if e.type.value.startswith("subagent")]
    expected = [AgentEventType.SUBAGENT_TASK_START,
        AgentEventType.SUBAGENT_LIMIT if limited else AgentEventType.SUBAGENT_TEXT,
        AgentEventType.SUBAGENT_COMPLETED]
    assert [e.type for e in child_events] == expected
    assert all(e.source == SUBAGENT_EVENT_SOURCE and e.session_id == agent.target.session_id for e in child_events)
    assert all(e.data.sub_session_id == result["session_id"] for e in child_events)
    assert child_events[0].data.task == "do work" and child_events[0].data.depth == 1
    assert child_events[0].data.preset == "c"
    assert timeline[0] == AgentEventType.SUBAGENT_TASK_START.value
    assert "child_llm" in timeline
    assert timeline.index("child_llm") > 0
    assert timeline[-1] == AgentEventType.SUBAGENT_COMPLETED.value
    assert timeline.index(AgentEventType.SUBAGENT_COMPLETED.value) > timeline.index("child_llm")
    if limited:
        assert result["rounds"] == 2
        assert result["usage"]["calls"] == 3
        assert child_events[1].data.max_rounds == 2
        assert child_events[-1].data.status == "failed"
        assert child_events[-1].data.rounds == 2
    else:
        assert result["rounds"] == 1
        assert result["usage"]["calls"] == 1
        assert child_events[1].data.final_text == "child out"
        assert child_events[1].data.rounds == 1
        assert child_events[-1].data.status == "completed"
        assert child_events[-1].data.rounds == 1


async def test_detached_child_parent_self_cancel_does_not_fabricate_completed_event(store, runtime_agent):
    """When a parent cancels only itself (scope='self') while a detached child is still running,
    the parent bus must NOT receive a fabricated SUBAGENT_COMPLETED claiming cancelled/failed;
    the detached child finishes independently and its original InputRef reads completed.
    """
    import asyncio

    from power_loop import AgentEventBus
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef
    from power_loop.core.agent_context import get_current_loop

    bus = AgentEventBus()
    events = []
    bus.subscribe(None, lambda e: events.append(e) if e.type.value.startswith("subagent") else None)
    child_started, release_child = asyncio.Event(), asyncio.Event()

    class _BlockingModel(_Scripted):
        async def complete(self, request, **kwargs):
            child_started.set()
            await release_child.wait()
            return LLMResponse(raw_text="detached child done")

    reg = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="detached-worker", system_prompt="CHILD", max_rounds=2, lifecycle="detached"),
            "long task",
            parent_loop=get_current_loop(),
        )

    reg.register(
        ToolDefinition(
            name="delegate", description="delegate",
            input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture"),
        ),
        delegate,
    )
    agent = await runtime_agent(
        model=_BlockingModel(), store=store, registry=reg, event_bus=bus,
        config=AgentLoopConfig(context_budget_tokens=10**9), command=("delegate", {}),
    )
    try:
        receipt = await agent.runtime.client.send(
            agent.target, "go", message_id="m-detached-parent", actor=agent.actor,
        )
        await asyncio.wait_for(child_started.wait(), 5)
        await agent.runtime.client.cancel(
            receipt.input_ref, actor=agent.actor, scope="self", reason="stop parent only",
        )
        parent_res = await agent.runtime.client.wait(
            receipt.input_ref, actor=agent.actor, deadline_ms=now_ms() + 5000,
        )
        assert parent_res.status == "cancelled"

        # While the detached child is still running, only SUBAGENT_TASK_START may have been published.
        assert [e.type for e in events] == [AgentEventType.SUBAGENT_TASK_START]
        assert events[0].source == SUBAGENT_EVENT_SOURCE
        assert events[0].session_id == agent.target.session_id
        child_sid = events[0].data.sub_session_id
        assert child_sid

        # Confirm the child is genuinely still running in the store, then release it and read its real result.
        srow = await store._db.fetchone(
            f"SELECT execution_state,start_input_message_id FROM {store.t.sessions} WHERE session_id=?",
            (child_sid,),
        )
        assert srow["execution_state"] == "running"
        child_ref = InputRef(store.store_id, child_sid, srow["start_input_message_id"])

        release_child.set()
        child_res = await agent.runtime.client.wait(
            child_ref, actor=agent.actor, deadline_ms=now_ms() + 5000,
        )
        assert child_res.status == "completed"
        assert child_res.final_text == "detached child done"
        assert child_res.rounds == 1
        assert [e.type for e in events] == [AgentEventType.SUBAGENT_TASK_START]
        await agent.drain()
    finally:
        release_child.set()


async def test_model_tool_delegation_publishes_subagent_events_on_parent_bus(store, runtime_agent):
    """Ordinary model tool call delegating via run_agent_spec also emits START -> TEXT -> COMPLETED."""
    from power_loop import AgentEventBus
    from power_loop.core.agent_context import get_current_loop
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract

    bus = AgentEventBus()
    child_events = []
    bus.subscribe(None, lambda e: child_events.append(e) if e.type.value.startswith("subagent") else None)

    reg = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="helper", system_prompt="CHILD", max_rounds=2),
            "summarize",
            parent_loop=get_current_loop(),
        )

    reg.register(
        ToolDefinition(name="delegate", description="delegate", input_schema={"type": "object", "properties": {}},
                       pricing=Free("test_fixture")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    class _ParentAndChild(_Scripted):
        submitted = False

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("CHILD"):
                return LLMResponse(raw_text="child summary")
            if not self.submitted:
                self.submitted = True
                return LLMResponse(raw_text="", tool_calls=[{
                    "id": "call-delegate", "type": "function",
                    "function": {"name": "delegate", "arguments": "{}"},
                }])
            return LLMResponse(raw_text="parent done")

    agent = await runtime_agent(
        model=_ParentAndChild(), store=store, registry=reg, event_bus=bus,
        config=AgentLoopConfig(context_budget_tokens=10**9),
    )
    res = await agent.send("go")
    assert res.status == "completed"
    await agent.drain()
    rows = await store.load_all_messages(agent.target.session_id)
    tool_payload = json.loads(next(r.content for r in rows if r.tool_call_id == "call-delegate"))
    assert tool_payload["status"] == "completed"
    assert [e.type for e in child_events] == [
        AgentEventType.SUBAGENT_TASK_START,
        AgentEventType.SUBAGENT_TEXT,
        AgentEventType.SUBAGENT_COMPLETED,
    ]
    assert all(e.source == SUBAGENT_EVENT_SOURCE and e.session_id == agent.target.session_id for e in child_events)
    assert all(e.data.sub_session_id == tool_payload["session_id"] for e in child_events)
    assert child_events[1].data.final_text == "child summary"
    assert child_events[-1].data.status == "completed"
