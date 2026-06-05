from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import (
    MAX_SPAWN_DEPTH,
    AgentLoopConfig,
    AgentSpec,
    AgentSpecError,
    SessionStore,
    StatefulAgentLoop,
    SubagentLifecycle,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.spec import filtered_registry, run_agent_spec
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None: return None


def _spawn_call(call_id: str, task: str) -> LLMResponse:
    import json
    return LLMResponse(
        raw_text="",
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "arguments": json.dumps({"task": task}),
            },
        }],
    )


def _run_agent_call(call_id: str, spec: dict, user_input: str) -> LLMResponse:
    import json
    return LLMResponse(
        raw_text="",
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "run_agent",
                "arguments": json.dumps({"spec": spec, "input": user_input}),
            },
        }],
    )


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


# ── AgentSpec validation ────────────────────────────────────────────────


def test_agent_spec_rejects_unknown_fields() -> None:
    with pytest.raises(AgentSpecError, match="unknown AgentSpec field"):
        AgentSpec.from_dict({"name": "x", "system_prompt": "y", "evil": True})


def test_agent_spec_rejects_empty_name() -> None:
    with pytest.raises(AgentSpecError):
        AgentSpec(name="", system_prompt="ok")


def test_agent_spec_rejects_bad_lifecycle() -> None:
    with pytest.raises(AgentSpecError, match="unknown lifecycle"):
        AgentSpec(name="n", system_prompt="p", lifecycle="forever")


def test_agent_spec_rejects_out_of_range_rounds() -> None:
    with pytest.raises(AgentSpecError):
        AgentSpec(name="n", system_prompt="p", max_rounds=0)
    with pytest.raises(AgentSpecError):
        AgentSpec(name="n", system_prompt="p", max_rounds=999)


def test_agent_spec_accepts_minimal() -> None:
    s = AgentSpec(name="n", system_prompt="p")
    assert s.tools is None
    assert s.lifecycle == SubagentLifecycle.EPHEMERAL.value


def test_agent_spec_from_json_string() -> None:
    s = AgentSpec.from_json('{"name":"n","system_prompt":"p","max_rounds":3}')
    assert s.max_rounds == 3


# ── filtered_registry ──────────────────────────────────────────────────


def test_filtered_registry_none_returns_none() -> None:
    assert filtered_registry(None, ["x"]) is None


def test_filtered_registry_no_whitelist_returns_same() -> None:
    reg = ToolRegistry()
    assert filtered_registry(reg, None) is reg


def test_filtered_registry_whitelist_subset() -> None:
    parent = ToolRegistry()
    for name in ("a", "b", "c"):
        parent.register(
            ToolDefinition(name=name, description=name, input_schema={"type": "object", "properties": {}}),
            lambda **_kw: name,
        )
    sub = filtered_registry(parent, ["a", "c", "missing"])
    assert sub is not None
    assert sub.has("a") and sub.has("c") and not sub.has("b")
    assert not sub.has("missing")


# ── spawn_agent / run_agent end-to-end ─────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_agent_creates_child_session_and_returns_text(store: SessionStore) -> None:
    """Parent LLM calls spawn_agent → child runs once → parent emits final."""
    parent_llm = _Scripted(responses=[
        _spawn_call("tc1", "summarize topic X"),
        LLMResponse(raw_text="parent done"),
    ])
    child_llm_responses = [LLMResponse(raw_text="child summary")]

    # We share one fake LLM but advance scripted; combine queues by injecting
    # extra responses after the spawn round. Parent sees 2 calls; child sees 1.
    parent_llm.responses.insert(1, child_llm_responses[0])
    # Reorder: 1st = spawn_call (parent), 2nd = child summary, 3rd = parent done
    parent_llm.responses = [
        _spawn_call("tc1", "summarize topic X"),
        LLMResponse(raw_text="child summary"),
        LLMResponse(raw_text="parent done"),
    ]

    registry = ToolRegistry()
    register_spawn_agent(registry)

    loop = StatefulAgentLoop(
        llm=parent_llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=4),
    )
    r = await loop.send("kick off")
    assert r.status == "completed"
    assert r.final_text == "parent done"

    # Child session should be EPHEMERAL-deleted on success.
    children = store.list_children(r.session_id)
    assert children == []


@pytest.mark.asyncio
async def test_run_agent_meta_tool_uses_strict_spec(store: SessionStore) -> None:
    spec = {
        "name": "summarizer",
        "system_prompt": "You summarize.",
        "max_rounds": 2,
        "max_tokens": 1000,
        "temperature": 0.0,
    }
    parent_llm = _Scripted(responses=[
        _run_agent_call("tc2", spec, "summarize this"),
        LLMResponse(raw_text="child output: ok"),  # child round 1
        LLMResponse(raw_text="parent wraps up"),    # parent round 2
    ])
    registry = ToolRegistry()
    register_spawn_agent(registry)

    loop = StatefulAgentLoop(
        llm=parent_llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=3),
    )
    r = await loop.send("go")
    assert r.status == "completed"
    assert r.final_text == "parent wraps up"


@pytest.mark.asyncio
async def test_run_agent_with_invalid_spec_returns_error(store: SessionStore) -> None:
    parent_llm = _Scripted(responses=[
        _run_agent_call("tc3", {"name": "x"}, "go"),  # missing system_prompt → spec error
        LLMResponse(raw_text="parent recovered"),
    ])
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(
        llm=parent_llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=2),
    )
    r = await loop.send("go")
    # The tool returns an error string; parent loop continues; final = "parent recovered".
    assert r.final_text == "parent recovered"

    # No child sessions should have been created.
    assert store.list_children(r.session_id) == []


@pytest.mark.asyncio
async def test_subagent_parent_link_recorded_when_lifecycle_linked(store: SessionStore) -> None:
    spec = AgentSpec(
        name="kept", system_prompt="P",
        lifecycle=SubagentLifecycle.LINKED.value,
    )
    parent_llm = _Scripted(responses=[
        LLMResponse(raw_text="parent immediate"),  # parent finishes round 1 with no tool
    ])
    parent_loop = StatefulAgentLoop(
        llm=parent_llm, store=store,
        config=AgentLoopConfig(max_rounds=1),
    )
    # Drive the parent once so a parent session exists in contextvars-scoped runs.
    pr = await parent_loop.send("hi")
    parent_sid = pr.session_id

    # Now call run_agent_spec directly with a fresh child LLM scripted.
    parent_loop.llm = _Scripted(responses=[LLMResponse(raw_text="child output")])
    # Re-establish the contextvar by entering parent_loop._run_loop semantics:
    from power_loop.core.agent_context import set_current_loop, set_session_id
    tok_loop = set_current_loop(parent_loop)
    tok_sid = set_session_id(parent_sid)
    try:
        child_result = await run_agent_spec(spec, "do work", parent_loop=parent_loop)
    finally:
        from power_loop.core.agent_context import reset_current_loop, reset_session_id
        reset_current_loop(tok_loop)
        reset_session_id(tok_sid)

    assert child_result["status"] == "completed"
    assert child_result["session_id"] is not None  # preserved (LINKED)
    children = store.list_children(parent_sid)
    assert len(children) == 1
    assert children[0].parent_session_id == parent_sid
    assert children[0].lifecycle is SubagentLifecycle.LINKED


@pytest.mark.asyncio
async def test_subagent_failure_preserves_session_even_when_ephemeral(store: SessionStore) -> None:
    """Hit_round_limit child must NOT be deleted (EPHEMERAL only deletes on completed)."""
    spec = AgentSpec(name="hopeless", system_prompt="P", max_rounds=1)

    parent_llm = _Scripted(responses=[LLMResponse(raw_text="parent done")])
    parent_loop = StatefulAgentLoop(
        llm=parent_llm, store=store, config=AgentLoopConfig(max_rounds=1),
    )
    pr = await parent_loop.send("hi")
    parent_sid = pr.session_id

    # Replace LLM so child always emits tool_calls → never completes → hit_round_limit.
    import json
    def _toolish() -> LLMResponse:
        return LLMResponse(
            raw_text="",
            tool_calls=[{
                "id": "tcz", "type": "function",
                "function": {"name": "spawn_agent", "arguments": json.dumps({"task": "loop"})},
            }],
        )
    parent_loop.llm = _Scripted(responses=[_toolish(), LLMResponse(raw_text="after limit")])
    # Child has no tool_registry → spawn_agent inside child returns an error string
    # but the child loop completes that round; we instead want hit_round_limit.
    # Simpler: spec.max_rounds=1, child LLM emits a tool_call → child hits limit.
    parent_loop.tool_registry = None

    from power_loop.core.agent_context import (
        reset_current_loop,
        reset_session_id,
        set_current_loop,
        set_session_id,
    )
    tok_loop = set_current_loop(parent_loop)
    tok_sid = set_session_id(parent_sid)
    try:
        result = await run_agent_spec(spec, "go", parent_loop=parent_loop)
    finally:
        reset_current_loop(tok_loop)
        reset_session_id(tok_sid)

    assert result["status"] != "completed"
    # Preserved for debugging despite EPHEMERAL lifecycle.
    assert result["session_id"] is not None
    assert store.get_session(result["session_id"]) is not None


@pytest.mark.asyncio
async def test_spawn_depth_rejection(store: SessionStore) -> None:
    """When depth would exceed MAX_SPAWN_DEPTH, run_agent_spec returns rejected
    instead of creating a child session."""
    # Pre-create a parent chain at depth MAX_SPAWN_DEPTH so the next spawn fails.
    sid = store.create_session()
    cur = sid
    for _ in range(MAX_SPAWN_DEPTH):
        cur = store.create_session(parent_session_id=cur)
    deep_parent_sid = cur  # at depth = MAX_SPAWN_DEPTH

    parent_loop = StatefulAgentLoop(
        llm=_Scripted(), store=store, config=AgentLoopConfig(max_rounds=1),
    )

    from power_loop.core.agent_context import (
        reset_current_loop,
        reset_session_id,
        set_current_loop,
        set_session_id,
    )
    tok_loop = set_current_loop(parent_loop)
    tok_sid = set_session_id(deep_parent_sid)
    try:
        result = await run_agent_spec(
            AgentSpec(name="n", system_prompt="p"),
            "go", parent_loop=parent_loop,
        )
    finally:
        reset_current_loop(tok_loop)
        reset_session_id(tok_sid)

    assert result["status"] == "rejected"
    assert result["session_id"] is None
    # No new sessions created beyond the chain we built.
    assert store.list_children(deep_parent_sid) == []


def test_spawn_agent_outside_loop_returns_clear_error() -> None:
    import asyncio

    from power_loop.tools.spawn_agent import _handle_spawn_agent
    out = asyncio.run(_handle_spawn_agent(task="x"))
    assert "must be invoked from inside" in out
