"""S2 (3.14): per-send tool allowlist propagates to child spawns.

``send(tools=...)`` publishes the run's effective allowlist via
``agent_context.get_effective_tools``; ``run_agent_spec`` intersects the child's
registry with it (``inherit_send_filter``); workflows capture the set at
submission and clamp every leaf at the SPEC level (executor-agnostic), journal
it, and reapply it on resume.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import (
    get_effective_tools,
    reset_effective_tools,
    set_effective_tools,
)
from power_loop.runtime.spec import run_agent_spec
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow
from power_loop.workflow.engine import WorkflowEngine
from power_loop.workflow.journal import new_journal
from power_loop.workflow.resume import _journaled_clamp
from power_loop.workflow.spec import WorkflowSpec


@dataclass
class _Capturing(LLMService):
    """Scripted LLM that records each request's advertised tool names."""

    responses: list[LLMResponse] = field(default_factory=list)
    seen_tools: list[list[str]] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.seen_tools.append(
            sorted(t["function"]["name"] for t in (request.tools or []))
        )
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

    async def close(self) -> None:
        return None


def _tool_def(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name, description=name,
        input_schema={"type": "object", "properties": {}},
    )


def _registry(*names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for n in names:
        reg.register(_tool_def(n), lambda **_kw: "ok")
    return reg


def _tool_call(call_id: str, name: str, args: dict | None = None) -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }],
    )


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


# ── contextvar published by _run_loop ────────────────────────────────────


@pytest.mark.asyncio
async def test_send_with_tools_publishes_effective_set(store: SessionStore) -> None:
    seen: list[frozenset[str] | None] = []

    def probe(**_kw: Any) -> str:
        seen.append(get_effective_tools())
        return "probed"

    reg = _registry("a", "b")
    reg.register(_tool_def("probe"), probe)
    llm = _Capturing(responses=[
        _tool_call("t1", "probe"),
        LLMResponse(raw_text="fin"),
    ])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=reg, config=AgentLoopConfig(max_rounds=3),
    )
    sid = await loop.new_session()
    r = await loop.send("go", session_id=sid, tools=["probe", "a"])
    assert r.status == "completed"
    assert seen == [frozenset({"probe", "a"})]
    # And it is reset once the run finishes.
    assert get_effective_tools() is None


@pytest.mark.asyncio
async def test_send_without_tools_publishes_none(store: SessionStore) -> None:
    seen: list[frozenset[str] | None] = []

    def probe(**_kw: Any) -> str:
        seen.append(get_effective_tools())
        return "probed"

    reg = _registry("a")
    reg.register(_tool_def("probe"), probe)
    llm = _Capturing(responses=[
        _tool_call("t1", "probe"),
        LLMResponse(raw_text="fin"),
    ])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=reg, config=AgentLoopConfig(max_rounds=3),
    )
    sid = await loop.new_session()
    await loop.send("go", session_id=sid)
    assert seen == [None]


# ── run_agent_spec inherits the filter ───────────────────────────────────


@pytest.mark.asyncio
async def test_child_inherit_mode_clamped_by_send_filter(store: SessionStore) -> None:
    """spec.tools=None (inherit all) under an active filter → child sees only
    the filtered names."""
    llm = _Capturing(responses=[LLMResponse(raw_text="child done")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry("a", "b", "c"),
        config=AgentLoopConfig(max_rounds=2),
    )
    token = set_effective_tools(frozenset({"a", "c"}))
    try:
        res = await run_agent_spec(
            AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
        )
    finally:
        reset_effective_tools(token)
    assert res["status"] == "completed"
    assert llm.seen_tools == [["a", "c"]]


@pytest.mark.asyncio
async def test_child_specify_mode_intersected(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text="child done")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry("a", "b", "c"),
        config=AgentLoopConfig(max_rounds=2),
    )
    token = set_effective_tools(frozenset({"a", "b"}))
    try:
        await run_agent_spec(
            AgentSpec(name="kid", system_prompt="p", tools=["b", "c"]),
            "task", parent_loop=loop,
        )
    finally:
        reset_effective_tools(token)
    assert llm.seen_tools == [["b"]]


@pytest.mark.asyncio
async def test_inherit_send_filter_false_escape_hatch(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text="child done")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry("a", "b"),
        config=AgentLoopConfig(max_rounds=2),
    )
    token = set_effective_tools(frozenset({"a"}))
    try:
        await run_agent_spec(
            AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
            inherit_send_filter=False,
        )
    finally:
        reset_effective_tools(token)
    assert llm.seen_tools == [["a", "b"]]


@pytest.mark.asyncio
async def test_nested_child_run_resets_filter(store: SessionStore) -> None:
    """A child run WITHOUT tools= resets the innermost filter to None: a
    grandchild is clamped only by the child's (already-clamped) registry, not
    by re-application of the outer names."""
    llm = _Capturing(responses=[
        _tool_call("t1", "spawnling"),        # child round 1: call the spawning tool
        LLMResponse(raw_text="grandchild"),   # grandchild run
        LLMResponse(raw_text="child done"),   # child round 2
    ])
    inner_seen: list[frozenset[str] | None] = []

    async def spawnling(**_kw: Any) -> str:
        # Runs inside the CHILD's run → the innermost filter must be None.
        inner_seen.append(get_effective_tools())
        from power_loop.core.agent_context import get_current_loop

        res = await run_agent_spec(
            AgentSpec(name="grandkid", system_prompt="p"),
            "task", parent_loop=get_current_loop(),
        )
        return str(res["status"])

    reg = _registry("a", "b")
    reg.register(_tool_def("spawnling"), spawnling)
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=reg, config=AgentLoopConfig(max_rounds=3),
    )
    token = set_effective_tools(frozenset({"spawnling", "a"}))
    try:
        await run_agent_spec(
            AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
        )
    finally:
        reset_effective_tools(token)
    assert inner_seen == [None]
    # Child advertised the clamped set; grandchild inherited the child's full
    # (clamped) registry — not the raw parent registry.
    assert llm.seen_tools[0] == ["a", "spawnling"]
    assert llm.seen_tools[1] == ["a", "spawnling"]


# ── workflow: capture at submission + leaf clamp + journal round-trip ────


def _leaf_spec(tools: list[str] | None = None) -> dict[str, Any]:
    node_spec: dict[str, Any] = {"name": "leaf", "system_prompt": "p"}
    if tools is not None:
        node_spec["tools"] = tools
    return {
        "name": "wf",
        "root": {"type": "agent", "id": "n1", "spec": node_spec},
    }


@pytest.mark.asyncio
async def test_workflow_captures_send_filter_at_submission(store: SessionStore) -> None:
    llm = _Capturing(responses=[])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry("a", "b"),
        config=AgentLoopConfig(max_rounds=2),
    )
    token = set_effective_tools(frozenset({"a"}))
    try:
        wf = create_workflow(_leaf_spec(), parent_loop=loop)
    finally:
        reset_effective_tools(token)
    assert wf._allowed_tools == frozenset({"a"})
    # Explicit override beats capture; None = unrestricted.
    wf2 = create_workflow(_leaf_spec(), parent_loop=loop, allowed_tools=["b"])
    assert wf2._allowed_tools == frozenset({"b"})
    wf3 = create_workflow(_leaf_spec(), parent_loop=loop, allowed_tools=None)
    assert wf3._allowed_tools is None


@pytest.mark.asyncio
async def test_workflow_engine_clamps_leaf_tools(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text="leaf out")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry("a", "b", "c"),
        config=AgentLoopConfig(max_rounds=2),
    )
    engine = WorkflowEngine(loop, allowed_tools=frozenset({"a", "b"}))
    result = await engine.run(WorkflowSpec.from_json(_leaf_spec()))
    assert result.status == "completed"
    assert llm.seen_tools == [["a", "b"]]  # inherit → clamp set becomes the whitelist


@pytest.mark.asyncio
async def test_workflow_engine_intersects_explicit_leaf_tools(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text="leaf out")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry("a", "b", "c"),
        config=AgentLoopConfig(max_rounds=2),
    )
    engine = WorkflowEngine(loop, allowed_tools=frozenset({"b", "c"}))
    result = await engine.run(WorkflowSpec.from_json(_leaf_spec(tools=["a", "b"])))
    assert result.status == "completed"
    assert llm.seen_tools == [["b"]]


def test_journal_persists_and_rehydrates_clamp() -> None:
    j = new_journal("r1", "wf", allowed_tools=["a", "b"])
    assert j["allowed_tools"] == ["a", "b"]
    assert _journaled_clamp(j) == frozenset({"a", "b"})
    # null and pre-3.14 (absent key) → unrestricted.
    assert _journaled_clamp(new_journal("r2", "wf")) is None
    legacy = new_journal("r3", "wf")
    legacy.pop("allowed_tools")
    assert _journaled_clamp(legacy) is None
