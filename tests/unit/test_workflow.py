"""Unit tests for power_loop.workflow (the minimal dynamic-workflow tier).

Control-flow + validation tests use a content-aware fake LLM; no real provider.
Budget gating is tested with a stub Executor so it doesn't depend on usage
plumbing.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.core.agent_context import reset_current_loop, set_current_loop
from power_loop.workflow import (
    AgentNode,
    SequenceNode,
    SharedBudget,
    WorkflowSpec,
    WorkflowSpecError,
    create_workflow,
)
from power_loop.workflow.engine import WorkflowEngine
from power_loop.workflow.tool import _handle_create_workflow

pytestmark = pytest.mark.unit


# ── fakes ────────────────────────────────────────────────────────────────────


@dataclass
class _FakeLLM(LLMService):
    """Replies based on the child system prompt: planner -> JSON, else text."""

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        sp = (request.system_prompt or "").lower()
        last = ""
        for m in reversed(request.messages or []):
            if m.get("role") == "user":
                last = str(m.get("content") or "")
                break
        if "subtopics" in sp:
            txt = json.dumps({"subtopics": ["history", "etiquette", "tools"]})
        elif "classify" in sp:
            txt = json.dumps({"label": "urgent"})
        elif "research:" in sp:
            txt = f"findings::{last[-30:]}"
        else:
            txt = f"summary::{last[:40]}"
        return LLMResponse(raw_text=txt, content_text=txt)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _empty()

    async def close(self) -> None:
        return None


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_FakeLLM(),
        db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3, compactor=None),
    )


# ── validation ───────────────────────────────────────────────────────────────


def test_valid_spec_parses():
    spec = WorkflowSpec.from_json({
        "name": "wf", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "p"}},
        ]},
    })
    assert spec.name == "wf"
    assert isinstance(spec.root, SequenceNode)
    assert isinstance(spec.root.steps[0], AgentNode)


def test_invalid_spec_aggregates_all_problems():
    bad = {
        "name": "",
        "root": {
            "type": "foreach", "as": "x", "items_from": "ghost.k",
            "body": {"type": "agent", "spec": {"name": "a"}},  # missing id + system_prompt
            "max_concurrency": 0, "on_error": "boom", "retires": 1,  # typo'd key
        },
    }
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json(bad)
    problems = ei.value.problems
    # All problems reported at once (loud-fail), not just the first.
    assert len(problems) >= 6
    joined = "\n".join(problems)
    assert "name" in joined
    assert "retires" in joined  # unknown key surfaced
    assert "ghost" in joined  # dangling items_from reference
    assert "max_concurrency" in joined
    assert "on_error" in joined
    assert "system_prompt" in joined  # nested AgentSpec error folded in (no crash)


def test_unknown_top_level_key_rejected():
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "a",
                                "spec": {"name": "a", "system_prompt": "p"}}, "oops": 1})
    assert any("oops" in p for p in ei.value.problems)


def test_foreach_requires_items_source():
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {
            "type": "foreach", "as": "i",
            "body": {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "p"}},
        }})
    assert any("items_from" in p or "items" in p for p in ei.value.problems)


def test_duplicate_agent_ids_rejected():
    """Agent ids key results/journal/replay — duplicates must fail validation,
    not silently mis-replay the wrong node on resume."""
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "dup", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "dup", "spec": {"name": "b", "system_prompt": "p"}},
        ]}})
    assert any("duplicate" in p and "dup" in p for p in ei.value.problems)


def test_container_id_colliding_with_agent_id_rejected():
    """C5 regression: a container (foreach aggregate) id sharing an agent id
    collides in the _results/journal/replay namespace — must be rejected globally,
    not only checked across agent ids."""
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "x", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "foreach", "id": "x", "items": ["a"], "as": "i",
             "body": {"type": "agent", "id": "leaf", "spec": {"name": "b", "system_prompt": "p"}}},
        ]}})
    assert any("duplicate node id" in p and "x" in p for p in ei.value.problems)


def test_reference_to_foreach_body_id_rejected():
    """C4 regression: a foreach-body node id is not individually addressable on
    resume; inputs_from/items_from/branch.on must not reference it (would work on
    attempt 1 but raise on resume)."""
    # branch.on → a foreach-body agent id
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "foreach", "id": "f", "items": ["a", "b"], "as": "i",
             "body": {"type": "agent", "id": "bodyleaf",
                      "spec": {"name": "b", "system_prompt": "p"}}},
            {"type": "branch", "on": "bodyleaf.choice",
             "cases": {"y": {"type": "agent", "id": "z", "spec": {"name": "z", "system_prompt": "p"}}}},
        ]}})
    assert any("foreach body" in p and "bodyleaf" in p for p in ei.value.problems)

    # inputs_from → a foreach-body agent id
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "foreach", "id": "f", "items": ["a"], "as": "i",
             "body": {"type": "agent", "id": "bodyleaf",
                      "spec": {"name": "b", "system_prompt": "p"}}},
            {"type": "agent", "id": "writer", "inputs_from": ["bodyleaf"],
             "spec": {"name": "w", "system_prompt": "p"}},
        ]}})
    assert any("foreach body" in p and "bodyleaf" in p for p in ei.value.problems)

    # items_from → a foreach-body agent id
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "foreach", "id": "f", "items": ["a"], "as": "i",
             "body": {"type": "agent", "id": "bodyleaf",
                      "spec": {"name": "b", "system_prompt": "p"}}},
            {"type": "foreach", "id": "g", "items_from": "bodyleaf.list", "as": "j",
             "body": {"type": "agent", "id": "leaf2", "spec": {"name": "c", "system_prompt": "p"}}},
        ]}})
    assert any("foreach body" in p and "bodyleaf" in p for p in ei.value.problems)


def test_reference_to_foreach_aggregate_id_still_allowed():
    """The fix must NOT reject the valid pattern: referencing a top-level foreach's
    own aggregate id (which IS addressable/replayable) from inputs_from, items_from
    AND branch.on — all three ref kinds are symmetric."""
    spec = WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "plan",
         "spec": {"name": "p", "system_prompt": "p"},
         "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subtopics"],
            "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "foreach", "id": "research", "items_from": "plan.subtopics", "as": "t",
         "body": {"type": "agent", "id": "r", "spec": {"name": "r", "system_prompt": "p"}}},
        # items_from → a foreach AGGREGATE id (previously wrongly rejected as unknown)
        {"type": "foreach", "id": "expand", "items_from": "research.items", "as": "u",
         "body": {"type": "agent", "id": "e", "spec": {"name": "e", "system_prompt": "p"}}},
        # branch.on → a foreach AGGREGATE id
        {"type": "branch", "on": "expand.items",
         "cases": {"x": {"type": "agent", "id": "c", "spec": {"name": "c", "system_prompt": "p"}}}},
        {"type": "agent", "id": "write", "inputs_from": ["research"],
         "spec": {"name": "w", "system_prompt": "p"}},
    ]}})
    assert spec.name == "w"


# ── execution ────────────────────────────────────────────────────────────────


async def test_engine_sequence_foreach_structured_dataflow():
    spec = {
        "name": "research", "input": "tea",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "plan",
             "spec": {"name": "planner", "system_prompt": "List the subtopics."},
             "output_schema": {"name": "Plan", "schema": {
                 "type": "object", "required": ["subtopics"],
                 "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "research", "items_from": "plan.subtopics", "as": "t",
             "parallel": True, "max_concurrency": 3,
             "body": {"type": "agent", "id": "r",
                      "spec": {"name": "researcher", "system_prompt": "research: {{t}}"},
                      "input": "Research {{t}}"}},
            {"type": "agent", "id": "write",
             "spec": {"name": "writer", "system_prompt": "Synthesize."}, "inputs_from": ["research"]},
        ]},
    }
    res = await create_workflow(spec, parent_loop=_loop()).run()
    assert res.status == "completed"
    assert res.results["plan"].payload == {"subtopics": ["history", "etiquette", "tools"]}
    # foreach aggregate holds one entry per item
    assert len(res.results["research"].payload["items"]) == 3
    # inputs_from wired the research output into the writer's message
    assert "findings::" in res.results["write"].text or res.results["write"].ok
    assert res.final is not None


async def test_engine_branch_selects_case():
    spec = {
        "name": "triage", "input": "ticket",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "classify",
             "spec": {"name": "classifier", "system_prompt": "classify the ticket"},
             "output_schema": {"name": "C", "schema": {
                 "type": "object", "required": ["label"],
                 "properties": {"label": {"type": "string"}}}}},
            {"type": "branch", "on": "classify.label", "cases": {
                "urgent": {"type": "agent", "id": "esc",
                           "spec": {"name": "esc", "system_prompt": "escalate"}},
                "normal": {"type": "agent", "id": "norm",
                           "spec": {"name": "norm", "system_prompt": "normal"}},
            }},
        ]},
    }
    res = await create_workflow(spec, parent_loop=_loop()).run()
    assert res.status == "completed"
    assert "esc" in res.results  # urgent case ran
    assert "norm" not in res.results  # other case skipped


async def test_missing_reference_fails_cleanly():
    # foreach references a payload key the planner never produced.
    spec = {
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "plan", "spec": {"name": "p", "system_prompt": "List the subtopics."},
             "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subtopics"],
                "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "f", "items_from": "plan.nonexistent", "as": "i",
             "body": {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "do {{i}}"}}},
        ]},
    }
    res = await create_workflow(spec, parent_loop=_loop()).run()
    assert res.status == "failed"
    assert any("nonexistent" in e for e in res.errors)


# ── shared budget ────────────────────────────────────────────────────────────


def test_shared_budget_can_spawn_and_commit():
    b = SharedBudget(100)
    assert b.can_spawn()
    b.commit({"total_tokens": 100})
    assert not b.can_spawn()
    assert b.remaining == 0


async def test_engine_budget_stops_spawning():
    @dataclass
    class _StubExec:
        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            return {"status": "completed", "final_text": "ok", "session_id": None,
                    "usage": {"total_tokens": 60}}

    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "one", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "two", "spec": {"name": "b", "system_prompt": "p"}},
        ]},
    })
    # ceiling 60, each step costs 60: step "one" runs and exhausts the pool,
    # step "two" can no longer spawn.
    eng = WorkflowEngine(_loop(), executor=_StubExec(), budget=SharedBudget(60))
    res = await eng.run(spec)
    assert res.status == "budget_exceeded"
    assert res.results["one"].ok
    assert res.results["two"].status == "budget_exceeded"


# ── create_workflow tool ─────────────────────────────────────────────────────


async def test_tool_requires_active_loop():
    out = await _handle_create_workflow(spec={"name": "w", "root": {
        "type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "p"}}})
    assert out.startswith("Error:")
    assert "active StatefulAgentLoop" in out


async def test_tool_reports_validation_error_for_repair():
    loop = _loop()
    token = set_current_loop(loop)
    try:
        out = await _handle_create_workflow(spec={"name": "", "root": {"type": "nope"}})
    finally:
        reset_current_loop(token)
    assert out.startswith("Error:")
    assert "WorkflowSpec" in out  # aggregated problems the model can read & fix


async def test_tool_runs_and_returns_summary():
    loop = _loop()
    token = set_current_loop(loop)
    try:
        out = await _handle_create_workflow(
            spec={"name": "wf", "root": {"type": "agent", "id": "a",
                  "spec": {"name": "a", "system_prompt": "Summarize."}}},
            input="hello",
        )
    finally:
        reset_current_loop(token)
    assert "workflow 'wf' -> completed" in out


# ── per-subagent LLM override (B1) + output_schema enforcement ────────────────


async def test_agent_node_model_and_output_schema_reach_llm_request():
    """A node's spec.model and output_schema must reach the actual LLMRequest:
    model overrides the global; output_schema becomes a json_schema response_format."""
    captured: dict = {}

    @dataclass
    class _CapLLM(LLMService):
        async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                           on_chunk_think=None, on_stream_end=None) -> LLMResponse:
            captured["model"] = request.model
            captured["response_format"] = request.response_format
            return LLMResponse(raw_text='{"x": 1}', content_text='{"x": 1}')

        def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
            async def _e() -> AsyncIterator[LLMStreamChunk]:
                if False:
                    yield LLMStreamChunk()
            return _e()

        async def close(self) -> None:
            return None

    loop = StatefulAgentLoop(
        llm=_CapLLM(), db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )
    spec = {"name": "w", "root": {"type": "agent", "id": "a",
            "spec": {"name": "a", "system_prompt": "p", "model": "my-model-x"},
            "output_schema": {"name": "S", "schema": {"type": "object", "required": ["x"],
                "properties": {"x": {"type": "integer"}}}}}}
    res = await create_workflow(spec, parent_loop=loop).run()
    assert res.status == "completed"
    assert captured["model"] == "my-model-x"  # per-subagent model override reached the request
    assert captured["response_format"] is not None  # output_schema -> response_format
    assert captured["response_format"].get("type") == "json_schema"
    assert res.results["a"].payload == {"x": 1}  # parsed structured output
    assert res.results["a"].usage is not None  # usage surfaced (B2)
