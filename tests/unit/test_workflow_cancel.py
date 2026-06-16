"""Cancellation tests for the workflow tier (stop_event forwarding).

Covers the end-to-end path opened by forwarding ``stop_event`` through
``run_agent_spec``:

* ``run_agent_spec`` honors a cancellation token (the leaf returns "cancelled");
* the engine refuses to spawn further leaves once cancelled and settles the run
  as "cancelled";
* ``Workflow.cancel()`` flips the same token before/around a run.

Control flow only — uses a content-agnostic fake LLM and stub executors, no
real provider.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    CancellationToken,
    StatefulAgentLoop,
    run_agent_spec,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.workflow import SharedBudget, WorkflowSpec, create_workflow
from power_loop.workflow.engine import WorkflowEngine

pytestmark = pytest.mark.unit


@dataclass
class _FakeLLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text="ok", content_text="ok")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_FakeLLM(),
        db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=3, compactor=None),
    )


# ── run_agent_spec forwards the token ─────────────────────────────────────────


async def test_run_agent_spec_honors_pre_cancelled_token():
    """A pre-cancelled stop_event reaches the child loop, which stops at round 0."""
    loop = _loop()
    sid = await loop.new_session()
    token = CancellationToken()
    token.cancel("stop now")

    set_token = set_session_id(sid)
    try:
        out = await run_agent_spec(
            AgentSpec(name="child", system_prompt="p"),
            "do the thing",
            parent_loop=loop,
            stop_event=token,
        )
    finally:
        reset_session_id(set_token)

    assert out["status"] == "cancelled"


# ── engine-level cancellation ─────────────────────────────────────────────────


async def test_engine_pre_cancelled_skips_every_step():
    token = CancellationToken()
    token.cancel()
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "one", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "two", "spec": {"name": "b", "system_prompt": "p"}},
        ]},
    })
    eng = WorkflowEngine(_loop(), stop_event=token)
    res = await eng.run(spec)
    assert res.status == "cancelled"
    assert res.results["one"].status == "cancelled"
    # the sequence short-circuits on cancellation: step two is never reached
    assert "two" not in res.results


async def test_engine_cancel_midway_stops_spawning():
    """First leaf flips the token mid-run; the next leaf must not spawn."""
    token = CancellationToken()
    spawned: list[str] = []

    @dataclass
    class _Exec:
        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            spawned.append(spec.name)
            token.cancel()  # simulate an external cancel arriving during step one
            return {"status": "completed", "final_text": "ok", "session_id": None,
                    "usage": {"total_tokens": 1}}

    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "one", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "two", "spec": {"name": "b", "system_prompt": "p"}},
        ]},
    })
    eng = WorkflowEngine(_loop(), executor=_Exec(), stop_event=token)
    res = await eng.run(spec)
    assert res.status == "cancelled"
    assert res.results["one"].ok           # step one completed before cancel landed
    assert res.results["two"].status == "cancelled"
    assert spawned == ["a"]                 # step two never reached the executor


async def test_engine_forwards_token_into_executor():
    """The engine passes its own token down to each leaf's run_agent."""
    seen: list[bool] = []

    @dataclass
    class _Exec:
        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            seen.append(stop_event is not None)
            return {"status": "completed", "final_text": "ok", "session_id": None, "usage": {}}

    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "agent", "id": "a",
                              "spec": {"name": "a", "system_prompt": "p"}}})
    eng = WorkflowEngine(_loop(), executor=_Exec())
    res = await eng.run(spec)
    assert res.status == "completed"
    assert seen == [True]  # a token (never-cancelled) was forwarded


# ── Workflow.cancel() ─────────────────────────────────────────────────────────


async def test_workflow_cancel_before_run():
    wf = create_workflow(
        {"name": "w", "input": "x", "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "one", "spec": {"name": "a", "system_prompt": "p"}},
        ]}},
        parent_loop=_loop(),
    )
    wf.cancel("user aborted")
    res = await wf.run()
    assert res.status == "cancelled"
    assert res.results["one"].status == "cancelled"


async def test_cancel_takes_precedence_over_budget():
    """When both fire, a cancelled run reports 'cancelled', not 'budget_exceeded'."""
    token = CancellationToken()
    token.cancel()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "agent", "id": "a",
                              "spec": {"name": "a", "system_prompt": "p"}}})
    eng = WorkflowEngine(_loop(), budget=SharedBudget(0), stop_event=token)
    res = await eng.run(spec)
    assert res.status == "cancelled"
