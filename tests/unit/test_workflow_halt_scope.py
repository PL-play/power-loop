"""C6: ``on_error="halt"`` must be scoped to its OWN siblings.

A nested halt used to flip the run-wide cancel token and set ``self._cancelled``,
which tore down unrelated in-flight branches of an enclosing ``on_error="continue"``
node and forced the whole run to ``"cancelled"``. Halt must cancel only the tasks
of its own ``_gather_branches`` call; the raised exception propagates normally so a
containing ``continue`` collects it and its other branches keep running.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.runtime.cancellation import CancellationToken
from power_loop.workflow import WorkflowSpec
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
        llm=_FakeLLM(), db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=3, compactor=None),
    )


@dataclass
class _PollingExec:
    """``boom`` raises immediately; slow leaves cooperatively poll their token
    (like a realistic sub-agent) and report completed unless cancelled."""

    started: list = None  # type: ignore[assignment]
    finished: list = None  # type: ignore[assignment]
    token_cancelled: list = None  # type: ignore[assignment]

    def __post_init__(self):
        self.started, self.finished, self.token_cancelled = [], [], []

    async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
        key = (user_input or "").strip() or spec.name
        self.started.append(key)
        if key == "boom":
            raise RuntimeError("branch failed")
        for _ in range(100):
            if stop_event is not None and stop_event.is_cancelled():
                self.token_cancelled.append(key)
                return {"status": "cancelled", "final_text": "", "session_id": None, "usage": {}}
            await asyncio.sleep(0.005)
        self.finished.append(key)
        return {"status": "completed", "final_text": "ok", "session_id": None, "usage": {}}


async def test_inner_halt_does_not_cancel_outer_continue_branch() -> None:
    ex = _PollingExec()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "parallel", "on_error": "continue", "branches": [
            {"type": "agent", "id": "outer_slow", "input": "outer_slow",
             "spec": {"name": "outer_slow", "system_prompt": "p"}},
            {"type": "parallel", "on_error": "halt", "branches": [
                {"type": "agent", "id": "boom", "input": "boom",
                 "spec": {"name": "boom", "system_prompt": "p"}},
                {"type": "agent", "id": "inner_slow", "input": "inner_slow",
                 "spec": {"name": "inner_slow", "system_prompt": "p"}},
            ]},
        ]}})
    # A REAL owned token: the old code's run-wide flip WOULD have cancelled outer_slow.
    eng = WorkflowEngine(_loop(), executor=ex, stop_event=CancellationToken())

    res = await eng.run(spec)

    # The enclosing continue completes; the inner halt is collected as a branch error.
    assert res.status == "completed"
    assert any("branch error" in e or "failed" in e for e in res.errors)
    # The unrelated outer branch ran to completion (NOT cancelled by the inner halt).
    assert "outer_slow" in ex.finished
    assert ex.token_cancelled == []  # nobody observed a flipped run-wide token


async def test_top_level_halt_reports_failed_not_cancelled() -> None:
    """A halt that raises a WorkflowRunError to the top settles as 'failed', not the
    spurious 'cancelled' the run-wide flag used to force."""
    spec = WorkflowSpec.from_json({"name": "w", "input": "x", "root": {"type": "sequence", "steps": [
        # branch on a payload key the planner never produces → WorkflowRunError under halt
        {"type": "agent", "id": "plan", "spec": {"name": "p", "system_prompt": "p"}},
        {"type": "parallel", "on_error": "halt", "branches": [
            {"type": "branch", "on": "plan.missing",
             "cases": {"y": {"type": "agent", "id": "z", "spec": {"name": "z", "system_prompt": "p"}}}},
        ]},
    ]}})
    res = await WorkflowEngine(_loop(), executor=_PollingExec()).run(spec)
    assert res.status == "failed"
    assert res.status != "cancelled"
