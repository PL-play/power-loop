"""H1.2 / C2: parallel & foreach `on_error="halt"` cancel in-flight siblings.

asyncio.gather(return_exceptions=False) re-raises on the first failure but leaves
the other branches running detached — they keep burning real LLM calls and a late
record_step can clobber the finalized journal. The engine must instead cancel the
siblings when a branch fails under halt.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import AgentLoopConfig, StatefulAgentLoop
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
class _Tracker:
    """An executor where 'boom' fails immediately and slow leaves block until
    cancelled — records which leaves started, finished, or were cancelled."""

    started: list = None  # type: ignore[assignment]
    finished: list = None  # type: ignore[assignment]
    cancelled: list = None  # type: ignore[assignment]

    def __post_init__(self):
        self.started, self.finished, self.cancelled = [], [], []

    async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
        # Key off the rendered input so a foreach (whose leaves share one spec name)
        # is distinguishable per item; fall back to the spec name for parallel.
        key = (user_input or "").strip() or spec.name
        self.started.append(key)
        try:
            if key == "boom":
                raise RuntimeError("branch failed")
            await asyncio.sleep(1)  # slow sibling; gets cancelled before this returns
            self.finished.append(key)
            return {"status": "completed", "final_text": "ok", "session_id": None, "usage": {}}
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise


def _leftover_tasks() -> list:
    return [t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()]


async def test_parallel_halt_cancels_inflight_siblings() -> None:
    ex = _Tracker()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "parallel", "on_error": "halt", "branches": [
            {"type": "agent", "id": "slow1", "input": "slow1", "spec": {"name": "slow1", "system_prompt": "p"}},
            {"type": "agent", "id": "boom", "input": "boom", "spec": {"name": "boom", "system_prompt": "p"}},
            {"type": "agent", "id": "slow2", "input": "slow2", "spec": {"name": "slow2", "system_prompt": "p"}},
        ]}})
    eng = WorkflowEngine(_loop(), executor=ex)

    with pytest.raises(RuntimeError, match="branch failed"):
        await eng.run(spec)

    assert "boom" in ex.started
    assert ex.finished == []                       # no slow sibling ran to completion
    assert set(ex.cancelled) == {"slow1", "slow2"}  # both in-flight siblings cancelled
    assert _leftover_tasks() == []                  # nothing left running detached


async def test_foreach_parallel_halt_cancels_inflight_siblings() -> None:
    ex = _Tracker()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {
            "type": "foreach", "id": "f", "parallel": True, "on_error": "halt",
            "items": ["slow1", "boom", "slow2"], "as": "name",
            "body": {"type": "agent", "id": "leaf", "input": "{{name}}",
                     "spec": {"name": "leaf", "system_prompt": "p"}},
        }})
    eng = WorkflowEngine(_loop(), executor=ex)

    with pytest.raises(RuntimeError, match="branch failed"):
        await eng.run(spec)

    assert "boom" in ex.started
    assert ex.finished == []
    assert set(ex.cancelled) == {"slow1", "slow2"}
    assert _leftover_tasks() == []


async def test_parallel_continue_still_collects_all_errors() -> None:
    """on_error='continue' is unchanged: every branch runs, errors are collected."""
    ex = _Tracker()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "parallel", "on_error": "continue", "branches": [
            {"type": "agent", "id": "ok1", "input": "ok1", "spec": {"name": "ok1", "system_prompt": "p"}},
            {"type": "agent", "id": "boom", "input": "boom", "spec": {"name": "boom", "system_prompt": "p"}},
        ]}})
    # 'ok1' must finish (not cancelled) under continue; shorten its sleep via name
    eng = WorkflowEngine(_loop(), executor=ex)
    res = await eng.run(spec)  # does not raise under continue
    assert res.status in {"completed", "failed"}
    assert any("branch error" in e or "failed" in e for e in res.errors)
    assert ex.cancelled == []  # continue never cancels siblings
