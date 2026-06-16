"""Regression tests for Tier-2 workflow fixes:

- C3: a task-level CancelledError on a subprocess leaf (host shutdown / engine task
  teardown) must KILL the worker process, not orphan it.
- C13: a run that collected branch errors under on_error="continue" settles
  status="completed" but result.ok must be False (a partial failure isn't success).
"""

from __future__ import annotations

import asyncio
import glob
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

pytestmark = pytest.mark.unit


@dataclass
class _OrchLLM(LLMService):
    async def complete(self, request: LLMRequest, **_) -> LLMResponse:
        return LLMResponse(raw_text="x", content_text="x")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_OrchLLM(), db_path=":memory:",
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )


# ── C3: task-level cancel kills the subprocess worker ────────────────────────


async def test_task_cancel_kills_subprocess_worker(tmp_path, monkeypatch) -> None:
    # Capture every worker process the executor spawns so we can assert it died.
    procs: list = []
    orig = asyncio.create_subprocess_exec

    async def _capture(*a, **k):
        p = await orig(*a, **k)
        procs.append(p)
        return p

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _capture)

    ex = SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True), runs_dir=str(tmp_path / "r"),
        env={"POWER_LOOP_PROVIDER": "echo", "POWER_LOOP_ECHO_REPLY": "ok",
             "POWER_LOOP_ECHO_SLEEP": "10"},  # worker sleeps long enough to cancel mid-flight
    )
    spec = WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "only",
                                   "spec": {"name": "only", "system_prompt": "p"}}})
    # NOTE: no stop_event — we cancel the asyncio TASK, not the cooperative token.
    task = asyncio.ensure_future(WorkflowEngine(_loop(), executor=ex, run_id="cx").run(spec))

    run_dir = tmp_path / "r" / "cx"
    for _ in range(200):
        if procs and run_dir.exists() and glob.glob(str(run_dir / "*.db")):
            break
        if task.done():
            break
        await asyncio.sleep(0.02)
    assert procs, "worker process never spawned"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # the worker must have been signalled to die (returncode set), not left running
    for _ in range(100):
        if procs[0].returncode is not None:
            break
        await asyncio.sleep(0.05)
    assert procs[0].returncode is not None, "worker orphaned after task cancellation"


# ── C13: continue-branch errors make ok False ────────────────────────────────


@dataclass
class _Exec:
    """One branch raises (a structural error collected under continue)."""

    async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
        if (user_input or "").strip() == "boom":
            raise RuntimeError("branch failed")
        return {"status": "completed", "final_text": "ok", "session_id": None, "usage": {}}


async def test_continue_branch_error_makes_ok_false() -> None:
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "parallel", "on_error": "continue", "branches": [
            {"type": "agent", "id": "good", "input": "fine",
             "spec": {"name": "good", "system_prompt": "p"}},
            {"type": "agent", "id": "bad", "input": "boom",
             "spec": {"name": "bad", "system_prompt": "p"}},
        ]}})
    res = await WorkflowEngine(_loop(), executor=_Exec()).run(spec)

    assert res.status == "completed"   # continue does not fail the run
    assert res.errors                  # but a branch error was collected
    assert res.ok is False             # so ok must NOT report success (was True)
