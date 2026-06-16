"""Phase 1c: SubprocessExecutor cancel / timeout / crash + resume integration.

Uses the echo provider's artificial latency (POWER_LOOP_ECHO_SLEEP) to make
cancel/timeout deterministic, and the env-gated worker fault injector
(POWER_LOOP_TEST_CRASH_NODES) to hard-crash a chosen leaf. The headline test
proves the subprocess executor composes with the resume tier: a crashed leaf is
re-run in a fresh process while the completed leaf is replayed, not re-run.
"""

from __future__ import annotations

import asyncio
import glob
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.runtime.cancellation import CancellationToken
from power_loop.workflow import (
    SubprocessExecutor,
    WorkerBootstrap,
    WorkflowSpec,
    resume_run,
)
from power_loop.workflow import journal as journal_mod
from power_loop.workflow.engine import WorkflowEngine
from power_loop.workflow.runner import make_on_step

pytestmark = pytest.mark.unit


@dataclass
class _OrchLLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text="x", content_text="x")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


async def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_OrchLLM(), store=await SessionStore.open(":memory:"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )


def _executor(runs_dir: str, *, reply="ok", sleep=None, crash_nodes=None, timeout_s=None):
    env = {"POWER_LOOP_PROVIDER": "echo", "POWER_LOOP_ECHO_REPLY": reply}
    if sleep is not None:
        env["POWER_LOOP_ECHO_SLEEP"] = str(sleep)
    if crash_nodes is not None:
        env["POWER_LOOP_TEST_CRASH_NODES"] = crash_nodes
    return SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True), runs_dir=runs_dir, env=env, timeout_s=timeout_s,
    )


_ONE_LEAF = {"name": "w", "root": {"type": "agent", "id": "only",
             "spec": {"name": "only", "system_prompt": "p"}}}


async def test_timeout_kills_worker_and_reports_failed(tmp_path) -> None:
    ex = _executor(str(tmp_path / "r"), sleep=10, timeout_s=0.5)  # worker sleeps >> timeout
    res = await WorkflowEngine(await _loop(), executor=ex, run_id="to").run(
        WorkflowSpec.from_json(_ONE_LEAF))
    leaf = res.results["only"]
    assert leaf.status == "failed"
    assert "timeout" in (leaf.error or "").lower()


async def test_cancel_midflight_terminates_worker(tmp_path) -> None:
    ex = _executor(str(tmp_path / "r"), sleep=10)  # long enough to cancel during
    token = CancellationToken()
    eng = WorkflowEngine(await _loop(), executor=ex, run_id="cx", stop_event=token)
    task = asyncio.ensure_future(eng.run(WorkflowSpec.from_json(_ONE_LEAF)))

    # Wait until the worker has actually started (created its db), then cancel.
    run_dir = tmp_path / "r" / "cx"
    for _ in range(100):
        if run_dir.exists() and glob.glob(str(run_dir / "*.db")):
            break
        if task.done():
            break
        await asyncio.sleep(0.05)
    token.cancel()
    res = await task

    assert res.status == "cancelled"
    assert res.results["only"].status == "cancelled"


async def test_hard_crash_reported_as_failed(tmp_path) -> None:
    ex = _executor(str(tmp_path / "r"), crash_nodes="only")  # worker os._exit's, no frame
    res = await WorkflowEngine(await _loop(), executor=ex, run_id="cr").run(
        WorkflowSpec.from_json(_ONE_LEAF))
    leaf = res.results["only"]
    assert leaf.status == "failed"
    assert "no result frame" in (leaf.error or "").lower()


async def test_subprocess_crash_then_resume_reruns_only_failed_leaf(tmp_path) -> None:
    """Headline: a→b in subprocesses; b's worker hard-crashes on attempt 1. On
    resume, a is REPLAYED (keeps its attempt-1 text) and only b re-runs (fresh
    process, new text)."""
    loop = await _loop()
    sid = await loop.new_session()
    run_id = "fr"
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "p"},
             "inputs_from": ["a"]},
        ]},
    })

    # Attempt 1: reply R1; crash leaf b. a completes (R1) and is journaled; b fails.
    ex1 = _executor(str(tmp_path / "r"), reply="R1", crash_nodes="b")
    await journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    eng1 = WorkflowEngine(loop, executor=ex1, on_step=make_on_step(loop.store, sid, run_id),
                          run_id=run_id)
    await eng1.run(spec)
    done = {s["node_id"]: s["status"] for s in (await journal_mod.read(loop.store, sid, run_id))["steps"]}
    assert done.get("a") == "completed" and done.get("b") != "completed"

    # Resume with a healthy executor that replies R2.
    ex2 = _executor(str(tmp_path / "r"), reply="R2")
    res = await resume_run(loop, sid, run_id, executor=ex2)

    assert res.status == "completed"
    assert res.results["a"].text == "R1"   # REPLAYED — not re-run (would be R2)
    assert res.results["b"].text == "R2"   # re-ran in a fresh process
