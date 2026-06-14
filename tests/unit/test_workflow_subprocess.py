"""Phase 1b: SubprocessExecutor — leaves run in real OS processes, own DBs.

Uses the config-selectable echo provider (POWER_LOOP_PROVIDER=echo) so each
spawned worker is fast and deterministic with no network. Drives everything
through the unchanged WorkflowEngine via the Executor seam.
"""

from __future__ import annotations

import glob
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

pytestmark = pytest.mark.unit


@dataclass
class _OrchestratorLLM(LLMService):
    """Unused by leaves (they run in subprocesses); just satisfies the loop ctor."""

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


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_OrchestratorLLM(), db_path=":memory:",
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )


def _echo_executor(runs_dir: str, reply: str) -> SubprocessExecutor:
    return SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True),
        runs_dir=runs_dir,
        env={"POWER_LOOP_PROVIDER": "echo", "POWER_LOOP_ECHO_REPLY": reply},
        timeout_s=60,
    )


async def test_two_leaves_run_in_separate_processes_and_dbs(tmp_path) -> None:
    runs = str(tmp_path / "runs")
    ex = _echo_executor(runs, "sub-ok")
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "p"}},
        ]},
    })
    res = await WorkflowEngine(_loop(), executor=ex, run_id="t1").run(spec)

    assert res.status == "completed"
    assert res.results["a"].text == "sub-ok" and res.results["b"].text == "sub-ok"
    # one db file per leaf, under runs/<run_id>/
    dbs = glob.glob(os.path.join(runs, "t1", "*.db"))
    assert len(dbs) == 2
    assert any(os.path.basename(p).startswith("a__") for p in dbs)
    assert any(os.path.basename(p).startswith("b__") for p in dbs)


async def test_foreach_fanout_gets_a_unique_db_per_item(tmp_path) -> None:
    """foreach bodies share a node id — each parallel item must still get its own
    db file (no collision = no shared-write)."""
    runs = str(tmp_path / "runs")
    ex = _echo_executor(runs, "item-done")
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "foreach", "id": "fan", "items": ["p", "q", "r"], "as": "s",
                 "parallel": True, "max_concurrency": 3,
                 "body": {"type": "agent", "id": "body",
                          "spec": {"name": "body", "system_prompt": "{{s}}"}, "input": "do {{s}}"}},
    })
    res = await WorkflowEngine(_loop(), executor=ex, run_id="t2").run(spec)

    assert res.status == "completed"
    assert len(res.results["fan"].payload["items"]) == 3
    # three distinct db files, all body__*.db — no two items shared a file
    dbs = glob.glob(os.path.join(runs, "t2", "body__*.db"))
    assert len(dbs) == 3 and len(set(dbs)) == 3


async def test_worker_failure_is_recorded_as_failed_leaf(tmp_path) -> None:
    """A worker that can't build its provider returns failed (not a crash); the
    engine records the leaf as non-completed → the resume tier would re-run it."""
    runs = str(tmp_path / "runs")
    # Force a provider build failure deterministically: openai provider with all
    # required fields emptied under both prefixes (overrides any inherited .env).
    fail_env = {
        "POWER_LOOP_PROVIDER": "openai",
        "POWER_LOOP_BASE_URL": "", "POWER_LOOP_API_KEY": "", "POWER_LOOP_MODEL": "",
        "OPENAI_COMPAT_BASE_URL": "", "OPENAI_COMPAT_API_KEY": "", "OPENAI_COMPAT_MODEL": "",
    }
    ex = SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True), runs_dir=runs, env=fail_env, timeout_s=60,
    )
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "agent", "id": "only",
                              "spec": {"name": "only", "system_prompt": "p"}}})
    res = await WorkflowEngine(_loop(), executor=ex, run_id="t3").run(spec)

    leaf = res.results["only"]
    assert leaf.status == "failed"
    assert "worker error" in leaf.text.lower() or "missing required" in (leaf.error or "").lower()
