"""Phase 1b: SubprocessExecutor — leaves run in real OS processes, own DBs.

Uses the config-selectable echo provider (POWER_LOOP_PROVIDER=echo) so each
spawned worker is fast and deterministic with no network. Drives everything
through the unchanged WorkflowEngine via the Executor seam.
"""

from __future__ import annotations

import asyncio
import glob
import os
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
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


# ── H4 (BUG_REVIEW_3.4): a leaked grandchild holding the pipe must not hang the leaf forever ──

@dataclass
class _ScriptLauncher:
    """Replace the worker command with an arbitrary script (env carries the params)."""

    script_path: str

    def build(self, *, base_argv, base_env, spec, db_path, workspace_dir):
        return [sys.executable, self.script_path], base_env


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
async def test_leaked_grandchild_does_not_hang_the_leaf(tmp_path) -> None:
    pidfile = str(tmp_path / "gc.pid")
    worker_py = str(tmp_path / "worker.py")
    # Worker spawns a grandchild that records its pid and sleeps 30s while INHERITING the worker's
    # stdout/stderr, then the worker exits — so communicate() never sees EOF. Pre-fix the executor
    # awaited the pipe forever; post-fix it bounds the drain and kills the whole process group.
    with open(worker_py, "w") as f:
        f.write(
            "import os, sys, subprocess, time\n"
            "code = (\"import os,time;\"\n"
            "        \"open(os.environ['GC_PIDFILE'],'w').write(str(os.getpid()));\"\n"
            "        \"time.sleep(30)\")\n"
            "subprocess.Popen([sys.executable, '-c', code])\n"
            "for _ in range(100):\n"
            "    try:\n"
            "        if open(os.environ['GC_PIDFILE']).read().strip():\n"
            "            break\n"
            "    except OSError:\n"
            "        pass\n"
            "    time.sleep(0.05)\n"
            "sys.exit(0)\n"
        )
    ex = SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True),
        runs_dir=str(tmp_path / "runs"),
        launcher=_ScriptLauncher(worker_py),
        env={"GC_PIDFILE": pidfile},
        term_grace_s=1.0,
        timeout_s=None,  # the HARD case: no timeout — only the returncode-grace path can settle it
    )
    # Must SETTLE (not hang); pre-fix this never returns. Generous bound vs the ~1-2s settle path.
    res = await asyncio.wait_for(
        ex.run_agent({"name": "leaf", "system_prompt": "p"}, "go",
                     parent_loop=_loop(), driver_sid="d"),
        timeout=20,
    )
    assert res["status"] == "failed"  # worker produced no result frame

    # The grandchild (in the worker's process group) was reaped by the killpg, not left to leak.
    gc_pid = int(open(pidfile).read().strip())
    for _ in range(40):
        try:
            os.kill(gc_pid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            break
    else:
        try:
            os.kill(gc_pid, 9)  # cleanup if somehow still alive
        except ProcessLookupError:
            pass
        raise AssertionError("leaked grandchild was not killed with the worker's process group")


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
