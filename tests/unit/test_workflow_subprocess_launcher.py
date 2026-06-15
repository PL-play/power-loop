"""WorkerLauncher seam: process-level sandbox hook for the subprocess executor.

power-loop stays sandbox-agnostic; an integrator injects a launcher that wraps
the worker command per leaf. These tests prove the seam: it receives per-leaf
context (the AgentSpec, so policy can key on spec.tools), it controls the child
argv and env, a wrapped command still runs end-to-end, and a launcher that
raises fails the leaf closed (no hang).
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

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


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_OrchLLM(), store=SessionStore.open(":memory:"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )


def _exec(runs_dir, *, launcher=None, env=None):
    base_env = {"POWER_LOOP_PROVIDER": "echo", "POWER_LOOP_ECHO_REPLY": "ok"}
    if env is not None:
        base_env = env
    return SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True, workspace_dir="/tmp/ws"),
        runs_dir=str(runs_dir), env=base_env, timeout_s=60, launcher=launcher,
    )


@dataclass
class _RecordingLauncher:
    calls: list = field(default_factory=list)
    wrap_if_tool: str | None = None

    def build(self, *, base_argv, base_env, spec, db_path, workspace_dir):
        wrapped = self.wrap_if_tool is not None and self.wrap_if_tool in (spec.tools or [])
        self.calls.append({
            "node": (spec.metadata or {}).get("workflow_node_id") or spec.name,
            "tools": list(spec.tools or []),
            "db_path": db_path,
            "workspace": workspace_dir,
            "base_argv": list(base_argv),
            "wrapped": wrapped,
        })
        return base_argv, base_env


async def test_launcher_receives_per_leaf_context(tmp_path) -> None:
    rec = _RecordingLauncher()
    res = await WorkflowEngine(_loop(), executor=_exec(tmp_path / "r", launcher=rec), run_id="ctx").run(
        WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "leaf",
                               "spec": {"name": "leaf", "system_prompt": "p"}}}))
    assert res.results["leaf"].status == "completed"
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["base_argv"] == [sys.executable, "-m", "power_loop.workflow.worker"]
    assert call["node"] == "leaf"
    assert call["workspace"] == "/tmp/ws"
    assert call["db_path"].endswith(".db") and "ctx" in call["db_path"]


async def test_launcher_can_decide_per_leaf_by_tools(tmp_path) -> None:
    """The headline capability: sandbox only leaves granted a risky tool."""
    rec = _RecordingLauncher(wrap_if_tool="bash")
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "safe", "spec": {"name": "safe", "system_prompt": "p"}},
            {"type": "agent", "id": "risky",
             "spec": {"name": "risky", "system_prompt": "p", "tools": ["bash"]}},
        ]},
    })
    res = await WorkflowEngine(_loop(), executor=_exec(tmp_path / "r", launcher=rec), run_id="pl").run(spec)
    assert res.status == "completed"
    by_node = {c["node"]: c for c in rec.calls}
    assert by_node["safe"]["wrapped"] is False   # no risky tool → run bare
    assert by_node["risky"]["wrapped"] is True    # has bash → would be sandboxed


async def test_launcher_controls_child_env(tmp_path) -> None:
    """A launcher can rewrite the child env (e.g. scrub secrets / inject proxy)."""

    @dataclass
    class _EnvLauncher:
        def build(self, *, base_argv, base_env, spec, db_path, workspace_dir):
            env = dict(base_env or {})
            env["POWER_LOOP_ECHO_REPLY"] = "from-launcher-env"
            return base_argv, env

    # executor env sets the provider but NOT the reply; the launcher injects it.
    ex = _exec(tmp_path / "r", launcher=_EnvLauncher(), env={"POWER_LOOP_PROVIDER": "echo"})
    res = await WorkflowEngine(_loop(), executor=ex, run_id="env").run(
        WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "leaf",
                               "spec": {"name": "leaf", "system_prompt": "p"}}}))
    assert res.results["leaf"].text == "from-launcher-env"


@pytest.mark.skipif(shutil.which("env") is None, reason="needs /usr/bin/env")
async def test_wrapped_command_still_runs(tmp_path) -> None:
    """A launcher that actually wraps the argv (here: prepend `env`) still runs
    the worker end-to-end — proving the spawn path honors the launcher's argv."""

    @dataclass
    class _EnvPrefixLauncher:
        def build(self, *, base_argv, base_env, spec, db_path, workspace_dir):
            return [shutil.which("env"), *base_argv], base_env

    ex = _exec(tmp_path / "r", launcher=_EnvPrefixLauncher(),
               env={"POWER_LOOP_PROVIDER": "echo", "POWER_LOOP_ECHO_REPLY": "wrapped-ok"})
    res = await WorkflowEngine(_loop(), executor=ex, run_id="wrap").run(
        WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "leaf",
                               "spec": {"name": "leaf", "system_prompt": "p"}}}))
    assert res.results["leaf"].text == "wrapped-ok"


async def test_launcher_failure_fails_leaf_closed(tmp_path) -> None:
    """If the launcher (e.g. a sandbox that can't start) raises, the leaf fails
    rather than hanging — fail closed."""

    @dataclass
    class _BrokenLauncher:
        def build(self, *, base_argv, base_env, spec, db_path, workspace_dir):
            raise RuntimeError("sandbox unavailable")

    res = await WorkflowEngine(_loop(), executor=_exec(tmp_path / "r", launcher=_BrokenLauncher()),
                               run_id="fc").run(
        WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "leaf",
                               "spec": {"name": "leaf", "system_prompt": "p"}}}))
    leaf = res.results["leaf"]
    assert leaf.status == "failed"
    assert "launch failed" in (leaf.error or "").lower()
    assert "sandbox unavailable" in (leaf.error or "")
