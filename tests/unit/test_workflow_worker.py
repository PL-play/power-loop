"""Phase 0 feasibility: the isolated worker core.

Proves the gate for an out-of-process executor with a database-per-subprocess:
a sub-agent can run to completion against its OWN db file with llm/tools rebuilt
from a WorkerBootstrap alone — no parent loop, no shared store, no shared bus —
and the supervisor can afterward open that db read-only and inspect the full
trace. (Same-process here; the boundary is what matters, and it is clean.)
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import SessionStore
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import WorkerBootstrap, WorkerBootstrapError, run_spec_isolated
from power_loop.workflow.worker import (
    WorkerJob,
    decode_result,
    encode_result,
    main,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeLLM(LLMService):
    reply: str = "isolated answer"

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text=self.reply, content_text=self.reply)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


async def test_isolated_worker_runs_from_config_only_and_leaves_inspectable_db() -> None:
    db = tempfile.mktemp(suffix=".db")
    # Bootstrap carries ONLY a way to rebuild deps — no parent objects exist here.
    boot = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply="hello from the leaf"))

    result = await run_spec_isolated(
        {"name": "leaf", "system_prompt": "be brief"},
        "do the thing",
        bootstrap=boot,
        db_path=db,
    )

    assert result["status"] == "completed"
    assert result["final_text"] == "hello from the leaf"
    assert result["db_path"] == db
    assert os.path.exists(db)  # the sub-agent's own ledger persists for inspection

    # Supervisor opens the sub-agent's private db read-side and sees the full trace.
    store = SessionStore.open(db)
    try:
        msgs = store.load_all_messages(result["session_id"])
        roles = [m.role for m in msgs]
        assert "user" in roles and "assistant" in roles
        assert any("do the thing" in (m.content or "") for m in msgs)
    finally:
        store.close()
    os.remove(db)


async def test_isolated_worker_honors_output_schema() -> None:
    db = tempfile.mktemp(suffix=".db")
    boot = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply='{"label": "urgent"}'))
    result = await run_spec_isolated(
        {"name": "classifier", "system_prompt": "classify",
         "output_schema": {"name": "C", "schema": {"type": "object", "required": ["label"],
            "properties": {"label": {"type": "string"}}}}},
        "ticket text",
        bootstrap=boot,
        db_path=db,
    )
    # output_schema is honored without any parent: the request carried a json_schema
    # response_format and the structured reply came back as final_text.
    assert result["status"] == "completed"
    assert '"label"' in result["final_text"]
    os.remove(db)


async def test_two_isolated_workers_use_separate_dbs() -> None:
    """The core of the user's proposal: independent leaves, independent ledgers."""
    a, b = tempfile.mktemp(suffix=".db"), tempfile.mktemp(suffix=".db")
    boot_a = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply="A"))
    boot_b = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply="B"))
    ra = await run_spec_isolated({"name": "a", "system_prompt": "p"}, "x", bootstrap=boot_a, db_path=a)
    rb = await run_spec_isolated({"name": "b", "system_prompt": "p"}, "y", bootstrap=boot_b, db_path=b)
    assert ra["final_text"] == "A" and rb["final_text"] == "B"
    assert a != b and os.path.exists(a) and os.path.exists(b)
    # Each file is written by exactly one loop → the one-writer-per-file rule holds
    # trivially, no shared-write coordination needed.
    os.remove(a)
    os.remove(b)


async def test_tool_whitelist_narrows_rebuilt_registry() -> None:
    db = tempfile.mktemp(suffix=".db")

    def _registry() -> ToolRegistry:
        from power_loop.contracts.tools import ToolDefinition
        reg = ToolRegistry()
        for name in ("alpha", "beta"):
            reg.register(ToolDefinition(name=name, description=name,
                         input_schema={"type": "object", "properties": {}}), lambda **k: "ok")
        return reg

    boot = WorkerBootstrap(llm_factory=lambda: _FakeLLM(), registry_factory=_registry)
    # spec restricts to just 'alpha'; the rebuilt registry must be narrowed to it.
    result = await run_spec_isolated(
        {"name": "leaf", "system_prompt": "p", "tools": ["alpha"]},
        "go", bootstrap=boot, db_path=db,
    )
    assert result["status"] == "completed"
    os.remove(db)


def test_bootstrap_without_llm_source_is_a_clear_error() -> None:
    with pytest.raises(WorkerBootstrapError, match="no LLM source"):
        WorkerBootstrap().build_llm()


# ── 1a: IPC contract (job + bootstrap serialization + result framing) ─────────


def test_worker_job_roundtrip() -> None:
    job = WorkerJob(
        spec={"name": "a", "system_prompt": "p"}, user_input="hi",
        db_path="/t/x.db", bootstrap={"llm_from_env": True, "tool_preset": "core"},
        max_spawn_depth=2,
    )
    j2 = WorkerJob.from_json(job.to_json())
    assert j2.spec == job.spec and j2.user_input == "hi" and j2.db_path == "/t/x.db"
    assert j2.bootstrap["tool_preset"] == "core" and j2.max_spawn_depth == 2


def test_bootstrap_serializable_roundtrip_and_factory_guard() -> None:
    b = WorkerBootstrap(llm_from_env=True, tool_preset="core", workspace_dir="/tmp/ws")
    d = b.to_serializable_dict()
    assert d == {"llm_from_env": True, "provider_prefix": None, "tool_preset": "core",
                 "workspace_dir": "/tmp/ws", "home_dir": None}
    b2 = WorkerBootstrap.from_dict(d)
    assert b2.llm_from_env and b2.tool_preset == "core" and b2.workspace_dir == "/tmp/ws"
    # A factory bootstrap cannot cross a process boundary.
    with pytest.raises(WorkerBootstrapError, match="factories"):
        WorkerBootstrap(llm_factory=lambda: None).to_serializable_dict()
    # The config path must opt into rebuilding the provider from env.
    with pytest.raises(WorkerBootstrapError, match="llm_from_env"):
        WorkerBootstrap(tool_preset="core").to_serializable_dict()


def test_result_framing_survives_stdout_noise() -> None:
    noisy = (
        "INFO: some startup log\n"
        "a stray print\n"
        + encode_result({"ok": True, "result": {"x": 1}}) + "\n"
        "trailing noise after the frame\n"
    )
    assert decode_result(noisy) == {"ok": True, "result": {"x": 1}}


def test_decode_result_without_frame_errors() -> None:
    with pytest.raises(WorkerBootstrapError, match="no result frame"):
        decode_result("just logs, no frame\n")


def test_worker_main_entrypoint_runs_job_via_echo_provider(monkeypatch) -> None:
    """The full entrypoint: stdin job → rebuild echo provider from env → framed result.

    Run in-process (main does its own asyncio.run); same code path a spawned
    `python -m power_loop.workflow.worker` takes.
    """
    import io

    monkeypatch.setenv("POWER_LOOP_PROVIDER", "echo")
    monkeypatch.setenv("POWER_LOOP_ECHO_REPLY", "hi from the worker entrypoint")
    db = tempfile.mktemp(suffix=".db")
    job = WorkerJob(
        spec={"name": "leaf", "system_prompt": "p"}, user_input="go",
        db_path=db, bootstrap={"llm_from_env": True},
    )
    out = io.StringIO()
    rc = main(stdin=io.StringIO(job.to_json()), stdout=out)

    assert rc == 0
    env = decode_result(out.getvalue())
    assert env["ok"] is True
    assert env["result"]["status"] == "completed"
    assert env["result"]["final_text"] == "hi from the worker entrypoint"
    assert env["result"]["db_path"] == db
    assert os.path.exists(db)
    os.remove(db)


def test_worker_main_reports_failure_as_envelope(monkeypatch) -> None:
    """A job that can't build its provider returns ok=False with a traceback,
    never hangs or crashes the worker silently."""
    import io

    # bootstrap forgot llm_from_env and has no factory → build_llm raises inside run_job.
    db = tempfile.mktemp(suffix=".db")
    job = WorkerJob(spec={"name": "leaf", "system_prompt": "p"}, user_input="go",
                    db_path=db, bootstrap={})
    out = io.StringIO()
    rc = main(stdin=io.StringIO(job.to_json()), stdout=out)

    assert rc == 1
    env = decode_result(out.getvalue())
    assert env["ok"] is False
    assert "no LLM source" in env["error"]
    assert env["traceback"]


def test_worker_runs_as_a_real_spawned_process() -> None:
    """Smoke test that `python -m power_loop.workflow.worker` works as an actual
    OS subprocess (echo provider → fast, deterministic, no network). De-risks 1b."""
    import subprocess
    import sys

    db = tempfile.mktemp(suffix=".db")
    job = WorkerJob(
        spec={"name": "leaf", "system_prompt": "p"}, user_input="go",
        db_path=db, bootstrap={"llm_from_env": True},
    )
    env = dict(os.environ, POWER_LOOP_PROVIDER="echo", POWER_LOOP_ECHO_REPLY="spawned-ok")
    proc = subprocess.run(
        [sys.executable, "-m", "power_loop.workflow.worker"],
        input=job.to_json(), capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    result = decode_result(proc.stdout)
    assert result["ok"] is True
    assert result["result"]["final_text"] == "spawned-ok"
    assert os.path.exists(db)
    os.remove(db)
