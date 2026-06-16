"""Phase 1d: leaf db_path journaling + retention/GC for per-leaf dbs."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.workflow import (
    SubprocessExecutor,
    WorkerBootstrap,
    WorkflowSpec,
    cleanup_run,
    get_workflow,
    reap_runs,
    register_wake_guard,  # noqa: F401  (import sanity for the package surface)
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


def _executor(runs_dir: str, *, delete_on_success=False) -> SubprocessExecutor:
    return SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True), runs_dir=runs_dir,
        env={"POWER_LOOP_PROVIDER": "echo", "POWER_LOOP_ECHO_REPLY": "ok"},
        timeout_s=60, delete_on_success=delete_on_success,
    )


_ONE_LEAF = {"name": "w", "root": {"type": "agent", "id": "only",
             "spec": {"name": "only", "system_prompt": "p"}}}


async def test_leaf_db_path_is_journaled_and_inspectable(tmp_path) -> None:
    loop = await _loop()
    sid = await loop.new_session()
    run_id = "dbp"
    spec = WorkflowSpec.from_json(_ONE_LEAF)
    await journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    ex = _executor(str(tmp_path / "r"))
    res = await WorkflowEngine(
        loop, executor=ex, on_step=make_on_step(loop.store, sid, run_id), run_id=run_id,
    ).run(spec)

    leaf = res.results["only"]
    assert leaf.db_path and os.path.exists(leaf.db_path)
    # journaled, so a supervisor can find + open the private db after the fact
    step = next(s for s in (await journal_mod.read(loop.store, sid, run_id))["steps"]
                if s["node_id"] == "only")
    assert step["db_path"] == leaf.db_path
    info = await get_workflow(loop, sid, run_id, detail=True)
    assert any(s.get("db_path") == leaf.db_path for s in info["steps"])

    store = await SessionStore.open(leaf.db_path)
    try:
        assert any(m.role == "assistant" for m in await store.load_all_messages(leaf.session_id))
    finally:
        await store.close()


async def test_delete_on_success_drops_completed_leaf_db(tmp_path) -> None:
    ex = _executor(str(tmp_path / "r"), delete_on_success=True)
    res = await WorkflowEngine(await _loop(), executor=ex, run_id="del").run(
        WorkflowSpec.from_json(_ONE_LEAF))
    leaf = res.results["only"]
    assert leaf.status == "completed"
    assert leaf.db_path is None  # dropped on success → nothing left to inspect


async def test_default_keeps_completed_leaf_db(tmp_path) -> None:
    ex = _executor(str(tmp_path / "r"))  # delete_on_success defaults False
    res = await WorkflowEngine(await _loop(), executor=ex, run_id="keep").run(
        WorkflowSpec.from_json(_ONE_LEAF))
    assert os.path.exists(res.results["only"].db_path)


def test_cleanup_run_removes_a_runs_directory(tmp_path) -> None:
    runs = tmp_path / "r"
    (runs / "run1").mkdir(parents=True)
    (runs / "run1" / "a__x.db").write_text("data")
    (runs / "run2").mkdir(parents=True)
    cleanup_run(str(runs), "run1")
    assert not (runs / "run1").exists()
    assert (runs / "run2").exists()  # other runs untouched


def test_reap_runs_removes_old_keeps_recent(tmp_path) -> None:
    runs = tmp_path / "r"
    old = runs / "old"
    old.mkdir(parents=True)
    (old / "a__1.db").write_text("x")
    recent = runs / "recent"
    recent.mkdir(parents=True)
    (recent / "b__2.db").write_text("y")

    # Backdate the 'old' run dir + its file well past the window.
    past = time.time() - 10_000
    os.utime(old / "a__1.db", (past, past))
    os.utime(old, (past, past))

    reaped = reap_runs(str(runs), older_than_s=3600)
    assert reaped == ["old"]
    assert not old.exists()
    assert recent.exists()


def test_reap_runs_survives_concurrent_unlink(tmp_path) -> None:
    """A file vanishing mid-sweep must not abort the whole sweep (H1.6).

    The worker processes and ``delete_on_success`` remove dbs + WAL sidecars
    concurrently, so ``f.stat()`` can race a real unlink. A dangling symlink is the
    deterministic stand-in: stat() follows it and raises ``FileNotFoundError``.
    Before the guard, that exception escaped and stranded every later run dir.
    """
    runs = tmp_path / "r"
    past = time.time() - 10_000

    racy = runs / "racy"
    racy.mkdir(parents=True)
    (racy / "real__1.db").write_text("x")
    os.symlink(str(tmp_path / "gone" / "missing.db"), racy / "vanished__2.db")
    os.utime(racy / "real__1.db", (past, past))
    os.utime(racy, (past, past))

    # A second old run dir that must still be reaped even though 'racy' raced.
    other = runs / "other"
    other.mkdir(parents=True)
    (other / "b__1.db").write_text("y")
    os.utime(other / "b__1.db", (past, past))
    os.utime(other, (past, past))

    reaped = reap_runs(str(runs), older_than_s=3600)  # must not raise

    assert set(reaped) == {"racy", "other"}
    assert not racy.exists()
    assert not other.exists()


def test_reap_runs_falls_back_to_dir_mtime_when_all_files_vanish(tmp_path) -> None:
    """If every file in a run dir is unstattable, fall back to the dir's own mtime
    rather than crashing or silently never reaping it (H1.6)."""
    runs = tmp_path / "r"
    past = time.time() - 10_000
    d = runs / "only-symlinks"
    d.mkdir(parents=True)
    os.symlink(str(tmp_path / "gone" / "a.db"), d / "a__1.db")
    os.utime(d, (past, past))

    reaped = reap_runs(str(runs), older_than_s=3600)
    assert reaped == ["only-symlinks"]
    assert not d.exists()
