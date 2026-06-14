"""Out-of-process executor (Phase 1b): one OS process + one DB per leaf.

Implements the :class:`~power_loop.workflow.engine.Executor` protocol by spawning
the Phase-1a worker entrypoint (``python -m power_loop.workflow.worker``) once per
leaf, piping it a :class:`WorkerJob`, and returning the framed result dict. The
engine, journal, and resume code are unchanged — this slots in behind the seam:

    eng = WorkflowEngine(loop, executor=SubprocessExecutor(bootstrap=...))

Each leaf runs in its **own process** against its **own SQLite file** (so the
one-writer-per-file rule holds trivially — no shared-write coordination). A
crashed/killed worker becomes a non-``completed`` result, which the resume tier
re-runs; a cancelled ``stop_event`` terminates the child (SIGTERM→SIGKILL).

Reserved seam — *shared scope → orchestrator*: a future shared blackboard would
be reached by handing the worker an address/token here so its ``board_*`` tools
RPC back to the orchestrator's board (the one place isolated leaves can share).
Not implemented in Phase 1.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.spec import AgentSpec

from .worker import WorkerBootstrap, WorkerJob, decode_result

__all__ = ["SubprocessExecutor"]


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:64] or "x"


class SubprocessExecutor:
    """Run each workflow leaf in a separate OS process with its own database."""

    def __init__(
        self,
        *,
        bootstrap: WorkerBootstrap,
        runs_dir: str | None = None,
        python_executable: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        term_grace_s: float = 3.0,
    ) -> None:
        # Validate up front that this bootstrap can cross a process boundary.
        self._bootstrap_dict = bootstrap.to_serializable_dict()
        self._runs_dir = Path(runs_dir) if runs_dir else Path(tempfile.gettempdir()) / "power_loop_wf_runs"
        self._py = python_executable or sys.executable
        self._env_overrides = dict(env) if env is not None else None
        self._timeout_s = timeout_s
        self._term_grace_s = term_grace_s

    async def run_agent(
        self,
        spec: AgentSpec,
        user_input: str,
        *,
        parent_loop: Any,
        driver_sid: str,
        stop_event: Any = None,
    ) -> dict[str, Any]:
        spec = spec if isinstance(spec, AgentSpec) else AgentSpec.from_json(spec)
        db_path = self._db_path_for(spec)
        job = WorkerJob(
            spec=asdict(spec),
            user_input=user_input,
            db_path=db_path,
            bootstrap=self._bootstrap_dict,
        )
        return await self._spawn_and_collect(job, CancellationToken.from_any(stop_event))

    # ── db path: unique per invocation (foreach bodies share a node id) ───────

    def _db_path_for(self, spec: AgentSpec) -> str:
        meta = spec.metadata or {}
        run_id = _safe(str(meta.get("workflow_run_id") or "adhoc"))
        node = _safe(str(meta.get("workflow_node_id") or spec.name or "leaf"))
        d = self._runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        # uuid suffix → unique even for parallel foreach iterations sharing node id.
        return str(d / f"{node}__{uuid.uuid4().hex[:8]}.db")

    # ── spawn + collect ───────────────────────────────────────────────────────

    def _child_env(self) -> dict[str, str] | None:
        if self._env_overrides is None:
            return None  # inherit parent environment
        return {**os.environ, **self._env_overrides}

    async def _spawn_and_collect(self, job: WorkerJob, token: CancellationToken) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            self._py, "-m", "power_loop.workflow.worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._child_env(),
        )
        comm = asyncio.ensure_future(proc.communicate(input=job.to_json().encode()))
        outcome, stdout_b, stderr_b = await self._await_proc(proc, comm, token)
        stdout = (stdout_b or b"").decode(errors="replace")
        stderr = (stderr_b or b"").decode(errors="replace")
        return self._to_result(job, outcome, proc.returncode, stdout, stderr)

    async def _await_proc(
        self, proc: asyncio.subprocess.Process, comm: asyncio.Future, token: CancellationToken
    ) -> tuple[str, bytes | None, bytes | None]:
        loop = asyncio.get_event_loop()
        deadline = (loop.time() + self._timeout_s) if self._timeout_s else None
        while True:
            done, _ = await asyncio.wait({comm}, timeout=0.1)
            if comm in done:
                stdout_b, stderr_b = comm.result()
                return "ok", stdout_b, stderr_b
            if token.is_cancelled():
                out = await self._terminate(proc, comm)
                return ("cancelled", *out)
            if deadline is not None and loop.time() > deadline:
                out = await self._terminate(proc, comm)
                return ("timeout", *out)

    async def _terminate(
        self, proc: asyncio.subprocess.Process, comm: asyncio.Future
    ) -> tuple[bytes | None, bytes | None]:
        """SIGTERM then SIGKILL, draining the (shielded) communicate() each time."""
        for stop in (proc.terminate, proc.kill):
            if proc.returncode is not None:
                break
            try:
                stop()
            except ProcessLookupError:
                break
            try:
                return await asyncio.wait_for(asyncio.shield(comm), self._term_grace_s)
            except asyncio.TimeoutError:
                continue
        try:
            return await asyncio.shield(comm)
        except Exception:  # noqa: BLE001
            return None, None

    def _to_result(
        self, job: WorkerJob, outcome: str, returncode: int | None, stdout: str, stderr: str
    ) -> dict[str, Any]:
        base = {"session_id": None, "rounds": 0, "usage": {}, "db_path": job.db_path}
        if outcome == "cancelled":
            return {**base, "status": "cancelled", "final_text": "[cancelled by orchestrator]"}
        if outcome == "timeout":
            return {**base, "status": "failed", "error": "worker timeout",
                    "final_text": f"[worker timeout]\n{stderr[-2000:]}"}
        try:
            env = decode_result(stdout)
        except Exception as exc:  # noqa: BLE001 — no frame ⇒ the worker died before reporting
            return {**base, "status": "failed", "error": f"no result frame: {exc}",
                    "final_text": f"[worker produced no result; rc={returncode}]\n{stderr[-2000:]}"}
        if not env.get("ok"):
            return {**base, "status": "failed", "error": env.get("error"),
                    "traceback": env.get("traceback"),
                    "final_text": f"[worker error] {env.get('error')}"}
        return env["result"]
