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
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.spec import AgentSpec

from .worker import WorkerBootstrap, WorkerJob, decode_result

__all__ = [
    "SubprocessExecutor",
    "WorkerLauncher",
    "DirectWorkerLauncher",
    "cleanup_run",
    "reap_runs",
]


@runtime_checkable
class WorkerLauncher(Protocol):
    """How a worker process is launched — the seam for process-level sandboxing.

    power-loop stays sandbox-agnostic: by default the worker runs bare
    (:class:`DirectWorkerLauncher`). An integrator injects a launcher that wraps
    the command (``runsc``/gVisor, ``docker run``, ``firejail``, ``nsjail`` …) so
    the *child* runs confined while the orchestrator stays unconfined — the only
    way to give a sub-agent stronger isolation than its parent. This mirrors how
    deeptalk swaps power-loop's in-process bash for a sandboxed ``ShellBackend``,
    one level up: there it wraps the *shell*, here it wraps the *whole worker*.

    ``build`` is called once per leaf with the leaf's :class:`AgentSpec` (so the
    policy can decide *per leaf* by inspecting ``spec.tools``) and the paths the
    worker will touch. It returns the ``(argv, env)`` to spawn.

    Mounts/paths are the launcher's responsibility: the worker uses ``db_path``
    (and the bootstrap's ``workspace_dir``) verbatim, so a sandbox must make those
    paths resolve to the same location inside it (e.g. an identity bind-mount).
    """

    def build(
        self,
        *,
        base_argv: list[str],
        base_env: dict[str, str] | None,
        spec: AgentSpec,
        db_path: str,
        workspace_dir: str | None,
    ) -> tuple[list[str], dict[str, str] | None]: ...


class DirectWorkerLauncher:
    """Default: run the worker bare (no sandbox), command/env unchanged."""

    def build(
        self,
        *,
        base_argv: list[str],
        base_env: dict[str, str] | None,
        spec: AgentSpec,
        db_path: str,
        workspace_dir: str | None,
    ) -> tuple[list[str], dict[str, str] | None]:
        return base_argv, base_env


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:64] or "x"


def _remove_db(db_path: str) -> None:
    """Remove a leaf db and its WAL sidecars (-wal / -shm)."""
    for p in (db_path, db_path + "-wal", db_path + "-shm"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def cleanup_run(runs_dir: str, run_id: str) -> None:
    """Delete all per-leaf dbs for one run (its ``runs_dir/<run_id>/`` directory).

    Call when the supervisor is done inspecting a finished run.
    """
    shutil.rmtree(Path(runs_dir) / _safe(run_id), ignore_errors=True)


def reap_runs(runs_dir: str, *, older_than_s: float, now: float | None = None) -> list[str]:
    """Reap run directories whose newest file is older than ``older_than_s``.

    Age is mtime-based (no run bookkeeping needed). Returns the run dir names
    removed. Use as a periodic GC so kept-for-inspection dbs don't accumulate
    forever.
    """
    base = Path(runs_dir)
    if not base.is_dir():
        return []
    cutoff = (time.time() if now is None else now) - float(older_than_s)
    reaped: list[str] = []
    for run_dir in base.iterdir():
        # A whole-dir guard: the leaf workers and ``delete_on_success`` remove
        # dbs + WAL sidecars concurrently, so any stat/iterdir here can race a
        # concurrent unlink. Swallow per-dir OS errors and move on — one vanished
        # file must never abort the sweep and strand every later run dir. Mirrors
        # ``_remove_db``'s swallow-and-continue.
        try:
            if not run_dir.is_dir():
                continue
            mtimes: list[float] = []
            for f in run_dir.iterdir():
                try:
                    mtimes.append(f.stat().st_mtime)
                except OSError:
                    continue  # file vanished mid-sweep
            try:
                fallback = run_dir.stat().st_mtime
            except OSError:
                continue  # the run dir itself vanished — nothing to reap
            newest = max(mtimes, default=fallback)
            if newest < cutoff:
                shutil.rmtree(run_dir, ignore_errors=True)
                reaped.append(run_dir.name)
        except OSError:
            continue
    return reaped


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
        delete_on_success: bool = False,
        launcher: WorkerLauncher | None = None,
    ) -> None:
        # Validate up front that this bootstrap can cross a process boundary.
        # to_serializable_dict is attached to WorkerBootstrap at import time in
        # worker.py (ergonomic method), so it's invisible to the type checker here.
        self._bootstrap_dict = bootstrap.to_serializable_dict()  # type: ignore[attr-defined]
        self._runs_dir = Path(runs_dir) if runs_dir else Path(tempfile.gettempdir()) / "power_loop_wf_runs"
        self._py = python_executable or sys.executable
        self._env_overrides = dict(env) if env is not None else None
        self._timeout_s = timeout_s
        self._term_grace_s = term_grace_s
        # How the worker process is launched. Default = bare; inject a launcher
        # to wrap it in a sandbox (per leaf — see WorkerLauncher).
        self._launcher: WorkerLauncher = launcher or DirectWorkerLauncher()
        # Default: keep every leaf db so the supervisor can inspect it afterward
        # (use reap_runs / cleanup_run to GC). Set True to drop a leaf db as soon
        # as it completes — useful for high-fan-out runs you won't inspect.
        self._delete_on_success = delete_on_success

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
        return await self._spawn_and_collect(job, spec, CancellationToken.from_any(stop_event))

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

    async def _spawn_and_collect(
        self, job: WorkerJob, spec: AgentSpec, token: CancellationToken
    ) -> dict[str, Any]:
        base: dict[str, Any] = {"session_id": None, "rounds": 0, "usage": {}, "db_path": job.db_path}
        try:
            argv, env = self._launcher.build(
                base_argv=[self._py, "-m", "power_loop.workflow.worker"],
                base_env=self._child_env(),
                spec=spec,
                db_path=job.db_path,
                workspace_dir=self._bootstrap_dict.get("workspace_dir"),
            )
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed: a launch/sandbox error is a failed leaf, not a hang
            return {**base, "status": "failed", "error": f"worker launch failed: {exc}",
                    "final_text": f"[worker launch failed] {type(exc).__name__}: {exc}"}
        comm = asyncio.ensure_future(proc.communicate(input=job.to_json().encode()))
        outcome, stdout_b, stderr_b = await self._await_proc(proc, comm, token)
        stdout = (stdout_b or b"").decode(errors="replace")
        stderr = (stderr_b or b"").decode(errors="replace")
        result = self._to_result(job, outcome, proc.returncode, stdout, stderr)
        # Retention: keep failed/cancelled leaves for debugging; optionally drop
        # a completed leaf's db right away.
        if self._delete_on_success and result.get("status") == "completed":
            _remove_db(job.db_path)
            result = {**result, "db_path": None}
        return result

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
        base: dict[str, Any] = {"session_id": None, "rounds": 0, "usage": {}, "db_path": job.db_path}
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
