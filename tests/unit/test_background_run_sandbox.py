"""C1: ``background_run`` must honor the injected ShellBackend, like ``bash``.

It used to run ``subprocess.run(command, shell=True)`` directly on the host —
bypassing an installed sandbox and leaking the host environment — while the docs
claimed it was sandboxed. It must now launch through ``env.shell_backend``.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Hashable
from pathlib import Path

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.default_tools import BackgroundManager

pytestmark = pytest.mark.unit


class _MarkerBackend:
    """A stand-in sandbox backend: tags the launch env with a unique marker and
    records that its launch hooks were consulted."""

    def launch_argv(self, workspace_dir: Path) -> list[str]:
        return ["/bin/bash", "--norc", "--noprofile"]

    def launch_cwd(self, workspace_dir: Path) -> str | None:
        return str(workspace_dir)

    def launch_env(self, workspace_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["PL_SANDBOX_MARKER"] = "sandbox-123"
        return env

    def session_key(self, workspace_dir: Path) -> Hashable:
        return ("marker", str(workspace_dir))


def _wait(mgr: BackgroundManager, task_id: str, timeout: float = 5.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = mgr.tasks.get(task_id, {}).get("status")
        if status and status != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("background task did not finish in time")


def test_background_run_launches_through_shell_backend(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        captured["input"] = kw.get("input")
        captured["env"] = kw.get("env")
        captured["cwd"] = kw.get("cwd")

        class _R:
            stdout = ""
            stderr = ""
            returncode = 0

        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    mgr = BackgroundManager()
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=_MarkerBackend())):
        started = mgr.run("echo hi")
    task_id = started.split()[2]
    _wait(mgr, task_id)

    # The command goes to the backend-launched shell's stdin — NOT
    # subprocess.run("echo hi", shell=True) on the host.
    assert captured["argv"] == ["/bin/bash", "--norc", "--noprofile"]
    assert captured["input"] == "echo hi"
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"].get("PL_SANDBOX_MARKER") == "sandbox-123"


def test_background_run_executes_in_backend_environment(tmp_path) -> None:
    """End-to-end with real bash: the command actually runs inside the backend's
    environment (the host env alone would never define PL_SANDBOX_MARKER)."""
    mgr = BackgroundManager()
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=_MarkerBackend())):
        started = mgr.run("echo marker=$PL_SANDBOX_MARKER")
    task_id = started.split()[2]
    _wait(mgr, task_id)
    assert "marker=sandbox-123" in (mgr.tasks[task_id]["result"] or "")
