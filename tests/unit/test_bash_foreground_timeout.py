"""6.44.0: a timed-out foreground ``bash`` command is killed where it runs, and the shell backend
gets a ``touch`` before every command.

Before 6.44.0 a timeout only terminated the local shell process. Under a sandbox backend that
process is the ``docker exec`` client: the command kept running inside the container (verified on
a gVisor sandbox: ``sleep 600`` survived the client). Here the "escape" is reproduced locally with
``setsid``, which takes the command out of the shell's process group the same way.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.exec_backend import LocalShellBackend
from power_loop.tools.default_tools import BashSession


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(") ", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_timeout_kills_the_commands_processes(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    sess = BashSession(tmp_path)
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            out = sess.execute(f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait", timeout=2)
        assert "timed out after 2s" in out
        pid = int(pidfile.read_text().strip())
        for _ in range(50):
            if not _alive(pid):
                break
            time.sleep(0.1)
        assert not _alive(pid), "the timed-out command is still running"
        # the session was restarted and works
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            assert "after" in sess.execute("echo after")
    finally:
        sess.close()


def test_commands_that_finish_keep_their_background_children(tmp_path: Path) -> None:
    """Only a TIMEOUT kills; a command that returns leaves what it started alone (dev servers)."""
    pidfile = tmp_path / "pid"
    sess = BashSession(tmp_path)
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            sess.execute(f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' &", timeout=5)
            for _ in range(50):
                if pidfile.exists() and pidfile.read_text().strip():
                    break
                time.sleep(0.1)
            pid = int(pidfile.read_text().strip())
            sess.execute("sleep 5", timeout=1)              # an unrelated command times out
        assert _alive(pid)
        os.kill(pid, 9)
    finally:
        sess.close()


class _TouchingBackend(LocalShellBackend):
    def __init__(self, fail: bool = False) -> None:
        self.touched: list[Path] = []
        self.fail = fail

    def touch(self, workspace_dir: Path) -> None:
        self.touched.append(workspace_dir)
        if self.fail:
            raise RuntimeError("liveness store down")


def test_backend_is_touched_before_every_command(tmp_path: Path) -> None:
    backend = _TouchingBackend()
    sess = BashSession(tmp_path, backend=backend)
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            sess.execute("echo one")
            sess.execute("echo two")
        assert backend.touched == [tmp_path, tmp_path]
    finally:
        sess.close()


def test_a_failing_touch_does_not_fail_the_command(tmp_path: Path) -> None:
    sess = BashSession(tmp_path, backend=_TouchingBackend(fail=True))
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            assert "still runs" in sess.execute("echo still runs")
    finally:
        sess.close()
