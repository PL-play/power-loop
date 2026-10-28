"""Local process ownership and an explicit shell-target control seam.

A bound target is captured once at launch. Control never asks a backend to
create/resolve another target. The default remote helper requires Python 3 and
Linux pidfds; unsupported or inaccessible targets remain unconfirmed.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShellControlResult:
    confirmed: bool
    error: str | None = None

    def __post_init__(self):
        if type(self.confirmed) is not bool or self.error is not None and not isinstance(self.error, str):
            raise ValueError('invalid shell control result')


class ShellProcessControl(ABC):
    """Host extension bound to the original execution resource, never its alias."""

    @abstractmethod
    def stop_tagged(self, tag: str, *, variable: str, grace_s: float) -> ShellControlResult:
        """Stop matching original processes; failures must remain unconfirmed."""
        raise NotImplementedError


def stop_shell_target(control, tag, *, variable, grace_s):
    try:
        result = control.stop_tagged(tag, variable=variable, grace_s=grace_s)
        if not isinstance(result, ShellControlResult):
            return ShellControlResult(False, 'invalid_control_result')
        return result
    except Exception:  # noqa: BLE001 — a host adapter failure is not a stop proof
        return ShellControlResult(False, 'original_target_control_failed')


class BoundShellControl(ShellProcessControl):
    def __init__(self, argv, cwd, env):
        from . import _tagged_processes
        self.argv, self.cwd, self.env = tuple(argv), cwd, dict(env)
        self._script = Path(_tagged_processes.__file__).read_text()
        command = shlex.join(['python3', '-c', self._script, '--probe'])
        result = subprocess.run(self.argv, input=command + '\n', text=True,
            capture_output=True, cwd=self.cwd, env=self.env, timeout=5)
        if result.returncode != 0:
            raise RuntimeError('original shell control probe failed')
        data = json.loads(result.stdout)
        if (not isinstance(data.get('boot_id'), str) or not data['boot_id']
                or type(data.get('not_before')) is not int or data['not_before'] <= 0):
            raise RuntimeError('invalid original shell control probe')
        self._boot_id, self._not_before = data['boot_id'], data['not_before']

    def stop_tagged(self, tag: str, *, variable: str, grace_s: float) -> ShellControlResult:
        command = shlex.join(['python3', '-c', self._script, variable, tag, str(grace_s),
                              str(self._not_before), self._boot_id])
        try:
            result = subprocess.run(self.argv, input=command + '\n', text=True,
                capture_output=True, cwd=self.cwd, env=self.env, timeout=max(0, grace_s) + 5)
            if result.returncode != 0:
                return ShellControlResult(False, 'original_target_control_failed')
            data = json.loads(result.stdout)
            if data.get('confirmed') is True:
                return ShellControlResult(True)
            return ShellControlResult(False, str(data.get('error', 'stop_unconfirmed')))
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, AttributeError):
            return ShellControlResult(False, 'original_target_control_unavailable')


def bind_shell_control(backend, workspace_dir, *, argv, cwd, env) -> ShellProcessControl:
    factory = getattr(backend, 'bind_process_control', None)
    control = (factory(workspace_dir, argv=tuple(argv), cwd=cwd, env=dict(env))
               if factory else BoundShellControl(argv, cwd, env))
    if not isinstance(control, ShellProcessControl):
        raise TypeError('bind_process_control must return ShellProcessControl')
    return control


class OwnedProcess:
    """Popen-compatible collection with cancellation via one original pipe."""

    def __init__(self, argv, *, cwd, env, grace_s=0.1):
        if os.name != 'posix':
            raise RuntimeError('owned shell processes require POSIX supervision')
        self._lock = threading.Lock()
        self._control = None
        self._result_fd = None
        self._exitcode = None
        self._collected = False
        self._stop_at = None
        control_read, control_write = os.pipe()
        result_read, result_write = os.pipe()
        guard = Path(__file__).parents[1] / 'workflow' / '_process_guard.py'
        try:
            self.process = subprocess.Popen([sys.executable, str(guard), '--result-fd',
                str(result_write), str(control_read), str(grace_s), *argv],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=cwd, env=env, text=True, start_new_session=True,
                pass_fds=(control_read, result_write))
        except BaseException:
            os.close(control_write)
            os.close(result_read)
            raise
        else:
            self._control, self._result_fd = control_write, result_read
        finally:
            os.close(control_read)
            os.close(result_write)

    @property
    def pid(self):
        return self.process.pid

    def request_stop(self):
        with self._lock:
            fd, self._control = self._control, None
            if fd is not None:
                self._stop_at = time.monotonic()
                os.close(fd)

    def _collect(self):
        with self._lock:
            if self._collected or self.process.returncode is None:
                return
            fd, self._result_fd = self._result_fd, None
            try:
                data = os.read(fd, 128)
                self._exitcode = int(data) if data else self.process.returncode
            finally:
                os.close(fd)
                self._collected = True
        self.request_stop()

    @property
    def returncode(self):
        self._collect()
        return self._exitcode

    def poll(self):
        self.process.poll()
        return self.returncode

    def wait(self, timeout=None):
        self.process.wait(timeout=timeout)
        return self.returncode

    def communicate(self, input=None, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else float('inf')
        while True:
            with self._lock:
                stop_deadline = self._stop_at + 1 if self._stop_at is not None else float('inf')
            bound = min(deadline, stop_deadline)
            try:
                result = self.process.communicate(input=input,
                    timeout=max(0, min(0.1, bound - time.monotonic())))
                self._collect()
                return result
            except subprocess.TimeoutExpired:
                input = None  # Popen retains the original input across drain retries.
                if time.monotonic() >= bound:
                    raise
