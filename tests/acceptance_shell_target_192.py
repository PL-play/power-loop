"""Explicit actual-container gate; only uniquely named resources created here."""
from __future__ import annotations

import asyncio
import os
import subprocess
import uuid

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.exec_backend import LocalShellBackend
from power_loop.runtime.process_control import BoundShellControl
from power_loop.tools.default_tools import BackgroundManager


def docker(*args):
    return subprocess.run(['docker', *args], capture_output=True, text=True, timeout=20)


def create(name, image, record_property, runtime):
    result = docker('run', '-d', '--runtime', runtime, '--name', name, '--network', 'none', image, 'sleep', '120')
    assert result.returncode == 0, result.stderr
    cid = result.stdout.strip()
    record_property('owned_container_id', cid)
    return cid


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["runc", "runsc"])
async def test_actual_target_stop_does_not_resolve_replacement_container(tmp_path, record_property, runtime):
    record_property('container_runtime', runtime)
    image = os.environ['ASYNC_PROCESS_TEST_IMAGE']
    assert image.startswith('sha256:')
    label = 'a192-shell-' + uuid.uuid4().hex[:12]
    original = create(label, image, record_property, runtime)
    replacement = None
    manager = BackgroundManager()

    class Backend(LocalShellBackend):
        target = original
        calls = 0

        def launch_argv(self, workspace_dir):
            self.calls += 1
            return ['docker', 'exec', '-i', self.target, '/bin/bash', '--norc', '--noprofile']

    backend = Backend()
    tid = None
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=backend)):
            msg = await manager.run('echo $$ > /tmp/original-pid; sleep 90')
        tid = msg.split()[2]
        for _ in range(100):
            if docker('exec', original, 'test', '-s', '/tmp/original-pid').returncode == 0:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError('original remote command never started')
        record_property('owned_process_pid', manager.tasks[tid]['proc'].pid)
        assert docker('rename', original, label + '-old').returncode == 0
        replacement = create(label, image, record_property, runtime)
        backend.target = replacement
        # The SAME tag deliberately exists in the replacement. Only the bound
        # original target may be controlled, even after its reusable name moves.
        shell = 'echo $$ > /tmp/foreign-pid; exec sleep 90'
        r = docker('exec', '-d', replacement, 'env', 'PL_BG_TAG=' + tid, 'bash', '-c', shell)
        assert r.returncode == 0, r.stderr
        assert await manager.cancel_task(tid, shell_term_s=0.05) == 'stopped'
        assert backend.calls == 1
        alive = docker('exec', replacement, 'bash', '-c', 'kill -0 "$(cat /tmp/foreign-pid)"')
        assert alive.returncode == 0
        dead = docker('exec', original, 'bash', '-c', 'kill -0 "$(cat /tmp/original-pid)"')
        assert dead.returncode != 0
        control = manager.tasks[tid]['control_target']
        assert isinstance(control, BoundShellControl)
        assert docker('rm', '-f', original).returncode == 0
        # Original target absence must not cause creation/adoption of replacement.
        assert control.stop_tagged(tid, variable='PL_BG_TAG', grace_s=0).confirmed is False
        assert docker('exec', replacement, 'bash', '-c', 'kill -0 "$(cat /tmp/foreign-pid)"').returncode == 0
    finally:
        if tid is not None:
            await manager.cancel_task(tid, shell_term_s=0)
        for cid in [original, replacement]:
            if cid is not None:
                docker('rm', '-f', cid)
    live = docker('ps', '-a', '--no-trunc', '--format', '{{.ID}}')
    assert live.returncode == 0
    assert original not in live.stdout and (replacement is None or replacement not in live.stdout)
