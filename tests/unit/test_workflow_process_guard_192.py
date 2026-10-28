"""Real OS ownership tests; no historical numeric PID is used for cancellation."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from power_loop.runtime.cancellation import CancellationToken
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap
from power_loop.workflow import _process_guard as guard
from power_loop.workflow.subprocess_executor import _ProcessControl
from tests.unit.test_workflow_subprocess import _loop, _ScriptLauncher

pytestmark = pytest.mark.unit


def _script(tmp_path, *, exit_worker=False):
    p = tmp_path / 'worker.py'
    p.write_text('''import json, os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
with open(os.environ['PID_FILE'], 'w') as f:
    json.dump([os.getppid(), os.getpid(), child.pid], f)
''' + ('sys.exit(0)\n' if exit_worker else 'time.sleep(60)\n'))
    return p


async def _pids(path):
    for _ in range(200):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            await asyncio.sleep(0.025)
    raise AssertionError('worker did not publish its original identities')


async def _gone(pids):
    for _ in range(200):
        if not any(Path('/proc', str(pid)).exists() for pid in pids):
            return
        await asyncio.sleep(0.025)
    raise AssertionError(f'original processes remain: {pids}')


def _executor(tmp_path, script):
    return SubprocessExecutor(bootstrap=WorkerBootstrap(llm_from_env=True),
        runs_dir=str(tmp_path / 'runs'), launcher=_ScriptLauncher(str(script)),
        env={'PID_FILE': str(tmp_path / 'pids.json')}, term_grace_s=0.1)


@pytest.mark.parametrize('trigger', ['token', 'task', 'worker_exit'])
async def test_guard_stops_original_descendants_and_preserves_unrelated_process(
    tmp_path, trigger, record_property,
):
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    record_property('owned_process_pid', unrelated.pid)
    token = CancellationToken()
    ex = _executor(tmp_path, _script(tmp_path, exit_worker=trigger == 'worker_exit'))
    task = asyncio.create_task(ex.run_agent({'name': 'guard', 'system_prompt': 'p'}, 'x',
        parent_loop=_loop(), driver_sid='d', stop_event=token))
    try:
        pids = await _pids(tmp_path / 'pids.json')
        for pid in pids:
            record_property('owned_process_pid', pid)
        if trigger == 'token':
            token.cancel()
            assert (await task)['status'] == 'cancelled'
        elif trigger == 'task':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert (await task)['status'] == 'failed'  # no result frame
        await _gone(pids)
        assert unrelated.poll() is None
    finally:
        token.cancel()
        if not task.done():
            task.cancel()
        unrelated.kill()
        unrelated.wait(timeout=5)


async def test_actual_owner_sigkill_closes_pipe_and_stops_original_group(tmp_path, record_property):
    script = _script(tmp_path)
    owner = tmp_path / 'owner.py'
    owner.write_text(f'''import asyncio
from tests.unit.test_workflow_process_guard_192 import _executor, _loop
from pathlib import Path
asyncio.run(_executor(Path({str(tmp_path)!r}), Path({str(script)!r})).run_agent(
    {{'name': 'guard', 'system_prompt': 'p'}}, 'x', parent_loop=_loop(), driver_sid='d'))
''')
    process = subprocess.Popen([sys.executable, str(owner)], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    record_property('owned_process_pid', process.pid)
    try:
        pids = await _pids(tmp_path / 'pids.json')
        for pid in pids:
            record_property('owned_process_pid', pid)
        process.kill()
        assert process.wait(timeout=5) == -signal.SIGKILL
        await _gone(pids)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize('stage', ['before_spawn', 'after_spawn'])
async def test_cancellation_during_spawn_keeps_original_pipe_until_creation_settles(
    tmp_path, monkeypatch, record_property, stage,
):
    original = asyncio.create_subprocess_exec
    created = asyncio.Event()
    release = asyncio.Event()
    spawned = asyncio.Event()
    processes = []

    async def delayed(*args, **kwargs):
        if stage == 'before_spawn':
            created.set()
            await release.wait()
        process = await original(*args, **kwargs)
        processes.append(process)
        record_property('owned_process_pid', process.pid)
        spawned.set()
        created.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed)
    ex = _executor(tmp_path, _script(tmp_path))
    task = asyncio.create_task(ex.run_agent({'name': 'guard', 'system_prompt': 'p'}, 'x',
        parent_loop=_loop(), driver_sid='d'))
    await asyncio.wait_for(created.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.wait_for(spawned.wait(), 5)
    await asyncio.wait_for(processes[0].wait(), 5)
    pidfile = tmp_path / 'pids.json'
    if pidfile.exists():
        pids = await _pids(pidfile)
        for pid in pids:
            record_property('owned_process_pid', pid)
        await _gone(pids)


def test_guard_with_already_closed_owner_pipe_never_starts_worker(tmp_path, record_property):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    marker = tmp_path / 'started'
    try:
        process = subprocess.Popen([sys.executable, guard.__file__, str(read_fd), '0.1',
            sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()'],
            pass_fds=(read_fd,), start_new_session=True)
        record_property('owned_process_pid', process.pid)
        assert process.wait(timeout=5) == 0
        assert not marker.exists()
    finally:
        os.close(read_fd)


@pytest.mark.parametrize('group,session', [(41, 42), (42, 41), (41, 41)])
def test_guard_refuses_signal_without_live_ownership(monkeypatch, group, session):
    calls = []
    monkeypatch.setattr(os, 'getpid', lambda: 42)
    monkeypatch.setattr(os, 'getpgrp', lambda: group)
    monkeypatch.setattr(os, 'getsid', lambda _pid: session)
    monkeypatch.setattr(os, 'killpg', lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match='does not own'):
        guard._signal_owned_group(signal.SIGKILL)
    assert calls == []


def test_guard_signals_only_its_current_live_group(monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'getpid', lambda: 42)
    monkeypatch.setattr(os, 'getpgrp', lambda: 42)
    monkeypatch.setattr(os, 'getsid', lambda _pid: 42)
    monkeypatch.setattr(os, 'killpg', lambda *args: calls.append(args))
    guard._signal_owned_group(signal.SIGTERM)
    assert calls == [(42, signal.SIGTERM)]


def test_repeated_cancel_cannot_close_reused_descriptor(monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'close', calls.append)
    control = _ProcessControl(42)
    control.close()
    control.close()
    assert calls == [42]
