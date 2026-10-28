"""Original shell targets, PID-safe signaling and cancellation races."""
from __future__ import annotations

import asyncio
import builtins
import io
import os
import shlex
import signal
import sys
import threading
from pathlib import Path

import pytest

from power_loop.runtime import _tagged_processes as tagged
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.exec_backend import LocalShellBackend
from power_loop.runtime.process_control import ShellControlResult, ShellProcessControl
from power_loop.tools import default_tools as dt

pytestmark = pytest.mark.unit


async def _settled(manager, tid):
    for _ in range(500):
        if tid not in manager._threads:
            return manager.tasks[tid]
        await asyncio.sleep(0.02)
    raise AssertionError('collector did not finish')


async def _file(path):
    for _ in range(300):
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        await asyncio.sleep(0.02)
    raise AssertionError('process did not publish original PID')


async def _gone(pid):
    for _ in range(300):
        if not Path('/proc', str(pid)).exists():
            return
        await asyncio.sleep(0.02)
    raise AssertionError('original process still exists')


class OriginalBackend(LocalShellBackend):
    def __init__(self):
        self.launches = 0

    def launch_argv(self, workspace_dir):
        self.launches += 1
        if self.launches > 1:
            raise AssertionError('stop re-resolved the launch target')
        return super().launch_argv(workspace_dir)


async def test_cancel_uses_original_target_and_reaches_detached_child(tmp_path, record_property):
    backend, manager = OriginalBackend(), dt.BackgroundManager()
    pidfile = tmp_path / 'child'
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=backend)):
        message = await manager.run(f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait")
    tid = message.split()[2]
    pid = await _file(pidfile)
    record_property('owned_process_pid', pid)
    record_property('owned_process_pid', manager.tasks[tid]['proc'].pid)
    assert await manager.cancel_task(tid, shell_term_s=0.01) == 'stopped'
    assert (await _settled(manager, tid))['status'] == 'cancelled'
    await _gone(pid)
    assert backend.launches == 1


async def test_cancel_during_backend_preparation_prevents_command_submission(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class DelayedBackend(LocalShellBackend):
        def launch_argv(self, workspace_dir):
            entered.set()
            assert release.wait(5)
            return super().launch_argv(workspace_dir)

    manager = dt.BackgroundManager()
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=DelayedBackend())):
        message = await manager.run('touch forbidden')
    tid = message.split()[2]
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert await manager.cancel_task(tid) == 'pending'
    finally:
        release.set()
    assert (await _settled(manager, tid))['status'] == 'cancelled'
    assert not (tmp_path / 'forbidden').exists()


async def test_unknown_stop_is_retained_and_retry_controls_same_original_target(
    tmp_path, monkeypatch, record_property,
):
    class RetryControl(ShellProcessControl):
        available = False

        def __init__(self, original):
            self.original = original

        def stop_tagged(self, tag, *, variable, grace_s):
            if not self.available:
                return ShellControlResult(False, 'transport_unavailable')
            return self.original.stop_tagged(tag, variable=variable, grace_s=grace_s)

    class Backend(OriginalBackend):
        def bind_process_control(self, workspace_dir, *, argv, cwd, env):
            from power_loop.runtime.process_control import BoundShellControl
            self.control = RetryControl(BoundShellControl(argv, cwd, env))
            return self.control

    backend, manager = Backend(), dt.BackgroundManager()
    pidfile = tmp_path / 'child'
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=backend)):
        msg = await manager.run(f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait")
    tid = msg.split()[2]
    pid = await _file(pidfile)
    record_property('owned_process_pid', pid)
    record_property('owned_process_pid', manager.tasks[tid]['proc'].pid)
    try:
        assert await manager.cancel_task(tid, shell_term_s=0) == 'unknown'
        assert manager.tasks[tid]['status'] == 'stop_unconfirmed'
        monkeypatch.setattr(dt, '_MAX_BG_TASKS', 0)
        with manager._lock:
            manager._prune_locked()
        assert tid in manager.tasks
    finally:
        backend.control.available = True
        assert await manager.cancel_task(tid, shell_term_s=0) == 'stopped'
    await _gone(pid)
    assert manager.tasks[tid]['status'] == 'cancelled'
    assert backend.launches == 1


@pytest.mark.parametrize('replacement', [True, False])
def test_pidfd_signal_cannot_follow_reused_numeric_pid(monkeypatch, replacement):
    streams = iter([b'PL_BG_TAG=original\0', b'PL_BG_TAG=other\0' if replacement else b'PL_BG_TAG=original\0'])
    calls, closed = [], []
    monkeypatch.setattr(os, 'listdir', lambda _path: ['42'])
    monkeypatch.setattr(os, 'getpid', lambda: 100)
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: io.BytesIO(next(streams)))
    monkeypatch.setattr(os, 'pidfd_open', lambda pid: calls.append(('open', pid)) or 77)
    monkeypatch.setattr(signal, 'pidfd_send_signal', lambda fd, sig: calls.append(('signal', fd, sig)))
    monkeypatch.setattr(os, 'close', closed.append)
    handles = tagged.signal_members('PL_BG_TAG', 'original', signal.SIGKILL)
    if replacement:
        assert handles == [] and calls == [('open', 42)] and closed == [77]
    else:
        assert handles == [77] and calls == [('open', 42), ('signal', 77, signal.SIGKILL)]


def test_permission_failure_for_matching_process_is_not_absence(monkeypatch):
    monkeypatch.setattr(os, 'listdir', lambda _path: ['42'])
    monkeypatch.setattr(os, 'getpid', lambda: 100)
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: io.BytesIO(b'PL_BG_TAG=original\0'))

    def denied(_pid):
        raise PermissionError('denied')

    monkeypatch.setattr(os, 'pidfd_open', denied)
    result = tagged.stop('PL_BG_TAG', 'original', 0)
    assert result == {'confirmed': False, 'error': 'process_control_failed'}


async def test_control_factory_requires_explicit_interface_before_launch(tmp_path):
    class Invalid(LocalShellBackend):
        def bind_process_control(self, *args, **kwargs):
            return object()

    manager = dt.BackgroundManager()
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=Invalid())):
        msg = await manager.run('touch forbidden')
    task = await _settled(manager, msg.split()[2])
    assert task['status'] == 'error'
    assert not (tmp_path / 'forbidden').exists()
    assert 'ShellProcessControl' in task['result']


async def test_timeout_kills_detached_process_and_drain_is_bounded(tmp_path, record_property):
    manager = dt.BackgroundManager()
    pidfile = tmp_path / 'child'
    code = 'import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
    code += f'open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(60)'
    command = 'setsid ' + shlex.join([sys.executable, '-c', code]) + ' & wait'
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, background_timeout_s=1)):
        msg = await manager.run(command)
    tid = msg.split()[2]
    pid = await _file(pidfile)
    record_property('owned_process_pid', pid)
    record_property('owned_process_pid', manager.tasks[tid]['proc'].pid)
    task = await asyncio.wait_for(_settled(manager, tid), 8)
    assert task['status'] == 'timeout'
    await _gone(pid)


async def test_unknown_stop_remains_in_jobs_and_blocks_completion(tmp_path, monkeypatch):
    from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
    from power_loop.runtime.completion import running_background, wait_background
    from tests.unit.test_stop import _LLM

    store = await SessionStore.open(str(tmp_path / 'unknown.db'))
    loop = StatefulAgentLoop(llm=_LLM(), store=store, config=AgentLoopConfig())
    monkeypatch.setattr(dt, 'BG', dt.BackgroundManager())
    try:
        sid = await loop.new_session()
        await store.upsert_background_task(sid, task_id='original', command='original command', status='running')
        assert (await loop.stop_job(sid, 'bg:original'))['result'] == 'unknown'
        row = await store.get_background_task(sid, 'original')
        assert row.status == 'stop_unconfirmed'
        assert any(j['id'] == 'bg:original' for j in await loop.list_jobs(sid))
        assert [r.task_id for r in await running_background(store, sid)] == ['original']
        assert (await wait_background(store, sid, ['original'], 0.01))[0].status == 'stop_unconfirmed'
        assert (await loop.stop_job(sid, 'bg:original'))['result'] == 'unknown'
    finally:
        await loop.aclose(drain_timeout_s=0.1)
        await store.close()


@pytest.mark.parametrize('broken', ['raises', 'wrong_result'])
def test_adapter_failure_cannot_become_a_stop_receipt(broken):
    from power_loop.runtime.process_control import stop_shell_target

    class Broken(ShellProcessControl):
        def stop_tagged(self, *args, **kwargs):
            if broken == 'raises':
                raise OSError('transport unavailable')
            return {'confirmed': True}

    result = stop_shell_target(Broken(), 'original', variable='PL_BG_TAG', grace_s=0)
    assert result.confirmed is False


async def test_actual_background_owner_sigkill_stops_original_local_group(tmp_path, record_property):
    import json
    import subprocess

    pids = tmp_path / 'pids.json'
    code = f'import os,json,time; open({str(pids)!r},"w").write(json.dumps([os.getppid(),os.getpid()])); time.sleep(60)'
    owner = tmp_path / 'owner.py'
    command = shlex.join([sys.executable, '-c', code])
    owner.write_text(f'''import asyncio
from pathlib import Path
from power_loop.runtime.env import RuntimeEnv,runtime_env_context
from power_loop.tools.default_tools import BackgroundManager
async def main():
    manager=BackgroundManager()
    with runtime_env_context(RuntimeEnv(workspace_dir=Path({str(tmp_path)!r}))):
        await manager.run({command!r})
    await asyncio.sleep(60)
asyncio.run(main())
''')
    process = subprocess.Popen([sys.executable, str(owner)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    record_property('owned_process_pid', process.pid)
    try:
        for _ in range(200):
            try:
                ids = json.loads(pids.read_text())
                break
            except (FileNotFoundError, ValueError):
                await asyncio.sleep(0.025)
        else:
            raise AssertionError('background worker not started')
        for pid in ids:
            record_property('owned_process_pid', pid)
        process.kill()
        assert process.wait(timeout=5) == -signal.SIGKILL
        for pid in ids:
            await _gone(pid)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_foreground_control_binding_failure_closes_new_pty(tmp_path, monkeypatch):
    import pty

    opened = []
    original = pty.openpty

    def capture():
        pair = original()
        opened.extend(pair)
        return pair

    class Invalid(LocalShellBackend):
        def bind_process_control(self, *args, **kwargs):
            raise OSError('binding unavailable')

    monkeypatch.setattr(pty, 'openpty', capture)
    with pytest.raises(OSError, match='binding unavailable'):
        dt.BashSession(tmp_path, backend=Invalid())
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_concurrent_stop_attempts_preserve_last_confirmed_result(tmp_path, record_property):
    class Serial(ShellProcessControl):
        calls = 0

        def __init__(self, original):
            self.original = original

        def stop_tagged(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ShellControlResult(False, 'first_ack_lost')
            return self.original.stop_tagged(*args, **kwargs)

    class Backend(LocalShellBackend):
        def bind_process_control(self, workspace_dir, *, argv, cwd, env):
            from power_loop.runtime.process_control import BoundShellControl
            self.control = Serial(BoundShellControl(argv, cwd, env))
            return self.control

    backend, manager = Backend(), dt.BackgroundManager()
    pidfile = tmp_path / 'child'
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=backend)):
        msg = await manager.run(f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait")
    tid = msg.split()[2]
    pid = await _file(pidfile)
    record_property('owned_process_pid', pid)
    record_property('owned_process_pid', manager.tasks[tid]['proc'].pid)
    results = await asyncio.gather(manager.cancel_task(tid, shell_term_s=0),
                                   manager.cancel_task(tid, shell_term_s=0))
    assert results == ['unknown', 'stopped']
    assert manager.tasks[tid]['status'] == 'cancelled'
    await _gone(pid)


@pytest.mark.parametrize('started,state', [(99, 'S'), (101, 'Z')])
def test_old_or_dead_process_does_not_make_new_command_uncontrollable(monkeypatch, started, state):
    monkeypatch.setattr(os, 'listdir', lambda _path: ['42'])
    monkeypatch.setattr(os, 'getpid', lambda: 1000)
    monkeypatch.setattr(tagged, 'process_info', lambda name: (started, state))

    def must_not_read(*args, **kwargs):
        raise AssertionError('older/dead process environment must not be read')

    monkeypatch.setattr(builtins, 'open', must_not_read)
    assert tagged.signal_members('PL_BG_TAG', 'original', signal.SIGKILL, not_before=100) == []


def test_unreadable_new_same_user_process_keeps_stop_unknown(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(os, 'listdir', lambda _path: ['42'])
    monkeypatch.setattr(os, 'getpid', lambda: 1000)
    monkeypatch.setattr(tagged, 'process_info', lambda name: (101, 'S'))
    monkeypatch.setattr(os, 'stat', lambda path: SimpleNamespace(st_uid=os.geteuid()))

    def denied(*args, **kwargs):
        raise PermissionError('new process unreadable')

    monkeypatch.setattr(builtins, 'open', denied)
    assert tagged.stop('PL_BG_TAG', 'original', 0, not_before=100)['confirmed'] is False


def test_target_reboot_is_not_original_resource_absence(monkeypatch):
    monkeypatch.setattr(tagged, 'probe', lambda: {'boot_id': 'new', 'not_before': 123})

    def no_scan(*args, **kwargs):
        raise AssertionError('do not scan a replacement boot')

    monkeypatch.setattr(tagged, 'signal_members', no_scan)
    assert tagged.stop('PL_BG_TAG', 'original', 0, not_before=100, boot_id='old') == {
        'confirmed': False, 'error': 'original_target_boot_changed'}


async def test_old_deferred_status_cannot_overwrite_new_confirmed_stop(tmp_path):
    from power_loop import SessionStore

    store = await SessionStore.open(str(tmp_path / 'deferred.db'))
    manager = dt.BackgroundManager()
    gate = asyncio.Lock()
    try:
        sid = await store.create_session()
        await store.upsert_background_task(sid, task_id='original', command='command', status='stop_unconfirmed')
        task = {'store': store, 'session_id': sid, 'command': 'command', 'status': 'stop_unconfirmed',
                'stopped': True, 'result': 'old unknown', 'stop_lock': gate}
        manager.tasks['original'] = task
        manager._orphaned.append({'store': store, 'sid': sid, 'task_id': 'original', 'command': 'command',
                                 'status': 'stop_unconfirmed', 'return_code': None, 'output_tail': 'old unknown'})
        await gate.acquire()
        replay = asyncio.create_task(manager.flush_orphaned(store, sid))
        await asyncio.sleep(0)
        assert not replay.done()
        task.update(status='cancelled', result='original stop confirmed', stop_confirmed=True)
        gate.release()
        await replay
        row = await store.get_background_task(sid, 'original')
        assert row.status == 'cancelled' and row.output_tail == 'original stop confirmed'
        assert not manager._orphaned
    finally:
        await store.close()
