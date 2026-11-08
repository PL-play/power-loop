"""Original shell targets, PID-safe signaling and cancellation races."""
from __future__ import annotations

import asyncio
import builtins
import contextlib
import io
import json
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


@contextlib.asynccontextmanager
async def _command(tmp_path, command, *, backend=None, control=None, timeout=300):
    from power_loop import AgentLoopConfig, InputRef, RuntimeEnv, SessionStore, create_default_tool_registry
    from tests.runtime_fixture import RunningAgent

    from .test_timers import _Scripted, _tool_resp
    store = await SessionStore.open(str(tmp_path/'shell.db'))
    registry = create_default_tool_registry(include=['background_run', 'bash'], workspace_dir=tmp_path)
    model = _Scripted([_tool_resp('start', 'background_run', json.dumps({'action':'run', 'command':command}))])
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None),
        runtime_env=RuntimeEnv(workspace_dir=tmp_path, shell_backend=backend, background_timeout_s=timeout),
        process_control=control).start()
    ref = None
    try:
        assert (await runner.send('start')).status == 'completed'
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(row.content for row in rows if row.tool_call_id == 'start'))
        ref = InputRef(**receipt['input_ref'])
        yield runner, ref
    finally:
        if ref is not None:
            await runner.runtime.client.cancel(ref, actor=runner.actor)
        await runner.aclose()
        await store.close()


async def _result(runner, ref):
    from power_loop.agent.runtime_store import now_ms
    return await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms()+10000)



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
    raise AssertionError('original process still exists: ' + Path('/proc', str(pid), 'stat').read_text())


class OriginalBackend(LocalShellBackend):
    def __init__(self):
        self.launches = 0

    def launch_argv(self, workspace_dir):
        self.launches += 1
        if self.launches > 1:
            raise AssertionError('stop re-resolved the launch target')
        return super().launch_argv(workspace_dir)


@pytest.mark.parametrize('concurrent', [False, True])
async def test_cancel_uses_original_target_and_reaches_detached_child(tmp_path, record_property, concurrent):
    backend, pidfile = OriginalBackend(), tmp_path/'child'
    async with _command(tmp_path, f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait", backend=backend) as (runner, ref):
        pid = await _file(pidfile)
        record_property('owned_process_pid', pid)
        receipts = await asyncio.gather(*(runner.runtime.client.cancel(ref, actor=runner.actor) for _ in range(2 if concurrent else 1)))
        assert all(receipt.input_ref == ref for receipt in receipts)
        result = await _result(runner, ref)
        assert result.status == 'cancelled', result
        assert (await runner.runtime.client.inspect(result.job_ref, actor=runner.actor)).resource_state == 'stopped', result
        assert await _result(runner, ref) == result
        await _gone(pid)
        assert backend.launches == 1



async def test_cancel_during_backend_preparation_prevents_command_submission(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class DelayedBackend(LocalShellBackend):
        def launch_argv(self, workspace_dir):
            entered.set()
            assert release.wait(5)
            return super().launch_argv(workspace_dir)
    async with _command(tmp_path, 'touch forbidden', backend=DelayedBackend()) as (runner, ref):
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await runner.runtime.client.cancel(ref, actor=runner.actor)
            # Wait until the worker has delivered the explicit cancellation.
            await asyncio.wait_for(runner.runtime.worker._active[ref.session_id].stop.wait(), 5)
        finally:
            release.set()
        assert (await _result(runner, ref)).status == 'cancelled'
        assert not (tmp_path/'forbidden').exists()



async def test_unknown_stop_is_visible_and_new_input_can_continue(tmp_path, record_property):
    from power_loop.runtime.process_control import BoundShellControl
    class Unavailable(ShellProcessControl):
        def __init__(self, original):
            self.original = original
            self.calls = []
        def stop_tagged(self, tag, *, variable, grace_s):
            self.calls.append((tag, variable))
            return ShellControlResult(False, 'transport_unavailable')
    controls = []
    def bind(workspace, *, argv, cwd, env):
        control = Unavailable(BoundShellControl(argv, cwd, env))
        controls.append(control)
        return control
    backend, pidfile = OriginalBackend(), tmp_path/'child'
    async with _command(tmp_path, f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait", backend=backend, control=bind) as (runner, ref):
        pid = await _file(pidfile)
        record_property('owned_process_pid', pid)
        try:
            await runner.runtime.client.cancel(ref, actor=runner.actor)
            result = await _result(runner, ref)
            assert result.status == 'cancelled' and 'original_shell_stop_unconfirmed' in result.final_text
            assert (await runner.runtime.client.inspect(result.job_ref, actor=runner.actor)).resource_state == 'unknown'
            assert (await runner.send('new input')).status == 'completed'
            assert backend.launches == 1 and len(controls[0].calls) == 1
        finally:
            tag, variable = controls[0].calls[0]
            assert controls[0].original.stop_tagged(tag, variable=variable, grace_s=0).confirmed
        await _gone(pid)



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


@pytest.mark.parametrize('exited', [True, False])
@pytest.mark.parametrize('first_read', [True, False])
def test_membership_read_denied_requires_original_pidfd_exit(monkeypatch, exited, first_read):
    from types import SimpleNamespace

    reads, waited, closed = [], [], []
    monkeypatch.setattr(os, 'listdir', lambda _path: ['42'])
    monkeypatch.setattr(os, 'getpid', lambda: 100)
    monkeypatch.setattr(os, 'pidfd_open', lambda pid: 77)
    monkeypatch.setattr(os, 'stat', lambda path: SimpleNamespace(st_uid=os.geteuid()))
    monkeypatch.setattr(os, 'close', closed.append)

    def environment(*args, **kwargs):
        reads.append(args[0])
        if not first_read and len(reads) == 1:
            return io.BytesIO(b'PL_BG_TAG=original\0')
        raise PermissionError('original is exiting or no longer readable')

    def wait_original(fds, writes, errors, timeout):
        waited.append((fds, writes, errors, timeout))
        return ([77] if exited else [], [], [])

    monkeypatch.setattr(builtins, 'open', environment)
    monkeypatch.setattr(tagged.select, 'select', wait_original)
    if exited:
        assert tagged.signal_members('PL_BG_TAG', 'original', signal.SIGKILL) == []
    else:
        with pytest.raises(PermissionError):
            tagged.signal_members('PL_BG_TAG', 'original', signal.SIGKILL)
    assert waited == [([77], [], [], 0.05)]
    assert closed == [77]


async def test_control_factory_requires_explicit_interface_before_launch(tmp_path):
    async with _command(tmp_path, 'touch forbidden', control=lambda *a, **kw: object()) as (runner, ref):
        result = await _result(runner, ref)
        assert result.status == 'failed' and 'ShellProcessControl' in result.final_text
        assert not (tmp_path/'forbidden').exists()



async def test_timeout_kills_detached_process_and_drain_is_bounded(tmp_path, record_property):
    pidfile = tmp_path/'child'
    code = 'import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
    code += f'open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(60)'
    command = 'setsid ' + shlex.join([sys.executable, '-c', code]) + ' & wait'
    async with _command(tmp_path, command, timeout=1) as (runner, ref):
        pid = await _file(pidfile)
        record_property('owned_process_pid', pid)
        result = await asyncio.wait_for(_result(runner, ref), 8)
        assert result.status == 'failed' and result.error == 'TimeoutError'
        await _gone(pid)






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
from power_loop.runtime.process_control import OwnedProcess
import os
async def main():
    process = OwnedProcess(["/bin/bash", "-c", {command!r}], cwd={str(tmp_path)!r}, env=os.environ.copy())
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
        with runtime_env_context(RuntimeEnv(process_control=lambda *a, **kw: (_ for _ in ()).throw(OSError("binding unavailable")))):
            dt.BashSession(tmp_path)
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)





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
    monkeypatch.setattr(os, 'pidfd_open', lambda pid: 77)
    monkeypatch.setattr(os, 'close', lambda fd: None)
    monkeypatch.setattr(tagged.select, 'select', lambda *args: ([], [], []))

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
