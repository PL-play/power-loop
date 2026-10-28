"""Original stop evidence must not inherit an already-spent pipe deadline."""
import io
import os
import shlex
import subprocess
import time
from types import SimpleNamespace

import pytest

from power_loop.runtime.process_control import OwnedProcess

pytestmark = pytest.mark.unit


def test_stopped_producer_can_drain_after_original_cancel_deadline(tmp_path, record_property):
    proc = OwnedProcess(['/bin/bash', '--norc', '--noprofile'], cwd=tmp_path, env=os.environ.copy())
    record_property('owned_process_pid', proc.pid)
    try:
        ready = tmp_path / 'output-ready'
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(input='echo original-output; touch '+shlex.quote(str(ready))+'; exec sleep 60\n', timeout=.1)
        deadline = time.monotonic()+5
        while not ready.exists():
            assert time.monotonic() < deadline, 'original producer did not emit its output'
            with pytest.raises(subprocess.TimeoutExpired):
                proc.communicate(timeout=.05)
        proc.request_stop()
        proc.wait(timeout=3)
        # Model target cleanup taking longer than the original drain budget,
        # without spending a real second sleeping in every test run.
        proc._stop_at = time.monotonic()-2
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=1)
        output, error = proc.drain_stopped_output(timeout=1)
        assert output == 'original-output\n' and not error
        assert all(s.closed for s in (proc.process.stdin, proc.process.stdout, proc.process.stderr))
    finally:
        proc.request_stop()
        proc.wait(timeout=3)
        for stream in (proc.process.stdin, proc.process.stdout, proc.process.stderr):
            stream.close()


def test_external_writer_cannot_hold_final_drain_or_leave_descriptors_open():
    called = []
    def unfinished(*, timeout):
        called.append(timeout)
        raise subprocess.TimeoutExpired('original producer pipes', timeout)
    proc = OwnedProcess.__new__(OwnedProcess)
    streams = [io.StringIO() for _ in range(3)]
    proc.process = SimpleNamespace(poll=lambda: 0, communicate=unfinished,
        stdin=streams[0], stdout=streams[1], stderr=streams[2])
    with pytest.raises(subprocess.TimeoutExpired):
        proc.drain_stopped_output(timeout=.25)
    assert called == [.25] and all(s.closed for s in streams)


def test_live_producer_is_not_mistaken_for_final_output_collection():
    proc = OwnedProcess.__new__(OwnedProcess)
    proc.process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(RuntimeError, match='has not exited'):
        proc.drain_stopped_output()


@pytest.mark.parametrize('timeout', [None, 0, -1, float('inf'), float('nan'), True, 4, '1'])
def test_final_drain_cannot_be_unbounded(timeout):
    proc = OwnedProcess.__new__(OwnedProcess)
    with pytest.raises(ValueError, match='finite budget'):
        proc.drain_stopped_output(timeout)
