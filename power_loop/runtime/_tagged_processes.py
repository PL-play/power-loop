"""Standalone stdlib-only control helper, executed inside the original target.

Open a pidfd before checking membership. Even if /proc/PID later refers to a
replacement, sending via the fd cannot signal that replacement. Tags identify
cooperative command descendants; they are not a sandbox for hostile programs.
"""
from __future__ import annotations

import json
import os
import select
import signal
import sys
import time


def process_info(name):
    with open('/proc/' + name + '/stat') as stream:
        fields = stream.read().rpartition(')')[2].split()
        return int(fields[19]), fields[0]


def start_ticks(name):
    return process_info(name)[0]


def probe():
    fd = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)
    with open('/proc/sys/kernel/random/boot_id') as stream:
        boot_id = stream.read().strip()
    return {'boot_id': boot_id, 'not_before': start_ticks('self')}


def _reap_ready_handles(handles):
    if not handles:
        return
    ready, _, _ = select.select(list(handles.values()), [], [], 0)
    if not ready:
        return
    ready_set = set(ready)
    for pid, fd in list(handles.items()):
        if fd in ready_set:
            del handles[pid]
            os.close(fd)


def signal_members(variable, tag, sig, not_before=0, handles=None):
    needle = (variable + '=' + tag).encode()
    owned = handles is None
    tracked = {} if owned else handles
    transferred = False
    signalled = 0
    try:
        # First re-signal already-tracked live pidfds without reopening them.
        _reap_ready_handles(tracked)
        for pid, existing_fd in list(tracked.items()):
            try:
                signal.pidfd_send_signal(existing_fd, sig)
                signalled += 1
            except ProcessLookupError:
                del tracked[pid]
                os.close(existing_fd)
        for name in os.listdir('/proc'):
            if not name.isdigit():
                continue
            pid = int(name)
            if pid == os.getpid() or pid in tracked:
                continue
            fd = None
            matched = False
            try:
                # The original probe predates command creation. Older processes
                # cannot be its descendants, including unreadable host services.
                if not_before:
                    started, state = process_info(name)
                    if started < not_before or state in ('Z', 'X', 'x'):
                        continue
                # Preliminary filtering avoids opening every unrelated process.
                with open('/proc/' + name + '/environ', 'rb') as stream:
                    if needle not in stream.read().split(b'\0'):
                        continue
                matched = True
                fd = os.pidfd_open(pid)
                # Recheck AFTER acquiring the immutable kernel handle.
                with open('/proc/' + name + '/environ', 'rb') as stream:
                    if needle not in stream.read().split(b'\0'):
                        continue
                signal.pidfd_send_signal(fd, sig)
                tracked[pid] = fd
                fd = None
                signalled += 1
            except (ProcessLookupError, FileNotFoundError):
                pass
            except PermissionError:
                if not matched:
                    try:
                        same_user = os.stat('/proc/' + name).st_uid == os.geteuid()
                    except FileNotFoundError:
                        continue
                    if not same_user:
                        continue
                # /proc/PID/environ can become unreadable during exit, before
                # either membership read. An immutable pidfd exit notification
                # proves absence; a live unreadable process remains unknown.
                if fd is None:
                    try:
                        fd = os.pidfd_open(pid)
                    except (ProcessLookupError, FileNotFoundError):
                        continue
                if not select.select([fd], [], [], 0.05)[0]:
                    raise
            finally:
                if fd is not None:
                    os.close(fd)
        transferred = True
        return list(tracked.values()) if owned else signalled
    finally:
        if owned and not transferred:
            for fd in tracked.values():
                os.close(fd)



def stop(variable, tag, grace_s, *, not_before=0, boot_id=None):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        return {'confirmed': False, 'error': 'pidfd_unavailable'}
    if not variable or not tag or '=' in variable or '\0' in variable + tag:
        return {'confirmed': False, 'error': 'invalid_process_tag'}
    handles = {}
    try:
        if boot_id is not None and probe()['boot_id'] != boot_id:
            return {'confirmed': False, 'error': 'original_target_boot_changed'}
        if signal_members(variable, tag, signal.SIGTERM, not_before, handles=handles):
            time.sleep(max(0, min(grace_s, 5)))
        deadline = time.monotonic() + 2
        # Re-scan for descendants forked before their original parents stopped,
        # reusing existing pidfds and closing exited handles promptly.
        while time.monotonic() < deadline:
            signalled = signal_members(variable, tag, signal.SIGKILL, not_before, handles=handles)
            _reap_ready_handles(handles)
            if not signalled and not handles:
                return {'confirmed': True, 'error': None}
            if not signalled:
                while handles and time.monotonic() < deadline:
                    ready, _, _ = select.select(list(handles.values()), [], [], 0.05)
                    ready_set = set(ready)
                    for pid, fd in list(handles.items()):
                        if fd in ready_set:
                            del handles[pid]
                            os.close(fd)
                return {'confirmed': not handles, 'error': 'exit_unconfirmed' if handles else None}
            time.sleep(0.02)
        return {'confirmed': False, 'error': 'processes_still_running'}
    except (OSError, ValueError):
        return {'confirmed': False, 'error': 'process_control_failed'}
    finally:
        for fd in list(handles.values()):
            os.close(fd)
        handles.clear()


if __name__ == '__main__':
    if sys.argv[1] == '--probe':
        print(json.dumps(probe()))
    else:
        print(json.dumps(stop(sys.argv[1], sys.argv[2], float(sys.argv[3]),
            not_before=int(sys.argv[4]), boot_id=sys.argv[5])))
