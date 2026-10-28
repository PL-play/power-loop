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


def signal_members(variable, tag, sig, not_before=0):
    needle = (variable + '=' + tag).encode()
    handles = []
    transferred = False
    try:
        for name in os.listdir('/proc'):
            if not name.isdigit() or int(name) == os.getpid():
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
                fd = os.pidfd_open(int(name))
                # Recheck AFTER acquiring the immutable kernel handle.
                with open('/proc/' + name + '/environ', 'rb') as stream:
                    if needle not in stream.read().split(b'\0'):
                        continue
                signal.pidfd_send_signal(fd, sig)
                handles.append(fd)
                fd = None
            except (ProcessLookupError, FileNotFoundError):
                pass
            except PermissionError:
                # A matching process we could open but cannot signal is unknown.
                if matched:
                    raise
                try:
                    same_user = os.stat('/proc/' + name).st_uid == os.geteuid()
                except FileNotFoundError:
                    same_user = False
                if same_user:
                    raise
            finally:
                if fd is not None:
                    os.close(fd)
        transferred = True
        return handles
    finally:
        if not transferred:
            for fd in handles:
                os.close(fd)



def stop(variable, tag, grace_s, *, not_before=0, boot_id=None):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        return {'confirmed': False, 'error': 'pidfd_unavailable'}
    if not variable or not tag or '=' in variable or '\0' in variable + tag:
        return {'confirmed': False, 'error': 'invalid_process_tag'}
    handles = []
    try:
        if boot_id is not None and probe()['boot_id'] != boot_id:
            return {'confirmed': False, 'error': 'original_target_boot_changed'}
        handles.extend(signal_members(variable, tag, signal.SIGTERM, not_before))
        if handles:
            time.sleep(max(0, min(grace_s, 5)))
        deadline = time.monotonic() + 2
        # Re-scan for descendants forked before their original parents stopped.
        while time.monotonic() < deadline:
            current = signal_members(variable, tag, signal.SIGKILL, not_before)
            handles.extend(current)
            if not current:
                waiting = list(handles)
                while waiting and time.monotonic() < deadline:
                    ready, _, _ = select.select(waiting, [], [], 0.05)
                    waiting = [fd for fd in waiting if fd not in ready]
                return {'confirmed': not waiting, 'error': 'exit_unconfirmed' if waiting else None}
            time.sleep(0.02)
        return {'confirmed': False, 'error': 'processes_still_running'}
    except (OSError, ValueError):
        return {'confirmed': False, 'error': 'process_control_failed'}
    finally:
        for fd in handles:
            os.close(fd)


if __name__ == '__main__':
    if sys.argv[1] == '--probe':
        print(json.dumps(probe()))
    else:
        print(json.dumps(stop(sys.argv[1], sys.argv[2], float(sys.argv[3]),
            not_before=int(sys.argv[4]), boot_id=sys.argv[5])))
