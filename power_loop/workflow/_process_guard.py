"""POSIX process-group owner, launched as a standalone stdlib-only script.

Only this live group leader signals its own group. The caller owns a pipe,
never a reusable PID/PGID cancellation capability. Worker stdout stays intact;
the guard's final SIGKILL exit is expected, including after a successful frame.
This is not containment for commands that create another session or remote work.
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time


def _owned_group() -> int:
    # Our live PID pins this session/group identity throughout the syscall.
    pid = os.getpid()
    if os.getpgrp() != pid or os.getsid(0) != pid:
        raise RuntimeError("worker guard does not own its process group")
    return pid


def _signal_owned_group(sig: int) -> None:
    os.killpg(_owned_group(), sig)


def main() -> int:
    control_fd, grace = int(sys.argv[1]), float(sys.argv[2])
    command = sys.argv[3:]
    if not command or grace < 0:
        raise ValueError("invalid worker guard arguments")
    _owned_group()
    cancelled = False

    def request_stop(_sig, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def pipe_closed(timeout: float) -> bool:
        ready, _, _ = select.select([control_fd], [], [], timeout)
        return bool(ready) and not os.read(control_fd, 4096)

    # Parent may have died while Python was starting; do not launch new work.
    if pipe_closed(0) or cancelled:
        return 0
    # close_fds keeps the control endpoint out of workers and their descendants.
    child = subprocess.Popen(command, close_fds=True)
    try:
        while child.poll() is None and not cancelled:
            if pipe_closed(0.05):
                cancelled = True
        if cancelled:
            _signal_owned_group(signal.SIGTERM)
            deadline = time.monotonic() + grace
            # Keep the group leader alive even if the direct child exits first:
            # grandchildren may ignore TERM and still own stdout or side effects.
            while time.monotonic() < deadline:
                child.poll()
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    finally:
        # A finished worker can leave descendants holding pipes. Kill the group
        # while we still pin its identity, even on a local supervision error.
        _signal_owned_group(signal.SIGKILL)
    return 0  # unreachable after a successful killpg


if __name__ == "__main__":
    raise SystemExit(main())
