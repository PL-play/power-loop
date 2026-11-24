"""One private shell process, stopped through its original backend target."""
import codecs
import os
import select
import shlex
import subprocess
import threading
import time
from collections import deque

from power_loop.core.agent_context import get_current_cancel_token, get_current_loop
from power_loop.runtime.env import get_runtime_env
from power_loop.runtime.exec_backend import DEFAULT_SHELL_BACKEND
from power_loop.runtime.process_control import OwnedProcess, bind_shell_control, stop_shell_target

BACKGROUND_SHELL_MAX_OUTPUT_BYTES = 2 * 1024 * 1024


class _BoundedStreamReader:
    """Drains a text stream to EOF while retaining at most `max_bytes` UTF-8 bytes."""

    def __init__(self, stream, max_bytes: int = BACKGROUND_SHELL_MAX_OUTPUT_BYTES) -> None:
        self._stream = stream
        self._max_bytes = max(1, int(max_bytes))
        self._head_budget = self._max_bytes // 2
        self._tail_budget = self._max_bytes - self._head_budget
        self._lock = threading.Lock()
        self._head: list[str] = []
        self._head_bytes = 0
        self._head_closed = False
        self._tail: deque[tuple[str, int]] = deque()
        self._tail_bytes = 0
        self.truncated = False
        self.omitted_bytes = 0
        self.eof = stream is None
        self._stop_event = threading.Event()
        self._eof_event = threading.Event()
        if self.eof:
            self._eof_event.set()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self._stream is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _append_text(self, chunk: str) -> None:
        if not chunk:
            return
        raw = chunk.encode("utf-8", errors="replace")
        size = len(raw)
        with self._lock:
            if not self._head_closed:
                remaining_head = self._head_budget - self._head_bytes
                if size <= remaining_head:
                    self._head.append(chunk)
                    self._head_bytes += size
                    return
                head_text = raw[:remaining_head].decode("utf-8", errors="ignore")
                head_len = len(head_text.encode("utf-8"))
                if head_len:
                    self._head.append(head_text)
                    self._head_bytes += head_len
                # Once any byte spills past the head budget, never append later
                # chunks back into _head or output order would invert.
                self._head_closed = True
                raw = raw[head_len:]
                chunk = raw.decode("utf-8", errors="replace")
                size = len(raw)
                if not size:
                    return
            if size > self._tail_budget:
                chunk = raw[-self._tail_budget :].decode("utf-8", errors="ignore")
                new_size = len(chunk.encode("utf-8"))
                self.omitted_bytes += size - new_size
                size = new_size
                self.truncated = True
            self._tail.append((chunk, size))
            self._tail_bytes += size
            while self._tail_bytes > self._tail_budget and len(self._tail) > 1:
                _, dropped_size = self._tail.popleft()
                self._tail_bytes -= dropped_size
                self.omitted_bytes += dropped_size
                self.truncated = True

    def _run_fd(self, fd: int) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        poller = select.poll()
        poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        while not self._stop_event.is_set():
            try:
                events = poller.poll(50)
            except (OSError, ValueError):
                break
            if not events:
                continue
            try:
                data = os.read(fd, 16384)
            except OSError:
                break
            if not data:
                tail_text = decoder.decode(b"", final=True)
                self._append_text(tail_text)
                self.eof = True
                self._eof_event.set()
                return
            self._append_text(decoder.decode(data, final=False))

    def _run(self) -> None:
        fd: int | None = None
        fileno = getattr(self._stream, "fileno", None)
        if callable(fileno):
            try:
                fd = fileno()
            except (OSError, ValueError):
                fd = None
        if fd is not None:
            self._run_fd(fd)
            return
        try:
            while not self._stop_event.is_set():
                chunk = self._stream.read(4096)
                if not chunk:
                    self.eof = True
                    self._eof_event.set()
                    return
                self._append_text(chunk)
        except (OSError, ValueError):
            pass

    def wait_eof(self, timeout: float | None = None) -> bool:
        return self._eof_event.wait(timeout=timeout)

    def join(self, timeout: float | None = None) -> bool:
        if not self._thread.is_alive():
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def text(self) -> str:
        with self._lock:
            head = "".join(self._head)
            tail = "".join(chunk for chunk, _ in self._tail)
            if not self.truncated:
                return head + tail
            return f"{head}\n... ({self.omitted_bytes} bytes omitted) ...\n{tail}"


def _stop_readers(readers: tuple[_BoundedStreamReader, ...], timeout: float = 0.2) -> None:
    for reader in readers:
        reader.stop()
    deadline = time.monotonic() + timeout
    for reader in readers:
        reader.join(timeout=max(0.0, deadline - time.monotonic()))


def _close_process_streams(process) -> None:
    proc = getattr(process, "process", None)
    if proc is None:
        return
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def drain_original(process, *, readers: tuple[_BoundedStreamReader, ...] = ()):
    try:
        process.wait(timeout=2)
        if not readers:
            process.drain_stopped_output(timeout=2)
            return
        deadline = time.monotonic() + 2
        for reader in readers:
            remaining = max(0.0, deadline - time.monotonic())
            if not reader.wait_eof(timeout=remaining) or not reader.join(timeout=max(0.0, deadline - time.monotonic())):
                raise subprocess.TimeoutExpired("original producer pipes", 2)
    except subprocess.TimeoutExpired as exc:
        from power_loop.runtime.task_runtime import mark_unconfirmed_resource
        mark_unconfirmed_resource()
        raise RuntimeError("original_shell_stop_unconfirmed") from exc
    finally:
        if readers:
            _stop_readers(readers)
            _close_process_streams(process)


def execute(command, timeout=None, **kwargs):
    from .command_policy import command_policy_reason
    from .default_tools import _dangerous_command_reason, _truncate_chars, _validate_bash_command_scope
    env = get_runtime_env()
    error = (_dangerous_command_reason(command, env.command_rules)
        or command_policy_reason(command, env.blocked_command_categories, env.command_rules)
        or _validate_bash_command_scope(command))
    if error:
        raise ValueError(error)
    context = get_current_loop()._runtime_worker_context
    token = get_current_cancel_token()
    token.raise_if_cancelled()
    workspace = env.require_workspace_dir()
    backend = env.shell_backend or DEFAULT_SHELL_BACKEND
    argv, cwd, launch_env = backend.launch_argv(workspace), backend.launch_cwd(workspace), backend.launch_env(workspace)
    target = bind_shell_control(backend, workspace, argv=argv, cwd=cwd, env=launch_env, factory=env.process_control)
    process = OwnedProcess(argv, cwd=cwd, env=launch_env)
    tag = context.session_id + "_" + str(context.send_index)
    payload = "exec env POWER_LOOP_BACKGROUND_SESSION=" + tag + " /bin/bash --norc --noprofile -c " + shlex.quote(command) + "\n"
    deadline = time.monotonic() + (env.background_timeout_s if timeout is None else timeout)
    stream_cap = max(1, BACKGROUND_SHELL_MAX_OUTPUT_BYTES // 2)
    stdout_reader = _BoundedStreamReader(process.process.stdout, max_bytes=stream_cap)
    stderr_reader = _BoundedStreamReader(process.process.stderr, max_bytes=stream_cap)
    readers = (stdout_reader, stderr_reader)
    stdout_reader.start()
    stderr_reader.start()
    if process.process.stdin is not None:
        try:
            process.process.stdin.write(payload)
            process.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.process.stdin.close()
            except OSError:
                pass
    try:
        while True:
            if token.is_cancelled() or time.monotonic() >= deadline:
                process.request_stop()
                stopped = stop_shell_target(target, tag, variable="POWER_LOOP_BACKGROUND_SESSION", grace_s=0.1)
                if not stopped.confirmed:
                    from power_loop.runtime.task_runtime import mark_unconfirmed_resource
                    mark_unconfirmed_resource()
                drain_original(process, readers=readers)
                if not stopped.confirmed:
                    raise RuntimeError("original_shell_stop_unconfirmed")
                token.raise_if_cancelled()
                raise TimeoutError("background shell timed out")
            wait_s = min(0.05, max(0.0, deadline - time.monotonic()))
            if process.poll() is None:
                try:
                    process.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    continue
            if all(reader.eof for reader in readers):
                for reader in readers:
                    reader.join(timeout=0.2)
                return _truncate_chars(
                    f"exit_code={process.returncode}\n{stdout_reader.text()}{stderr_reader.text()}"
                )
            for reader in readers:
                if not reader.eof:
                    reader.wait_eof(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
                    break
    finally:
        if process.poll() is None:
            process.request_stop()
            drain_original(process, readers=readers)
        else:
            _stop_readers(readers)
            _close_process_streams(process)
