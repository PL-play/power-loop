"""Opt-in capture of normalized terminal output before bash display truncation.

This preserves every line delivered by the persistent terminal reader, not raw PTY
bytes (ANSI cleanup and terminal line normalization still apply). No capture by default.
"""
from contextvars import ContextVar
from pathlib import Path

BASH_CAPTURE_MAX_BYTES = 2 * 1024 * 1024

current_bash_capture: ContextVar['BashOutputCapture | None'] = ContextVar('bash_output_capture', default=None)


class BashOutputCapture:
    def __init__(self, path: Path, *, max_bytes: int = BASH_CAPTURE_MAX_BYTES):
        self.path = Path(path)
        self.max_bytes = max(1, int(max_bytes))
        self.error: str | None = None
        self.lines = 0
        self.bytes_written = 0
        self.truncated = False
        self.dropped_bytes = 0

    def __enter__(self):
        self.stream = self.path.open('x', encoding='utf-8')
        self.token = current_bash_capture.set(self)
        return self

    def write_line(self, line: str) -> None:
        if self.error:
            return
        payload = (line + '\n').encode('utf-8', errors='replace')
        if self.truncated:
            self.dropped_bytes += len(payload)
            return
        remaining = self.max_bytes - self.bytes_written
        if remaining <= 0:
            self.truncated = True
            self.dropped_bytes += len(payload)
            return
        if len(payload) > remaining:
            clipped = payload[:remaining].decode('utf-8', errors='ignore')
            clipped_len = len(clipped.encode('utf-8'))
            try:
                if clipped:
                    self.stream.write(clipped)
                    self.bytes_written += clipped_len
                    self.lines += 1
            except OSError as exc:
                # Never interrupt command draining and leave a shell running with stale output.
                self.error = type(exc).__name__
                return
            self.truncated = True
            self.dropped_bytes += len(payload) - clipped_len
            return
        try:
            self.stream.write(line + '\n')
            self.bytes_written += len(payload)
            self.lines += 1
        except OSError as exc:
            # Never interrupt command draining and leave a shell running with stale output.
            self.error = type(exc).__name__

    def read_text(self) -> str:
        with self.path.open('rb') as fh:
            data = fh.read(self.max_bytes)
        text = data.decode('utf-8', errors='replace')
        if self.truncated:
            return (
                f"{text}\n... (bash capture truncated at {self.max_bytes} bytes; "
                f"{self.dropped_bytes} bytes omitted)"
            )
        return text

    def __exit__(self, *exc):
        current_bash_capture.reset(self.token)
        try:
            self.stream.close()
        except OSError as error:
            self.error = type(error).__name__
