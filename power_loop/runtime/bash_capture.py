"""Opt-in capture of normalized terminal output before bash display truncation.

This preserves every line delivered by the persistent terminal reader, not raw PTY
bytes (ANSI cleanup and terminal line normalization still apply). No capture by default.
"""
from contextvars import ContextVar
from pathlib import Path

current_bash_capture: ContextVar['BashOutputCapture | None'] = ContextVar('bash_output_capture', default=None)


class BashOutputCapture:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.error: str | None = None
        self.lines = 0

    def __enter__(self):
        self.stream = self.path.open('x', encoding='utf-8')
        self.token = current_bash_capture.set(self)
        return self

    def write_line(self, line: str) -> None:
        if self.error:
            return
        try:
            self.stream.write(line + '\n')
            self.lines += 1
        except OSError as exc:
            # Never interrupt command draining and leave a shell running with stale output.
            self.error = type(exc).__name__

    def __exit__(self, *exc):
        current_bash_capture.reset(self.token)
        try:
            self.stream.close()
        except OSError as error:
            self.error = type(error).__name__
