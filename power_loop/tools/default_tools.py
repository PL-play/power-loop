from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
import logging
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from power_loop.core.agent_context import get_ctx
from power_loop.runtime.env import get_runtime_env, safe_path
from power_loop.runtime.exec_backend import DEFAULT_SHELL_BACKEND, ShellBackend
from power_loop.runtime.human_input import request_user_input
from power_loop.runtime.runtime_state import get_tool_runtime_context
from power_loop.runtime.skills import get_default_loader
from power_loop.tools.command_policy import command_policy_reason

logger = logging.getLogger(__name__)

RESULT_MAX_CHARS = 50000
TEXT_FILE_MAX_BYTES = 5 * 1024 * 1024
READ_DEFAULT_LIMIT = 2000
READ_MAX_LIMIT = 20000
SEARCH_MAX_RESULTS = 500
GLOB_MAX_RESULTS = 500
SENTINEL_PREFIX = "___POWER_LOOP_CMD_DONE_"

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COMMON_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

# Process-global optimistic read-before-write stamps. Bounded (LRU) so a long-lived
# process that reads many distinct files can't grow it without limit (C9).
_MAX_FILE_READ_STATE = 4096
FILE_READ_STATE: OrderedDict[str, tuple[int, int]] = OrderedDict()
_FILE_READ_STATE_LOCK = threading.Lock()


def _file_stamp(fp: Path) -> tuple[int, int]:
    stat = fp.stat()
    return stat.st_mtime_ns, stat.st_size


def _display_path(path: Path) -> str:
    runtime_env = get_runtime_env()
    resolved = path.resolve()
    # Compare against RESOLVED roots: ``resolved`` has symlinks/.. collapsed, so an
    # unresolved (symlinked) workspace_dir would otherwise fail is_relative_to and
    # render an absolute path for files that are in fact inside the workspace.
    ws = runtime_env.workspace_dir.resolve() if runtime_env.workspace_dir is not None else None
    home = runtime_env.home_dir.resolve() if runtime_env.home_dir is not None else None
    if ws is not None and resolved.is_relative_to(ws):
        return str(resolved.relative_to(ws))
    if home is not None and resolved.is_relative_to(home):
        return f"@home/{resolved.relative_to(home)}"
    return str(resolved)


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _truncate_chars(text: str, limit: int = RESULT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head - 80
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... ({omitted} chars omitted) ...\n\n{text[-tail:]}"


# In-memory caps for persistent-bash output accumulation (default-tools-4). Both exceed
# _truncate_lines' 30+170 display window, so capping here never changes the rendered output.
_BASH_HEAD_KEEP = 256
_BASH_TAIL_KEEP = 2048


def _truncate_lines(lines: list[str], head: int = 30, tail: int = 170) -> str:
    limit = head + tail
    if len(lines) <= limit:
        return "\n".join(lines)
    omitted = len(lines) - limit
    return (
        "\n".join(lines[:head])
        + f"\n\n... ({omitted} lines omitted) ...\n\n"
        + "\n".join(lines[-tail:])
    )


def _clean_terminal_line(line: str) -> str:
    return _ANSI_RE.sub("", line).rstrip("\r")


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    if not data:
        return False
    sample = data[:4096]
    # Fast path: mostly-ASCII text (incl. common whitespace) is clearly text.
    ascii_ish = sum(byte in b"\n\r\t\b\f" or 32 <= byte <= 126 for byte in sample)
    if ascii_ish / len(sample) >= 0.70:
        return False
    # Otherwise it may be NON-Latin UTF-8 text (CJK / emoji, whose bytes are mostly >126) —
    # NOT binary. Decode leniently (a multibyte char truncated at the 4KB sample boundary is
    # negligible) and flag as binary only if the text is littered with undecodable bytes
    # (U+FFFD) or C0/C1 control chars (excluding normal whitespace), which real text lacks.
    text = sample.decode("utf-8", errors="replace")
    bad = sum(
        1
        for ch in text
        if ch == "�" or ord(ch) == 0x7f or (ord(ch) < 32 and ch not in "\n\r\t\f\b")
    )
    return bad / len(text) > 0.15


def _read_text(fp: Path, *, max_bytes: int | None = TEXT_FILE_MAX_BYTES) -> str:
    if not fp.exists():
        raise FileNotFoundError(fp)
    if fp.is_dir():
        raise IsADirectoryError(fp)
    size = fp.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise ValueError(
            f"File is too large to read as text ({size} bytes > {max_bytes} bytes): {_display_path(fp)}. "
            "Use read_file with offset/limit for previews or grep/glob for targeted search."
        )
    data = fp.read_bytes()
    if _looks_binary(data):
        raise ValueError(f"Refusing to read binary-looking file as text: {_display_path(fp)}")
    return data.decode("utf-8", errors="replace")


def _remember_read(fp: Path) -> None:
    key = str(fp.resolve())
    with _FILE_READ_STATE_LOCK:
        FILE_READ_STATE[key] = _file_stamp(fp)
        FILE_READ_STATE.move_to_end(key)
        while len(FILE_READ_STATE) > _MAX_FILE_READ_STATE:
            FILE_READ_STATE.popitem(last=False)  # evict least-recently-read


def _check_read_state(fp: Path) -> str | None:
    key = str(fp.resolve())
    stamp = FILE_READ_STATE.get(key)  # .get(): tolerate a concurrent LRU eviction
    if stamp is None:
        return f"Error: File has not been read yet. Use read_file first before modifying: {_display_path(fp)}"
    if fp.exists() and _file_stamp(fp) != stamp:
        return f"Error: File changed since last read. Re-read it before modifying: {_display_path(fp)}"
    return None


def _detect_line_ending(content: str) -> str:
    """Pick the file's dominant line ending by COUNT (not first occurrence).

    Returns ``\\r\\n``, ``\\r`` (classic Mac), or ``\\n``. The old first-occurrence
    heuristic flattened a CRLF file to LF whenever a stray lone-LF appeared before
    the first CRLF, and never recognized CR-only files. A truly mixed file still
    normalizes to its dominant ending (the edit tools normalize→edit→restore a
    single ending; per-line fidelity for mixed files is out of scope)."""
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf      # lone LF (each \r\n contributes one \n)
    cr = content.count("\r") - crlf      # lone CR (each \r\n contributes one \r)
    if crlf >= lf and crlf >= cr and crlf > 0:
        return "\r\n"
    if cr > lf:
        return "\r"
    return "\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(text: str, ending: str) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    if ending == "\r":
        return text.replace("\n", "\r")
    return text


def _split_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def _normalized_for_diagnostic(text: str) -> str:
    result = _normalize_to_lf(text)
    result = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", result)
    result = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', result)
    result = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", result)
    result = re.sub(r"[\u00a0\u2002-\u200a\u202f\u205f\u3000]", " ", result)
    return result


def _line_numbers_for_offsets(content: str, needle: str, max_positions: int = 5) -> list[str]:
    positions: list[str] = []
    start = 0
    while len(positions) < max_positions:
        idx = content.find(needle, start)
        if idx == -1:
            break
        positions.append(str(content[:idx].count("\n") + 1))
        start = idx + max(1, len(needle))
    return positions


def _generate_diff(old_content: str, new_content: str, context: int = 3) -> str:
    diff = difflib.unified_diff(
        old_content.splitlines(),
        new_content.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
        n=context,
    )
    return "\n".join(diff)


def _is_hidden_path(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return any(part.startswith(".") and part not in (".", "..") for part in rel_parts)


def _pattern_mentions_hidden(pattern: str) -> bool:
    return any(part.startswith(".") for part in Path(pattern).parts)


def _should_skip_dir(path: Path, root: Path, include_hidden: bool, pattern: str = "") -> bool:
    name = path.name
    if name in _COMMON_SKIP_DIRS:
        return True
    return not include_hidden and not _pattern_mentions_hidden(pattern) and _is_hidden_path(path, root)


def _safe_relative_for_subprocess(path: Path) -> str:
    # Resolve the workspace too (see safe_path): comparing a resolved candidate
    # against an unresolved/symlinked workspace_dir would wrongly fall through to
    # the absolute path for an in-workspace target.
    workspace_dir = get_runtime_env().require_workspace_dir().resolve()
    resolved = path.resolve()
    if resolved.is_relative_to(workspace_dir):
        rel = resolved.relative_to(workspace_dir)
        return "." if str(rel) == "." else rel.as_posix()
    return str(resolved)


_BASH_WRITE_HINTS = (
    " >",
    ">>",
    " tee ",
    " sed -i",
    " rm ",
    " mv ",
    " cp ",
    " touch ",
    " mkdir ",
)
_BASH_READ_HINTS = (
    " cat ",
    " less ",
    " head ",
    " tail ",
    " grep ",
    " rg ",
    " find ",
)


def _path_referenced(haystack_lower: str, target_lower: str) -> bool:
    """True iff ``target_lower`` appears in ``haystack_lower`` as a PATH — itself or ``target/<sub>``
    — not merely as a substring of a longer path component (default-tools-3): home ``/var/lib/agent``
    must NOT match a sibling dir ``/var/lib/agent-extra``. A match requires the next char to be a
    path separator or a shell delimiter (or end-of-string)."""
    if not target_lower:
        return False
    idx = haystack_lower.find(target_lower)
    while idx != -1:
        after = haystack_lower[idx + len(target_lower): idx + len(target_lower) + 1]
        if after == "" or after in "/\\ \t\n\r'\"`;:|&()<>":
            return True
        idx = haystack_lower.find(target_lower, idx + 1)
    return False


def _is_agent_path_allowed_for_bash(command: str) -> bool:
    runtime_env = get_runtime_env()
    if runtime_env.home_dir is None:
        return True
    lowered = command.lower()
    if not _path_referenced(lowered, str(runtime_env.home_dir).lower()):
        return True
    return any(_path_referenced(lowered, str(path).lower()) for path in runtime_env.home_rw_allowlist)


def _validate_bash_command_scope(command: str) -> str | None:
    runtime_env = get_runtime_env()
    if runtime_env.home_dir is None:
        return None
    lowered = f" {command.lower()} "
    if not _path_referenced(lowered, str(runtime_env.home_dir).lower()):
        return None
    if _is_agent_path_allowed_for_bash(command):
        return None

    if any(hint in lowered for hint in _BASH_WRITE_HINTS):
        return (
            "Error: Writing under POWER_LOOP_HOME is blocked outside allowlisted paths (.cache/logs/skills). "
            "Use workspace files or allowlisted agent paths only."
        )
    if any(hint in lowered for hint in _BASH_READ_HINTS):
        return (
            "Error: Reading agent-home internals is blocked outside allowlisted paths (.cache/logs/skills). "
            "Use load_skill(name) for skill content instead of direct file reads."
        )
    # DEFAULT-DENY: the command references POWER_LOOP_HOME, is NOT under an allowlisted path, and
    # matched none of the verb hints above — but the hint lists are not exhaustive (awk / base64 / od
    # / python -c / dd of= / truncate / ln -s all reach agent-home undetected). Refuse rather than
    # fall through to "allow", since the resolved target is provably an un-allowlisted home path.
    return (
        "Error: Accessing POWER_LOOP_HOME is blocked outside allowlisted paths (.cache/logs/skills). "
        "Use workspace files or allowlisted agent paths only."
    )


def _dangerous_command_reason(command: str) -> str | None:
    lowered = command.strip().lower()
    compact = re.sub(r"\s+", " ", lowered)
    # Recursive/force rm targeting root, home, or a top-level SYSTEM directory.
    # The target alternatives below deliberately allow /tmp and relative paths
    # (scratch is fine) while blocking '/', '~', '$HOME', '~/…', '$HOME/…', and
    # '/etc', '/usr/local', '/var/…' etc. The flag group repeats so 'rm -r -f /x'
    # is caught too. (Older regex only matched the BARE root/home — it missed
    # 'rm -rf /etc', a real false-negative.)
    if re.search(
        r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+"
        r"(?:/(?:\s|$)"                                       # bare /
        r"|~(?:/|\s|$)"                                       # ~ or ~/…
        r"|\$home"                                           # $HOME or $HOME/…
        r"|/(?:bin|boot|dev|etc|home|lib|lib64|opt|proc|root|run|sbin|srv|sys|usr|var)(?:/|\s|$))",  # /sysdir…
        compact,
    ):
        return "refusing recursive deletion of a root/home/system path"
    if re.search(r">\s*/dev/(sd|disk|rdisk|nvme|zero|mem)", compact):
        return "refusing redirection to raw device paths"
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        tokens = command.split()
    commands = {Path(token).name for token in tokens if token and not token.startswith("-")}
    blocked = {
        "sudo",
        "su",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "mkfs",
        "diskutil",
        "dd",
    }
    used = sorted(commands & blocked)
    if used:
        return f"refusing privileged or device-level command: {', '.join(used)}"
    return None


class BashSession:
    """Persistent bash process with stdout/stderr merged through a pty.

    How the shell process is launched is delegated to a ``ShellBackend`` so a
    host can route execution into a sandbox; the default backend runs an
    in-process ``/bin/bash`` in ``cwd`` (original behavior).
    """

    def __init__(self, cwd: Path, backend: ShellBackend | None = None):
        self._cwd = cwd
        self._backend: ShellBackend = backend or DEFAULT_SHELL_BACKEND
        self._proc: subprocess.Popen[str] | None = None
        self._q: queue.Queue[str] = queue.Queue()
        self._master_fd: int | None = None
        self._lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        import pty

        master_fd, slave_fd = pty.openpty()
        try:
            import termios

            attrs = termios.tcgetattr(master_fd)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(master_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        env = self._backend.launch_env(self._cwd)

        self._proc = subprocess.Popen(
            self._backend.launch_argv(self._cwd),
            stdin=subprocess.PIPE,
            stdout=slave_fd,
            stderr=slave_fd,
            text=True,
            bufsize=0,
            cwd=self._backend.launch_cwd(self._cwd),
            env=env,
        )
        os.close(slave_fd)
        self._master_fd = master_fd
        self._q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        buf = ""
        fd = self._master_fd
        if fd is None:
            return
        try:
            while True:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                buf += data.decode("utf-8", errors="replace")
                while True:
                    idx_n = buf.find("\n")
                    idx_r = buf.find("\r")
                    if idx_n == -1 and idx_r == -1:
                        break
                    cut = min(i for i in (idx_n, idx_r) if i != -1)
                    line, buf = buf[:cut], buf[cut + 1 :]
                    if line:
                        self._q.put(line)
            if buf:
                self._q.put(buf)
        except Exception:
            pass

    def _drain_until(self, sentinel: str, timeout: int, idle_timeout: float = 5.0) -> tuple[list[str], str | None]:
        # Bound in-memory accumulation (default-tools-4): the displayed output keeps only
        # head+tail (via _truncate_lines), so a command emitting millions of lines must not buffer
        # them all. Keep the first _BASH_HEAD_KEEP fully + the last _BASH_TAIL_KEEP in a ring,
        # dropping the middle (both bounds comfortably exceed _truncate_lines' 30+170 window).
        head: list[str] = []
        tail: deque[str] = deque(maxlen=_BASH_TAIL_KEEP)
        dropped = 0
        exit_code: str | None = None
        deadline = time.monotonic() + timeout

        def _accumulate(line: str) -> None:
            nonlocal dropped
            if len(head) < _BASH_HEAD_KEEP:
                head.append(line)
                return
            if len(tail) == _BASH_TAIL_KEEP:
                dropped += 1
            tail.append(line)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = min(remaining, idle_timeout)
            try:
                raw_line = self._q.get(timeout=wait)
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    break
                continue

            line = _clean_terminal_line(raw_line)
            if sentinel in line:
                match = re.search(re.escape(sentinel) + r"\s+(-?\d+)", line)
                if match:
                    exit_code = match.group(1)
                break
            _accumulate(line)

        while not self._q.empty() and exit_code is None:
            try:
                line = _clean_terminal_line(self._q.get_nowait())
            except queue.Empty:
                break
            if sentinel in line:
                match = re.search(re.escape(sentinel) + r"\s+(-?\d+)", line)
                if match:
                    exit_code = match.group(1)
                break
            _accumulate(line)

        middle = [f"... ({dropped} line(s) omitted) ..."] if dropped else []
        return head + middle + list(tail), exit_code

    def execute(self, command: str, timeout: int = 120) -> str:
        reason = _dangerous_command_reason(command)
        if reason:
            return f"Error: Dangerous command blocked ({reason}). Run it manually if you really intend it."
        policy_err = command_policy_reason(command, get_runtime_env().blocked_command_categories)
        if policy_err:
            return policy_err

        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()

            assert self._proc is not None and self._proc.stdin is not None
            sentinel = f"{SENTINEL_PREFIX}{uuid.uuid4().hex}___"
            full_cmd = f"{command}\nprintf '\\n{sentinel} %s\\n' $?\n"
            try:
                self._proc.stdin.write(full_cmd)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._restart_locked()
                return "Error: Bash session crashed and was restarted. Please retry the command."

            lines, exit_code = self._drain_until(sentinel, timeout=timeout)
            timed_out = exit_code is None
            if timed_out:
                self._restart_locked()

        header = f"exit_code={exit_code or '?'}"
        if timed_out:
            header += f" (timed out after {timeout}s; bash session was restarted)"
        body = _truncate_lines(lines) if lines else "(no output)"
        return _truncate_chars(f"{header}\n{body}")

    def _restart_locked(self) -> None:
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._start()

    def restart(self) -> str:
        with self._lock:
            self._restart_locked()
        return "Bash session restarted."

    @property
    def backend(self) -> ShellBackend:
        return self._backend

    def close(self) -> None:
        with self._lock:
            if self._master_fd is not None:
                try:
                    os.close(self._master_fd)
                except OSError:
                    pass
                self._master_fd = None
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
            self._proc = None


# Process-global cache of persistent shells, keyed by backend.session_key. Bounded
# (LRU) and lock-guarded: sync tool handlers run under asyncio.to_thread, so without
# the lock concurrent calls for one key would each spawn a BashSession and orphan all
# but the last (C8); without the bound, many distinct workspaces would leak processes/
# threads/fds without limit (C9). Evicting a session closes its process+thread+fd.
_MAX_BASH_SESSIONS = 64
_BASH_SESSIONS: OrderedDict[Any, BashSession] = OrderedDict()
_BASH_SESSIONS_LOCK = threading.Lock()


def _bash_for_current_workspace() -> BashSession:
    env = get_runtime_env()
    workspace_dir = env.require_workspace_dir()
    backend = env.shell_backend or DEFAULT_SHELL_BACKEND
    # Key by the backend's session_key: same execution target (workspace +
    # backend, e.g. a specific sandbox container) reuses one persistent shell;
    # different targets get distinct sessions.
    key = backend.session_key(workspace_dir)
    # Double-checked, lock-guarded get-or-create: exactly one session per key is ever
    # constructed even under concurrent to_thread calls (C8). Creation holds the lock
    # (it spawns a subprocess) — rare enough that serializing it is fine.
    with _BASH_SESSIONS_LOCK:
        session = _BASH_SESSIONS.get(key)
        if session is not None:
            _BASH_SESSIONS.move_to_end(key)  # mark most-recently-used
            return session
        session = BashSession(workspace_dir, backend)
        _BASH_SESSIONS[key] = session
        while len(_BASH_SESSIONS) > _MAX_BASH_SESSIONS:
            _, evicted = _BASH_SESSIONS.popitem(last=False)  # drop least-recently-used
            try:
                evicted.close()
            except Exception:
                pass
        return session


def close_all_bash_sessions() -> None:
    """Close and drop every cached persistent bash session (each owns a subprocess +
    reader thread + pty fd). A host can call this on shutdown to reclaim resources;
    the next bash call lazily recreates whatever it needs."""
    with _BASH_SESSIONS_LOCK:
        for session in list(_BASH_SESSIONS.values()):
            try:
                session.close()
            except Exception:
                pass
        _BASH_SESSIONS.clear()


def run_bash(command: str | None = None, restart: bool = False, timeout: int = 120) -> str:
    bash = _bash_for_current_workspace()
    if restart:
        return bash.restart()
    if not command or not command.strip():
        return "Error: command is required (or set restart=true)."
    timeout = _clamp_int(timeout, 120, 1, 600)
    scope_err = _validate_bash_command_scope(command)
    if scope_err:
        return scope_err
    return bash.execute(command, timeout=timeout)


def _list_directory(dp: Path, limit: int = 200) -> str:
    entries = sorted(dp.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = [f"Directory: {_display_path(dp)}/"]
    for entry in entries[:limit]:
        prefix = "d " if entry.is_dir() else "f "
        size = ""
        if entry.is_file():
            try:
                size = f" ({entry.stat().st_size} bytes)"
            except OSError:
                size = " (size unavailable)"
        lines.append(f"  {prefix}{entry.name}{size}")
    if len(entries) > limit:
        lines.append(f"  ... and {len(entries) - limit} more entries")
    return "\n".join(lines)


def run_read(path: str, offset: int | None = None, limit: int | None = None) -> str:
    try:
        fp = safe_path(path)
        if fp.is_dir():
            return _list_directory(fp)

        # Always enforce the size cap (M-default-tools-1): passing a `limit` used to disable it
        # (max_bytes=None), so paging a multi-GB file read the WHOLE file into memory. offset/limit
        # page OUTPUT within a readable (<=cap) text file; files past the cap use grep/glob.
        text = _read_text(fp, max_bytes=TEXT_FILE_MAX_BYTES)
        all_lines = text.splitlines()
        total = len(all_lines)
        start = max(0, (offset or 1) - 1)
        page_limit = _clamp_int(limit, READ_DEFAULT_LIMIT, 1, READ_MAX_LIMIT) if limit is not None else READ_DEFAULT_LIMIT
        end = min(total, start + page_limit)
        selected = all_lines[start:end]
        numbered = [f"{start + i + 1:>6}|{line}" for i, line in enumerate(selected)]
        header = f"({_display_path(fp)}: showing lines {start + 1}-{end} of {total})"
        if end < total:
            header += f" Use offset={end + 1} to continue."
        _remember_read(fp)
        return _truncate_chars(header + ("\n" + "\n".join(numbered) if numbered else "\n(empty file)"))
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, overwrite: bool = True) -> str:
    try:
        fp = safe_path(path)
        if fp.exists() and fp.is_dir():
            return f"Error: Cannot write file because path is a directory: {_display_path(fp)}"
        existed = fp.exists()
        if existed and not overwrite:
            return f"Error: File already exists and overwrite=false: {_display_path(fp)}"
        if existed:
            read_err = _check_read_state(fp)
            if read_err:
                return read_err.replace("modifying", "overwriting")

        fp.parent.mkdir(parents=True, exist_ok=True)
        old_size = fp.stat().st_size if existed else 0
        fp.write_text(content, encoding="utf-8")
        _remember_read(fp)
        line_count = len(content.splitlines()) if content else 0
        display = _display_path(fp)
        if existed:
            return f"Wrote {len(content)} bytes ({line_count} lines) to {display} (overwritten, was {old_size} bytes)"
        return f"Wrote {len(content)} bytes ({line_count} lines) to {display} (new file)"
    except Exception as e:
        return f"Error: {e}"


def _find_unique_edit_span(content: str, old_text: str, path: str, replace_all: bool) -> tuple[int, int, int] | str:
    if old_text == "":
        return "Error: old_text must not be empty."
    count = content.count(old_text)
    if count == 0:
        diagnostic_content = _normalized_for_diagnostic(content)
        diagnostic_old = _normalized_for_diagnostic(old_text)
        if diagnostic_old and diagnostic_content.count(diagnostic_old) > 0:
            return (
                f"Error: Text was only found after fuzzy normalization in {path}. "
                "Use read_file and provide the exact current text to avoid unintended edits."
            )
        return f"Error: Text not found in {path}. Use read_file and provide a larger exact snippet."
    if not replace_all and count > 1:
        lines = ", ".join(_line_numbers_for_offsets(content, old_text))
        return (
            f"Error: old_text matches {count} locations in {path} (first lines: {lines}). "
            "Provide more surrounding context to make it unique, or set replace_all=true."
        )
    start = content.find(old_text)
    return start, start + len(old_text), count


def run_edit(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    try:
        fp = safe_path(path)
        read_err = _check_read_state(fp)
        if read_err:
            return read_err
        raw_content = _read_text(fp, max_bytes=None)
        bom, content = _split_bom(raw_content)
        original_ending = _detect_line_ending(content)
        normalized = _normalize_to_lf(content)
        norm_old = _normalize_to_lf(old_text)
        norm_new = _normalize_to_lf(new_text)

        if norm_old == norm_new:
            return "Error: old_text and new_text are identical."

        span_or_error = _find_unique_edit_span(normalized, norm_old, path, replace_all)
        if isinstance(span_or_error, str):
            return span_or_error
        start, end, count = span_or_error

        if replace_all:
            updated = normalized.replace(norm_old, norm_new)
        else:
            updated = normalized[:start] + norm_new + normalized[end:]

        final = bom + _restore_line_endings(updated, original_ending)
        fp.write_text(final, encoding="utf-8")
        _remember_read(fp)
        label = "replace_all" if replace_all else "exact"
        diff_output = _generate_diff(normalized, updated)
        return (
            f"Edited {_display_path(fp)} ({label}, {count} replacement{'s' if count != 1 else ''})"
            + _json_syntax_warning(fp, updated)
            + (f"\n{diff_output}" if diff_output else "")
        )
    except Exception as e:
        return f"Error: {e}"


def _json_syntax_warning(fp: Path, updated: str) -> str:
    """Post-edit syntax check for .json files.

    A patch/edit that leaves a dangling comma silently corrupts the file; the
    agent then burns rounds diagnosing a downstream "not valid JSON" error.
    The write is kept (the agent may be mid-way through multi-step edits) but
    the result carries a loud, positioned warning. Empty string when fine.
    """
    if fp.suffix.lower() != ".json":
        return ""
    import json as _json

    try:
        _json.loads(updated)
        return ""
    except ValueError as exc:
        return (
            f"\n⚠️ WARNING: {_display_path(fp)} is NOT valid JSON after this change "
            f"({exc}). The file was written anyway — fix the syntax before using it."
        )


@dataclass
class PatchHunk:
    changes: list[tuple[str, str]]
    old_start: int | None = None
    anchor: str | None = None


_UNIFIED_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@(?:\s*(.*))?$")


def _parse_patch(patch_text: str) -> list[PatchHunk]:
    hunks: list[PatchHunk] = []
    current: PatchHunk | None = None

    for raw_line in patch_text.splitlines():
        if raw_line.startswith(("--- ", "+++ ", "diff --git ", "index ", "new file mode ", "deleted file mode ")):
            continue
        if raw_line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            match = _UNIFIED_HEADER_RE.match(raw_line)
            if match:
                current = PatchHunk(changes=[], old_start=int(match.group(1)))
            else:
                anchor = raw_line[2:].strip()
                current = PatchHunk(changes=[], anchor=anchor or None)
            continue
        if current is None:
            continue
        if raw_line.startswith(("+", "-", " ")):
            current.changes.append((raw_line[0], raw_line[1:]))
        elif raw_line == r"\ No newline at end of file":
            continue

    if current is not None:
        hunks.append(current)
    return [hunk for hunk in hunks if hunk.changes]


def _line_matches(file_line: str, patch_line: str) -> bool:
    return file_line == patch_line or file_line.rstrip() == patch_line.rstrip()


def _find_sequence(lines: list[str], sequence: list[str], start: int = 0) -> tuple[int | None, bool]:
    if not sequence:
        return start, False
    positions: list[int] = []
    max_start = len(lines) - len(sequence)
    for idx in range(max(0, start), max_start + 1):
        if all(_line_matches(lines[idx + j], expected) for j, expected in enumerate(sequence)):
            positions.append(idx)
            if len(positions) > 1:
                return positions[0], True
    if positions:
        return positions[0], False
    return None, False


def _apply_hunks(lines: list[str], hunks: list[PatchHunk], path: str) -> list[str] | str:
    cursor = 0
    for index, hunk in enumerate(hunks, 1):
        old_sequence = [text for op, text in hunk.changes if op in (" ", "-")]

        search_from = cursor
        if hunk.old_start is not None:
            search_from = max(0, hunk.old_start - 1)
        if hunk.anchor and not old_sequence:
            anchor_pos, ambiguous = _find_sequence(lines, [hunk.anchor], cursor)
            if anchor_pos is None:
                return f"Error: Could not locate anchor for hunk {index} in {path}: {hunk.anchor!r}"
            if ambiguous:
                return f"Error: Anchor for hunk {index} is ambiguous in {path}: {hunk.anchor!r}"
            search_from = anchor_pos + 1

        pos, ambiguous = _find_sequence(lines, old_sequence, search_from)
        if pos is None and hunk.old_start is not None:
            pos, ambiguous = _find_sequence(lines, old_sequence, cursor)
        if pos is None:
            preview = "\\n".join(old_sequence[:5])
            return f"Error: Could not locate original lines for hunk {index} in {path}. First expected lines:\n{preview}"
        if ambiguous:
            return f"Error: Hunk {index} matches multiple locations in {path}. Add more context lines."

        # Build the replacement from the FILE's actual matched lines for context
        # (' ') positions — not the patch's rendering. The match may be fuzzy
        # (rstrip-equal, see _line_matches), so reusing the patch text for context
        # lines would silently rewrite untouched lines (e.g. strip trailing
        # whitespace). Only '+' lines come from the patch; '-' lines are dropped.
        matched = lines[pos : pos + len(old_sequence)]
        replacement: list[str] = []
        fi = 0
        for op, text in hunk.changes:
            if op == " ":
                replacement.append(matched[fi])
                fi += 1
            elif op == "-":
                fi += 1
            elif op == "+":
                replacement.append(text)
        lines[pos : pos + len(old_sequence)] = replacement
        cursor = pos + len(replacement)
    return lines


def run_apply_patch(path: str, patch: str) -> str:
    try:
        fp = safe_path(path)
        read_err = _check_read_state(fp)
        if read_err:
            return read_err
        raw_content = _read_text(fp, max_bytes=None)
        bom, content = _split_bom(raw_content)
        original_ending = _detect_line_ending(content)
        normalized = _normalize_to_lf(content)
        lines = normalized.split("\n")

        hunks = _parse_patch(patch)
        if not hunks:
            return (
                "Error: No valid hunks found. Use unified diff hunks like "
                "'@@ -1,2 +1,2 @@' with space/context, -delete, and +add lines."
            )

        result = _apply_hunks(list(lines), hunks, path)
        if isinstance(result, str):
            return result
        updated = "\n".join(result)
        if updated == normalized:
            return f"No changes applied to {_display_path(fp)}."

        final = bom + _restore_line_endings(updated, original_ending)
        fp.write_text(final, encoding="utf-8")
        _remember_read(fp)
        diff_output = _generate_diff(normalized, updated)
        return (
            f"Patched {_display_path(fp)} ({len(hunks)} hunk{'s' if len(hunks) != 1 else ''})"
            f"{_json_syntax_warning(fp, updated)}\n{diff_output}"
        )
    except Exception as e:
        return f"Error: {e}"


def _iter_files_for_glob(base: Path, pattern: str, include_hidden: bool) -> list[Path]:
    matches: list[Path] = []
    patterns = [pattern]
    if pattern.startswith("**/"):
        patterns.append(pattern[3:])
    if "**" in pattern:
        for root, dirs, files in os.walk(base):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not _should_skip_dir(root_path / d, base, include_hidden, pattern)]
            names = files + dirs
            for name in names:
                candidate = root_path / name
                if not include_hidden and not _pattern_mentions_hidden(pattern) and _is_hidden_path(candidate, base):
                    continue
                rel = candidate.relative_to(base).as_posix()
                if any(fnmatch.fnmatch(rel, candidate_pattern) for candidate_pattern in patterns):
                    matches.append(candidate)
    else:
        for candidate in base.glob(pattern):
            # Mirror the ** branch's whole-directory pruning: base.glob only filters
            # the matched leaf, so a pattern like "*/*.py" would otherwise return
            # files under node_modules/.git/.venv/... Reject if the leaf OR ANY
            # ancestor directory is a _COMMON_SKIP_DIRS dir.
            rel_parts = candidate.relative_to(base).parts
            if any(part in _COMMON_SKIP_DIRS for part in rel_parts):
                continue
            if not include_hidden and not _pattern_mentions_hidden(pattern) and _is_hidden_path(candidate, base):
                continue
            matches.append(candidate)
    return matches


def run_glob(pattern: str, path: str = ".", max_results: int = 100, include_hidden: bool = False) -> str:
    try:
        base = safe_path(path)
        if not base.is_dir():
            return f"Error: {path} is not a directory"
        if not pattern.strip():
            return "Error: pattern is required"
        max_results = _clamp_int(max_results, 100, 1, GLOB_MAX_RESULTS)
        search_pattern = pattern if pattern.startswith("**/") or "/" in pattern else f"**/{pattern}"
        matches = _iter_files_for_glob(base, search_pattern, include_hidden)
        matches = sorted(
            set(matches),
            key=lambda p: p.stat().st_mtime_ns if p.exists() else 0,
            reverse=True,
        )
        if not matches:
            return f"No files matching '{search_pattern}' in {_display_path(base)}"
        limited = matches[:max_results]
        lines = [f"[searched in: {_display_path(base)}]"] + [_display_path(match) for match in limited]
        if len(matches) > max_results:
            lines.append(f"... and {len(matches) - max_results} more")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def _grep_with_python(
    pattern: str,
    base: Path,
    include: str | None,
    max_results: int,
    literal: bool,
    case_sensitive: bool,
    include_hidden: bool,
) -> str:
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(re.escape(pattern) if literal else pattern, flags)
    roots = [base] if base.is_file() else []
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not _should_skip_dir(root_path / d, base, include_hidden)]
            for filename in files:
                fp = root_path / filename
                if include and not fnmatch.fnmatch(fp.relative_to(base).as_posix(), include):
                    continue
                if not include_hidden and _is_hidden_path(fp, base):
                    continue
                roots.append(fp)

    results: list[str] = []
    for fp in roots:
        try:
            data = fp.read_bytes()
            if _looks_binary(data):
                continue
            text = data.decode("utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if compiled.search(line):
                if len(results) >= max_results:
                    # One more match than we'll show → results were truncated. Emit
                    # the same notice the rg path appends so output doesn't silently
                    # differ by whether ripgrep happens to be installed.
                    results.append(f"... results truncated at {max_results}")
                    return "\n".join(results)
                results.append(f"{_display_path(fp)}:{line_no}:{line.rstrip()}")
    return "\n".join(results)


def run_grep(
    pattern: str,
    path: str = ".",
    include: str | None = None,
    max_results: int = 50,
    literal: bool = False,
    case_sensitive: bool = True,
    include_hidden: bool = False,
) -> str:
    try:
        base = safe_path(path)
        if not base.exists():
            return f"Error: path does not exist: {path}"
        if not pattern:
            return "Error: pattern is required"
        max_results = _clamp_int(max_results, 50, 1, SEARCH_MAX_RESULTS)

        rg_cmd = [
            "rg",
            "--no-heading",
            "--line-number",
            "--with-filename",
            "--color=never",
            "--max-filesize",
            "5M",
        ]
        if include_hidden:
            rg_cmd.append("--hidden")
        if literal:
            rg_cmd.append("--fixed-strings")
        if not case_sensitive:
            rg_cmd.append("--ignore-case")
        if include:
            rg_cmd.extend(["--glob", include])
        # Exclude the same directories the Python fallback prunes
        # (_COMMON_SKIP_DIRS), at ANY depth, and AFTER the include glob so the
        # exclusion wins (rg uses last-glob-wins precedence). The old root-
        # anchored "!node_modules/**" globs missed nested copies and could be
        # overridden by a later include, so results diverged depending on
        # whether rg or the Python fallback ran. "**/" matches zero or more
        # leading dirs, so this covers both top-level and nested matches.
        for skip in sorted(_COMMON_SKIP_DIRS):
            rg_cmd.extend(["--glob", f"!**/{skip}/**"])
        if not include_hidden:
            rg_cmd.extend(["--glob", "!.*", "--glob", "!**/.*"])
        rg_cmd.extend(["--", pattern, _safe_relative_for_subprocess(base)])

        try:
            workspace_dir = get_runtime_env().require_workspace_dir()
            result = subprocess.run(
                rg_cmd,
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode > 1:
                return f"Error: {result.stderr.strip() or 'rg failed'}"
            # Count and slice the SAME (non-blank) set so the truncation notice
            # reflects what's actually shown — counting raw splitlines (incl.
            # blanks) could claim truncation when nothing was dropped.
            all_lines = [line for line in result.stdout.splitlines() if line]
            lines = all_lines[:max_results]
            if not lines:
                return f"No matches for '{pattern}' in {_display_path(base)}"
            if len(all_lines) > max_results:
                lines.append(f"... results truncated at {max_results}")
            return "\n".join(lines)
        except FileNotFoundError:
            output = _grep_with_python(pattern, base, include, max_results, literal, case_sensitive, include_hidden)
            return output if output else f"No matches for '{pattern}' in {_display_path(base)}"
    except re.error as e:
        return f"Error: Invalid regex: {e}"
    except Exception as e:
        return f"Error: {e}"


def run_load_skill(name: str) -> str:
    runtime_ctx = get_tool_runtime_context()
    skills_dir = getattr(runtime_ctx.config, "skills_dir", None) if runtime_ctx.config is not None else None
    if skills_dir:
        from power_loop.runtime.skills import SkillLoader

        return SkillLoader(skills_dir).get_content(name)
    return get_default_loader().get_content(name)


def _current_store_and_session() -> tuple[Any | None, str | None]:
    runtime_ctx = get_tool_runtime_context()
    return runtime_ctx.store, runtime_ctx.session_id


# 6.16.0：校验与渲染收敛到 runtime/todos.py 一处——这两个函数原来在 core/state.py 里
# 还有一份**逐字重复**的实现，改一处漏一处就是清单在工具层和上下文层各说各话。
from power_loop.runtime.todos import render_todos as _render_todos  # noqa: E402
from power_loop.runtime.todos import todo_counts as _todo_counts  # noqa: E402
from power_loop.runtime.todos import validate_todos as _validate_todos  # noqa: E402

#: Cap on retained in-memory background-task records. The durable record lives in the store; this
#: is just a cache for `check`, so a long-lived loop must not accumulate task dicts (each pinning a
#: store + event_loop reference) forever (default-tools-2).
_MAX_BG_TASKS = 256


class BackgroundManager:
    """Background command runner with task tracking."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Live daemon threads (so shutdown can drain them) and terminal write-backs that
        # could not be delivered to their owning loop (so a later check / shutdown can
        # still persist them instead of leaving the row stuck at 'running' forever).
        self._threads: dict[str, threading.Thread] = {}
        self._orphaned: list[dict[str, Any]] = []

    def _prune_locked(self) -> None:
        """Evict oldest TERMINAL (non-running) task records beyond ``_MAX_BG_TASKS`` so the cache —
        and the store/event_loop refs each entry pins — can't grow without bound (default-tools-2).
        Running tasks are never evicted. Caller must hold ``self._lock``."""
        while len(self.tasks) > _MAX_BG_TASKS:
            for tid, t in self.tasks.items():  # insertion order = oldest first
                if t.get("status") != "running":
                    del self.tasks[tid]
                    self._threads.pop(tid, None)
                    break
            else:
                break  # everything still running — nothing safe to evict yet

    async def run(self, command: str) -> str:
        reason = _dangerous_command_reason(command)
        if reason:
            return f"Error: Dangerous command blocked ({reason}). Run it manually if you really intend it."
        policy_err = command_policy_reason(command, get_runtime_env().blocked_command_categories)
        if policy_err:
            return policy_err

        scope_err = _validate_bash_command_scope(command)
        if scope_err:
            return scope_err

        task_id = str(uuid.uuid4())[:8]
        env = get_runtime_env()
        workspace_dir = env.require_workspace_dir()
        # Capture the injected ShellBackend HERE (the daemon thread below runs
        # outside this contextvar). background_run must honor the same sandbox seam
        # as `bash` — running on the host would bypass an installed sandbox and leak
        # the host environment (C1).
        shell_backend = env.shell_backend
        store, sid = _current_store_and_session()
        # Capture the running event loop so the daemon thread can drive the async
        # store's writes back on it (the store's transaction/lock are loop-bound).
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None
        if store is not None and sid is not None:
            await store.upsert_background_task(
                sid,
                task_id=task_id,
                command=command,
                status="running",
                output_tail=None,
            )
        with self._lock:
            self.tasks[task_id] = {
                "status": "running",
                "result": None,
                "command": command,
                "session_id": sid,
                "store": store,
                "event_loop": event_loop,
                "workspace_dir": workspace_dir,
                "shell_backend": shell_backend,
            }
            self._prune_locked()

        thread = threading.Thread(target=self._execute, args=(task_id, command), daemon=True)
        with self._lock:
            self._threads[task_id] = thread
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str) -> None:
        store = None
        sid = None
        event_loop = None
        shell_backend: ShellBackend | None = None
        with self._lock:
            task = self.tasks.get(task_id)
            if task is not None:
                store = task.get("store")
                sid = task.get("session_id")
                event_loop = task.get("event_loop")
                workspace_dir = task.get("workspace_dir")
                shell_backend = task.get("shell_backend")
            else:
                workspace_dir = None
        try:
            if not isinstance(workspace_dir, Path):
                raise RuntimeError("Background task is missing workspace_dir")
            # Launch through the ShellBackend (same seam as `bash`): feed the command
            # to the backend-launched shell's stdin so an installed sandbox actually
            # contains it, instead of running shell=True on the host (C1). For the
            # default LocalShellBackend this is an in-process bash, equivalent to the
            # prior behavior; for a sandbox backend it runs inside the sandbox.
            backend = shell_backend or DEFAULT_SHELL_BACKEND
            result = subprocess.run(
                backend.launch_argv(workspace_dir),
                input=command,
                cwd=backend.launch_cwd(workspace_dir),
                env=backend.launch_env(workspace_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = _truncate_chars((result.stdout + result.stderr).strip())
            status = "completed" if result.returncode == 0 else f"failed({result.returncode})"
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
            return_code = None
        except Exception as e:  # pragma: no cover
            output = f"Error: {e}"
            status = "error"
            return_code = None

        with self._lock:
            task = self.tasks.get(task_id)
            if task is not None:
                task["status"] = status
                task["result"] = output or "(no output)"
        if store is not None and sid is not None:
            payload = {
                "store": store,
                "sid": sid,
                "task_id": task_id,
                "command": command,
                "status": status,
                "return_code": return_code,
                "output_tail": output or "(no output)",
            }
            # Drive the async store write on the OWNING event loop from this daemon thread
            # (the store's lock/transaction — and a PG/MySQL pool — are loop-bound, so we
            # cannot just spin a fresh loop here). If that loop is gone (closed during
            # shutdown / a per-call asyncio.run that already returned), the write can't be
            # delivered now: stash it so a later check()/aclose persists it instead of
            # leaving the durable row stuck at 'running'. Never silently swallow.
            delivered = False
            if event_loop is not None and not event_loop.is_closed():
                try:
                    fut = asyncio.run_coroutine_threadsafe(
                        store.upsert_background_task(
                            sid,
                            task_id=task_id,
                            command=command,
                            status=status,
                            return_code=return_code,
                            output_tail=output or "(no output)",
                        ),
                        event_loop,
                    )
                    fut.result(timeout=30)
                    delivered = True
                except Exception:
                    logger.warning(
                        "background task %s: terminal status write-back failed on its "
                        "owning event loop; deferring for recovery", task_id, exc_info=True
                    )
            if not delivered:
                with self._lock:
                    self._orphaned.append(payload)
                logger.warning(
                    "background task %s (session %s) finished as %s but its owning event "
                    "loop was unavailable; status deferred (recovered on next "
                    "background_run action=check or aclose)", task_id, sid, status,
                )
        with self._lock:
            self._threads.pop(task_id, None)

    async def flush_orphaned(self, store: Any, sid: str | None = None) -> None:
        """Persist deferred terminal write-backs via the (live) ``store`` — for one ``sid``
        (``check``) or all sessions on that store (``aclose``).

        Runs on the caller's loop with a store that is valid there (``check`` is invoked
        mid-turn, ``aclose`` before the store closes), so it sidesteps the loop-binding
        that defeated the daemon thread's original write-back."""
        with self._lock:
            pending: list[dict[str, Any]] = []
            rest: list[dict[str, Any]] = []
            for o in self._orphaned:
                mine = o["store"] is store and (sid is None or o["sid"] == sid)
                (pending if mine else rest).append(o)
            self._orphaned = rest
        for o in pending:
            try:
                await store.upsert_background_task(
                    o["sid"], task_id=o["task_id"], command=o["command"],
                    status=o["status"], return_code=o["return_code"],
                    output_tail=o["output_tail"],
                )
            except Exception:  # pragma: no cover - keep for a later retry
                logger.warning(
                    "failed to flush deferred background status for task %s",
                    o["task_id"], exc_info=True,
                )
                with self._lock:
                    self._orphaned.append(o)

    def join_pending(self, timeout: float | None = None) -> int:
        """Wait (up to ``timeout`` total) for in-flight background threads to finish so a
        finishing task's terminal write-back lands before the owning store/loop is torn
        down. Returns the count still running afterward (best-effort drain)."""
        with self._lock:
            threads = list(self._threads.values())
        deadline = None if timeout is None else time.monotonic() + timeout
        for t in threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)
        with self._lock:
            return sum(1 for t in self._threads.values() if t.is_alive())

    async def run_tool(self, tool_name: str, args: dict[str, Any] | None) -> str:
        """6.8.0 ``action=tool``：把一个 async_capable 工具作为后台任务跑，立即返回 task_id。

        与 shell 分支不同：工具 handler 本来就是协程，直接 ``asyncio.create_task`` 在
        本进程事件循环上跑——**contextvars 随 task 创建自动拷贝**（PEP 567），所以宿主的
        计费/活动打标上下文天然跟着走，不需要捕获重播。结果持久化进同一张
        background 任务表（command 字段以 ``tool:`` 前缀区分）；完成后调用宿主注册的
        ``on_tool_task_complete``（best-effort），由宿主决定要不要唤醒睡着的 agent。
        进程重启 = 任务丢失，行卡 running，由 check 的 flush_orphaned 语义外**另行**
        诚实兜底：这里在启动时登记 in-memory，check 未命中 in-memory 且行仍 running
        时按孤儿报告（复用 shell 分支已有的自愈路径）。
        """
        from power_loop.core.agent_context import get_current_loop

        name = str(tool_name or "").strip()
        if not name:
            raise ValueError("background_run action=tool requires tool")
        if name in ("background_run", "workflow", "spawn_agent"):
            raise ValueError(f"{name} 不允许后台化（递归/编排类入口）")
        loop = get_current_loop()
        registry = getattr(loop, "tool_registry", None) if loop is not None else None
        reg = registry.get(name) if registry is not None else None
        if reg is None:
            raise ValueError(f"unknown tool: {name}")
        definition = getattr(reg, "definition", reg)
        from power_loop.tools.registry import async_capable_actions, async_capable_for

        if not async_capable_for(definition, args):
            acts = async_capable_actions(definition)
            if acts:
                # 工具本身可异步、只是这次这个 action 不行——把能异步的那几个报出来，
                # 别让它以为整个工具都不能后台跑（报错即出路）。
                got = str((args or {}).get("action") or "(没传 action)")
                raise ValueError(
                    f"{name} 只有 action={'/'.join(acts)} 可以后台跑，这次传的是 {got}；"
                    "其余 action 有副作用，直接同步调用即可。"
                )
            raise ValueError(
                f"{name} 未标记可异步（async_capable=False）——直接同步调用即可；"
                "只有无副作用的长耗时工具才开放后台化。"
            )
        store, sid = _current_store_and_session()
        if store is not None and sid is not None:
            rows = await store.list_background_tasks(sid)
            running = sum(1 for r in rows if r.status == "running" and r.command.startswith("tool:"))
            if running >= 8:
                raise ValueError("后台工具任务已达并发上限（8）——先 check 收结果再发新的。")
        task_id = str(uuid.uuid4())[:8]
        args = dict(args or {})
        label = f"tool:{name} " + json.dumps(args, ensure_ascii=False)[:160]
        with self._lock:
            self.tasks[task_id] = {"command": label, "status": "running", "result": None}
        if store is not None and sid is not None:
            await store.upsert_background_task(
                sid, task_id=task_id, command=label, status="running",
                return_code=None, output_tail="(running)",
            )

        async def _worker() -> None:
            status, output = "completed", ""
            try:
                result = await registry.invoke_async(name, args)
                output = str(result if result is not None else "")[:8000]
            except Exception as exc:  # noqa: BLE001 — 失败是任务结果，不是崩溃
                status = "failed"
                output = f"{type(exc).__name__}: {exc}"[:2000]
            with self._lock:
                task = self.tasks.get(task_id)
                if task is not None:
                    task["status"] = status
                    task["result"] = output or "(no output)"
            if store is not None and sid is not None:
                try:
                    await store.upsert_background_task(
                        sid, task_id=task_id, command=label, status=status,
                        return_code=0 if status == "completed" else 1,
                        output_tail=output or "(no output)",
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("tool task %s: status write-back failed", task_id, exc_info=True)
            cb = _TOOL_TASK_ON_COMPLETE
            if cb is not None:
                try:
                    await cb(sid, task_id, status)
                except Exception:  # noqa: BLE001 — 宿主回调失败不毁任务结果
                    logger.warning("tool task %s: on_complete callback failed", task_id, exc_info=True)

        asyncio.create_task(_worker(), name=f"bg-tool-{task_id}")
        return (
            f"后台任务已启动：task_id={task_id}（{label[:80]}）。你可以继续做别的；"
            "完成后会收到通知，background_run(action=\"check\", task_id=\"" + task_id + "\") 取结果。"
        )

    async def check(self, task_id: str | None = None) -> str:
        store, sid = _current_store_and_session()
        if store is not None and sid is not None:
            # Recover any terminal status a daemon thread couldn't deliver to a now-gone
            # loop, so a stuck 'running' row self-heals on the next read.
            await self.flush_orphaned(store, sid)
            if task_id:
                row = await store.get_background_task(sid, task_id)
                if row is None:
                    return f"Error: Unknown task {task_id}"
                return f"[{row.status}] {row.command[:80]}\n{row.output_tail or '(running)'}"
            rows = await store.list_background_tasks(sid)
            if not rows:
                return "No background tasks."
            return "\n".join(f"{row.task_id}: [{row.status}] {row.command[:80]}" for row in rows)
        with self._lock:
            if task_id:
                task = self.tasks.get(task_id)
                if not task:
                    return f"Error: Unknown task {task_id}"
                return f"[{task['status']}] {task['command'][:80]}\n{task.get('result') or '(running)'}"

            lines: list[str] = []
            for tid, task in self.tasks.items():
                lines.append(f"{tid}: [{task['status']}] {task['command'][:80]}")
            return "\n".join(lines) if lines else "No background tasks."


BG = BackgroundManager()

#: 6.8.0 宿主 seam：后台**工具**任务完成时回调 ``(session_id, task_id, status)``。
#: 宿主用它决定要不要唤醒一个已 pass_turn 的 agent（在忙的 session 下一轮开轮时
#: BackgroundRuntimeProjector 本来就会注入更新，无需回调介入）。
_TOOL_TASK_ON_COMPLETE: Any = None


def register_tool_task_callback(cb: Any) -> None:
    global _TOOL_TASK_ON_COMPLETE
    _TOOL_TASK_ON_COMPLETE = cb


async def run_background(
    action: str, command: str | None = None, task_id: str | None = None,
    tool: str | None = None, args: dict[str, Any] | None = None,
) -> str:
    """Execute one action of the merged ``background_run`` tool."""
    operation = str(action or "").strip().lower()
    if operation == "run":
        if not (command or "").strip():
            raise ValueError("background_run action=run requires command")
        return await BG.run(str(command))
    if operation == "check":
        return await BG.check(task_id)
    if operation == "tool":
        return await BG.run_tool(tool, args)
    raise ValueError("background_run action must be one of: run, check, tool")


async def run_todo(items: list[dict[str, Any]]) -> str:
    validated = _validate_todos(items)
    store, sid = _current_store_and_session()
    if store is not None and sid is not None:
        rendered = _render_todos(validated)
        await store.set_runtime_state(
            sid,
            "todo",
            {"items": validated, "rendered": rendered, "counts": _todo_counts(validated)},
        )
        # Keep the in-process context warm for UI/event compatibility.
        get_ctx().todo.update(validated)
        return rendered
    return get_ctx().todo.update(validated)


def _notes_policy() -> Any:
    """Resolve the active NotesPolicy: loop config override or the default."""
    from power_loop.runtime.notes import DEFAULT_NOTES_POLICY

    runtime_ctx = get_tool_runtime_context()
    policy = getattr(runtime_ctx.config, "notes_policy", None) if runtime_ctx.config else None
    return policy or DEFAULT_NOTES_POLICY


def _notes_store_and_session() -> tuple[Any, str]:
    store, sid = _current_store_and_session()
    if store is None or sid is None:
        raise RuntimeError("note tool needs an active session (StatefulAgentLoop)")
    return store, sid


async def run_note(
    action: str,
    note_id: int | None = None,
    content: str | None = None,
    pinned: bool | None = None,
) -> str:
    """Execute one action of the model-facing persistent-note tool."""
    from power_loop.runtime.notes import add_note_checked, render_notes, update_note_checked

    operation = str(action or "").strip().lower()
    store, sid = _notes_store_and_session()
    policy = _notes_policy()
    if operation == "add":
        if content is None:
            raise ValueError("note action=add requires content")
        row = await add_note_checked(store, sid, content, pinned=bool(pinned), policy=policy)
        count = await store.count_notes(sid)
        return f"noted as #{row.note_id} ({count}/{policy.max_notes} notes used)"
    if operation == "update":
        if note_id is None:
            raise ValueError("note action=update requires note_id")
        await update_note_checked(
            store, sid, int(note_id), content=content, pinned=pinned, policy=policy
        )
        return f"note #{note_id} updated"
    if operation == "delete":
        if note_id is None:
            raise ValueError("note action=delete requires note_id")
        if not await store.delete_note(sid, int(note_id)):
            raise ValueError(f"note #{note_id} does not exist")
        return f"note #{note_id} deleted ({await store.count_notes(sid)} remaining)"
    if operation == "list":
        notes = await store.list_notes(sid)
        if not notes:
            return "(no notes yet — use note(action=add, content=...) to remember durable facts)"
        return render_notes(notes, policy=policy)
    raise ValueError("note action must be one of: add, update, delete, list")


def _timers_store_and_session() -> tuple[Any, str]:
    store, sid = _current_store_and_session()
    if store is None or sid is None:
        raise RuntimeError("wakeup tools need an active session (StatefulAgentLoop)")
    return store, sid


WAKEUP_MIN_DELAY_S = 5
WAKEUP_MAX_DELAY_S = 30 * 86400
WAKEUP_MAX_LIVE = 10  # live (armed) timers per session


async def run_schedule_wakeup(
    action: str,
    delay_seconds: int | None = None,
    note: str | None = None,
    every_seconds: int | None = None,
    timer_id: int | None = None,
) -> str:
    """Execute one action of the merged ``schedule_wakeup`` tool.

    ``action`` 可省略（6.13.0）：按参数形状推断——有 delay_seconds 就是 schedule，只有 timer_id
    就是 cancel，什么都没有就是 list。省略 action 是模型的高频写法（与同族 schedule_followup 的
    ``operation='schedule' (default)`` 一致），此前被必填校验硬拒，每次多烧一轮。"""
    operation = str(action or "").strip().lower()
    if not operation:
        if delay_seconds is not None:
            operation = "schedule"
        elif timer_id is not None:
            operation = "cancel"
        else:
            operation = "list"
    if operation == "schedule":
        if delay_seconds is None:
            raise ValueError("schedule_wakeup action=schedule requires delay_seconds")
        if note is None:
            raise ValueError("schedule_wakeup action=schedule requires note")
        return await _schedule_wakeup(int(delay_seconds), note, every_seconds)
    if operation == "list":
        return await _list_wakeups()
    if operation == "cancel":
        if timer_id is None:
            raise ValueError("schedule_wakeup action=cancel requires timer_id")
        return await _cancel_wakeup(int(timer_id))
    raise ValueError("schedule_wakeup action must be one of: schedule, list, cancel")


async def _schedule_wakeup(delay_seconds: int, note: str, every_seconds: int | None = None) -> str:
    import time as _time

    store, sid = _timers_store_and_session()
    delay = int(delay_seconds)
    if not (WAKEUP_MIN_DELAY_S <= delay <= WAKEUP_MAX_DELAY_S):
        raise ValueError(
            f"delay_seconds must be between {WAKEUP_MIN_DELAY_S} and {WAKEUP_MAX_DELAY_S}"
        )
    interval = int(every_seconds) if every_seconds else None
    if interval is not None and not (WAKEUP_MIN_DELAY_S <= interval <= WAKEUP_MAX_DELAY_S):
        raise ValueError(
            f"every_seconds must be between {WAKEUP_MIN_DELAY_S} and {WAKEUP_MAX_DELAY_S}"
        )
    if not (note or "").strip():
        raise ValueError("note is required — write what future-you should do")
    live = [t for t in await store.list_timers(sid) if t.status == "armed"]
    if len(live) >= WAKEUP_MAX_LIVE:
        return (
            f"Budget exceeded: {len(live)} wake-ups already scheduled. "
            "Cancel or merge some first (schedule_wakeup action=list / action=cancel)."
        )
    timer = await store.create_timer(
        sid, due_at=int(_time.time() * 1000 + delay * 1000), note=note.strip(),
        interval_s=interval,
    )
    if interval:
        return (
            f"Recurring wake-up #{timer.timer_id} scheduled: first in {delay}s, "
            f"then every {interval}s until you cancel it."
        )
    return f"Wake-up #{timer.timer_id} scheduled in {delay}s. You'll receive your note."


async def _list_wakeups() -> str:
    import time as _time

    store, sid = _timers_store_and_session()
    timers = await store.list_timers(sid)
    if not timers:
        return "No wake-ups scheduled."
    now = _time.time() * 1000
    lines = [
        f"#{t.timer_id} in {max(0, int((t.due_at - now) / 1000))}s"
        + (f" (every {t.interval_s}s, fired {t.fire_count}x)" if t.interval_s else "")
        + f" ({t.status}): {t.note}"
        for t in timers
    ]
    return "\n".join(lines)


async def _cancel_wakeup(timer_id: int) -> str:
    store, sid = _timers_store_and_session()
    if await store.transition_timer(sid, int(timer_id), from_status="armed", to_status="cancelled"):
        return f"Wake-up #{timer_id} cancelled."
    return f"Wake-up #{timer_id} not found or already fired."


def run_current_time() -> str:
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc).astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


# ── recall_compacted: pull back detail that compaction folded out ──────────

RECALL_COMPACTED_MAX_LIMIT = 50
RECALL_COMPACTED_CONTENT_CHARS = 2000


async def run_recall_compacted(
    query: str | None = None,
    from_seq: int | None = None,
    to_seq: int | None = None,
    limit: int = 20,
) -> str:
    """Surface THIS session's compacted-out (folded) messages on demand.

    Compaction summarizes old turns into a ``compact_note`` and marks the originals
    ``compacted_out`` — they leave the active window but are NOT deleted. When the
    note isn't enough, this retrieves the originals (read-only, current session only),
    filtered by keyword (``query``, case-insensitive substring) and/or seq range.
    """
    from power_loop.runtime.store.types import MessageState

    ctx = get_tool_runtime_context(required=True)
    store, sid = ctx.store, ctx.session_id
    assert store is not None and sid is not None  # required=True guarantees this

    limit = max(1, min(int(limit or 20), RECALL_COMPACTED_MAX_LIMIT))
    rows = [r for r in await store.load_all_messages(sid) if r.state is MessageState.COMPACTED_OUT]
    if from_seq is not None:
        rows = [r for r in rows if r.seq >= int(from_seq)]
    if to_seq is not None:
        rows = [r for r in rows if r.seq <= int(to_seq)]
    if query and query.strip():
        q = query.strip().lower()
        rows = [r for r in rows if q in (r.content or "").lower()]

    total = len(rows)
    if total == 0:
        return "No compacted (folded-out) messages match this query/range."

    shown = rows[-limit:]  # rows are seq-ascending → most recent matches
    blocks: list[str] = []
    for r in shown:
        body = r.content or ""
        if len(body) > RECALL_COMPACTED_CONTENT_CHARS:
            body = body[:RECALL_COMPACTED_CONTENT_CHARS] + " …[truncated]"
        head = f"[seq {r.seq} · {r.role}"
        if r.round_index is not None:
            head += f" · round {r.round_index}"
        head += "]"
        blocks.append(f"{head}\n{body}")

    header = f"{len(shown)} of {total} compacted message(s)"
    if total > len(shown):
        header += f" (showing the {len(shown)} most recent; narrow with query/from_seq/to_seq)"
    return header + ":\n\n" + "\n\n".join(blocks)


# ── recall_send: re-expand a send the projection layer summarized ──────────────

RECALL_SEND_CONTENT_CHARS = 2000
# Single-row recall (``recall_send(send_index, seq=…)``) returns the ORIGINAL body — a skill file,
# a read file, a vision answer — so its cap is generous. The send-level view keeps the small cap
# because it lists every row of the send.
RECALL_SEND_ROW_CHARS = 40_000
# The send-level listing is a MAP, not a payload. Real logs: no-seq calls averaged ~31.5K chars
# against ~1.5K for seq'd ones — a listing of a tool-heavy send costs more context than the thing
# the model was trying to find. So the whole listing lives inside one budget: every row still
# appears (with its «sS» coordinate and its true size), bodies shrink together until they fit.
RECALL_SEND_TOTAL_CHARS = 12_000
RECALL_SEND_MIN_ROW_CHARS = 240


async def run_recall_send(send_index: int, seq: int | None = None) -> str:
    """Re-expand one past send the send-context projection summarized.

    When ``AgentLoopConfig.history_projector`` is set, finished sends appear in context as a
    compact projected summary while their FULL detail stays in ``pl_messages`` (the immutable
    audit). This returns that send's original messages — assistant text, tool calls (by name)
    and their results — read-only, current session, by the ``send_index`` (the ``#N`` the
    summary shows). With ``seq`` it returns just THAT row (the ``sS`` coordinate a projected
    tool line carries) with a much larger body cap — plus the assistant call that produced it.
    """
    ctx = get_tool_runtime_context(required=True)
    store, sid = ctx.store, ctx.session_id
    assert store is not None and sid is not None  # required=True guarantees this

    try:
        target = int(send_index)
    except (TypeError, ValueError):
        return f"Invalid send_index: {send_index!r} (expected an integer)."

    rows = [r for r in await store.load_all_messages(sid) if r.send_index == target]
    if not rows:
        return f"No messages found for send #{target} in this session."

    if seq is not None:
        try:
            want = int(seq)
        except (TypeError, ValueError):
            return f"Invalid seq: {seq!r} (expected an integer)."
        hit = next((r for r in rows if r.seq == want), None)
        if hit is None:
            lo, hi = min(r.seq for r in rows), max(r.seq for r in rows)
            return (f"No row seq {want} in send #{target} (its rows span seq {lo}–{hi}). "
                    "Use the exact «sS» coordinate from the summary line, or omit seq to list the whole send.")
        return _render_recall_row(hit, rows, cap=RECALL_SEND_ROW_CHARS, head=f"send #{target} · seq {want}")

    def _tc_name(tc: dict[str, Any]) -> str:
        # ``function`` is normally a dict, but a malformed/imported tool_call can carry a
        # non-dict (e.g. a bare string); guard so name extraction never raises (the tool
        # must always return a string).
        fn = tc.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        return str(name or tc.get("name") or "?")

    # 蒸馏（而不是原样吐 content）：多模态行的 content 是 JSON，原样返回等于把序列化的
    # 块列表——内联 data URL 的话就是整个 base64——塞给模型。列整个 send 时不注入图片：
    # 一次注入十几张会失控；要看某一行，recall 那一行（带 seq）。
    bodies = [_recall_row_body(r)[0] for r in rows]
    # One budget for the whole listing: rows all stay visible, bodies shrink together to fit.
    cap = RECALL_SEND_CONTENT_CHARS
    if sum(min(len(b), cap) for b in bodies) > RECALL_SEND_TOTAL_CHARS:
        cap = max(RECALL_SEND_MIN_ROW_CHARS, RECALL_SEND_TOTAL_CHARS // max(1, len(rows)))
    blocks: list[str] = []
    trimmed = 0
    for r, body in zip(rows, bodies, strict=True):
        suffix = ""
        if r.tool_calls:
            names = ", ".join(_tc_name(tc) for tc in r.tool_calls)
            suffix = f"[tool_calls: {names}]"
        full = len(body)
        # Truncate the CONTENT first (the body is the unbounded part), THEN append the
        # tool-calls suffix — otherwise a ~2000-char message would have its suffix cut, making
        # a tool-bearing send look tool-free in the inspector.
        if full > cap:
            trimmed += 1
            body = body[:cap] + f" …[truncated: {full} chars — recall_send(send_index={target}, seq={r.seq}) for the original]"
        if suffix:
            body = f"{body}\n{suffix}" if body else suffix
        head = f"[seq {r.seq} · {r.role}"
        if r.name:
            head += f" · {r.name}"
        if r.round_index is not None:
            head += f" · round {r.round_index}"
        head += "]"
        blocks.append(f"{head}\n{body}")
    header = f"send #{target} — {len(rows)} message(s)"
    if trimmed:
        header += (f"; {trimmed} body(ies) trimmed to {cap} chars to keep this listing small. "
                   f"This is a MAP — for one original, call recall_send(send_index={target}, seq=S) "
                   "with the seq you want (returns that row in full).")
    return f"{header}:\n\n" + "\n\n".join(blocks)


def _recall_row_body(row: Any) -> tuple[str, list[str]]:
    """One stored row → (text a tool may return, ``[(path, ref)]`` of any images in it).

    A multimodal row's ``content`` column holds JSON, and returning it raw hands the model a
    serialized block list — with an inlined data URL that is the whole base64 payload. So the
    text side is DISTILLED (same helper the projection uses), and the images are reported
    separately for the caller to put back in front of the model.
    """
    from power_loop.runtime.representation import distill_multimodal_text
    from power_loop.runtime.store.types import decode_row_content

    text = distill_multimodal_text(getattr(row, "content", None), getattr(row, "meta", None)) or ""
    decoded = decode_row_content(getattr(row, "content", None), getattr(row, "meta", None))
    images: list[tuple[str, str]] = []
    if isinstance(decoded, list):
        for block in decoded:
            if not isinstance(block, dict) or block.get("type") != "attachment":
                continue
            att = block.get("attachment") or {}
            if att.get("kind") == "image" and att.get("path"):
                images.append((str(att["path"]), str(att.get("ref") or "")))
    return text, images


def _render_recall_row(hit: Any, rows: list[Any], *, cap: int, head: str) -> str:
    """One original row in full (capped at ``cap`` chars, remainder counted), preceded by the
    assistant call that produced it when the row is a tool result."""
    blocks: list[str] = []
    if hit.role == "tool" and hit.tool_call_id:
        for a in rows:
            for tc in (a.tool_calls or []) if a.role == "assistant" else []:
                if str(tc.get("id") or "") == str(hit.tool_call_id):
                    fn = tc.get("function")
                    fn = fn if isinstance(fn, dict) else {}
                    blocks.append(
                        f"[seq {a.seq} · assistant call · {fn.get('name') or tc.get('name') or '?'}]\n"
                        f"{fn.get('arguments') or ''}"
                    )
                    break
    body, row_images = _recall_row_body(hit)
    total = len(body)
    if total > cap:
        body = body[:cap] + f" …[truncated: {total - cap} more chars of {total}]"
    label = f"[seq {hit.seq} · {hit.role}" + (f" · {hit.name}" if hit.name else "") + "]"
    blocks.append(f"{label}\n{body}")
    # This row held image(s): put them back in front of the model for ONE round. A tool result
    # cannot carry an image (OpenAI-compatible `tool` messages are text-only), so the picture
    # arrives as its own user message and this text just says it is there.
    if row_images:
        from power_loop.core.agent_context import get_session_id
        from power_loop.runtime.image_recall import queue_image_for_next_round

        sid = get_session_id()
        shown = [
            path for path, ref in row_images
            if queue_image_for_next_round(
                sid, path=path, ref=ref, note=f"（{head} 里的图，已放回你眼前）"
            )
        ]
        if shown:
            blocks.append(
                f"[{len(shown)} 张图已放到你眼前——它们只在这一轮可见，看完就没了；"
                "还要再看就再 recall 一次。]"
            )
        else:
            blocks.append(
                "[这一行有图，但没能放进上下文（会话不可用或本轮排队已满）；"
                "可以用 see_image 直接看它的路径。]"
            )
    return f"{head} — original row ({total} chars):\n\n" + "\n\n".join(blocks)


# Async tool handlers are registered as ``async def`` adapters (NOT sync lambdas that
# merely return a coroutine): the registry uses ``inspect.iscoroutinefunction`` to decide
# sync-vs-async dispatch and to keep the per-call ``runtime_env_context`` held across the
# await — a sync lambda returning a coroutine would let the env reset before the body runs.
async def _h_todo(**kw: Any) -> Any:
    return await run_todo(kw["items"])


async def _h_note(**kw: Any) -> Any:
    return await run_note(
        kw["action"], kw.get("note_id"), kw.get("content"), kw.get("pinned")
    )


async def _h_schedule_wakeup(**kw: Any) -> Any:
    return await run_schedule_wakeup(
        kw["action"],
        kw.get("delay_seconds"),
        kw.get("note"),
        kw.get("every_seconds"),
        kw.get("timer_id"),
    )


async def _h_recall_compacted(**kw: Any) -> Any:
    return await run_recall_compacted(
        kw.get("query"), kw.get("from_seq"), kw.get("to_seq"), kw.get("limit", 20)
    )


async def _h_recall_send(**kw: Any) -> Any:
    return await run_recall_send(kw["send_index"], kw.get("seq"))


async def _h_background_run(**kw: Any) -> Any:
    return await run_background(kw["action"], kw.get("command"), kw.get("task_id"),
                                kw.get("tool"), kw.get("args"))


DEFAULT_TOOL_HANDLERS: dict[str, Any] = {
    "bash": lambda **kw: run_bash(kw.get("command"), kw.get("restart", False), kw.get("timeout", 120)),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("offset"), kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"], kw.get("overwrite", True)),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"], kw.get("replace_all", False)),
    "apply_patch": lambda **kw: run_apply_patch(kw["path"], kw["patch"]),
    "glob": lambda **kw: run_glob(
        kw["pattern"],
        kw.get("path", "."),
        kw.get("max_results", 100),
        kw.get("include_hidden", False),
    ),
    "grep": lambda **kw: run_grep(
        kw["pattern"],
        kw.get("path", "."),
        kw.get("include"),
        kw.get("max_results", 50),
        kw.get("literal", False),
        kw.get("case_sensitive", True),
        kw.get("include_hidden", False),
    ),
    "load_skill": lambda **kw: run_load_skill(kw["name"]),
    "todo": _h_todo,
    "note": _h_note,
    "schedule_wakeup": _h_schedule_wakeup,
    "current_time": lambda **kw: run_current_time(),
    "recall_compacted": _h_recall_compacted,
    "recall_send": _h_recall_send,
    "background_run": _h_background_run,
    "request_user_input": lambda **kw: request_user_input(
        kind=kw.get("kind", "text"),
        prompt=kw["prompt"],
        options=kw.get("options") or [],
        metadata=kw.get("metadata") or {},
    ),
}
