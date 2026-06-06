from __future__ import annotations

import difflib
import fnmatch
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from power_loop.core.agent_context import get_ctx, get_current_loop, get_session_id
from power_loop.runtime.env import AGENT_DIR, AGENT_RW_ALLOWLIST, WORKSPACE_DIR, safe_path
from power_loop.runtime.skills import get_default_loader

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

FILE_READ_STATE: dict[str, tuple[int, int]] = {}


def _file_stamp(fp: Path) -> tuple[int, int]:
    stat = fp.stat()
    return stat.st_mtime_ns, stat.st_size


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(WORKSPACE_DIR):
        return str(resolved.relative_to(WORKSPACE_DIR))
    if resolved.is_relative_to(AGENT_DIR):
        return f"@agent/{resolved.relative_to(AGENT_DIR)}"
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
    textish = sum(byte in b"\n\r\t\b\f" or 32 <= byte <= 126 for byte in sample)
    return textish / len(sample) < 0.70


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
    FILE_READ_STATE[str(fp.resolve())] = _file_stamp(fp)


def _check_read_state(fp: Path) -> str | None:
    key = str(fp.resolve())
    if key not in FILE_READ_STATE:
        return f"Error: File has not been read yet. Use read_file first before modifying: {_display_path(fp)}"
    if fp.exists() and _file_stamp(fp) != FILE_READ_STATE[key]:
        return f"Error: File changed since last read. Re-read it before modifying: {_display_path(fp)}"
    return None


def _detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1 or crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx == lf_idx - 1 else "\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


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
    resolved = path.resolve()
    if resolved.is_relative_to(WORKSPACE_DIR):
        rel = resolved.relative_to(WORKSPACE_DIR)
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


def _is_agent_path_allowed_for_bash(command: str) -> bool:
    lowered = command.lower()
    if str(AGENT_DIR).lower() not in lowered:
        return True
    return any(str(path).lower() in lowered for path in AGENT_RW_ALLOWLIST)


def _validate_bash_command_scope(command: str) -> str | None:
    lowered = f" {command.lower()} "
    if str(AGENT_DIR).lower() not in lowered:
        return None
    if _is_agent_path_allowed_for_bash(command):
        return None

    if any(hint in lowered for hint in _BASH_WRITE_HINTS):
        return (
            "Error: Writing under agent home is blocked outside allowlisted paths (.cache/logs/skills). "
            "Use workspace files or allowlisted agent paths only."
        )
    if any(hint in lowered for hint in _BASH_READ_HINTS):
        return (
            "Error: Reading agent-home internals is blocked outside allowlisted paths (.cache/logs/skills). "
            "Use load_skill(name) for skill content instead of direct file reads."
        )
    return None


def _dangerous_command_reason(command: str) -> str | None:
    lowered = command.strip().lower()
    compact = re.sub(r"\s+", " ", lowered)
    if re.search(r"\brm\s+(-[a-z]*[rf][a-z]*\s+|-[a-z]*[rf][a-z]*.*\s)(/|~|\$home)(\s|$)", compact):
        return "refusing recursive deletion of a root/home path"
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
    """Persistent bash process with stdout/stderr merged through a pty."""

    def __init__(self, cwd: Path):
        self._cwd = cwd
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

        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"

        self._proc = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=slave_fd,
            stderr=slave_fd,
            text=True,
            bufsize=0,
            cwd=str(self._cwd),
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
        lines: list[str] = []
        exit_code: str | None = None
        deadline = time.monotonic() + timeout

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
            lines.append(line)

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
            lines.append(line)
        return lines, exit_code

    def execute(self, command: str, timeout: int = 120) -> str:
        reason = _dangerous_command_reason(command)
        if reason:
            return f"Error: Dangerous command blocked ({reason}). Run it manually if you really intend it."

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


BASH = BashSession(WORKSPACE_DIR)


def run_bash(command: str | None = None, restart: bool = False, timeout: int = 120) -> str:
    if restart:
        return BASH.restart()
    if not command or not command.strip():
        return "Error: command is required (or set restart=true)."
    timeout = _clamp_int(timeout, 120, 1, 600)
    scope_err = _validate_bash_command_scope(command)
    if scope_err:
        return scope_err
    return BASH.execute(command, timeout=timeout)


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

        text = _read_text(fp, max_bytes=None if limit is not None else TEXT_FILE_MAX_BYTES)
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
            + (f"\n{diff_output}" if diff_output else "")
        )
    except Exception as e:
        return f"Error: {e}"


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
        replacement = [text for op, text in hunk.changes if op in (" ", "+")]

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
        return f"Patched {_display_path(fp)} ({len(hunks)} hunk{'s' if len(hunks) != 1 else ''})\n{diff_output}"
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
            if _should_skip_dir(candidate, base, include_hidden, pattern):
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
                results.append(f"{_display_path(fp)}:{line_no}:{line.rstrip()}")
                if len(results) >= max_results:
                    return "\n".join(results)
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
            "--glob",
            "!.git/**",
            "--glob",
            "!node_modules/**",
            "--glob",
            "!dist/**",
            "--glob",
            "!build/**",
        ]
        if include_hidden:
            rg_cmd.append("--hidden")
        if literal:
            rg_cmd.append("--fixed-strings")
        if not case_sensitive:
            rg_cmd.append("--ignore-case")
        if include:
            rg_cmd.extend(["--glob", include])
        if not include_hidden:
            rg_cmd.extend(["--glob", "!.*", "--glob", "!**/.*"])
        rg_cmd.extend(["--", pattern, _safe_relative_for_subprocess(base)])

        try:
            result = subprocess.run(
                rg_cmd,
                cwd=str(WORKSPACE_DIR),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode > 1:
                return f"Error: {result.stderr.strip() or 'rg failed'}"
            lines = [line for line in result.stdout.splitlines() if line][:max_results]
            if not lines:
                return f"No matches for '{pattern}' in {_display_path(base)}"
            if len(result.stdout.splitlines()) > max_results:
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
    loop = get_current_loop()
    skills_dir = getattr(getattr(loop, "config", None), "skills_dir", None) if loop is not None else None
    if skills_dir:
        from power_loop.runtime.skills import SkillLoader

        return SkillLoader(skills_dir).get_content(name)
    return get_default_loader().get_content(name)


def _current_store_and_session() -> tuple[Any | None, str | None]:
    loop = get_current_loop()
    sid = get_session_id()
    store = getattr(loop, "store", None) if loop is not None else None
    return store, sid


def _render_todos(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No todos."
    lines: list[str] = []
    for item in items:
        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
        lines.append(f"{marker} #{item['id']}: {item['text']}")
    done = sum(1 for item in items if item["status"] == "completed")
    lines.append(f"\n({done}/{len(items)} completed)")
    return "\n".join(lines)


def _validate_todos(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(items) > 20:
        raise ValueError("Max 20 todos allowed")
    validated: list[dict[str, Any]] = []
    in_progress_count = 0
    for i, item in enumerate(items):
        text = str(item.get("text", "")).strip()
        status = str(item.get("status", "pending")).lower()
        item_id = str(item.get("id", str(i + 1)))
        if not text:
            raise ValueError(f"Item {item_id}: text required")
        if status not in ("pending", "in_progress", "completed"):
            raise ValueError(f"Item {item_id}: invalid status '{status}'")
        if status == "in_progress":
            in_progress_count += 1
        validated.append({"id": item_id, "text": text, "status": status})
    if in_progress_count > 1:
        raise ValueError("Only one task can be in_progress at a time")
    return validated


class BackgroundManager:
    """Background command runner with task tracking."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        reason = _dangerous_command_reason(command)
        if reason:
            return f"Error: Dangerous command blocked ({reason}). Run it manually if you really intend it."

        scope_err = _validate_bash_command_scope(command)
        if scope_err:
            return scope_err

        task_id = str(uuid.uuid4())[:8]
        store, sid = _current_store_and_session()
        if store is not None and sid is not None:
            store.upsert_background_task(
                sid,
                task_id=task_id,
                command=command,
                status="running",
                output_tail=None,
            )
        with self._lock:
            self.tasks[task_id] = {"status": "running", "result": None, "command": command, "session_id": sid, "store": store}

        threading.Thread(target=self._execute, args=(task_id, command), daemon=True).start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str) -> None:
        store = None
        sid = None
        with self._lock:
            task = self.tasks.get(task_id)
            if task is not None:
                store = task.get("store")
                sid = task.get("session_id")
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(WORKSPACE_DIR),
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
            try:
                store.upsert_background_task(
                    sid,
                    task_id=task_id,
                    command=command,
                    status=status,
                    return_code=return_code,
                    output_tail=output or "(no output)",
                )
            except Exception:
                pass

    def check(self, task_id: str | None = None) -> str:
        store, sid = _current_store_and_session()
        if store is not None and sid is not None:
            if task_id:
                row = store.get_background_task(sid, task_id)
                if row is None:
                    return f"Error: Unknown task {task_id}"
                return f"[{row.status}] {row.command[:80]}\n{row.output_tail or '(running)'}"
            rows = store.list_background_tasks(sid)
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


def run_background(command: str) -> str:
    return BG.run(command)


def check_background(task_id: str | None = None) -> str:
    return BG.check(task_id)


def run_todo(items: list[dict[str, Any]]) -> str:
    validated = _validate_todos(items)
    store, sid = _current_store_and_session()
    if store is not None and sid is not None:
        rendered = _render_todos(validated)
        done = sum(1 for item in validated if item["status"] == "completed")
        store.set_runtime_state(
            sid,
            "todo",
            {
                "items": validated,
                "rendered": rendered,
                "counts": {"total": len(validated), "completed": done},
            },
        )
        # Keep the in-process context warm for UI/event compatibility.
        get_ctx().todo.update(validated)
        return rendered
    return get_ctx().todo.update(validated)


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
    "todo": lambda **kw: run_todo(kw["items"]),
    "background_run": lambda **kw: run_background(kw["command"]),
    "check_background": lambda **kw: check_background(kw.get("task_id")),
}
