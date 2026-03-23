from __future__ import annotations

# NOTE: This module intentionally copies tool implementations from zero-code/core/tools.py
# with only import-path adjustments for power-loop package layout.

import difflib
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from power_loop.core.agent_context import get_ctx
from power_loop.runtime.env import AGENT_DIR, AGENT_RW_ALLOWLIST, WORKSPACE_DIR, safe_path
from power_loop.runtime.skills import SKILL_LOADER

RESULT_MAX_CHARS = 50000
_HEAD_LINES = 30
_TAIL_LINES = 170
SENTINEL = "___ZERO_CODE_CMD_DONE___"

FILE_READ_STATE: dict[str, float] = {}


def _truncate_output(lines: list[str], head: int = _HEAD_LINES, tail: int = _TAIL_LINES) -> str:
    limit = head + tail
    if len(lines) <= limit:
        return "\n".join(lines)
    omitted = len(lines) - limit
    return (
        "\n".join(lines[:head])
        + f"\n\n... ({omitted} lines omitted) ...\n\n"
        + "\n".join(lines[-tail:])
    )


class BashSession:
    """Persistent bash process with merged stdout/stderr via pty."""

    def __init__(self, cwd: Path):
        self._cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[str] = queue.Queue()
        self._master_fd: int | None = None
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
                    candidates = [i for i in (idx_n, idx_r) if i != -1]
                    cut = min(candidates)
                    line, buf = buf[:cut], buf[cut + 1 :]
                    if line:
                        self._q.put(line)
            if buf:
                self._q.put(buf)
        except Exception:
            pass

    def _drain(self, timeout: float, idle_timeout: float = 5.0) -> tuple[list[str], str | None]:
        lines: list[str] = []
        exit_code: str | None = None
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = min(remaining, idle_timeout)
            try:
                line = self._q.get(timeout=wait)
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    while not self._q.empty():
                        try:
                            line = self._q.get_nowait()
                            cleaned = re.sub(r"\\x1b\\[[0-9;]*[A-Za-z]", "", line).rstrip("\r")
                            if SENTINEL in cleaned:
                                parts = cleaned.strip().split()
                                if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                                    exit_code = parts[-1]
                                break
                            lines.append(cleaned)
                        except queue.Empty:
                            break
                    break
                break

            cleaned = re.sub(r"\\x1b\\[[0-9;]*[A-Za-z]", "", line).rstrip("\r")
            if SENTINEL in cleaned:
                parts = cleaned.strip().split()
                if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                    exit_code = parts[-1]
                break
            lines.append(cleaned)

        return lines, exit_code

    def execute(self, command: str, timeout: int = 120) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return (
                "Error: Dangerous command blocked.\n"
                "For safety, interactive or privileged commands (like sudo / shutdown) "
                "must be run manually in your own terminal, not via the agent bash tool."
            )

        if self._proc is None or self._proc.poll() is not None:
            self._start()

        full_cmd = f"{command}\\necho {SENTINEL} $?\\n"
        try:
            self._proc.stdin.write(full_cmd)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._start()
            return "Error: Bash session crashed, restarted. Please retry."

        lines, exit_code = self._drain(timeout, idle_timeout=5.0)
        timed_out = exit_code is None

        if timed_out and self._proc is not None and self._proc.poll() is not None:
            exit_code = str(self._proc.returncode)
            timed_out = False

        header = f"exit_code={exit_code or '?'}"
        if timed_out:
            header += f"  (timed out after {timeout}s — command may still be running)"

        body = _truncate_output(lines) if lines else "(no output)"
        return f"{header}\\n{body}"[:RESULT_MAX_CHARS]

    def restart(self) -> str:
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._start()
        return "Bash session restarted."


BASH = BashSession(WORKSPACE_DIR)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(WORKSPACE_DIR):
        return str(resolved.relative_to(WORKSPACE_DIR))
    if resolved.is_relative_to(AGENT_DIR):
        return f"@agent/{resolved.relative_to(AGENT_DIR)}"
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
            "Error: Writing under agent home is blocked outside allowlisted paths (.cache/logs). "
            "Use workspace files or allowlisted agent paths only."
        )
    if any(hint in lowered for hint in _BASH_READ_HINTS):
        return (
            "Error: Reading agent-home internals is blocked outside allowlisted paths (.cache/logs). "
            "Use load_skill(name) for skill content instead of direct file reads."
        )
    return None


def run_bash(command: str = None, restart: bool = False, timeout: int = 120) -> str:
    if restart:
        return BASH.restart()
    if not command:
        return "Error: command is required (or set restart=true)"
    timeout = max(5, min(int(timeout), 600))
    scope_err = _validate_bash_command_scope(command)
    if scope_err:
        return scope_err
    return BASH.execute(command, timeout=timeout)


def _list_directory(dp: Path) -> str:
    entries = sorted(dp.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = [f"Directory: {_display_path(dp)}/"]
    for entry in entries[:100]:
        prefix = "d " if entry.is_dir() else "f "
        size = ""
        if entry.is_file():
            size = f" ({entry.stat().st_size} bytes)"
        lines.append(f"  {prefix}{entry.name}{size}")
    if len(entries) > 100:
        lines.append(f"  ... and {len(entries) - 100} more entries")
    return "\\n".join(lines)


def run_read(path: str, offset: int = None, limit: int = None) -> str:
    try:
        fp = safe_path(path)
        if fp.is_dir():
            return _list_directory(fp)
        text = fp.read_text()
        all_lines = text.splitlines()
        total = len(all_lines)
        start = max(0, (offset or 1) - 1)
        end = min(total, start + limit) if limit else total
        selected = all_lines[start:end]
        numbered = [f"{start + i + 1:>6}|{line}" for i, line in enumerate(selected)]
        header = f"({total} lines total)"
        if start > 0 or end < total:
            header = f"(showing lines {start+1}-{end} of {total})"
        FILE_READ_STATE[str(fp)] = fp.stat().st_mtime
        return header + "\\n" + "\\n".join(numbered)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        existed = fp.exists()
        old_size = fp.stat().st_size if existed else 0
        fp.write_text(content)
        line_count = content.count("\\n") + (1 if content and not content.endswith("\\n") else 0)
        display = _display_path(fp)
        if existed:
            return f"Wrote {len(content)} bytes ({line_count} lines) to {display} (overwritten, was {old_size} bytes) [workspace: {WORKSPACE_DIR}]"
        return f"Wrote {len(content)} bytes ({line_count} lines) to {display} (new file) [workspace: {WORKSPACE_DIR}]"
    except Exception as e:
        return f"Error: {e}"


def _check_read_state(fp: Path) -> str | None:
    key = str(fp)
    if key not in FILE_READ_STATE:
        return f"Error: File has not been read yet. Use read_file first before editing: {fp.name}"
    if fp.exists():
        current_mtime = fp.stat().st_mtime
        if current_mtime > FILE_READ_STATE[key]:
            return f"Error: File was modified since last read. Re-read it first: {fp.name}"
    return None


def _detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\\r\\n")
    lf_idx = content.find("\\n")
    if lf_idx == -1:
        return "\\n"
    if crlf_idx == -1:
        return "\\n"
    return "\\r\\n" if crlf_idx < lf_idx else "\\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\\r\\n", "\\n").replace("\\r", "\\n")


def _restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\\n", "\\r\\n") if ending == "\\r\\n" else text


def _strip_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def _normalize_unicode(text: str) -> str:
    lines = text.split("\\n")
    stripped = "\\n".join(line.rstrip() for line in lines)
    result = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", stripped)
    result = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', result)
    result = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", result)
    result = re.sub(r"[\u00a0\u2002-\u200a\u202f\u205f\u3000]", " ", result)
    return result


def _fuzzy_find(content: str, old_text: str) -> tuple[int, int, str] | None:
    idx = content.find(old_text)
    if idx != -1:
        return idx, idx + len(old_text), content

    lf_content = _normalize_to_lf(content)
    lf_old = _normalize_to_lf(old_text)
    idx = lf_content.find(lf_old)
    if idx != -1:
        return idx, idx + len(lf_old), lf_content

    uni_content = _normalize_unicode(lf_content)
    uni_old = _normalize_unicode(lf_old)
    idx = uni_content.find(uni_old)
    if idx != -1:
        return idx, idx + len(uni_old), uni_content

    trim_content = "\\n".join(line.strip() for line in lf_content.split("\\n"))
    trim_old = "\\n".join(line.strip() for line in lf_old.split("\\n"))
    idx = trim_content.find(trim_old)
    if idx != -1:
        return idx, idx + len(trim_old), trim_content

    return None


def _generate_diff(old_content: str, new_content: str, context: int = 3) -> str:
    old_lines = old_content.split("\\n")
    new_lines = new_content.split("\\n")
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="", n=context)
    return "\\n".join(diff)


def run_edit(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    try:
        fp = safe_path(path)

        read_err = _check_read_state(fp)
        if read_err:
            return read_err

        raw_content = fp.read_text()
        bom, content = _strip_bom(raw_content)
        original_ending = _detect_line_ending(content)
        normalized = _normalize_to_lf(content)
        norm_old = _normalize_to_lf(old_text)
        norm_new = _normalize_to_lf(new_text)

        if norm_old == norm_new:
            return "Error: old_text and new_text are identical."

        if replace_all:
            match = _fuzzy_find(normalized, norm_old)
            if match is None:
                return f"Error: Text not found in {path}. Provide a larger unique snippet."
            _, _, base = match
            if base == normalized:
                count = base.count(norm_old)
                updated = base.replace(norm_old, norm_new)
            else:
                search_key = _normalize_unicode(norm_old)
                count = base.count(search_key)
                updated = base.replace(search_key, norm_new)
        else:
            match = _fuzzy_find(normalized, norm_old)
            if match is None:
                return f"Error: Text not found in {path}. Provide a larger unique snippet."
            start, end, base = match

            fuzzy_base = _normalize_unicode(_normalize_to_lf(base))
            fuzzy_old = _normalize_unicode(norm_old)
            occurrence_count = fuzzy_base.split(fuzzy_old)
            count = len(occurrence_count) - 1
            if count > 1:
                positions = []
                search_start = 0
                for _ in range(min(count, 5)):
                    idx = fuzzy_base.find(fuzzy_old, search_start)
                    if idx == -1:
                        break
                    line_no = fuzzy_base[:idx].count("\\n") + 1
                    positions.append(str(line_no))
                    search_start = idx + 1
                return (
                    f"Error: old_text matches {count} locations in {path} (lines: {', '.join(positions)}). "
                    "Provide more surrounding context to make it unique, or use replace_all=true."
                )

            updated = base[:start] + norm_new + base[end:]
            count = 1

        diff_output = _generate_diff(base, updated)
        final = bom + _restore_line_endings(updated, original_ending)
        fp.write_text(final)
        FILE_READ_STATE[str(fp)] = fp.stat().st_mtime

        label = "replace_all" if replace_all else ("fuzzy match" if base != normalized else "exact")
        return f"Edited {path} ({label}, {count} replacement{'s' if count != 1 else ''})\\n{diff_output}"
    except Exception as e:
        return f"Error: {e}"


def _seek_context(lines: list[str], context_lines: list[str], start_from: int = 0) -> int | None:
    def _match_line(file_line: str, ctx_line: str) -> bool:
        if file_line == ctx_line:
            return True
        if _normalize_unicode(file_line) == _normalize_unicode(ctx_line):
            return True
        if file_line.rstrip() == ctx_line.rstrip():
            return True
        if file_line.strip() == ctx_line.strip():
            return True
        return False

    if not context_lines:
        return start_from

    for i in range(start_from, len(lines)):
        if _match_line(lines[i], context_lines[0]):
            if len(context_lines) == 1:
                return i
            all_match = True
            for j, ctx in enumerate(context_lines[1:], 1):
                if i + j >= len(lines) or not _match_line(lines[i + j], ctx):
                    all_match = False
                    break
            if all_match:
                return i
    return None


def _parse_patch(patch_text: str) -> list[dict[str, Any]]:
    hunks = []
    current_context = []
    current_changes = []

    for raw_line in patch_text.split("\\n"):
        if raw_line.startswith("@@"):
            if current_changes:
                hunks.append({"context": current_context, "changes": current_changes})
                current_context = []
                current_changes = []
            ctx_text = raw_line[2:].strip() if len(raw_line) > 2 else ""
            if ctx_text:
                current_context.append(ctx_text)
        elif raw_line.startswith("-"):
            current_changes.append(("-", raw_line[1:]))
        elif raw_line.startswith("+"):
            current_changes.append(("+", raw_line[1:]))
        elif raw_line.startswith(" "):
            current_changes.append((" ", raw_line[1:]))

    if current_changes:
        hunks.append({"context": current_context, "changes": current_changes})

    return hunks


def run_apply_patch(path: str, patch: str) -> str:
    try:
        fp = safe_path(path)

        read_err = _check_read_state(fp)
        if read_err:
            return read_err

        raw_content = fp.read_text()
        bom, content = _strip_bom(raw_content)
        original_ending = _detect_line_ending(content)
        normalized = _normalize_to_lf(content)
        lines = normalized.split("\\n")

        hunks = _parse_patch(patch)
        if not hunks:
            return "Error: No valid hunks found in patch. Use @@ for context and +/- for changes."

        cursor = 0
        for hi, hunk in enumerate(hunks):
            ctx = hunk["context"]
            changes = hunk["changes"]

            if ctx:
                pos = _seek_context(lines, ctx, cursor)
                if pos is None:
                    return (
                        f"Error: Could not locate context for hunk {hi + 1} in {path}. "
                        f"Context: {ctx!r}"
                    )
                cursor = pos + len(ctx)
            else:
                if hi == 0:
                    cursor = 0

            apply_at = cursor
            i = apply_at
            result_insert = []

            for op, text in changes:
                if op == "-":
                    if i >= len(lines):
                        return (
                            f"Error: Hunk {hi + 1} tries to delete beyond end of file. "
                            f"Expected: {text!r}"
                        )
                    file_line = lines[i]
                    if not (
                        file_line == text
                        or file_line.strip() == text.strip()
                        or _normalize_unicode(file_line) == _normalize_unicode(text)
                    ):
                        return (
                            f"Error: Hunk {hi + 1} delete mismatch at line {i + 1}. "
                            f"Expected: {text!r}, Found: {file_line!r}"
                        )
                    i += 1
                elif op == "+":
                    result_insert.append(text)
                elif op == " ":
                    if i >= len(lines):
                        return (
                            f"Error: Hunk {hi + 1} context line beyond end of file. "
                            f"Expected: {text!r}"
                        )
                    i += 1
                    result_insert.append(lines[i - 1])

            lines[apply_at:i] = result_insert
            cursor = apply_at + len(result_insert)

        new_content = "\\n".join(lines)
        diff_output = _generate_diff(normalized, new_content)
        final = bom + _restore_line_endings(new_content, original_ending)
        fp.write_text(final)
        FILE_READ_STATE[str(fp)] = fp.stat().st_mtime

        return f"Patched {path} ({len(hunks)} hunk{'s' if len(hunks) != 1 else ''})\\n{diff_output}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str, path: str = ".") -> str:
    try:
        base = safe_path(path)
        if not base.is_dir():
            return f"Error: {path} is not a directory"
        if not pattern.startswith("**/") and "/" not in pattern:
            pattern = "**/" + pattern
        matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return f"No files matching '{pattern}' in {_display_path(base)} [workspace: {WORKSPACE_DIR}]"
        header = f"[searched in: {_display_path(base)}, workspace: {WORKSPACE_DIR}]"
        lines = [header] + [_display_path(m) for m in matches[:50]]
        result = "\\n".join(lines)
        if len(matches) > 50:
            result += f"\\n... and {len(matches) - 50} more"
        return result
    except Exception as e:
        return f"Error: {e}"


def run_grep(pattern: str, path: str = ".", include: str = None, max_results: int = 50) -> str:
    try:
        base = safe_path(path)
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", str(max_results), pattern, str(base)]
        if include:
            cmd.extend(["--glob", include])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode > 1:
                err = r.stderr.strip() or "rg failed"
                return f"Error: {err}"
            out = r.stdout.strip()
            if not out:
                return f"No matches for '{pattern}'"
            lines = out.splitlines()[:max_results]
            return "\\n".join(lines)
        except FileNotFoundError:
            compiled = re.compile(pattern)
            results = []
            search_dir = base if base.is_dir() else base.parent
            glob_pat = include or "**/*"
            for fp in search_dir.glob(glob_pat):
                if not fp.is_file():
                    continue
                try:
                    for i, line in enumerate(fp.read_text().splitlines(), 1):
                        if compiled.search(line):
                            results.append(f"{_display_path(fp)}:{i}:{line.rstrip()}")
                            if len(results) >= max_results:
                                break
                except (UnicodeDecodeError, PermissionError):
                    continue
                if len(results) >= max_results:
                    break
            return "\\n".join(results) if results else f"No matches for '{pattern}'"
    except Exception as e:
        return f"Error: {e}"


def run_load_skill(name: str) -> str:
    return SKILL_LOADER.get_content(name)

class BackgroundManager:
    """Background command runner with task tracking."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return (
                "Error: Dangerous command blocked.\n"
                "Background tasks do not support interactive or privileged commands "
                "(like sudo / shutdown). Please run these manually in your own shell."
            )

        scope_err = _validate_bash_command_scope(command)
        if scope_err:
            return scope_err

        task_id = str(uuid.uuid4())[:8]
        with self._lock:
            self.tasks[task_id] = {"status": "running", "result": None, "command": command}

        threading.Thread(target=self._execute, args=(task_id, command), daemon=True).start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str) -> None:
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=str(WORKSPACE_DIR),
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:  # pragma: no cover
            output = f"Error: {e}"
            status = "error"

        with self._lock:
            task = self.tasks.get(task_id)
            if task is not None:
                task["status"] = status
                task["result"] = output or "(no output)"

    def check(self, task_id: str | None = None) -> str:
        with self._lock:
            if task_id:
                task = self.tasks.get(task_id)
                if not task:
                    return f"Error: Unknown task {task_id}"
                return f"[{task['status']}] {task['command'][:60]}\n{task.get('result') or '(running)'}"

            lines: list[str] = []
            for tid, task in self.tasks.items():
                lines.append(f"{tid}: [{task['status']}] {task['command'][:60]}")
            return "\n".join(lines) if lines else "No background tasks."


BG = BackgroundManager()


def run_background(command: str) -> str:
    return BG.run(command)


def check_background(task_id: str | None = None) -> str:
    return BG.check(task_id)


def run_todo(items: list[dict[str, Any]]) -> str:
    return get_ctx().todo.update(items)


# Default tool handlers copied from zero-code mapping style.
DEFAULT_TOOL_HANDLERS: dict[str, Any] = {
    "bash": lambda **kw: run_bash(kw.get("command"), kw.get("restart", False), kw.get("timeout", 120)),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("offset"), kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"], kw.get("replace_all", False)),
    "apply_patch": lambda **kw: run_apply_patch(kw["path"], kw["patch"]),
    "glob": lambda **kw: run_glob(kw["pattern"], kw.get("path", ".")),
    "grep": lambda **kw: run_grep(kw["pattern"], kw.get("path", "."), kw.get("include"), kw.get("max_results", 50)),
    "load_skill": lambda **kw: run_load_skill(kw["name"]),
    "todo": lambda **kw: run_todo(kw["items"]),
    "background_run": lambda **kw: run_background(kw["command"]),
    "check_background": lambda **kw: check_background(kw.get("task_id")),
}
