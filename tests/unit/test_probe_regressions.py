"""Regression tests for 7 bugs found by a coverage-driven probe of under-tested
modules (default file/search tools, budget, context state). Each asserts the
CORRECT post-fix behavior and fails on the pre-fix code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from power_loop.core.state import ContextManager
from power_loop.runtime.budget import estimate_tokens, trim_history
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools import default_tools
from power_loop.tools.default_tools import (
    FILE_READ_STATE,
    run_edit,
    run_glob,
    run_grep,
    run_read,
)

pytestmark = pytest.mark.unit


def _ws(tmp_path: Path):
    FILE_READ_STATE.clear()
    return runtime_env_context(RuntimeEnv(workspace_dir=tmp_path.resolve()))


# ── #1 glob: skip-dirs pruned for slash patterns without ** ──────────────────


def test_glob_prunes_skip_dirs_for_slash_pattern(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.py").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x")
    with _ws(tmp_path):
        out = run_glob("*/*.py")
        out_star = run_glob("**/*.py")
    assert "src/app.py" in out
    assert "node_modules" not in out  # was leaked by the non-** branch
    # the two glob branches must agree on which files match
    assert ("node_modules" in out) == ("node_modules" in out_star)


# ── #2 apply_patch: untouched context-line bytes preserved ───────────────────


def test_apply_patch_preserves_context_line_trailing_whitespace(tmp_path: Path) -> None:
    with _ws(tmp_path):
        (tmp_path / "f.txt").write_text("keep   \nold\ntail\n")
        run_read("f.txt")
        # context line is given WITHOUT the trailing spaces the file has (fuzzy match)
        patch = "@@ -1,2 +1,2 @@\n keep\n-old\n+new\n"
        default_tools.run_apply_patch("f.txt", patch)
        assert (tmp_path / "f.txt").read_text() == "keep   \nnew\ntail\n"


# ── #3 grep python fallback: truncation notice like the rg path ──────────────


def test_grep_fallback_emits_truncation_notice(tmp_path: Path, monkeypatch) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("NEEDLE\n")

    def _no_rg(*a, **k):  # force the python fallback
        raise FileNotFoundError("rg not installed")

    monkeypatch.setattr(subprocess, "run", _no_rg)
    with _ws(tmp_path):
        out = run_grep("NEEDLE", path=".", max_results=3)
    assert "truncated" in out
    assert len([ln for ln in out.splitlines() if ":NEEDLE" in ln]) == 3


# ── #6 edit: line endings preserved for CR-only and CRLF-with-stray-LF ───────


@pytest.mark.parametrize(
    "before,expected",
    [
        (b"a\rb\rc\r", b"a\rB\rc\r"),              # classic-Mac CR-only (was flattened to LF)
        (b"x\na\r\nb\r\n", b"x\r\na\r\nB\r\n"),    # CRLF-majority w/ early stray LF (was flattened)
        (b"a\r\nb\r\n", b"a\r\nB\r\n"),            # pure CRLF unchanged
        (b"a\nb\n", b"a\nB\n"),                    # pure LF unchanged
    ],
)
def test_edit_preserves_dominant_line_ending(tmp_path: Path, before: bytes, expected: bytes) -> None:
    with _ws(tmp_path):
        (tmp_path / "f").write_bytes(before)
        run_read("f")
        run_edit("f", "b", "B")
        assert (tmp_path / "f").read_bytes() == expected


# ── #4 trim_history: never drops the system block to empty ───────────────────


def test_trim_history_keeps_system_when_oversized(tmp_path: Path) -> None:
    msgs = [
        {"role": "system", "content": "s" * 4000},
        {"role": "system", "content": "t" * 4000},
    ]
    budget = estimate_tokens(msgs) // 2  # system block alone exceeds budget
    result = trim_history(msgs, max_tokens=budget, keep_system=True, keep_last_n=2)
    assert result, "must never return an empty history for a non-empty input"
    assert all(m["role"] == "system" for m in result)


# ── #5 update_usage: derive total when the provider omits it ─────────────────


def test_update_usage_derives_total_tokens() -> None:
    from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage

    cm = ContextManager()
    cm.update_usage(LLMResponse(raw_text="x", token_usage=LLMTokenUsage(prompt_tokens=100, completion_tokens=50)))
    assert cm.usage_totals["total_tokens"] == 150  # was 0 → broke SharedBudget


# ── #7 microcompact: micro_hot_tail=0 spills ALL oversized outputs ───────────


def test_microcompact_hot_tail_zero_spills_all(tmp_path: Path) -> None:
    cm = ContextManager(micro_hot_tail=0, micro_size_limit=100)
    cm.cache_dir = tmp_path / "cache"
    big = "Z" * 500
    msgs: list = [{"role": "user", "content": "hi"}]
    for i in range(4):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"id{i}", "type": "function",
                                     "function": {"name": f"t{i}", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"id{i}", "name": f"t{i}", "content": big})
    cm.microcompact(msgs)
    spilled = [m for m in msgs if m.get("role") == "tool" and str(m.get("content", "")).startswith("[tool output saved")]
    assert len(spilled) == 4  # was 0 due to the [:-0] off-by-one
