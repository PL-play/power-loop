"""Unit tests for M1.2 — trim_history.

Covers the invariants from ROADMAP §M1.2:
- Already fits → no change.
- Leading system block preserved.
- Last N user-bounded exchanges preserved.
- assistant(tool_calls) ↔ tool atomic pair not split.
- Empty / zero budget edges.
- Tail-only when budget is too tight.
"""

from __future__ import annotations

from power_loop import trim_history


def _msg(role: str, content: str = "x", **kw) -> dict:
    m: dict = {"role": role, "content": content}
    m.update(kw)
    return m


def _est(msgs: list[dict]) -> int:
    from power_loop.runtime.budget import estimate_tokens
    return estimate_tokens(msgs)


# ── No-op: budget already fits ──────────────────────────────────────────


def test_trim_already_fits_returns_unchanged() -> None:
    msgs = [_msg("system", "sys"), _msg("user", "hi"), _msg("assistant", "ok")]
    result = trim_history(msgs, max_tokens=10_000)
    assert result == msgs


def test_trim_empty_returns_empty() -> None:
    assert trim_history([], 100) == []


def test_trim_zero_budget_returns_empty() -> None:
    assert trim_history([_msg("user", "hi")], 0) == []


# ── System block preservation ────────────────────────────────────────────


def test_trim_keeps_leading_system() -> None:
    msgs = [
        _msg("system", "short sys"),
        _msg("system", "another note"),
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
        _msg("user", "q3"), _msg("assistant", "a3"),
        _msg("user", "q4"), _msg("assistant", "a4"),
    ]
    # Budget: exactly system + last 2 exchanges (q3/a3 + q4/a4), no room for q1/a1.
    tight = _est(msgs[:2]) + _est(msgs[-4:])
    result = trim_history(msgs, max_tokens=tight, keep_last_n=2)
    assert result[0] == msgs[0]
    assert result[1] == msgs[1]
    # Last 2 exchanges preserved.
    assert msgs[-2] in result
    assert msgs[-1] in result
    assert msgs[-4] in result
    assert msgs[-3] in result
    # q1/a1 dropped.
    assert msgs[2] not in result


def test_trim_keep_system_false_drops_system() -> None:
    msgs = [
        _msg("system", "s" * 200),
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
        _msg("user", "q3"), _msg("assistant", "a3"),
    ]
    # Budget: enough for last 2 exchanges but not full history.
    result = trim_history(msgs, max_tokens=_est(msgs[-4:]) + 5, keep_system=False, keep_last_n=2)
    assert not any(m.get("role") == "system" for m in result)
    # Last 2 exchanges still present.
    assert msgs[-2] in result
    assert msgs[-1] in result


# ── Tool-call atomicity ─────────────────────────────────────────────────


def test_trim_does_not_split_tool_call_pair() -> None:
    """If the cut would separate tool from its assistant(tool_calls), we
    walk back to include the assistant."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "call",
                                 tool_calls=[{"id": "tc1", "function": {"name": "e", "arguments": "{}"}}]),
        _msg("tool", "result", tool_call_id="tc1", name="e"),
        _msg("assistant", "done"),
        _msg("user", "q3"), _msg("assistant", "a3"),
    ]
    # Tight budget: only fit system + last exchange (q3/a3).
    tight = _est(msgs[-2:]) + _est(msgs[:1]) + 10
    result = trim_history(msgs, max_tokens=tight, keep_last_n=1)
    # The tool/tool_call_id pair near the boundary must be intact.
    found_tool = any(m.get("role") == "tool" for m in result)
    found_tc = any(m.get("tool_calls") for m in result)
    assert found_tool == found_tc  # either both present or both absent


def test_trim_tool_call_id_pair_preserved_when_at_boundary() -> None:
    """The assistant(tool_calls) + tool pair is the last exchange before
    a new user message. The pair must stay together."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"),
        _msg("assistant", "call", tool_calls=[{"id": "tc1", "function": {"name": "e", "arguments": "{}"}}]),
        _msg("tool", "r", tool_call_id="tc1", name="e"),
        _msg("assistant", "final"),
        _msg("user", "q3"), _msg("assistant", "a3"),
    ]
    # Budget: system + last exchange, but NOT enough for the tool_call pair.
    # The pair should be preserved as an atomic unit — if we keep q2, we
    # must also keep the tool/tool pair that follows it.
    result = trim_history(msgs, max_tokens=_est(msgs) // 2, keep_last_n=2)
    # q3/a3 must be present (last exchange).
    assert msgs[-2] in result
    assert msgs[-1] in result
    # The tool pair must be either fully present or fully absent.
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    tc_msgs = [m for m in result if m.get("tool_calls")]
    assert (bool(tool_msgs)) == (bool(tc_msgs))


# ── Tail-only when budget too tight ─────────────────────────────────────


def test_trim_tail_only_when_system_too_big() -> None:
    msgs = [
        _msg("system", "s" * 500),
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
    ]
    # Budget too small for system + any body but fits last exchange.
    tail_cost = _est(msgs[-2:])
    result = trim_history(msgs, max_tokens=tail_cost + 5, keep_last_n=1)
    assert result
    assert not any(m.get("role") == "system" for m in result)
    # Last exchange must be present.
    assert msgs[-2] in result
    assert msgs[-1] in result


# ── Non-mutation ────────────────────────────────────────────────────────


def test_trim_does_not_mutate_input() -> None:
    msgs = [
        _msg("system", "sys"),
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
        _msg("user", "q3"), _msg("assistant", "a3"),
    ]
    original = [dict(m) for m in msgs]
    trim_history(msgs, max_tokens=_est(msgs[:1]) + _est(msgs[-4:]) + 5, keep_last_n=2)
    assert msgs == original