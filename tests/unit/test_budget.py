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


def _no_dangling_tool_calls(msgs: list[dict]) -> bool:
    """Every assistant(tool_calls) id must be answered by some tool message in the result."""
    answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    return all(
        tc.get("id") in answered
        for m in msgs
        if m.get("role") == "assistant" and m.get("tool_calls")
        for tc in (m.get("tool_calls") or [])
    )


def test_trim_body_front_drops_dangling_tool_call() -> None:
    """H5 (BUG_REVIEW_3.4): the body front-fill keeps a contiguous prefix, so it can stop right
    after an assistant(tool_calls) but before its (large) tool result — leaving a dangling
    tool_call immediately before the tail. Both providers 400 on that; it must be dropped."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "big"),
        _msg("assistant", "", tool_calls=[{"id": "call_1", "function": {"name": "e", "arguments": "{}"}}]),
        _msg("tool", "R" * 8000, tool_call_id="call_1", name="e"),  # ~2000 tok — won't fit
        _msg("assistant", "done"),
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
    ]
    result = trim_history(msgs, max_tokens=70, keep_system=True, keep_last_n=2)
    assert _no_dangling_tool_calls(result), result
    # The call_1 assistant must be dropped (result didn't fit), not kept as a dangling tool_call.
    kept_tc_ids = {tc.get("id") for m in result if m.get("tool_calls") for tc in m["tool_calls"]}
    assert "call_1" not in kept_tc_ids


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


# ── Parallel tool-call group atomicity (R07) ────────────────────────────


def _tc(call_id: str, name: str = "tool") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _assert_valid_tool_groups(msgs: list[dict]) -> list[tuple[list[str], list[str]]]:
    """Verify strict protocol order: every assistant(tool_calls) is immediately
    followed by tool results covering all declared IDs (no dangling calls, no
    orphan results, no cross-user pairing). Returns kept ``(call_ids, result_ids)``."""
    groups: list[tuple[list[str], list[str]]] = []
    i = 0
    n = len(msgs)
    while i < n:
        m = msgs[i]
        role = m.get("role")
        assert role != "tool", f"orphan or misordered tool message at index {i}: {msgs}"
        if role == "assistant" and m.get("tool_calls"):
            call_ids = [tc["id"] for tc in m["tool_calls"]]
            result_ids: list[str] = []
            j = i + 1
            while j < n and msgs[j].get("role") == "tool":
                tid = msgs[j].get("tool_call_id")
                assert tid in call_ids, f"unexpected tool_call_id {tid!r} for {call_ids}: {msgs}"
                result_ids.append(tid)
                j += 1
            assert set(call_ids) == set(result_ids), (
                f"dangling tool_calls {call_ids} with results {result_ids}: {msgs}"
            )
            groups.append((call_ids, result_ids))
            i = j
            continue
        i += 1
    return groups


def test_trim_front_fill_parallel_tool_group_between_results_and_full_fit() -> None:
    """R07 front-fill: A short, B long; cutting between A and B drops the whole group,
    and fitting the full group keeps [assistant(A,B), tool(A), tool(B)] intact."""
    for with_reply in (True, False):
        msgs = [
            _msg("system", "sys"),
            _msg("user", "q1"),
            _msg("assistant", "", tool_calls=[_tc("call_A", "ta"), _tc("call_B", "tb")]),
            _msg("tool", "short_A", tool_call_id="call_A", name="ta"),
            _msg("tool", "B" * 2000, tool_call_id="call_B", name="tb"),
            *([_msg("assistant", "ans1")] if with_reply else []),
            _msg("user", "q2"),
            _msg("assistant", "ans2"),
        ]
        tail_idx = 6 if with_reply else 5
        tail = msgs[tail_idx:]

        # Budget fits sys + q1 + assistant(A,B) + tool(A) + tail, but NOT tool(B).
        budget_between = _est(msgs[:4]) + _est(tail)
        res_cut = trim_history(msgs, max_tokens=budget_between, keep_system=True, keep_last_n=1)
        assert _assert_valid_tool_groups(res_cut) == []
        assert _est(res_cut) <= budget_between
        assert res_cut[0] == msgs[0]
        assert res_cut[-2:] == tail

        # Budget fits the complete group (plus sys, q1, tail), even if not ans1.
        budget_full_group = _est(msgs[:5]) + _est(tail)
        res_full = trim_history(msgs, max_tokens=budget_full_group, keep_system=True, keep_last_n=1)
        assert _assert_valid_tool_groups(res_full) == [(["call_A", "call_B"], ["call_A", "call_B"])]
        assert _est(res_full) <= budget_full_group
        assert res_full[0] == msgs[0]
        assert res_full[-2:] == tail
        group_pos = res_full.index(msgs[2])
        assert res_full[group_pos : group_pos + 3] == msgs[2:5]


def test_trim_front_fill_parallel_tool_group_long_a_short_b() -> None:
    """R07 front-fill with A long and B short: cutting before full group drops the whole group."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "q1"),
        _msg("assistant", "", tool_calls=[_tc("call_A", "ta"), _tc("call_B", "tb")]),
        _msg("tool", "A" * 2000, tool_call_id="call_A", name="ta"),
        _msg("tool", "short_B", tool_call_id="call_B", name="tb"),
        _msg("assistant", "ans1"),
        _msg("user", "q2"),
        _msg("assistant", "ans2"),
    ]
    tail = msgs[6:]
    # Fits sys + q1 + assistant(A,B) + tool(B) + tail, not tool(A).
    budget_between = _est(msgs[:3]) + _est([msgs[4]]) + _est(tail)
    res_cut = trim_history(msgs, max_tokens=budget_between, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_cut) == []
    assert res_cut == [msgs[0], msgs[1], *tail]

    budget_full = _est(msgs[:5]) + _est(tail)
    res_full = trim_history(msgs, max_tokens=budget_full, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_full) == [(["call_A", "call_B"], ["call_A", "call_B"])]
    assert res_full[2:5] == msgs[2:5]


def test_trim_front_fill_no_cross_user_boundary_pairing() -> None:
    """Matching tool_call_id in a later user exchange must never satisfy an earlier incomplete cut."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "q1"),
        _msg("assistant", "", tool_calls=[_tc("call_A", "ta"), _tc("call_B", "tb")]),
        _msg("tool", "short_A", tool_call_id="call_A", name="ta"),
        _msg("tool", "B" * 2000, tool_call_id="call_B", name="tb"),
        _msg("assistant", "ans1"),
        _msg("user", "q2"),
        _msg("assistant", "", tool_calls=[_tc("call_B", "tb")]),
        _msg("tool", "short_tail_B", tool_call_id="call_B", name="tb"),
        _msg("assistant", "ans2"),
    ]
    tail = msgs[6:]
    budget_between = _est(msgs[:4]) + _est(tail)
    result = trim_history(msgs, max_tokens=budget_between, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(result) == [(["call_B"], ["call_B"])]
    assert result == [msgs[0], msgs[1], *tail]


def test_trim_reverse_tail_parallel_tool_group_boundaries() -> None:
    """R07 tail-over-budget reverse path: budgets between A/B and at the assistant boundary
    never emit orphan tools or dangling calls; fitting the full group preserves it in order."""
    # Case 1: A long, B short — reverse scan meets tool(B) first, then oversized tool(A).
    msgs_ab = [
        _msg("system", "sys"),
        _msg("user", "q_tail"),
        _msg("assistant", "", tool_calls=[_tc("call_A", "ta"), _tc("call_B", "tb")]),
        _msg("tool", "A" * 2000, tool_call_id="call_A", name="ta"),
        _msg("tool", "short_B", tool_call_id="call_B", name="tb"),
        _msg("assistant", "done"),
    ]
    budget_between_ab = _est(msgs_ab[4:]) + 2  # fits tool(B) + done, not tool(A)
    res_between_ab = trim_history(msgs_ab, max_tokens=budget_between_ab, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_between_ab) == []
    assert res_between_ab == [msgs_ab[-1]]

    budget_exact_ab = _est(msgs_ab[2:])  # fits group + done, not user + sys
    res_exact_ab = trim_history(msgs_ab, max_tokens=budget_exact_ab, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_exact_ab) == [(["call_A", "call_B"], ["call_A", "call_B"])]
    assert res_exact_ab == msgs_ab[2:]

    # Case 2: A short, B long — test budget between results AND budget fitting both results
    # plus done but 1 token short of assistant(A,B).
    msgs_ba = [
        _msg("system", "sys"),
        _msg("user", "q_tail"),
        _msg("assistant", "", tool_calls=[_tc("call_A", "ta"), _tc("call_B", "tb")]),
        _msg("tool", "short_A", tool_call_id="call_A", name="ta"),
        _msg("tool", "B" * 400, tool_call_id="call_B", name="tb"),
        _msg("assistant", "done"),
    ]
    budget_before_b = _est([msgs_ba[3], msgs_ba[5]])
    res_before_b = trim_history(msgs_ba, max_tokens=budget_before_b, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_before_b) == []
    assert res_before_b == [msgs_ba[-1]]

    budget_results_only = _est(msgs_ba[3:])  # fits tool(A) + tool(B) + done, NOT assistant(A,B)
    res_results_only = trim_history(msgs_ba, max_tokens=budget_results_only, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_results_only) == []
    assert res_results_only == [msgs_ba[-1]]

    budget_exact_ba = _est(msgs_ba[2:])  # fits complete group + done
    res_exact_ba = trim_history(msgs_ba, max_tokens=budget_exact_ba, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_exact_ba) == [(["call_A", "call_B"], ["call_A", "call_B"])]
    assert res_exact_ba == msgs_ba[2:]

    # Case 3: trailing parallel group with no final assistant reply yet.
    msgs_open = msgs_ab[:-1]
    budget_open_cut = _est([msgs_open[4]]) + 2
    res_open_cut = trim_history(msgs_open, max_tokens=budget_open_cut, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_open_cut) == []
    assert res_open_cut == []

    budget_open_full = _est(msgs_open[2:])
    res_open_full = trim_history(msgs_open, max_tokens=budget_open_full, keep_system=True, keep_last_n=1)
    assert _assert_valid_tool_groups(res_open_full) == [(["call_A", "call_B"], ["call_A", "call_B"])]
    assert res_open_full == msgs_open[2:]


def test_trim_reverse_tail_no_cross_user_boundary_pairing() -> None:
    """Reverse tail trim across two exchanges with reused tool_call_ids never pairs across user."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "q1"),
        _msg("assistant", "", tool_calls=[_tc("call_A", "ta"), _tc("call_B", "tb")]),
        _msg("tool", "A" * 2000, tool_call_id="call_A", name="ta"),
        _msg("tool", "short_B1", tool_call_id="call_B", name="tb"),
        _msg("assistant", "a1"),
        _msg("user", "q2"),
        _msg("assistant", "", tool_calls=[_tc("call_A", "ta"), _tc("call_B", "tb")]),
        _msg("tool", "short_A2", tool_call_id="call_A", name="ta"),
        _msg("tool", "short_B2", tool_call_id="call_B", name="tb"),
        _msg("assistant", "a2"),
    ]
    # Fits q2 exchange + a1 + short_B1, but NOT tool(A1).
    budget = _est(msgs[4:])
    result = trim_history(msgs, max_tokens=budget, keep_system=True, keep_last_n=2)
    assert _assert_valid_tool_groups(result) == [(["call_A", "call_B"], ["call_A", "call_B"])]
    assert result == msgs[5:]
