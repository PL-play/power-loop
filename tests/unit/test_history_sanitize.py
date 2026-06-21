"""align_tool_calls: the assembled-history robustness backstop.

Repairs tool-call/result pairing so a corrupt row can't make a provider reject the prompt and
brick the session. Must be a NO-OP on a healthy history, drop orphan tool results (reporting
their seqs for optional DB repair), synthesize placeholders for mid-history unanswered calls,
keep the parallel (messages, seqs, ords) lists aligned, and NEVER touch a TRAILING pending call.
"""

from __future__ import annotations

from power_loop.runtime.history_sanitize import align_tool_calls


def _u(text):
    return {"role": "user", "content": text}


def _a(text=None, calls=None):
    m = {"role": "assistant"}
    if text is not None:
        m["content"] = text
    if calls is not None:
        m["tool_calls"] = [{"id": c, "type": "function", "function": {"name": "f", "arguments": "{}"}} for c in calls]
    return m


def _t(call_id, content="ok"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _run(msgs):
    seqs = list(range(len(msgs)))  # distinct real seqs
    ords = list(seqs)
    return align_tool_calls(msgs, seqs, ords)


def test_valid_history_is_noop():
    msgs = [_u("hi"), _a(calls=["a"]), _t("a"), _a("done")]
    out_m, out_s, out_o, dropped, synth = _run(msgs)
    assert out_m == msgs and dropped == [] and synth == 0
    assert out_s == [0, 1, 2, 3] and out_o == [0, 1, 2, 3]


def test_orphan_tool_result_dropped_and_seq_reported():
    msgs = [_u("hi"), _t("ghost"), _a("done")]  # tool with no preceding call
    out_m, out_s, out_o, dropped, synth = _run(msgs)
    assert [m["role"] for m in out_m] == ["user", "assistant"]  # orphan removed
    assert dropped == [1] and synth == 0  # seq 1 was the orphan
    assert out_s == [0, 2] and out_o == [0, 2]  # lists stay aligned, kept rows keep their seq


def test_mid_history_unanswered_call_gets_placeholder():
    msgs = [_a(calls=["a"]), _u("next")]  # call never answered, a non-tool message follows
    out_m, out_s, out_o, dropped, synth = _run(msgs)
    assert [m["role"] for m in out_m] == ["assistant", "tool", "user"]
    assert out_m[1]["tool_call_id"] == "a" and "repaired" in out_m[1]["content"]
    assert synth == 1 and dropped == []
    assert out_s == [0, None, 1] and out_o == [0, None, 1]  # placeholder carries seq/ord None


def test_trailing_pending_call_is_preserved():
    # The LAST assistant call (in-flight send's pending tools the loop will execute) is NOT touched.
    msgs = [_u("hi"), _a(calls=["a"])]
    out_m, out_s, out_o, dropped, synth = _run(msgs)
    assert out_m == msgs and synth == 0 and dropped == []


def test_partial_results_filled_for_remaining_ids():
    msgs = [_a(calls=["a", "b"]), _t("a"), _u("next")]  # b unanswered
    out_m, _, _, dropped, synth = _run(msgs)
    assert [m["role"] for m in out_m] == ["assistant", "tool", "tool", "user"]
    assert out_m[2]["tool_call_id"] == "b" and synth == 1 and dropped == []


def test_duplicate_tool_result_second_is_orphan():
    msgs = [_a(calls=["a"]), _t("a", "first"), _t("a", "second"), _u("next")]
    out_m, _, _, dropped, synth = _run(msgs)
    # first result matches the open id; the second has no open id left → orphan, dropped
    assert [m["role"] for m in out_m] == ["assistant", "tool", "user"]
    assert out_m[1]["content"] == "first" and dropped == [2] and synth == 0


def test_malformed_tool_calls_do_not_crash_or_synthesize():
    # tool_calls with no usable id (None / non-dict) → treated as declaring no ids.
    bad = {"role": "assistant", "tool_calls": [{"id": None}, "notadict", {"function": {}}]}
    out_m, out_s, out_o, dropped, synth = align_tool_calls([bad, _u("x")], [0, 1], [0, 1])
    assert [m["role"] for m in out_m] == ["assistant", "user"]  # no placeholder added
    assert synth == 0 and dropped == []


def test_lengths_stay_aligned_under_mixed_repairs():
    msgs = [_t("orphan"), _a(calls=["a", "b"]), _t("a"), _u("next"), _a(calls=["c"])]
    out_m, out_s, out_o, dropped, synth = _run(msgs)
    assert len(out_m) == len(out_s) == len(out_o)
    assert dropped == [0]  # the leading orphan
    assert synth == 1  # placeholder for unanswered "b" (trailing "c" preserved)
    assert out_m[-1]["tool_calls"][0]["id"] == "c"  # trailing pending call untouched
