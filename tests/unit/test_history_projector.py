"""HistoryProjector trio: IdentityProjector (verbatim) + DefaultDeterministicProjector
(generic structured summary via ToolDefinition.project, no LLM). Covers project_send
(tool dispatch, fallback, truncation, follow-up list, determinism), render (no tool-protocol
leak), the Identity round-trip, and compact()."""

from __future__ import annotations

from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.history_projector import (
    DefaultDeterministicProjector,
    IdentityProjector,
    _row_to_loop_dict,
)
from power_loop.runtime.store.types import MessageRow, MessageState, ProjectMessageRow
from power_loop.tools.registry import ToolRegistry


def _mr(seq, role, content=None, *, tool_calls=None, tool_call_id=None, name=None) -> MessageRow:
    return MessageRow(
        session_id="s", seq=seq, role=role, name=name, content=content,
        tool_calls=tool_calls, tool_call_id=tool_call_id, round_index=None,
        state=MessageState.ACTIVE, meta={}, created_at=0,
    )


def _tc(call_id, name, arguments="{}"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _pm(send_index, kind, content) -> ProjectMessageRow:
    return ProjectMessageRow(
        session_id="s", send_index=send_index, kind=kind, content=content, rendered_text=None,
        source_seq_lo=None, source_seq_hi=None, compact_from_send=None, compact_to_send=None,
        projector_version=1, token_estimate=None, created_at=0,
    )


def test_default_project_send_basic() -> None:
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "hi"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "bash", '{"command":"ls"}')]),
        _mr(3, "tool", "exit 0\nfiles", tool_call_id="c1", name="bash"),
        _mr(4, "assistant", "done"),
    ]
    out = p.project_send(rows, send_index=1, tool_registry=None)
    assert out.source_seq_lo == 1 and out.source_seq_hi == 4
    by_kind = {r.kind: r.content for r in out.rows}
    assert by_kind["user"]["human"] == ["hi"]
    assert by_kind["project"]["final_text"] == "done"
    assert by_kind["project"]["tools"] == [{"name": "bash", "result": "exit 0\nfiles"}]


def test_tool_project_hook_used_over_fallback() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="write_file", description="w",
            project=lambda args, result: {"file": args.get("path")},
        ),
        lambda **kw: "",
    )
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "go"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "write_file", '{"path":"x.py","content":"..."}')]),
        _mr(3, "tool", "wrote 3 bytes", tool_call_id="c1", name="write_file"),
    ]
    out = p.project_send(rows, send_index=1, tool_registry=reg)
    tools = next(r.content for r in out.rows if r.kind == "project")["tools"]
    assert tools == [{"name": "write_file", "file": "x.py"}]  # hook output, not the raw result


def test_duplicate_tool_call_id_preserves_both_results() -> None:
    # Two tool calls share id "c1" (malformed/imported transcript). A plain dict[id]->result
    # would collapse both onto the SECOND result; the multimap must keep them paired in order.
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "go"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "bash", '{"c":"a"}'), _tc("c1", "bash", '{"c":"b"}')]),
        _mr(3, "tool", "RESULT-A", tool_call_id="c1", name="bash"),
        _mr(4, "tool", "RESULT-B", tool_call_id="c1", name="bash"),
        _mr(5, "assistant", "done"),
    ]
    tools = next(r.content for r in p.project_send(rows, send_index=1, tool_registry=None).rows
                 if r.kind == "project")["tools"]
    assert [t["result"] for t in tools] == ["RESULT-A", "RESULT-B"]


def test_missing_tool_result_distinct_from_empty() -> None:
    # A call with NO tool row renders "<missing>"; a produced-but-empty ("") result stays "".
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "go"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "gone"), _tc("c2", "empty")]),
        _mr(3, "tool", "", tool_call_id="c2", name="empty"),
        _mr(4, "assistant", "done"),
    ]
    tools = next(r.content for r in p.project_send(rows, send_index=1, tool_registry=None).rows
                 if r.kind == "project")["tools"]
    assert tools[0] == {"name": "gone", "result": "<missing>"}
    assert tools[1] == {"name": "empty", "result": ""}


def test_project_hook_receives_none_for_missing_result() -> None:
    # The hook must be able to tell "no result produced" (None) from a produced-but-empty result.
    seen: list[object] = []
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="t", description="t",
                       project=lambda args, result: (seen.append(result), {"k": "v"})[1]),
        lambda **kw: "",
    )
    p = DefaultDeterministicProjector()
    rows = [_mr(1, "user", "go"), _mr(2, "assistant", tool_calls=[_tc("c1", "t")]), _mr(3, "assistant", "x")]
    p.project_send(rows, send_index=1, tool_registry=reg)
    assert seen == [None]  # missing result → None passed through (not "")


def test_malformed_function_field_does_not_raise() -> None:
    # A tool_call whose "function" is a non-dict (bare string) must not crash name extraction.
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "go"),
        _mr(2, "assistant", tool_calls=[{"id": "c1", "function": "not-a-dict", "name": "weird"}]),
        _mr(3, "tool", "r", tool_call_id="c1"),
        _mr(4, "assistant", "ok"),
    ]
    tools = next(r.content for r in p.project_send(rows, send_index=1, tool_registry=None).rows
                 if r.kind == "project")["tools"]
    assert tools[0]["name"] == "weird"


def test_duplicate_tool_call_id_three_results_fifo() -> None:
    # The multimap deque must preserve FIFO order for 3+ results sharing one id.
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "go"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "bash"), _tc("c1", "bash"), _tc("c1", "bash")]),
        _mr(3, "tool", "R1", tool_call_id="c1"),
        _mr(4, "tool", "R2", tool_call_id="c1"),
        _mr(5, "tool", "R3", tool_call_id="c1"),
        _mr(6, "assistant", "done"),
    ]
    tools = next(r.content for r in p.project_send(rows, send_index=1, tool_registry=None).rows
                 if r.kind == "project")["tools"]
    assert [t["result"] for t in tools] == ["R1", "R2", "R3"]


def test_missing_result_hook_exception_falls_back_to_missing() -> None:
    # A hook that raises on a MISSING result must still yield "<missing>" (not crash, not "").
    def boom(args, result):
        raise ValueError("misbehaving hook")
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="broken", description="b", project=boom), lambda **kw: "")
    p = DefaultDeterministicProjector()
    rows = [_mr(1, "user", "go"), _mr(2, "assistant", tool_calls=[_tc("c1", "broken")]),
            _mr(3, "assistant", "done")]
    tools = next(r.content for r in p.project_send(rows, send_index=1, tool_registry=reg).rows
                 if r.kind == "project")["tools"]
    assert tools[0] == {"name": "broken", "result": "<missing>"}


def test_identity_projector_declares_trigger_ratio() -> None:
    # IdentityProjector satisfies the Protocol's trigger_ratio; a custom projector that OMITS it
    # still works via the loop's getattr(..., 0.75) fallback.
    assert IdentityProjector().trigger_ratio == 0.75

    class _MinimalProjector:
        version = 1
        keep_last_sends = 0  # no trigger_ratio attribute on purpose
        def project_send(self, send_rows, *, send_index, tool_registry): ...  # noqa: E704
        def render(self, rows): return []  # noqa: E704
        def compact(self, rows): return None  # noqa: E704

    assert getattr(_MinimalProjector(), "trigger_ratio", 0.75) == 0.75


def test_default_fallback_truncates_result() -> None:
    p = DefaultDeterministicProjector(max_chars=5)
    rows = [
        _mr(1, "assistant", tool_calls=[_tc("c1", "bash")]),
        _mr(2, "tool", "0123456789", tool_call_id="c1", name="bash"),
    ]
    out = p.project_send(rows, send_index=1, tool_registry=None)
    tools = next(r.content for r in out.rows if r.kind == "project")["tools"]
    assert tools[0]["result"] == "01234…"


def test_project_send_is_deterministic() -> None:
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "hi"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "bash", '{"command":"ls"}')]),
        _mr(3, "tool", "ok", tool_call_id="c1", name="bash"),
    ]
    assert p.project_send(rows, send_index=1, tool_registry=None) == p.project_send(
        rows, send_index=1, tool_registry=None
    )


def test_follow_up_user_rows_collapse_to_a_list() -> None:
    p = DefaultDeterministicProjector()
    rows = [_mr(1, "user", "first"), _mr(2, "assistant", "ok"), _mr(3, "user", "follow up")]
    out = p.project_send(rows, send_index=1, tool_registry=None)
    assert next(r.content for r in out.rows if r.kind == "user")["human"] == ["first", "follow up"]


def test_render_emits_plain_text_only_no_tool_protocol() -> None:
    p = DefaultDeterministicProjector()
    rows = [
        _mr(1, "user", "hi"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "bash", '{"command":"ls"}')]),
        _mr(3, "tool", "ok", tool_call_id="c1", name="bash"),
        _mr(4, "assistant", "done"),
    ]
    out = p.project_send(rows, send_index=1, tool_registry=None)
    msgs = p.render(out.rows)  # ProjectedRow duck-types on .kind/.content
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    for m in msgs:
        assert set(m) <= {"role", "content"}  # NO tool_calls / tool_call_id leak
        assert isinstance(m["content"], str)
    assert "[tools] bash" in msgs[1]["content"] and "done" in msgs[1]["content"]


def test_identity_round_trips_verbatim() -> None:
    ip = IdentityProjector()
    rows = [
        _mr(1, "user", "hi"),
        _mr(2, "assistant", tool_calls=[_tc("c1", "bash")]),
        _mr(3, "tool", "ok", tool_call_id="c1", name="bash"),
        _mr(4, "assistant", "done"),
    ]
    out = ip.project_send(rows, send_index=1, tool_registry=None)
    assert ip.render(out.rows) == [_row_to_loop_dict(r) for r in rows]  # byte-identical history


def test_default_compact_folds_sends() -> None:
    p = DefaultDeterministicProjector()
    rows = [
        _pm(1, "user", {"human": ["a"]}),
        _pm(1, "project", {"tools": [], "final_text": "hi"}),
        _pm(2, "user", {"human": ["b"]}),
        _pm(2, "project", {"tools": [], "final_text": "bye"}),
    ]
    c = p.compact(rows)
    assert c is not None and c.from_send == 1 and c.to_send == 2
    assert "#1 user: a" in c.content["summary"] and "#1 agent: hi" in c.content["summary"]
    assert "#2 agent: bye" in c.content["summary"]
    assert IdentityProjector().compact(rows) is None  # verbatim never folds


def test_default_compact_rolls_prior_compact_forward() -> None:
    p = DefaultDeterministicProjector()
    rows = [
        _pm(1, "compact", {"summary": "#1 agent: hi"}),  # prior fold (covers send 1)
        _pm(3, "user", {"human": ["c"]}),
        _pm(3, "project", {"tools": [], "final_text": "yo"}),
    ]
    c = p.compact(rows)
    assert c is not None
    assert c.from_send == 3 and c.to_send == 3  # derived from the NEW user/project sends only
    # nothing lost: prior summary rolled in + the newly folded send
    assert "#1 agent: hi" in c.content["summary"] and "#3 agent: yo" in c.content["summary"]


# ── pre-release hardening: param validation, bounded compact, max_chars default ──


def test_default_max_chars_default_is_300() -> None:
    # The per-field truncation budget bumped 200 → 300 (it was already the `max_chars` field).
    assert DefaultDeterministicProjector().max_chars == 300
    p = DefaultDeterministicProjector()
    big = "x" * 500
    out = p.project_send([_mr(1, "user", big)], send_index=1, tool_registry=None)
    human = next(r for r in out.rows if r.kind == "user").content["human"][0]
    assert len(human) == 301 and human.endswith("…")  # 300 chars + the ellipsis


def test_projector_params_validated_on_construction() -> None:
    import math

    import pytest

    # trigger_ratio must be in (0, 1] — 0, >1, and NaN all rejected (NaN would crash int(max_tokens*nan)).
    for bad in (0.0, 1.5, -0.1, math.nan):
        with pytest.raises(ValueError, match="trigger_ratio"):
            DefaultDeterministicProjector(trigger_ratio=bad)
        with pytest.raises(ValueError, match="trigger_ratio"):
            IdentityProjector(trigger_ratio=bad)
    with pytest.raises(ValueError, match="keep_last_sends"):
        DefaultDeterministicProjector(keep_last_sends=-1)
    with pytest.raises(ValueError, match="version"):
        DefaultDeterministicProjector(version=0)
    with pytest.raises(ValueError, match="max_compact_chars"):
        DefaultDeterministicProjector(max_compact_chars=-1)
    # valid edge values are accepted (ratio==1, keep==0, max_chars<=0 = no truncation,
    # max_compact_chars==0 = unbounded)
    DefaultDeterministicProjector(
        trigger_ratio=1.0, keep_last_sends=0, max_chars=0, max_compact_chars=0
    )


def test_compact_bounded_by_max_compact_chars() -> None:
    # The no-LLM compact concatenates; without a cap it grows unbounded. With max_compact_chars set,
    # it keeps the most-recent tail + a drop marker (dropped detail stays in pl_messages → recall_send).
    p = DefaultDeterministicProjector(max_compact_chars=80)
    rows = [_pm(n, "project", {"tools": [], "final_text": "Z" * 60}) for n in range(1, 9)]
    c = p.compact(rows)
    assert c is not None
    summary = c.content["summary"]
    assert len(summary) <= 80 + 80  # marker + the kept tail, bounded
    assert "recall_send" in summary  # the drop marker is present
    assert "#8 agent" in summary  # the most-recent folded send is kept (tail), not the oldest


def test_compact_unbounded_when_cap_disabled() -> None:
    # max_compact_chars=0 → no cap (the pre-fix behavior), so a long fold is fully retained.
    p = DefaultDeterministicProjector(max_compact_chars=0)
    rows = [_pm(n, "project", {"tools": [], "final_text": "Z" * 60}) for n in range(1, 9)]
    c = p.compact(rows)
    assert c is not None and "recall_send" not in c.content["summary"]
    assert "#1 agent" in c.content["summary"]  # oldest retained too
