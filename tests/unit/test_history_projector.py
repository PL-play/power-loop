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
