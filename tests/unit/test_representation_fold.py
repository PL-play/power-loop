"""power-loop 3.0 — the two orthogonal context axes: Representation × FoldStrategy.

Unit-level: representations build+render (incl. compact rows); fold strategies turn
ProjectMessageRows into one compact via a scripted LLM (no store/session needed). End-to-end
wiring + real-LLM coverage lands in later phases.
"""

from __future__ import annotations

import pytest

from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse
from power_loop.runtime.fold import (
    AgenticFold,
    FoldContext,
    LLMSummaryFold,
    NoteOp,
)
from power_loop.runtime.representation import (
    ProjectedRepresentation,
    ProjectionRenderConfig,
    VerbatimRepresentation,
)
from power_loop.runtime.store.types import MessageRow, MessageState, ProjectMessageRow

# ── fixtures ──────────────────────────────────────────────────────────────


def _mr(seq, role, *, content=None, tool_calls=None, tool_call_id=None, send_index=1):
    return MessageRow(
        session_id="s1", seq=seq, role=role, name=None, content=content,
        tool_calls=tool_calls, tool_call_id=tool_call_id, round_index=0,
        state=MessageState.ACTIVE, meta={}, created_at=0, send_index=send_index,
    )


def _pmr(send_index, kind, content, *, rendered_text=None):
    return ProjectMessageRow(
        session_id="s1", send_index=send_index, kind=kind, content=content,
        rendered_text=rendered_text, source_seq_lo=None, source_seq_hi=None,
        compact_from_send=None, compact_to_send=None, projector_version=1,
        token_estimate=None, created_at=0,
    )


def _tc(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class _ScriptLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return self.responses.pop(0)


# ── VerbatimRepresentation ─────────────────────────────────────────────────


def test_verbatim_roundtrip_byte_identical():
    rep = VerbatimRepresentation()
    assert (rep.kind, rep.version) == ("verbatim", 1)
    send = [
        _mr(1, "user", content="hi"),
        _mr(2, "assistant", content="hello", tool_calls=[_tc("a", "grep", "{}")]),
        _mr(3, "tool", content="result", tool_call_id="a"),
    ]
    projected = rep.project_send(send, send_index=1, tool_registry=None)
    assert projected.source_seq_lo == 1 and projected.source_seq_hi == 3
    # one project row carrying the verbatim message list
    rows = [_pmr(1, projected.rows[0].kind, projected.rows[0].content)]
    rendered = rep.render(rows)
    assert rendered == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": [_tc("a", "grep", "{}")]},
        {"role": "tool", "content": "result", "tool_call_id": "a"},
    ]


def test_verbatim_renders_compact_row():
    rep = VerbatimRepresentation()
    rows = [_pmr(5, "compact", {"summary": "older stuff folded"})]
    assert rep.render(rows) == [{"role": "user", "content": "older stuff folded"}]


# ── ProjectedRepresentation ────────────────────────────────────────────────


def test_projection_project_send_and_render():
    rep = ProjectedRepresentation()
    assert (rep.kind, rep.version) == ("projection", 1)
    send = [
        _mr(1, "user", content="do a search"),
        _mr(2, "assistant", content="ok", tool_calls=[_tc("a", "grep", '{"q": "foo"}')]),
        _mr(3, "tool", content="3 hits", tool_call_id="a"),
    ]
    projected = rep.project_send(send, send_index=1, tool_registry=None)
    kinds = [r.kind for r in projected.rows]
    assert kinds == ["user", "project"]
    user_row, project_row = projected.rows
    assert user_row.content == {"input": ["do a search"]}
    assert project_row.content["final_text"] == "ok"
    # 5.4.0: every tool entry carries its ROW COORDINATES — the result row's seq and the
    # issuing assistant row's seq — so a host renderer can print recall_send(send_index, seq).
    assert project_row.content["tools"] == [{"name": "grep", "result": "3 hits", "seq": 3, "call_seq": 2}]
    rendered = rep.render([_pmr(1, "user", user_row.content), _pmr(1, "project", project_row.content)])
    # each send is tagged with its #N so recall_send(send_index=N) is discoverable
    assert rendered[0] == {"role": "user", "content": "[#1] do a search"}
    assert rendered[1]["role"] == "assistant" and rendered[1]["content"].startswith("#1 ")
    assert "grep(result=3 hits)" in rendered[1]["content"] and "ok" in rendered[1]["content"]


def test_projection_renders_compact_row():
    rep = ProjectedRepresentation()
    assert rep.render([_pmr(9, "compact", {"summary": "folded"})]) == [
        {"role": "user", "content": "folded"}
    ]


def test_projection_keeps_input_full_but_truncates_tool_output():
    # The INPUT/user turn is kept verbatim (it's the real conversation, short, high-value); only the
    # assistant's WORK (tool result / final_text) is compressed to max_chars.
    rep = ProjectedRepresentation(max_chars=5)
    send = [
        _mr(1, "user", content="abcdefghij"),
        _mr(2, "assistant", content="ok", tool_calls=[_tc("a", "grep", "{}")]),
        _mr(3, "tool", content="0123456789", tool_call_id="a"),
    ]
    projected = rep.project_send(send, send_index=1, tool_registry=None)
    user_row, project_row = projected.rows
    assert user_row.content == {"input": ["abcdefghij"]}  # input verbatim, NOT truncated
    assert project_row.content["tools"][0]["result"] == "01234…"  # tool output IS truncated


def test_projection_render_reads_legacy_human_key():
    # Pre-3.3 projection rows stored the user input under "human"; render must still emit them so a
    # session with old rows keeps working after upgrade.
    rep = ProjectedRepresentation()
    out = rep.render([_pmr(1, "user", {"human": ["legacy input"]})])
    assert out == [{"role": "user", "content": "[#1] legacy input"}]


# ── ProjectedRepresentation render extensibility (config + subclass) ─────────


def _render(rows, rep=None):
    return [m["content"] for m in (rep or ProjectedRepresentation()).render(rows)]


def _compact_pmr(lo, hi, summary):
    return ProjectMessageRow(
        session_id="s1", send_index=hi, kind="compact", content={"summary": summary},
        rendered_text=None, source_seq_lo=None, source_seq_hi=None,
        compact_from_send=lo, compact_to_send=hi, projector_version=1,
        token_estimate=None, created_at=0,
    )


def test_compact_recall_hint_emitted_for_migration_seeded_fold():
    # M-projection-1: a MATURE fold whose range starts at send 0 (migration-seeded from_send=0)
    # must still show the recall_send hint. The old `lo > 0` permanently hid it for those folds.
    out = _render([_compact_pmr(0, 5, "older stuff")])[0]
    assert out.startswith("[older sends ")
    assert "#0" in out and "#5" in out and "recall_send" in out
    assert "older stuff" in out


def test_compact_no_hint_for_degenerate_seed():
    # The degenerate seed compact covers NO real send (from_send=0, to_send=0) → no recall note.
    out = _render([_compact_pmr(0, 0, "seed")])[0]
    assert "folded" not in out and out == "seed"


def test_render_config_coerces_bools_from_json_strings():
    # projection-2: a string "false" must read as False (JSON/templated config), not stay truthy.
    cfg = ProjectionRenderConfig.from_dict({"include_tools": "false", "include_final_text": "0"})
    assert cfg.include_tools is False and cfg.include_final_text is False
    assert ProjectionRenderConfig.from_dict({"include_tools": "true"}).include_tools is True
    # string knobs still coerced via str() and unknown keys ignored
    cfg2 = ProjectionRenderConfig.from_dict({"user_tag": 7, "nope": "x"})
    assert cfg2.user_tag == "7"


def test_render_user_row_handles_non_list_input():
    # projection-3: a bare-string `input` must not be iterated char-by-char.
    out = _render([_pmr(1, "user", {"input": "hello world"})])
    assert out == ["[#1] hello world"]


def test_render_config_defaults_match_builtin():
    # The default ProjectionRenderConfig must reproduce the historical rendering byte-for-byte.
    rows = [
        _pmr(1, "user", {"input": ["hi"]}),
        _pmr(1, "project", {"tools": [{"name": "grep", "result": "3 hits"}], "final_text": "done"}),
    ]
    assert _render(rows) == ["[#1] hi", "#1 [tools] grep(result=3 hits)\ndone"]
    assert _render(rows, ProjectedRepresentation(render_config=ProjectionRenderConfig())) == _render(rows)


def test_render_config_tags_and_separators():
    cfg = ProjectionRenderConfig(
        user_tag="U#{n}: ", project_tag="A#{n}: ", tools_header="tools→ ", tool_sep=" | ",
        tool_arg_sep=" ",
    )
    rows = [
        _pmr(2, "user", {"input": ["q"]}),
        _pmr(2, "project", {"tools": [{"name": "a", "result": "x"}, {"name": "b", "said": "y"}], "final_text": None}),
    ]
    assert _render(rows, ProjectedRepresentation(render_config=cfg)) == [
        "U#2: q",
        "A#2: tools→ a(result=x) | b(said=y)",
    ]


def test_render_config_include_flags():
    rows = [_pmr(1, "project", {"tools": [{"name": "t", "result": "r"}], "final_text": "ft"})]
    assert _render(rows, ProjectedRepresentation(render_config=ProjectionRenderConfig(include_tools=False))) == ["#1 ft"]
    assert _render(rows, ProjectedRepresentation(render_config=ProjectionRenderConfig(include_final_text=False))) == ["#1 [tools] t(result=r)"]
    cfg = ProjectionRenderConfig(include_tools=False, include_final_text=False, empty_project="(空)")
    assert _render(rows, ProjectedRepresentation(render_config=cfg)) == ["#1 (空)"]


def test_render_config_empty_tag_and_none_send_index():
    cfg = ProjectionRenderConfig(user_tag="", project_tag="#{n} ")
    rep = ProjectedRepresentation(render_config=cfg)
    assert _render([_pmr(1, "user", {"input": ["x"]})], rep) == ["x"]  # empty tag → no prefix
    # None send_index → no tag regardless of a non-empty template
    assert _render([_pmr(None, "project", {"tools": [], "final_text": "z"})], rep) == ["z"]


def test_render_config_fold_note():
    cfg = ProjectionRenderConfig(fold_note="[折叠 {range}]")
    row = ProjectMessageRow(
        session_id="s", send_index=5, kind="compact", content={"summary": "old"}, rendered_text=None,
        source_seq_lo=None, source_seq_hi=None, compact_from_send=1, compact_to_send=3,
        projector_version=1, token_estimate=None, created_at=0,
    )
    assert ProjectedRepresentation(render_config=cfg).render([row]) == [
        {"role": "user", "content": "[折叠 #1–#3]\nold"}
    ]


def test_render_config_from_dict_ignores_unknown_keys():
    cfg = ProjectionRenderConfig.from_dict({"project_tag": ">>", "bogus": 1})
    assert cfg.project_tag == ">>"
    assert cfg.user_tag == "[#{n}] "  # untouched default
    assert ProjectionRenderConfig.from_dict(None) == ProjectionRenderConfig()


def test_render_config_dict_coerced_in_constructor():
    rep = ProjectedRepresentation(render_config={"user_tag": "» "})
    assert isinstance(rep.render_config, ProjectionRenderConfig)
    assert _render([_pmr(1, "user", {"input": ["a"]})], rep) == ["» a"]


def test_render_row_skips_unknown_kind():
    # an unhandled kind → render_row returns None → skipped (matches the old if/elif fall-through)
    rows = [_pmr(1, "user", {"input": ["keep"]}), _pmr(1, "weird", {"x": 1})]
    assert _render(rows) == ["[#1] keep"]


def test_render_subclass_overrides_one_shape():
    # The decomposition lets a subclass override exactly ONE per-kind method (the "provide your own
    # render" path) without copy-pasting render — the other kinds keep the built-in shape.
    class MyRep(ProjectedRepresentation):
        def render_project_row(self, r):
            return {"role": "assistant", "content": f"[custom #{r.send_index}]"}

    rows = [_pmr(1, "user", {"input": ["hi"]}), _pmr(1, "project", {"tools": [], "final_text": "x"})]
    assert _render(rows, MyRep()) == ["[#1] hi", "[custom #1]"]


# ── LLMSummaryFold ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_summary_fold_basic():
    rep = ProjectedRepresentation()
    rows = [
        _pmr(1, "user", {"human": ["q1"]}),
        _pmr(1, "project", {"tools": [], "final_text": "a1"}),
        _pmr(2, "user", {"human": ["q2"]}),
        _pmr(2, "project", {"tools": [], "final_text": "a2"}),
    ]
    llm = _ScriptLLM([LLMResponse(raw_text="<summary>folded q1/q2</summary>")])
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=llm, max_tokens=8000)
    out = await LLMSummaryFold().fold(rows, context=ctx)
    assert out is not None
    assert out.content == {"summary": "folded q1/q2"}
    assert out.folded_to_send == 2  # max folded send_index
    assert out.note_ops == ()
    # the rendered span (q1/a1/q2/a2) was sent to the summarizer
    sent = llm.calls[0].messages[0]["content"]
    assert "q1" in sent and "a2" in sent


@pytest.mark.asyncio
async def test_llm_summary_fold_rolls_prior_compact_forward():
    rep = ProjectedRepresentation()
    rows = [
        _pmr(0, "compact", {"summary": "PRIOR SUMMARY"}),
        _pmr(1, "user", {"human": ["new q"]}),
        _pmr(1, "project", {"tools": [], "final_text": "new a"}),
    ]
    llm = _ScriptLLM([LLMResponse(raw_text="<summary>merged</summary>")])
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=llm, max_tokens=8000)
    out = await LLMSummaryFold().fold(rows, context=ctx)
    assert out.folded_to_send == 1
    # the prior compact's text was rolled into the summarization prompt
    assert "PRIOR SUMMARY" in llm.calls[0].messages[0]["content"]


@pytest.mark.asyncio
async def test_llm_summary_fold_none_when_nothing_foldable():
    rep = ProjectedRepresentation()
    llm = _ScriptLLM([])
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=llm, max_tokens=8000)
    # only a prior compact row, no user/project sends → decline
    assert await LLMSummaryFold().fold([_pmr(0, "compact", {"summary": "x"})], context=ctx) is None


@pytest.mark.asyncio
async def test_llm_summary_fold_soft_fails_to_none():
    class _Boom:
        async def complete(self, request):
            raise RuntimeError("provider down")

    rep = ProjectedRepresentation()
    rows = [_pmr(1, "user", {"human": ["q"]}), _pmr(1, "project", {"tools": [], "final_text": "a"})]
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=_Boom(), max_tokens=8000)
    assert await LLMSummaryFold().fold(rows, context=ctx) is None


# ── AgenticFold ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agentic_fold_captures_notes_then_summarizes():
    rep = ProjectedRepresentation()
    rows = [_pmr(1, "user", {"human": ["I live in Berlin, terse please"]}),
            _pmr(1, "project", {"tools": [], "final_text": "ok"})]
    llm = _ScriptLLM([
        LLMResponse(raw_text="saving", tool_calls=[_tc("1", "note", '{"action": "add", "content": "lives in Berlin"}')]),
        LLMResponse(raw_text="<summary>Berlin; terse.</summary>"),
    ])
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=llm, max_tokens=8000)
    out = await AgenticFold().fold(rows, context=ctx)
    assert out.content == {"summary": "Berlin; terse."}
    assert out.folded_to_send == 1
    assert out.note_ops == (NoteOp(op="add", content="lives in Berlin", pinned=False),)
    # the agentic call carried the note tools
    assert {t["function"]["name"] for t in (llm.calls[0].tools or [])} == {"note"}


@pytest.mark.asyncio
async def test_agentic_fold_falls_back_to_single_call():
    class _FailOnTools:
        def __init__(self):
            self.calls = []

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls.append(request)
            if request.tools:
                raise RuntimeError("tools rejected")
            return LLMResponse(raw_text="<summary>fallback</summary>")

    rep = ProjectedRepresentation()
    rows = [_pmr(1, "user", {"human": ["q"]}), _pmr(1, "project", {"tools": [], "final_text": "a"})]
    llm = _FailOnTools()
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=llm, max_tokens=8000)
    out = await AgenticFold().fold(rows, context=ctx)
    assert out.content == {"summary": "fallback"}
    assert out.note_ops == ()
    assert any(c.tools for c in llm.calls) and any(not c.tools for c in llm.calls)


@pytest.mark.asyncio
async def test_agentic_fold_bad_note_id_does_not_abort():
    # A note update action with a non-integer note_id must be answered with an error (NOT raise), so the
    # round continues and the summary is still produced. Review #17.
    rep = ProjectedRepresentation()
    rows = [_pmr(1, "user", {"human": ["q"]}), _pmr(1, "project", {"tools": [], "final_text": "a"})]
    llm = _ScriptLLM([
        LLMResponse(raw_text="", tool_calls=[_tc("1", "note", '{"action": "update", "note_id": "abc"}')]),
        LLMResponse(raw_text="<summary>ok</summary>"),
    ])
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=llm, max_tokens=8000)
    out = await AgenticFold(max_rounds=3).fold(rows, context=ctx)
    assert out.content == {"summary": "ok"}  # did not abort
    assert out.note_ops == ()  # the malformed update was skipped, not captured


@pytest.mark.asyncio
async def test_agentic_fold_preserves_notes_on_empty_final_summary():
    # Notes captured during the agentic loop must survive even when the exhausted-rounds final
    # summary comes back empty and the fold falls back to a plain summary. Review #18.
    rep = ProjectedRepresentation()
    rows = [_pmr(1, "user", {"human": ["q"]}), _pmr(1, "project", {"tools": [], "final_text": "a"})]
    llm = _ScriptLLM([
        LLMResponse(raw_text="", tool_calls=[_tc("1", "note", '{"action": "add", "content": "KEEP-ME"}')]),
        LLMResponse(raw_text=""),  # rounds exhausted → forced final summary is empty → None
        LLMResponse(raw_text="<summary>fallback-summary</summary>"),  # plain-summary fallback
    ])
    ctx = FoldContext(session_id="s1", round_index=0, representation=rep, llm=llm, max_tokens=8000)
    out = await AgenticFold(max_rounds=1).fold(rows, context=ctx)
    assert out.content == {"summary": "fallback-summary"}
    assert out.note_ops == (NoteOp(op="add", content="KEEP-ME", pinned=False),)  # NOT discarded


# ── validation ─────────────────────────────────────────────────────────────


def test_fold_rejects_keep_last_sends_zero():
    with pytest.raises(ValueError):
        LLMSummaryFold(keep_last_sends=0)
    with pytest.raises(ValueError):
        AgenticFold(keep_last_sends=0)


def test_fold_rejects_bad_summary_max_tokens():
    with pytest.raises(ValueError):
        LLMSummaryFold(summary_max_tokens=0)
    with pytest.raises(ValueError):
        AgenticFold(summary_max_tokens=-1)


def test_representation_rejects_bad_version():
    with pytest.raises(ValueError):
        VerbatimRepresentation(version=0)


def test_representation_max_chars_zero_means_unlimited():
    # max_chars <= 0 is VALID and disables the library's per-field truncation (a host can do its own).
    from power_loop.runtime.representation import _truncate

    ProjectedRepresentation(max_chars=0)
    ProjectedRepresentation(max_chars=-5)
    big = "x" * 5000
    assert _truncate(big, 0) == big  # unlimited
    assert _truncate(big, -1) == big  # unlimited
    assert _truncate(big, 10) == "x" * 10 + "…"  # positive cap unchanged
