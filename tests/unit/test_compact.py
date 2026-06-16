from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.runtime.budget import estimate_message_tokens, estimate_tokens
from power_loop.runtime.compact import CompactionPlan, DefaultCompactor

# ── Fake LLMs ───────────────────────────────────────────────────────────


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    calls: list = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(request)
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None: return None


# ── budget ──────────────────────────────────────────────────────────────


def test_estimate_text_message_tokens_monotonic() -> None:
    short = {"role": "user", "content": "hi"}
    long = {"role": "user", "content": "a" * 4000}
    assert estimate_message_tokens(long) > estimate_message_tokens(short)


def test_estimate_tokens_sums_messages() -> None:
    msgs = [{"role": "user", "content": "abcd" * 100}] * 5
    assert estimate_tokens(msgs) > 5 * 100


# ── DefaultCompactor: trigger ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_compactor_does_not_trigger_below_threshold() -> None:
    cp = DefaultCompactor(trigger_ratio=0.75)
    msgs = [{"role": "user", "content": "hi"}]
    plan = await cp.maybe_compact(msgs, llm=_Scripted(), max_tokens=10_000, round_index=0)
    assert plan is None


@pytest.mark.asyncio
async def test_compactor_triggers_above_threshold(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    summary_llm = _Scripted(responses=[LLMResponse(raw_text="<summary>folded</summary>")])
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 5000},
        {"role": "assistant", "content": "y" * 5000},
        {"role": "user", "content": "z" * 5000},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "last"},
    ]
    plan = await cp.maybe_compact(msgs, llm=summary_llm, max_tokens=4000, round_index=2)
    assert plan is not None
    assert plan.summary_text == "folded"
    assert plan.fold_start_idx == 1
    # Must keep last `user` ("last") + everything from it onward.
    assert plan.fold_end_idx < len(msgs) - 1


@pytest.mark.asyncio
async def test_leading_compact_note_is_refolded_not_orphaned(monkeypatch) -> None:
    """C7 regression: a leading compact_note must be FOLDABLE (merged into the new
    note), not mistaken for the original system_prompt (which is NOT in history) and
    preserved forever. A second compaction over a history that already starts with a
    compact_note must yield exactly ONE compact_note, not two."""
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    summary_llm = _Scripted(responses=[LLMResponse(raw_text="<summary>merged</summary>")])
    msgs: list[dict[str, Any]] = [
        {"role": "system", "name": "compact_note", "content": "older summary"},
        {"role": "user", "content": "x" * 5000},
        {"role": "assistant", "content": "y" * 5000},
        {"role": "user", "content": "last"},
    ]
    plan = await cp.maybe_compact(msgs, llm=summary_llm, max_tokens=4000, round_index=3)
    assert plan is not None
    assert plan.fold_start_idx == 0  # the leading compact_note is INCLUDED → merged
    # Applying the plan collapses [old note + cold turns] into ONE new note.
    note = {"role": "system", "name": "compact_note", "content": plan.summary_text}
    result = msgs[: plan.fold_start_idx] + [note] + msgs[plan.fold_end_idx + 1 :]
    notes = [m for m in result if m.get("name") == "compact_note"]
    assert len(notes) == 1
    assert notes[0]["content"] == "merged"


def test_compactable_span_preserves_genuine_system_and_memory() -> None:
    """The fix must still preserve a genuine injected system row (empty name) and
    recalled memory_* rows — only a leading compact_note becomes foldable."""
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "real system prompt"},
        {"role": "system", "name": "memory_1", "content": "recalled fact"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "last"},
    ]
    span = cp._compactable_span(msgs)
    assert span is not None
    assert span[0] == 2  # both the system prompt and memory_1 are preserved


def test_compactable_span_folds_leading_note_before_memory() -> None:
    """Real reload+recall order is [compact_note, memory_*, conversation] (recall
    injects memory AFTER leading system rows). The leading compact_note must still be
    foldable (span starts at 0); the memory_* rows fall INSIDE the span and are
    handled by the sink (their non-persisted placeholders are skipped when marking)."""
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    msgs: list[dict[str, Any]] = [
        {"role": "system", "name": "compact_note", "content": "old"},
        {"role": "system", "name": "memory_1", "content": "recalled"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "last"},
    ]
    span = cp._compactable_span(msgs)
    assert span is not None
    assert span[0] == 0  # the leading compact_note folds (not preserved as a phantom prompt)


# ── DefaultCompactor: atomic pair preservation ─────────────────────────


def _assert_no_orphan_tools(messages: list[dict[str, Any]]) -> None:
    """Every ``tool`` message must follow an ``assistant(tool_calls)`` whose ids
    include its tool_call_id — i.e. the history is protocol-valid (no orphan)."""
    open_ids: set = set()
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            open_ids |= {tc.get("id") for tc in m["tool_calls"]}
        elif role == "tool":
            assert m.get("tool_call_id") in open_ids, (
                f"orphan tool {m.get('tool_call_id')!r} (no preceding assistant.tool_calls)"
            )


@pytest.mark.asyncio
async def test_compactor_never_orphans_tool_when_user_follows_pair(monkeypatch) -> None:
    """Regression: a completed assistant↔tool pair followed by a new user turn
    must not be split. The fold boundary used to walk back over the trailing
    tool, folding the assistant but KEEPING its tool → an orphan tool that the
    provider rejects (HTTP 400). Now the whole pair folds together."""
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    summary_llm = _Scripted(responses=[LLMResponse(raw_text="<summary>ok</summary>")])
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1" * 1000},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2" * 1000},
        {"role": "assistant", "content": "call", "tool_calls": [{"id": "tc1"}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "out" * 1000},
        {"role": "user", "content": "u3"},
    ]
    plan = await cp.maybe_compact(msgs, llm=summary_llm, max_tokens=2000, round_index=1)
    assert plan is not None  # there is foldable history here
    note = {"role": "system", "name": "compact_note", "content": plan.summary_text}
    result = msgs[: plan.fold_start_idx] + [note] + msgs[plan.fold_end_idx + 1 :]
    _assert_no_orphan_tools(result)  # would FAIL before the fix (orphan tc1)
    # u3 is the kept tail, so the whole tool pair must be folded away.
    assert not any(m.get("role") == "tool" for m in result)


@pytest.mark.asyncio
async def test_compactor_expands_back_to_atomic_when_tail_starts_at_tool(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    summary_llm = _Scripted(responses=[LLMResponse(raw_text="<summary>s</summary>")])
    # Tail boundary will fall on the trailing tool. Expansion must pull the
    # boundary back to the matching assistant so the pair stays kept.
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "u0" * 3000},
        {"role": "assistant", "content": "a0", "tool_calls": [{"id": "tc"}]},
        {"role": "tool", "tool_call_id": "tc", "content": "x" * 3000},
    ]
    plan = await cp.maybe_compact(msgs, llm=summary_llm, max_tokens=2000, round_index=0)
    # No safe fold here: the assistant(tool_calls) at 1 and matching tool at 2
    # must both be preserved as a unit; the only foldable msg is the user at 0,
    # which is the kept exchange itself → nothing to fold.
    assert plan is None or plan.fold_end_idx < 1


# ── DefaultCompactor: summary failure ──────────────────────────────────


@pytest.mark.asyncio
async def test_compactor_returns_none_on_summary_error(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    @dataclass
    class _RaisingLLM(LLMService):
        async def complete(
            self,
            request: LLMRequest,
            *,
            on_chunk_delta_text: Callable[[str], Any] | None = None,
            on_chunk_think: Callable[[str], Any] | None = None,
            on_stream_end: Callable[[LLMResponse], Any] | None = None,
        ) -> LLMResponse:
            raise RuntimeError("boom")

        def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
            async def _empty() -> AsyncIterator[LLMStreamChunk]:
                if False:
                    yield LLMStreamChunk()

            return _empty()

        async def close(self) -> None: return None

    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1, summary_llm=_RaisingLLM())
    msgs = [
        {"role": "user", "content": "x" * 5000},
        {"role": "assistant", "content": "y" * 5000},
        {"role": "user", "content": "z"},
    ]
    plan = await cp.maybe_compact(
        msgs, llm=_Scripted(), max_tokens=2000, round_index=0,
    )
    assert plan is None


# ── DefaultCompactor: env override ─────────────────────────────────────


@pytest.mark.asyncio
async def test_env_threshold_overrides_ratio(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "10")
    cp = DefaultCompactor(keep_last_n=1)
    summary_llm = _Scripted(responses=[LLMResponse(raw_text="<summary>x</summary>")])
    msgs = [
        {"role": "user", "content": "a longer prompt here please"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "tail"},
    ]
    plan = await cp.maybe_compact(
        msgs, llm=summary_llm, max_tokens=1_000_000, round_index=0,
    )
    assert plan is not None  # absolute threshold beat the ratio


# ── end-to-end: SQLiteSink persistence ─────────────────────────────────


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_compaction_persists_to_store_and_marks_state(store: SessionStore, monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    """Long history forces compaction during the next send; verify the store
    has compacted_out rows + a compact_note + a compactions audit row."""
    sid = await store.create_session(system_prompt="S")
    # Seed history: 5 user/assistant pairs of fat content + nothing pending.
    for i in range(5):
        await store.append_message(sid, role="user", content="u" * 4000, round_index=i)
        await store.append_message(sid, role="assistant", content="a" * 4000, round_index=i)

    summary = "<summary>folded earlier turns</summary>"
    final_text = "after compact"
    llm = _Scripted(responses=[LLMResponse(raw_text=summary), LLMResponse(raw_text=final_text)])

    compactor = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    cfg = AgentLoopConfig(system_prompt="S", max_rounds=2, max_tokens=4000, compactor=compactor)
    loop = StatefulAgentLoop(llm=llm, store=store, config=cfg)

    r = await loop.send("kick another round", session_id=sid)
    assert r.status == "completed"
    assert r.final_text == final_text

    compactions = await store.list_compactions(sid)
    assert len(compactions) == 1
    rec = compactions[0]
    assert rec.from_seq >= 1 and rec.to_seq > rec.from_seq

    # Verify the corresponding range is now compacted_out and a compact_note
    # exists between them.
    all_rows = await store.load_all_messages(sid)
    by_seq = {row.seq: row for row in all_rows}
    for s in range(rec.from_seq, rec.to_seq + 1):
        assert by_seq[s].state is MessageState.COMPACTED_OUT

    notes = [r for r in all_rows if r.name == "compact_note"]
    assert len(notes) == 1
    assert "folded earlier turns" in (notes[0].content or "")


@pytest.mark.asyncio
async def test_no_compactor_means_no_compaction(store: SessionStore) -> None:
    cfg = AgentLoopConfig(max_rounds=1, max_tokens=10, compactor=None)
    llm = _Scripted(responses=[LLMResponse(raw_text="ok")])
    loop = StatefulAgentLoop(llm=llm, store=store, config=cfg)
    sid = await loop.new_session()
    r = await loop.send("hi" * 5000, session_id=sid)
    assert await store.list_compactions(r.session_id) == []


@pytest.mark.asyncio
async def test_compactor_plan_dataclass_shape() -> None:
    p = CompactionPlan(
        fold_start_idx=1, fold_end_idx=3, summary_text="x",
        before_tokens=100, after_tokens=10,
    )
    assert p.fold_end_idx - p.fold_start_idx == 2
    assert p.summary_text == "x"


# ── H1.1 / C1: recall must not shift which DB rows get compacted_out ────────


@dataclass
class _FakeMemory:
    """Recalls canned messages; never persists (the provider owns its memory)."""

    to_recall: list[dict] = field(default_factory=list)

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [dict(m) for m in self.to_recall]

    async def remember(self, *, snapshot, session_id):
        return None


async def _run_compaction(store: SessionStore, *, memory=None, hooks=None, compactor=None) -> tuple[str, list, set]:
    """Seed a fat 5-pair history, send one turn (forcing compaction), return
    (sid, all_rows, set-of-compacted_out-seqs)."""
    sid = await store.create_session(system_prompt="S")
    for i in range(5):
        await store.append_message(sid, role="user", content="u" * 4000, round_index=i)
        await store.append_message(sid, role="assistant", content="a" * 4000, round_index=i)
    llm = _Scripted(responses=[
        LLMResponse(raw_text="<summary>folded earlier turns</summary>"),
        LLMResponse(raw_text="done"),
    ])
    cfg = AgentLoopConfig(
        system_prompt="S", max_rounds=2, max_tokens=4000,
        compactor=compactor if compactor is not None else DefaultCompactor(keep_last_n=1),
        memory=memory,
    )
    loop = StatefulAgentLoop(llm=llm, store=store, config=cfg, hooks=hooks)
    await loop.send("kick another round", session_id=sid)
    rows = await store.load_all_messages(sid)
    compacted = {r.seq for r in rows if r.state is MessageState.COMPACTED_OUT}
    return sid, rows, compacted


# ── H7 Phase 2: optional CompactionContext + back-compat ───────────────────


def test_maybe_compact_context_introspection() -> None:
    """The pipeline gate decides which compactors get the optional context."""
    from power_loop.core.pipeline import _maybe_compact_accepts_context

    assert _maybe_compact_accepts_context(DefaultCompactor.maybe_compact) is True

    class _Old:
        async def maybe_compact(self, messages, *, llm, max_tokens, round_index): ...

    class _Kwargs:
        async def maybe_compact(self, messages, **kw): ...

    assert _maybe_compact_accepts_context(_Old.maybe_compact) is False
    assert _maybe_compact_accepts_context(_Kwargs.maybe_compact) is True


@pytest.mark.asyncio
async def test_context_aware_compactor_receives_populated_context(monkeypatch) -> None:
    """A context-accepting compactor is handed session_id + the configured memory
    + a working read-only fetch_messages accessor."""
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "100")
    from power_loop.runtime.compact import CompactionContext

    class _RecordingCompactor(DefaultCompactor):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.seen: CompactionContext | None = None

        async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
            self.seen = context
            return await super().maybe_compact(
                messages, llm=llm, max_tokens=max_tokens, round_index=round_index, context=context)

    mem = _FakeMemory(to_recall=[])
    cmp = _RecordingCompactor(keep_last_n=1)
    store = await SessionStore.open(":memory:")
    try:
        sid, _, compacted = await _run_compaction(store, memory=mem, compactor=cmp)
        assert compacted, "should have compacted"
        ctx = cmp.seen
        assert ctx is not None
        assert ctx.session_id == sid
        assert ctx.memory is mem
        assert callable(ctx.fetch_messages)
        fetched = await ctx.fetch_messages(1, 10)
        assert fetched and all("role" in m and "seq" in m for m in fetched)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_old_signature_compactor_still_works(monkeypatch) -> None:
    """Back-compat: a compactor with the OLD signature (no `context`) must NOT be
    passed context — it runs end-to-end without a TypeError and still compacts."""
    class _OldStyleCompactor:
        async def maybe_compact(self, messages, *, llm, max_tokens, round_index):
            if len(messages) < 4:
                return None
            return CompactionPlan(
                fold_start_idx=1, fold_end_idx=len(messages) - 2,
                summary_text="old-style summary", before_tokens=100, after_tokens=10)

    store = await SessionStore.open(":memory:")
    try:
        sid, _, compacted = await _run_compaction(store, compactor=_OldStyleCompactor())
        assert compacted, "old-signature compactor should still fold rows"
        assert await store.list_compactions(sid), "and persist the compaction"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_recall_does_not_shift_compacted_rows(monkeypatch) -> None:
    """C1 regression: recalled memory_* messages are injected at the FRONT of
    history; with the index↔seq map kept aligned, the SAME real conversation rows
    are marked compacted_out whether or not memory was recalled. Before the fix the
    range was shifted by len(recalled) → the wrong rows were marked (silent
    persisted corruption)."""
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "100")  # tiny → always fires on the fat seed
    store_a = await SessionStore.open(":memory:")
    store_b = await SessionStore.open(":memory:")
    try:
        sid_a, _, compacted_no_mem = await _run_compaction(store_a, memory=None)
        mem = _FakeMemory(to_recall=[
            {"content": "user prefers concise replies"},
            {"content": "user lives in Shanghai"},
        ])
        sid_b, rows_b, compacted_with_mem = await _run_compaction(store_b, memory=mem)

        assert compacted_no_mem, "baseline run should have compacted some rows"
        # The exact same real rows are folded — recall shifts indices, not seqs.
        assert compacted_with_mem == compacted_no_mem
        # the recorded audit range matches too
        assert [(c.from_seq, c.to_seq) for c in await store_a.list_compactions(sid_a)] == \
               [(c.from_seq, c.to_seq) for c in await store_b.list_compactions(sid_b)]
        # recalled memory is NEVER persisted as a session row
        assert not any((r.name or "").startswith("memory_") for r in rows_b)
        # and the kept tail (most-recent rows) is NOT compacted out
        active = {r.seq for r in rows_b if r.state is MessageState.ACTIVE}
        assert active and active.isdisjoint(compacted_with_mem)
    finally:
        await store_a.close()
        await store_b.close()


@pytest.mark.asyncio
async def test_compaction_skips_persistence_when_history_seqs_desync(monkeypatch) -> None:
    """C9 safety net: a SESSION_START hook that replaces history wholesale desyncs
    the sink's index↔seq map. on_compaction must then REFUSE to persist (rather than
    mark wrong rows compacted_out); the run still completes, active rows untouched."""
    from power_loop import AgentHooks
    from power_loop.contracts.hooks import HookPoint

    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "100")
    store = await SessionStore.open(":memory:")
    try:
        hooks = AgentHooks()

        def prepend_msg(ctx) -> None:
            # add a message the sink was never told about → length desync
            ctx.messages = [{"role": "system", "content": "injected by hook"}, *ctx.messages]

        hooks.register(HookPoint.SESSION_START, prepend_msg)

        sid, rows, compacted = await _run_compaction(store, hooks=hooks)

        # safety net engaged: nothing persisted as compacted, no audit row
        assert compacted == set()
        assert await store.list_compactions(sid) == []
        # the original rows remain ACTIVE (a resume would be correct)
        assert all(r.state is MessageState.ACTIVE for r in rows if r.name != "compact_note")
    finally:
        await store.close()
