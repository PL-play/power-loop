"""Regression suite for the double-compaction-within-one-run corruption (C1 / 0.14.1).

Root cause: a ``compact_note`` was allocated a fresh HIGH identity ``seq`` but placed
at a LOW logical position in the in-memory history. That made the sink's index→seq
map (``_history_seqs``) non-monotonic, so:

* **Facet A (corruption):** a *second* fold in the same run translated its fold
  boundary through the non-monotonic map and called
  ``record_compaction(from_seq, to_seq)`` with ``from_seq > to_seq`` → the
  ``BETWEEN`` mark folded NOTHING in the DB while the in-memory fold proceeded →
  in-memory history diverged from the persisted active set, and an inverted
  ``(from,to)`` row was written to the ``compactions`` audit table.
* **Facet B (ordering):** even a single fold sank the note below the kept tail on
  reload (``ORDER BY seq`` put its high id last), so the summary of OLD turns
  appeared AFTER newer kept messages.

The fix decouples identity (high ``seq``) from logical position: the sink marks the
EXACT folded seq-set (no range), and the note carries a logical ``ord`` that both the
sink's index map and ``load_active_messages`` order by. These tests pin the invariant
that the in-memory active set and order ALWAYS equal what reload reconstructs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.agent.sink import SQLiteSink
from power_loop.runtime.compact import CompactionPlan

pytestmark = pytest.mark.unit


# ── helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


def _append(sink: SQLiteSink, n: int, *, start_round: int = 0) -> None:
    for i in range(n):
        sink.on_message_appended(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"},
            round_index=start_round + i,
        )


def _inmem_active(sink: SQLiteSink) -> list[int]:
    """The in-memory active seqs, in logical (history) order."""
    return [s for s in sink._history_seqs if s is not None]


def _db_active_seqs(store: SessionStore, sid: str) -> list[int]:
    return [r.seq for r in store.load_active_messages(sid)]


def _fold(sink: SQLiteSink, start: int, end: int, *, summary: str, rnd: int) -> None:
    sink.on_compaction(
        fold_start_idx=start,
        fold_end_idx=end,
        summary_text=summary,
        before_tokens=100,
        after_tokens=10,
        round_index=rnd,
        expected_history_len=len(sink._history_seqs),
    )


def _assert_consistent(store: SessionStore, sid: str, sink: SQLiteSink) -> None:
    """The strong invariant: reload reproduces the EXACT in-memory active history
    — same membership AND same order — and no audit row is inverted."""
    assert _inmem_active(sink) == _db_active_seqs(store, sid), (
        "in-memory active history diverged from the persisted/reloaded active set"
    )
    for c in store.list_compactions(sid):
        assert c.from_seq <= c.to_seq, f"inverted audit row ({c.from_seq},{c.to_seq})"


# ── Facet A: the corruption ──────────────────────────────────────────────────


def test_two_folds_one_run_no_divergence(store: SessionStore) -> None:
    """Two folds within ONE run must keep in-memory history == persisted active set
    (the regression: previously 5 in-mem vs 9 in DB)."""
    sid = store.create_session(session_id="s")
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs([])

    _append(sink, 6)
    _fold(sink, 0, 2, summary="S1", rnd=6)
    _assert_consistent(store, sid, sink)

    _append(sink, 4, start_round=6)
    _fold(sink, 0, 3, summary="S2", rnd=10)  # folds the prior note + tail head

    _assert_consistent(store, sid, sink)
    # exactly one active compact_note survives (each fold merges the prior note in)
    notes = [r for r in store.load_active_messages(sid) if r.name == "compact_note"]
    assert len(notes) == 1


def test_three_folds_one_run_stay_consistent(store: SessionStore) -> None:
    """Stress the same-run regime past two folds."""
    sid = store.create_session(session_id="s")
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs([])
    rnd = 0
    for _ in range(3):
        _append(sink, 4, start_round=rnd)
        rnd += 4
        # fold everything but the last message
        _fold(sink, 0, len(sink._history_seqs) - 2, summary=f"S{rnd}", rnd=rnd)
        _assert_consistent(store, sid, sink)


def test_no_inverted_audit_rows(store: SessionStore) -> None:
    """The compactions audit table never records from_seq > to_seq."""
    sid = store.create_session(session_id="s")
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs([])
    _append(sink, 6)
    _fold(sink, 0, 2, summary="S1", rnd=6)
    _append(sink, 4, start_round=6)
    _fold(sink, 0, 3, summary="S2", rnd=10)
    audit = [(c.from_seq, c.to_seq) for c in store.list_compactions(sid)]
    assert audit and all(a <= b for a, b in audit), audit


# ── Facet B: reload ordering ─────────────────────────────────────────────────


def test_single_fold_note_reloads_at_front(store: SessionStore) -> None:
    """A prefix fold's note summarizes the oldest turns → it must reload FIRST,
    not sink below the kept tail (the high-identity-seq bug)."""
    sid = store.create_session(session_id="s")
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs([])
    _append(sink, 6)
    _fold(sink, 0, 2, summary="S", rnd=6)

    rows = store.load_active_messages(sid)
    assert rows[0].name == "compact_note", [r.name or r.role for r in rows]
    _assert_consistent(store, sid, sink)


def test_middle_fold_note_reloads_between_head_and_tail(store: SessionStore) -> None:
    """A note that folds a MIDDLE span sorts after the kept head and before the
    kept tail on reload (logical position, not identity seq)."""
    sid = store.create_session(session_id="s")
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs([])
    _append(sink, 6)  # seqs 1..6
    # keep head [1,2], fold middle [3,4,5] (indices 2..4), keep tail [6]
    _fold(sink, 2, 4, summary="MID", rnd=6)

    rows = store.load_active_messages(sid)
    names = [r.name or r.role for r in rows]
    note_pos = names.index("compact_note")
    seqs = [r.seq for r in rows]
    assert seqs[:note_pos] == [1, 2], seqs
    assert rows[-1].seq == 6
    _assert_consistent(store, sid, sink)


# ── resume regime: reload then fold again ────────────────────────────────────


def _reseed_sink(store: SessionStore, sid: str) -> SQLiteSink:
    """Build a fresh sink seeded from the DB exactly as StatefulAgentLoop does."""
    rows = store.load_active_messages(sid)
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs(
        [r.seq for r in rows],
        [
            int(r.meta["ord"])
            if r.name == "compact_note" and r.meta.get("ord") is not None
            else r.seq
            for r in rows
        ],
    )
    return sink


def test_reload_then_fold_consistent(store: SessionStore) -> None:
    """Fold, simulate a process restart (reseed the sink from the DB), append more,
    fold again — the reloaded logical map must keep marking correct."""
    sid = store.create_session(session_id="s")
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs([])
    _append(sink, 6)
    _fold(sink, 0, 2, summary="S1", rnd=6)

    sink2 = _reseed_sink(store, sid)
    # the reseeded in-memory order must already match the DB
    assert _inmem_active(sink2) == _db_active_seqs(store, sid)
    _append(sink2, 3, start_round=10)
    _fold(sink2, 0, 2, summary="S2", rnd=13)
    _assert_consistent(store, sid, sink2)


# ── record_compaction: explicit fold-set marking ─────────────────────────────


def test_fold_over_only_placeholders_keeps_maps_aligned(store: SessionStore) -> None:
    """A fold whose span is entirely in-memory-only (recalled ``None``) placeholders
    persists nothing, but must still mirror the pipeline's in-memory fold so the
    index maps stay length-aligned — and a subsequent real fold stays consistent."""
    sid = store.create_session(session_id="s")
    sink = SQLiteSink(store, sid)
    sink.init_history_seqs([])
    # two recalled placeholders at the front, then two real messages
    sink.on_messages_inserted(index=0, count=2)
    _append(sink, 2)
    assert len(sink._history_seqs) == len(sink._history_ord) == 4

    # fold the two placeholders (indices 0..1) — nothing to persist
    _fold(sink, 0, 1, summary="P", rnd=2)
    assert len(sink._history_seqs) == len(sink._history_ord)  # stayed aligned
    assert store.list_compactions(sid) == []  # nothing persisted
    _assert_consistent(store, sid, sink)

    # a subsequent real fold still persists correctly (safety net not tripped)
    _append(sink, 2, start_round=2)
    real_start = next(i for i, s in enumerate(sink._history_seqs) if s is not None)
    _fold(sink, real_start, len(sink._history_seqs) - 2, summary="R", rnd=4)
    _assert_consistent(store, sid, sink)
    assert store.list_compactions(sid), "the real fold should have persisted"


def test_record_compaction_explicit_set_marks_exactly_those(store: SessionStore) -> None:
    sid = store.create_session(session_id="s")
    seqs = [store.append_message(sid, role="user", content=f"m{i}") for i in range(5)]
    # fold a NON-contiguous subset; only those become compacted_out
    store.record_compaction(
        sid, from_seq=min(seqs[1], seqs[3]), to_seq=max(seqs[1], seqs[3]),
        note_content="n", before_tokens=1, after_tokens=1, round_index=0,
        fold_seqs=[seqs[1], seqs[3]], order_key=seqs[1],
    )
    by_seq = {r.seq: r.state for r in store.load_all_messages(sid)}
    assert by_seq[seqs[1]] is MessageState.COMPACTED_OUT
    assert by_seq[seqs[3]] is MessageState.COMPACTED_OUT
    assert by_seq[seqs[0]] is MessageState.ACTIVE
    assert by_seq[seqs[2]] is MessageState.ACTIVE  # the GAP row stays active
    assert by_seq[seqs[4]] is MessageState.ACTIVE


def test_record_compaction_unordered_fold_seqs_not_inverted(store: SessionStore) -> None:
    """Passing fold_seqs whose translated boundary would invert under BETWEEN
    (high-then-low) still marks every one and records a sane audit span."""
    sid = store.create_session(session_id="s")
    seqs = [store.append_message(sid, role="user", content=f"m{i}") for i in range(5)]
    store.record_compaction(
        sid, from_seq=seqs[0], to_seq=seqs[3], note_content="n",
        before_tokens=1, after_tokens=1, round_index=0,
        fold_seqs=[seqs[3], seqs[0], seqs[1]],  # deliberately unordered
        order_key=seqs[0],
    )
    compacted = {r.seq for r in store.load_all_messages(sid)
                 if r.state is MessageState.COMPACTED_OUT}
    assert compacted == {seqs[0], seqs[1], seqs[3]}
    c = store.list_compactions(sid)[0]
    assert c.from_seq <= c.to_seq


# ── property: any append/fold schedule keeps reload == in-memory ─────────────


@dataclass
class _Op:
    pass


@st.composite
def _schedule(draw: Any) -> list[tuple[str, int]]:
    """A sequence of ('append', k) and ('fold', _) ops; fold indices are resolved
    against the live length at apply time."""
    n = draw(st.integers(min_value=3, max_value=14))
    ops: list[tuple[str, int]] = []
    for _ in range(n):
        kind = draw(st.sampled_from(["append", "append", "fold"]))
        if kind == "append":
            ops.append(("append", draw(st.integers(min_value=1, max_value=4))))
        else:
            ops.append(("fold", draw(st.integers(min_value=0, max_value=100))))
    return ops


@given(ops=_schedule())
@settings(max_examples=200, deadline=None)
def test_random_schedule_keeps_reload_equal_to_inmemory(ops: list[tuple[str, int]]) -> None:
    store = SessionStore.open(":memory:")
    try:
        sid = store.create_session(session_id="p")
        sink = SQLiteSink(store, sid)
        sink.init_history_seqs([])
        rnd = 0
        for kind, val in ops:
            if kind == "append":
                _append(sink, val, start_round=rnd)
                rnd += val
                continue
            # fold: need at least 2 real messages to pick a non-trivial span
            length = len(sink._history_seqs)
            if length < 2:
                continue
            # resolve a valid [start, end] from the seed value
            start = val % (length - 1)
            end = start + (val % (length - start))
            end = min(end, length - 1)
            if end < start:
                continue
            _fold(sink, start, end, summary=f"S{rnd}", rnd=rnd)
            rnd += 1
            # INVARIANT: reload reproduces the exact in-memory active history
            assert _inmem_active(sink) == _db_active_seqs(store, sid)
            for c in store.list_compactions(sid):
                assert c.from_seq <= c.to_seq
    finally:
        store.close()


# ── end-to-end: two compactions in ONE send through the real pipeline ────────


@dataclass
class _ToolLooper(LLMService):
    """Emits a tool call for the first ``tool_rounds`` responses (each tool result
    is fat, keeping tokens above the tiny compaction threshold every round), then a
    final answer — so a single send spans several rounds and folds more than once."""

    tool_rounds: int = 3
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        i = self._idx
        self._idx += 1
        # The summary LLM call (compaction) also routes here; it has no tools and a
        # single user message — answer it with a summary so the fold proceeds.
        if not request.tools:
            return LLMResponse(raw_text="<summary>folded</summary>")
        if i < self.tool_rounds:
            return LLMResponse(
                raw_text="",
                tool_calls=[{
                    "id": f"tc{i}", "type": "function",
                    "function": {"name": "blob", "arguments": "{}"},
                }],
            )
        return LLMResponse(raw_text="final answer")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _blob_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="blob", description="returns fat content",
                       input_schema={"type": "object", "properties": {}}),
        lambda **kw: "T" * 4000,
    )
    return reg


class _EveryRoundCompactor:
    """Folds the front of history (keeping the last 2 messages) on every round once
    there is enough to fold — so a single multi-round send folds repeatedly. The
    DefaultCompactor keys its kept tail on user-bounded *exchanges*, so it folds at
    most once per single-user-turn send; this exercises the multi-fold-in-one-run
    regime the corruption needed. The bug lived in the sink/store, not the
    compactor, so any plan-emitting compactor drives the same persistence path."""

    async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
        if len(messages) < 4:
            return None
        return CompactionPlan(
            fold_start_idx=0, fold_end_idx=len(messages) - 3,
            summary_text="folded", before_tokens=100, after_tokens=10,
        )


@pytest.mark.asyncio
async def test_two_compactions_in_one_send_end_to_end(store: SessionStore, monkeypatch) -> None:
    """The real bug regime, end-to-end: a multi-round send that folds twice must
    leave a coherent, correctly-ordered session that reloads and recalls cleanly."""
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "100")  # tiny → folds every round
    sid = store.create_session(system_prompt="S")
    for i in range(4):
        store.append_message(sid, role="user", content="u" * 4000, round_index=i)
        store.append_message(sid, role="assistant", content="a" * 4000, round_index=i)

    cfg = AgentLoopConfig(
        system_prompt="S", max_rounds=6, max_tokens=4000,
        compactor=_EveryRoundCompactor(),
    )
    loop = StatefulAgentLoop(
        llm=_ToolLooper(tool_rounds=3), store=store, config=cfg,
        tool_registry=_blob_registry(),
    )
    r = await loop.send("go", session_id=sid, tools=["blob"])
    assert r.status == "completed"

    # more than one fold actually happened (the regime that used to corrupt)
    assert len(store.list_compactions(sid)) >= 2, store.list_compactions(sid)

    # exactly one active compact_note, and it reloads at the FRONT
    active = store.load_active_messages(sid)
    notes = [m for m in active if m.name == "compact_note"]
    assert len(notes) == 1
    assert active[0].name == "compact_note", [m.name or m.role for m in active]

    # no inverted audit rows
    assert all(c.from_seq <= c.to_seq for c in store.list_compactions(sid))

    # the folded originals are retained (recall path) and consistent
    compacted = [m for m in store.load_all_messages(sid)
                 if m.state is MessageState.COMPACTED_OUT]
    assert compacted, "folded originals must be retained for recall"

    # a follow-up send on the reloaded session still works
    loop2 = StatefulAgentLoop(
        llm=_ToolLooper(tool_rounds=0), store=store, config=cfg,
        tool_registry=_blob_registry(),
    )
    r2 = await loop2.send("again", session_id=sid, tools=["blob"])
    assert r2.status == "completed"
    assert store.load_active_messages(sid)[0].name == "compact_note"
