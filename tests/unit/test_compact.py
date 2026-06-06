from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
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


# ── DefaultCompactor: atomic pair preservation ─────────────────────────


@pytest.mark.asyncio
async def test_compactor_never_splits_assistant_tool_pair(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    summary_llm = _Scripted(responses=[LLMResponse(raw_text="<summary>ok</summary>")])
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "u1" * 1000},
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "tc"}]},
        {"role": "tool", "tool_call_id": "tc", "content": "out" * 1000},
        {"role": "user", "content": "u2"},
    ]
    plan = await cp.maybe_compact(msgs, llm=summary_llm, max_tokens=2000, round_index=1)
    # If tail keep_last_n=1 lands on the user u2 only, the assistant/tool
    # pair must NOT straddle: either both folded or both kept.
    if plan is not None:
        # If fold_end_idx == 2 (the tool), assistant at 1 must also be in fold
        assert not (plan.fold_end_idx == 2 and plan.fold_start_idx > 1)


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
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


@pytest.mark.asyncio
async def test_compaction_persists_to_store_and_marks_state(store: SessionStore, monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    """Long history forces compaction during the next send; verify the store
    has compacted_out rows + a compact_note + a compactions audit row."""
    sid = store.create_session(system_prompt="S")
    # Seed history: 5 user/assistant pairs of fat content + nothing pending.
    for i in range(5):
        store.append_message(sid, role="user", content="u" * 4000, round_index=i)
        store.append_message(sid, role="assistant", content="a" * 4000, round_index=i)

    summary = "<summary>folded earlier turns</summary>"
    final_text = "after compact"
    llm = _Scripted(responses=[LLMResponse(raw_text=summary), LLMResponse(raw_text=final_text)])

    compactor = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    cfg = AgentLoopConfig(system_prompt="S", max_rounds=2, max_tokens=4000, compactor=compactor)
    loop = StatefulAgentLoop(llm=llm, store=store, config=cfg)

    r = await loop.send("kick another round", session_id=sid)
    assert r.status == "completed"
    assert r.final_text == final_text

    compactions = store.list_compactions(sid)
    assert len(compactions) == 1
    rec = compactions[0]
    assert rec.from_seq >= 1 and rec.to_seq > rec.from_seq

    # Verify the corresponding range is now compacted_out and a compact_note
    # exists between them.
    all_rows = store.load_all_messages(sid)
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
    sid = loop.new_session()
    r = await loop.send("hi" * 5000, session_id=sid)
    assert store.list_compactions(r.session_id) == []


@pytest.mark.asyncio
async def test_compactor_plan_dataclass_shape() -> None:
    p = CompactionPlan(
        fold_start_idx=1, fold_end_idx=3, summary_text="x",
        before_tokens=100, after_tokens=10,
    )
    assert p.fold_end_idx - p.fold_start_idx == 2
    assert p.summary_text == "x"
