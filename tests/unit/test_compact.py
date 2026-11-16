from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    SessionStore,
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


def test_malformed_compact_threshold_env_is_fail_soft(monkeypatch) -> None:
    # compaction-1: a malformed CONTEXT_COMPACT_THRESHOLD must NOT crash the send — fall back to the
    # configured trigger instead of raising ValueError from int().
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "not-a-number")
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    assert cp._should_trigger(before_tokens=6000, max_tokens=10_000) is True   # ratio path: >= 5000
    assert cp._should_trigger(before_tokens=4000, max_tokens=10_000) is False


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
async def test_no_compactor_means_no_compaction(store: SessionStore, runtime_agent) -> None:
    cfg = AgentLoopConfig(max_rounds=1, max_tokens=10, context_budget_tokens=10**9)
    llm = _Scripted(responses=[LLMResponse(raw_text="ok")])
    loop = await runtime_agent(model=llm, store=store, config=cfg)
    await loop.send("hi" * 5000)
    assert await store.list_compactions(loop.target.session_id) == []


@pytest.mark.asyncio
async def test_compactor_plan_dataclass_shape() -> None:
    p = CompactionPlan(
        fold_start_idx=1, fold_end_idx=3, summary_text="x",
        before_tokens=100, after_tokens=10,
    )
    assert p.fold_end_idx - p.fold_start_idx == 2
    assert p.summary_text == "x"


