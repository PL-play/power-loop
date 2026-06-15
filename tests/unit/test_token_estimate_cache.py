"""SCALE-4: the pipeline's incremental token estimate is correct + the compactor
uses it instead of re-scanning every round.

The cache must ALWAYS equal a full recompute (it only changes *when* the scan happens,
never the value): the append path updates it incrementally; folds, recall front-inserts,
and hook history-replacement invalidate it so the next read recomputes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, ToolRegistry
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.state import ContextManager
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.compact import CompactionContext, DefaultCompactor

pytestmark = pytest.mark.unit


def _pipeline() -> AgentPipeline:
    return AgentPipeline(
        llm=None,
        config=AgentLoopConfig(),
        tool_registry=ToolRegistry(),
        hooks=AgentHooks(),
        bus=AgentEventBus(),
        ctx=ContextManager(),
    )


def _msg(n: int) -> dict[str, Any]:
    return {"role": "user", "content": "x" * n}


async def test_incremental_estimate_always_matches_full_recompute() -> None:
    p = _pipeline()
    assert p._estimate_history_tokens() == 0  # empty

    # append path (incremental bump)
    for i in range(6):
        await p._append_message(_msg(100 + i * 10))
    assert p._estimate_history_tokens() == estimate_tokens(p.history)

    # fold: replace a prefix range with one note (length shrinks → recompute path)
    note = {"role": "system", "name": "compact_note", "content": "summary", "meta": {}}
    p.history = [note, *p.history[4:]]
    assert p._estimate_history_tokens() == estimate_tokens(p.history)

    # recall front-insert (in-place, length grows but NOT at the tail → recompute path)
    p.history[0:0] = [{"role": "system", "name": "memory_a", "content": "recalled fact"}]
    assert p._estimate_history_tokens() == estimate_tokens(p.history)

    # more appends after a non-append mutation: the bump re-syncs from the recomputed base
    await p._append_message(_msg(500))
    assert p._estimate_history_tokens() == estimate_tokens(p.history)


# ── compactor uses the hint instead of scanning ─────────────────────────────


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=lambda: [LLMResponse(raw_text="<summary>s</summary>")])
    _i: int = 0

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        r = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _foldable() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "last"},
    ]


@pytest.mark.asyncio
async def test_compactor_uses_current_tokens_hint_below_threshold(monkeypatch) -> None:
    """A current_tokens hint BELOW the threshold suppresses compaction even though the
    message list (if scanned) might say otherwise — proving the hint is used, not a scan."""
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)  # use the ratio path
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    ctx = CompactionContext(current_tokens=10)  # 10 < 4000*0.5
    plan = await cp.maybe_compact(_foldable(), llm=_Scripted(), max_tokens=4000, round_index=1, context=ctx)
    assert plan is None


@pytest.mark.asyncio
async def test_compactor_uses_current_tokens_hint_above_threshold(monkeypatch) -> None:
    """A current_tokens hint ABOVE the threshold triggers compaction even though the
    actual (tiny) message list would not — again proving the hint drives the trigger."""
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)  # use the ratio path
    cp = DefaultCompactor(trigger_ratio=0.5, keep_last_n=1)
    ctx = CompactionContext(current_tokens=10_000)  # 10000 > 4000*0.5
    plan = await cp.maybe_compact(_foldable(), llm=_Scripted(), max_tokens=4000, round_index=1, context=ctx)
    assert plan is not None
