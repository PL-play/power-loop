"""Regression tests for Tier-2/3 pipeline fixes from the deep review:

- C12: the pending_tools return (tool_registry=None but the model emitted tool_calls)
  must still drive the terminal lifecycle (SESSION_ENDED + memory snapshot).
- C20: ctx.tool_calls counts a real, validated dispatch — not a rejected call, and
  not twice for a SHORT_CIRCUIT retry of the same logical call.
- C19/C22: prepare_round invalidates the SCALE-4 incremental token estimate after
  microcompact (content shrink, same length) and after a single-message fold.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    ToolDefinition,
    ToolRegistry,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.core.hooks import AgentHooks
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.state import ContextManager
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.compact import CompactionPlan

pytestmark = pytest.mark.unit


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _i: int = 0

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        if self._i >= len(self.responses):
            return LLMResponse(raw_text="done", content_text="done")
        r = self.responses[self._i]
        self._i += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _tool_call(cid: str, name: str) -> LLMResponse:
    return LLMResponse(raw_text="", tool_calls=[
        {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}])


def _pipeline(*, llm=None, config=None, tool_registry=None, ctx=None, bus=None) -> AgentPipeline:
    return AgentPipeline(
        llm=llm,
        config=config or AgentLoopConfig(),
        tool_registry=tool_registry,
        hooks=AgentHooks(),
        bus=bus or AgentEventBus(),
        ctx=ctx or ContextManager(),
    )


# ── C12: pending_tools drives the terminal lifecycle ─────────────────────────


async def test_pending_tools_return_finalizes_and_remembers() -> None:
    class _Mem:
        def __init__(self) -> None:
            self.remembered: list = []

        async def recall(self, *, messages, session_id, budget_tokens=1500):
            return []

        async def remember(self, *, snapshot, session_id) -> None:
            self.remembered.append(snapshot)

    mem = _Mem()
    bus = AgentEventBus()
    events: list = []
    bus.subscribe(None, events.append)
    p = _pipeline(
        llm=_Scripted(responses=[_tool_call("c1", "doit")]),
        config=AgentLoopConfig(memory=mem),
        tool_registry=None,  # registry-less: the pending_tools contract path
        bus=bus,
    )
    res = await p.run([{"role": "user", "content": "hi"}])

    assert res.status == "pending_tools"
    types = [e.type for e in events]
    assert AgentEventType.SESSION_STARTED in types
    assert AgentEventType.SESSION_ENDED in types  # was dropped pre-fix
    assert len(mem.remembered) == 1  # end-of-run snapshot was skipped pre-fix


# ── C20: tool_calls counts only a real, validated, non-retried dispatch ──────


async def test_tool_calls_counter_excludes_rejected_and_retry() -> None:
    reg = ToolRegistry()

    calls: list[str] = []

    def _ok(**kwargs):
        calls.append("ok")
        return "ok"

    reg.register(ToolDefinition(name="ok", description="d", input_schema={"type": "object"}), _ok)

    # unknown tool → rejected by validate, must NOT count
    p = _pipeline(tool_registry=reg)
    out, failed = await p.execute_tool("ghost", {})
    assert failed and p.ctx.tool_calls == 0

    # a real dispatch counts once
    await p.execute_tool("ok", {})
    assert p.ctx.tool_calls == 1

    # a SHORT_CIRCUIT-style retry of the same logical call (count=False) must not double
    await p.execute_tool("ok", {}, count=False)
    assert p.ctx.tool_calls == 1


# ── C19: microcompact shrink invalidates the estimate ────────────────────────


async def test_prepare_round_invalidates_estimate_after_microcompact(tmp_path) -> None:
    ctx = ContextManager(micro_hot_tail=0, micro_size_limit=100)
    ctx.cache_dir = tmp_path
    p = _pipeline(
        config=AgentLoopConfig(compactor=None, microcompact_enabled=True, microcompact_hot_tail=0,
                               microcompact_size_limit=100),
        tool_registry=ToolRegistry(), ctx=ctx,
    )
    await p._append_message({"role": "user", "content": "hi"})
    await p._append_message({"role": "assistant", "content": "",
                             "tool_calls": [{"id": "t", "type": "function",
                                             "function": {"name": "x", "arguments": "{}"}}]})
    await p._append_message({"role": "tool", "tool_call_id": "t", "name": "x", "content": "Z" * 5000})
    primed = p._estimate_history_tokens()  # large, cached

    await p.prepare_round(1)  # microcompact spills the 5000-char output → shrinks content

    recomputed = estimate_tokens(p.history)
    assert p._estimate_history_tokens() == recomputed  # cache must match (was stale-high)
    assert recomputed < primed  # content actually shrank


@pytest.mark.asyncio
async def test_microcompact_off_by_default_does_not_spill(tmp_path) -> None:
    """3.1.x: microcompact is OFF by default — prepare_round must NOT spill or
    rewrite content when config.microcompact_enabled is False."""
    ctx = ContextManager(micro_hot_tail=0, micro_size_limit=100)
    ctx.cache_dir = tmp_path
    p = _pipeline(config=AgentLoopConfig(compactor=None), tool_registry=ToolRegistry(), ctx=ctx)
    await p._append_message({"role": "user", "content": "hi"})
    await p._append_message({"role": "assistant", "content": "",
                             "tool_calls": [{"id": "t", "type": "function",
                                             "function": {"name": "x", "arguments": "{}"}}]})
    await p._append_message({"role": "tool", "tool_call_id": "t", "name": "x", "content": "Z" * 5000})

    await p.prepare_round(1)

    tool_msg = next(m for m in p.history if m.get("role") == "tool")
    assert tool_msg["content"] == "Z" * 5000  # untouched — no spill
    assert not any(tmp_path.iterdir())  # nothing written to disk


# ── C22: a single-message fold invalidates the estimate ──────────────────────


@dataclass
class _SingleMsgFoldCompactor:
    """Folds exactly one message into one note → history length is unchanged, so the
    incremental estimate would stay stale without invalidation."""

    fired: bool = False

    async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
        if self.fired or len(messages) < 3:
            return None
        self.fired = True
        return CompactionPlan(fold_start_idx=1, fold_end_idx=1, summary_text="s",
                              before_tokens=0, after_tokens=1)


async def test_prepare_round_invalidates_estimate_after_single_message_fold() -> None:
    p = _pipeline(
        llm=_Scripted(),
        config=AgentLoopConfig(max_tokens=4000, compactor=_SingleMsgFoldCompactor()),
        tool_registry=ToolRegistry(),
    )
    for _ in range(4):
        await p._append_message({"role": "user", "content": "word " * 80})
    primed = p._estimate_history_tokens()  # cached over 4 fat messages

    await p.prepare_round(1)  # folds index 1 (one msg) → one tiny note; length unchanged

    recomputed = estimate_tokens(p.history)
    assert p._estimate_history_tokens() == recomputed  # was the stale pre-fold value
    assert recomputed < primed
