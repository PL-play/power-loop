"""StatefulResult.usage accumulation + send(heal_pending=True) recovery.

Both features exist because orchestrators (one loop, many sessions, runs that
can be killed mid tool-call) need per-run token accounting and crash recovery
without spelunking the event bus / store internals.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import SessionPendingError, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk, LLMTokenUsage
from power_loop.core.state import ContextManager


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
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

    async def close(self) -> None:
        return None


def _resp(text: str, *, prompt: int, completion: int) -> LLMResponse:
    r = LLMResponse(raw_text=text, content_text=text)
    r.token_usage = LLMTokenUsage(prompt_tokens=prompt, completion_tokens=completion,
                                  total_tokens=prompt + completion)
    return r


def _tool_resp(call_id: str, name: str, *, prompt: int, completion: int) -> LLMResponse:
    r = LLMResponse(
        raw_text="",
        tool_calls=[{"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": "{}"}}],
    )
    r.token_usage = LLMTokenUsage(prompt_tokens=prompt, completion_tokens=completion,
                                  total_tokens=prompt + completion)
    return r


# ── ContextManager.usage_totals ─────────────────────────────────────────


def test_update_usage_keeps_last_call_and_accumulates_totals():
    ctx = ContextManager()
    ctx.update_usage(_resp("a", prompt=100, completion=10))
    ctx.update_usage(_resp("b", prompt=200, completion=20))
    # token_usage = last call (documented overwrite semantics)
    assert ctx.token_usage["prompt_tokens"] == 200
    # usage_totals = running sum + call count
    assert ctx.usage_totals["prompt_tokens"] == 300
    assert ctx.usage_totals["completion_tokens"] == 30
    assert ctx.usage_totals["calls"] == 2


def test_usage_totals_counts_calls_without_usage():
    ctx = ContextManager()
    ctx.update_usage(LLMResponse(raw_text="no usage attached"))
    assert ctx.usage_totals["calls"] == 1
    assert ctx.usage_totals["prompt_tokens"] == 0


# ── StatefulResult.usage ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_result_carries_summed_usage(tmp_path):
    from power_loop import AgentLoopConfig, create_default_tool_registry

    llm = _Scripted(responses=[
        _tool_resp("c1", "todo_write", prompt=100, completion=10),  # round 1: tool call
        _resp("final answer", prompt=150, completion=5),            # round 2: text
    ])
    reg = create_default_tool_registry(preset="core", bind=False)
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
        tool_registry=reg,
    )
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "completed"
    assert res.usage["prompt_tokens"] == 250
    assert res.usage["completion_tokens"] == 15
    assert res.usage["calls"] == 2
    loop.close()


@pytest.mark.asyncio
async def test_tool_rounds_also_persist_per_round_usage_rows(tmp_path):
    """每一轮都要有 usage_rounds 行——包括带工具调用的回合。

    5.2.2 之前 on_round_ended(usage=…) 只在无工具回合的收尾路径上发：agent 会话里几乎
    每轮都带工具，于是一个 34 轮的叶子在表里只剩最后一轮那一行（2026-08-19 线上实锤）。
    总量从来没错（session_stats 用 send 结束时的内存聚合 bump），丢的是这张表存在的
    唯一意义——逐轮明细。"""
    import sqlite3

    from power_loop import AgentLoopConfig, create_default_tool_registry

    llm = _Scripted(responses=[
        _tool_resp("c1", "todo_write", prompt=100, completion=10),  # round 1: tool round
        _tool_resp("c2", "todo_write", prompt=120, completion=12),  # round 2: tool round
        _resp("final answer", prompt=150, completion=5),            # round 3: text
    ])
    reg = create_default_tool_registry(preset="core", bind=False)
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=6),
        tool_registry=reg,
    )
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "completed"
    loop.close()
    rows = sqlite3.connect(str(tmp_path / "s.db")).execute(
        "SELECT round_index, prompt_tokens, completion_tokens FROM pl_usage_rounds "
        "WHERE session_id=? ORDER BY round_index", (sid,)
    ).fetchall()
    assert [r[0] for r in rows] == [0, 1, 2], f"每轮一行，含工具回合；实际 {rows}"
    assert [r[1] for r in rows] == [100, 120, 150]
    assert [r[2] for r in rows] == [10, 12, 5]


@pytest.mark.asyncio
async def test_each_send_gets_its_own_usage(tmp_path):
    from power_loop import AgentLoopConfig

    llm = _Scripted(responses=[
        _resp("one", prompt=10, completion=1),
        _resp("two", prompt=20, completion=2),
    ])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t"),
    )
    sid = await loop.new_session()
    r1 = await loop.send("a", session_id=sid)
    r2 = await loop.send("b", session_id=sid)
    assert r1.usage["prompt_tokens"] == 10
    assert r2.usage["prompt_tokens"] == 20  # NOT cumulative across sends
    loop.close()


# ── send(heal_pending=True) ─────────────────────────────────────────────


async def _wedge_session(loop: StatefulAgentLoop, sid: str) -> None:
    """Simulate a run killed mid tool-call: pending tool_calls in the store."""
    await loop.store.append_message(
        sid, role="assistant", content="",
        tool_calls=[{"id": "dead1", "type": "function",
                     "function": {"name": "echo", "arguments": "{}"}}],
    )
    await loop.store.set_pending(sid, {
        "assistant_seq": 1, "round_index": 0,
        "tool_call_ids": ["dead1"],
        "tool_calls": [{"id": "dead1", "type": "function",
                        "function": {"name": "echo", "arguments": "{}"}}],
    })


@pytest.mark.asyncio
async def test_send_raises_pending_by_default_with_guidance(tmp_path):
    from power_loop import AgentLoopConfig

    loop = StatefulAgentLoop(
        llm=_Scripted(responses=[_resp("ok", prompt=1, completion=1)]),
        db_path=str(tmp_path / "s.db"), config=AgentLoopConfig(system_prompt="t"),
    )
    sid = await loop.new_session()
    await _wedge_session(loop, sid)
    with pytest.raises(SessionPendingError) as ei:
        await loop.send("hello", session_id=sid)
    assert "heal_pending=True" in str(ei.value)
    loop.close()


@pytest.mark.asyncio
async def test_send_heal_pending_aborts_and_proceeds(tmp_path):
    from power_loop import AgentLoopConfig

    loop = StatefulAgentLoop(
        llm=_Scripted(responses=[_resp("recovered", prompt=1, completion=1)]),
        db_path=str(tmp_path / "s.db"), config=AgentLoopConfig(system_prompt="t"),
    )
    sid = await loop.new_session()
    await _wedge_session(loop, sid)
    res = await loop.send("hello again", session_id=sid, heal_pending=True)
    assert res.status == "completed"
    assert res.final_text == "recovered"
    # The dead tool_call got a synthetic <aborted> tool message.
    msgs = await loop.get_messages(sid)
    aborted = [m for m in msgs if m.get("role") == "tool" and "aborted" in str(m.get("content"))]
    assert len(aborted) == 1
    # Healthy session: heal_pending is a no-op.
    res2 = await loop.send("again", session_id=sid, heal_pending=True)
    assert res2.status == "completed"
    loop.close()
