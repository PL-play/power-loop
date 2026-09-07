"""6.10.0 上下文三旋钮：折叠预算解耦 / 上下文检查点 / send 内保险丝（投影替换）。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

import power_loop
from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    seen: list[list[dict[str, Any]]] = field(default_factory=list)
    _idx: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text: Callable[[str], Any] | None = None,
                       on_chunk_think: Callable[[str], Any] | None = None,
                       on_stream_end: Callable[[LLMResponse], Any] | None = None) -> LLMResponse:
        # 拍快照：pipeline 会原地改写历史行的 content（这正是保险丝的行为），引用会失真
        snap = [dict(m) for m in (getattr(request, "messages", None) or []) if isinstance(m, dict)]
        self.seen.append(snap)
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


def _resp(text: str, *, prompt: int = 10, completion: int = 2) -> LLMResponse:
    r = LLMResponse(raw_text=text, content_text=text)
    r.token_usage = LLMTokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
    return r


def _tool_resp(call_id: str, *, prompt: int = 10, completion: int = 2) -> LLMResponse:
    r = LLMResponse(raw_text="", tool_calls=[{"id": call_id, "type": "function",
                                              "function": {"name": "echo", "arguments": "{\"text\": \"x\"}"}}])
    r.token_usage = LLMTokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
    return r


def _echo_registry(out: str) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="echo", description="echo",
                       input_schema={"type": "object", "properties": {"text": {"type": "string"}}}),
        lambda **kw: out,
    )
    return reg


def _tool_rows(msgs: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("content") or "") for m in msgs if m.get("role") == "tool"]


# ── ① 折叠预算独立于输出上限 ─────────────────────────────────────────────

def test_context_budget_is_independent_from_output_max_tokens():
    assert AgentLoopConfig(system_prompt="t", max_tokens=40000).effective_context_budget() == 40000  # 兼容回退
    assert AgentLoopConfig(system_prompt="t", max_tokens=40000,
                           context_budget_tokens=30000).effective_context_budget() == 30000
    assert AgentLoopConfig(system_prompt="t", max_tokens=40000,
                           context_budget_tokens=0).effective_context_budget() == 40000  # 0 = 未设


# ── ② 上下文检查点：按上一轮真实 prompt_tokens，轮边界优雅收尾 ────────────

@pytest.mark.asyncio
async def test_context_checkpoint_ends_send_when_real_prompt_reaches_threshold(tmp_path):
    llm = _Scripted(responses=[
        _tool_resp("c1", prompt=100),     # round 0：100 < 4000 → 继续
        _tool_resp("c2", prompt=5000),    # round 1：真实 prompt 5000 ≥ 4000 → 边界收尾
        _resp("should never be reached"),
    ])
    events: list = []
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.STATUS_CHANGED, events.append)
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, context_checkpoint_tokens=4000),
        tool_registry=_echo_registry("ok"), event_bus=bus,
    )
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "context_checkpoint"
    assert res.rounds == 2
    assert llm._idx == 2, "第三个响应不该被消费：检查点在轮边界停"
    kinds = [(e.payload or {}).get("kind") for e in events]
    assert "context_checkpoint" in kinds
    ev = next(e for e in events if (e.payload or {}).get("kind") == "context_checkpoint")
    assert ev.payload["spent_tokens"] == 5000 and ev.payload["budget_tokens"] == 4000
    # 不留悬空 tool_calls：下一个 send 直接能跑（投影/续接靠宿主）
    res2 = await loop.send("again", session_id=sid)
    assert res2.status == "completed"
    await loop.aclose()


@pytest.mark.asyncio
async def test_context_checkpoint_off_by_default(tmp_path):
    llm = _Scripted(responses=[_tool_resp("c1", prompt=900000), _tool_resp("c2", prompt=900000), _resp("fin")])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=8),
                             tool_registry=_echo_registry("ok"))
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "completed" and res.final_text == "fin"
    await loop.aclose()


# ── ③ send 内保险丝：最早 n 条工具结果 → 投影行（内存替换，逐轮递进）───────

@pytest.mark.asyncio
async def test_insend_distill_replaces_oldest_tool_rows_progressively(tmp_path):
    big = "R" * 400  # > 300 字符才算「值得蒸馏」
    llm = _Scripted(responses=[
        _tool_resp("c1", prompt=10),     # round 0 → prepare_round(1)：10 < 50，不动
        _tool_resp("c2", prompt=100),    # round 1 → prepare_round(2)：100 ≥ 50，蒸馏最早 1 条(c1)
        _tool_resp("c3", prompt=100),    # round 2 → prepare_round(3)：仍 ≥ 50，再蒸馏下一条(c2)
        _resp("done"),
    ])
    kwargs: dict[str, Any] = {}
    rep_cls = getattr(power_loop, "ProjectedRepresentation", None)
    if rep_cls is not None:
        kwargs["representation"] = rep_cls()
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, insend_distill_tokens=50,
                               insend_distill_batch=1, insend_distill_hot_tail=0, **kwargs),
        tool_registry=_echo_registry(big),
    )
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "completed"
    assert len(llm.seen) == 4
    # round 1 请求：c1 原文
    assert _tool_rows(llm.seen[1]) == [big]
    # round 2 请求：c1 已换成投影行（带 send_index+seq 的 recall 坐标），c2 原文
    r2 = _tool_rows(llm.seen[2])
    assert len(r2) == 2 and r2[0].startswith("[distilled #1 seq=") and "recall_send(send_index=1, seq=" in r2[0]
    assert r2[1] == big
    # round 3 请求：c2 也被蒸馏，c3 原文——每轮只推进一批
    r3 = _tool_rows(llm.seen[3])
    assert [x.startswith("[distilled #1 seq=") for x in r3] == [True, True, False]
    assert len(r3[0]) < 400
    # 存储里的原文不动（pl_messages 是真相；这里通过再次装配上一 send 的投影/原文侧面验证不抛错）
    res2 = await loop.send("again", session_id=sid)
    assert res2.status == "completed"
    await loop.aclose()


@pytest.mark.asyncio
async def test_insend_distill_respects_hot_tail(tmp_path):
    big = "R" * 400
    llm = _Scripted(responses=[_tool_resp("c1", prompt=100), _tool_resp("c2", prompt=100), _resp("done")])
    kwargs: dict[str, Any] = {}
    rep_cls = getattr(power_loop, "ProjectedRepresentation", None)
    if rep_cls is not None:
        kwargs["representation"] = rep_cls()
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, insend_distill_tokens=50,
                               insend_distill_batch=10, insend_distill_hot_tail=2, **kwargs),
        tool_registry=_echo_registry(big),
    )
    sid = await loop.new_session()
    await loop.send("hi", session_id=sid)
    # 只有两条工具结果且 hot_tail=2 → 一条都不许动
    assert _tool_rows(llm.seen[2]) == [big, big]
    await loop.aclose()
