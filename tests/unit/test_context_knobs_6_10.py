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

@pytest.mark.asyncio
async def test_insend_distill_only_parses_the_rows_it_picks(tmp_path, monkeypatch):
    """反查 (name, args) 只解析本批选中的那几条，不是每轮把全历史重建一遍。

    旧写法每次触发都遍历全部 assistant 行、json.loads 每一个 tool_calls 的 arguments，
    只为查其中 batch 条（默认 10）。保险丝一旦启动几乎每轮都触发，于是每轮做一次
    O(历史) 的反序列化。这里数 json.loads 的次数：新写法 = 触发次数 × batch，
    旧写法 = 1+2+3+…（随轮次线性增长）。
    """
    import power_loop.core.pipeline as pipeline_mod

    # 只数蒸馏函数**内部**的解析：同一个 json.loads 在工具执行那条路径上也会被调用。
    calls: list[str] = []
    inside: list[bool] = []
    real_loads = pipeline_mod.json.loads
    real_distill = pipeline_mod.AgentPipeline._distill_oldest_tool_rows

    class _CountingJson:
        dumps = staticmethod(pipeline_mod.json.dumps)

        @staticmethod
        def loads(x, *a, **kw):
            if inside and isinstance(x, str):
                calls.append(x)
            return real_loads(x, *a, **kw)

    def _spy(self, batch, hot_tail):
        inside.append(True)
        try:
            return real_distill(self, batch, hot_tail)
        finally:
            inside.pop()

    monkeypatch.setattr(pipeline_mod, "json", _CountingJson)
    monkeypatch.setattr(pipeline_mod.AgentPipeline, "_distill_oldest_tool_rows", _spy)

    big = "R" * 400
    llm = _Scripted(responses=[
        _tool_resp("c1", prompt=10),    # prepare_round(1)：10 < 50，不触发
        _tool_resp("c2", prompt=100),   # prepare_round(2)：触发，选 1 条
        _tool_resp("c3", prompt=100),   # prepare_round(3)：触发，选 1 条
        _tool_resp("c4", prompt=100),   # prepare_round(4)：触发，选 1 条
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
    assert (await loop.send("hi", session_id=sid)).status == "completed"
    await loop.aclose()

    args_parsed = [c for c in calls if c == '{"text": "x"}']
    # 3 次触发 × batch=1。旧写法这里是 1+2+3=6。
    assert len(args_parsed) == 3, f"解析了 {len(args_parsed)} 次，应当只解析选中的那 3 条"
@pytest.mark.asyncio
async def test_insend_distill_protects_everything_when_fewer_rows_than_hot_tail(tmp_path):
    """工具行少于 hot_tail 时一条都不许动（负数切片曾让它绕回去）。

    `tool_idx[len(tool_idx) - hot_tail:]`：条数不足时起点是负数，Python 当成「倒数第
    |x| 条」。hot_tail=8 时，5 条只保住 3 条、6 条只保住 2 条、7 条只保住 1 条——被放开的
    正是模型手边刚拿到的结果。三次大工具结果就能把上下文顶到阈值，这条路真能走到。
    本意是「不足 hot_tail 条就全保住」：保险丝的前提是有旧结果可回收，没有旧的就什么都
    不该动，让它涨到切 send 阈值优雅收尾（断片比切 send 贵得多）。
    """
    big = "R" * 400
    llm = _Scripted(responses=[
        _tool_resp("c1", prompt=100),   # 从 prepare_round(1) 起每轮都超阈值
        _tool_resp("c2", prompt=100),
        _tool_resp("c3", prompt=100),
        _tool_resp("c4", prompt=100),
        _tool_resp("c5", prompt=100),
        _resp("done"),
    ])
    kwargs: dict[str, Any] = {}
    rep_cls = getattr(power_loop, "ProjectedRepresentation", None)
    if rep_cls is not None:
        kwargs["representation"] = rep_cls()
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        # 工具行最多 5 条 < hot_tail=8：旧写法在第 5 条时只保住最后 3 条
        config=AgentLoopConfig(system_prompt="t", max_rounds=10, insend_distill_tokens=50,
                               insend_distill_batch=10, insend_distill_hot_tail=8, **kwargs),
        tool_registry=_echo_registry(big),
    )
    sid = await loop.new_session()
    assert (await loop.send("hi", session_id=sid)).status == "completed"
    for i, req in enumerate(llm.seen):
        rows = _tool_rows(req)
        assert all(r == big for r in rows), f"第 {i} 次请求里有被蒸馏的行：{rows}"
    await loop.aclose()

def _write_call(cid: str, path: str, body: str) -> LLMResponse:
    import json as _json
    args = _json.dumps({"path": path, "content": body, "mode": "overwrite"}, ensure_ascii=False)
    return LLMResponse(raw_text="", tool_calls=[{
        "id": cid, "type": "function", "function": {"name": "echo", "arguments": args},
    }])


def _args_of(msgs: list[dict[str, Any]]) -> list[str]:
    out = []
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            out.append(((tc or {}).get("function") or {}).get("arguments") or "")
    return out


@pytest.mark.asyncio
async def test_insend_distill_also_slims_tool_call_arguments(tmp_path):
    """保险丝也回收**调用参数**，不只是结果（6.25.0）。

    工具在上下文里占两块：结果在 tool 行、参数在 assistant 行的 tool_calls 里。参数此前
    从没被碰过，而实测占全库上下文 29%——write_file 一个人 12.6MB、单次最大 5.4 万字符：
    文件已经落盘，正文却还在 prompt 里一轮轮重发。跨 send 的投影早就两块一起收，缺的是
    send 内这一层。规则通用：短字段（path/mode）原样留下，长字符串换成带 recall_send 坐标
    的说明，且**仍是合法 JSON**。
    """
    body = "B" * 3000
    llm = _Scripted(responses=[
        _write_call("w1", "a.txt", body),   # prepare_round(1)：10 < 50 不动
        _write_call("w2", "b.txt", body),   # prepare_round(2)：触发
        _write_call("w3", "c.txt", body),
        _resp("done"),
    ])
    llm.responses[0].token_usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11)
    for r in llm.responses[1:3]:
        r.token_usage = LLMTokenUsage(prompt_tokens=100, completion_tokens=1, total_tokens=101)
    kwargs: dict[str, Any] = {}
    rep_cls = getattr(power_loop, "ProjectedRepresentation", None)
    if rep_cls is not None:
        kwargs["representation"] = rep_cls()
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, insend_distill_tokens=50,
                               insend_distill_batch=1, insend_distill_hot_tail=0, **kwargs),
        tool_registry=_echo_registry("ok"),
    )
    sid = await loop.new_session()
    assert (await loop.send("hi", session_id=sid)).status == "completed"

    last = _args_of(llm.seen[-1])
    assert last, "最后一次请求里应当有工具调用"
    slimmed = [a for a in last if "⟨已移出上下文" in a]
    assert slimmed, f"参数没有被回收：{[a[:80] for a in last]}"
    import json as _json
    for a in slimmed:
        d = _json.loads(a)                      # 仍是合法 JSON
        assert d["mode"] == "overwrite"         # 短字段原样保留
        assert d["path"] in ("a.txt", "b.txt", "c.txt")
        assert "recall_send" in d["content"]    # 长字段换成带坐标的指针
        assert body not in a
    assert any(body in a for a in last), "热的那次调用参数不该被动"
    await loop.aclose()



@pytest.mark.asyncio
async def test_insend_distill_measures_growth_within_the_send_not_the_whole_context(tmp_path):
    """6.27.0：触发量 = 本 send 内的增长，不是整个上下文。

    真实事故（conv-237）：系统提示词 + 技能 + 工具目录一开局 3 万 token，trigger 配 30000，按整体判
    第 1 轮就超、之后 44 次连烧；模型自己刚发的 send_message 参数被瘦成占位符，模型照着占位符
    又发了九条进聊天。这里整个上下文一直远超阈值（4 万），但 send 内只涨了 20，保险丝不该动。
    """
    big = "R" * 400
    llm = _Scripted(responses=[
        _tool_resp("c1", prompt=40000),   # 第一轮真实 prompt = 基线
        _tool_resp("c2", prompt=40010),   # 增长 10 < 50
        _tool_resp("c3", prompt=40020),   # 增长 20 < 50
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
    assert (await loop.send("hi", session_id=sid)).status == "completed"
    for req in llm.seen[1:]:
        assert all(not r.startswith("[distilled #") for r in _tool_rows(req)), "整体大但 send 内没涨，不该蒸馏"
    # 同样的配置，send 内真涨了就该动（基线 100 → 100+60）
    llm2 = _Scripted(responses=[_tool_resp("c1", prompt=100), _tool_resp("c2", prompt=160), _resp("done")])
    loop2 = StatefulAgentLoop(
        llm=llm2, db_path=str(tmp_path / "s2.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, insend_distill_tokens=50,
                               insend_distill_batch=1, insend_distill_hot_tail=0, **kwargs),
        tool_registry=_echo_registry(big),
    )
    sid2 = await loop2.new_session()
    assert (await loop2.send("hi", session_id=sid2)).status == "completed"
    assert _tool_rows(llm2.seen[2])[0].startswith("[distilled #")
    await loop.aclose()
    await loop2.aclose()


@pytest.mark.asyncio
async def test_insend_distill_keeps_arguments_of_registered_speech_tools(tmp_path):
    """6.27.0：insend_distill_keep_tools 里的工具，调用参数永不瘦身（结果照旧）。

    send_message 的参数就是模型的发言；瘦成「⟨已移出上下文 N 字符…⟩」后，模型下一轮照着这个样子
    再发一遍——占位符进了聊天。write_file 的正文该瘦照瘦。
    """
    speech = "S" * 3000
    body = "B" * 3000

    import json as _json

    def _say_call(call_id: str) -> LLMResponse:
        r = LLMResponse(raw_text="", tool_calls=[{"id": call_id, "type": "function",
                                                  "function": {"name": "say", "arguments": _json.dumps({"text": speech})}}])
        r.token_usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11)
        return r

    llm = _Scripted(responses=[
        _say_call("s1"),                    # 基线
        _write_call("w1", "a.txt", body),   # 增长 90 → 触发
        _write_call("w2", "b.txt", body),
        _resp("done"),
    ])
    for r in llm.responses[1:3]:
        r.token_usage = LLMTokenUsage(prompt_tokens=100, completion_tokens=1, total_tokens=101)
    kwargs: dict[str, Any] = {}
    rep_cls = getattr(power_loop, "ProjectedRepresentation", None)
    if rep_cls is not None:
        kwargs["representation"] = rep_cls()
    reg = _echo_registry("ok")
    async def _say(**kw: Any) -> str:
        return "said"
    reg.register(ToolDefinition(name="say", description="speak", input_schema={"type": "object", "properties": {"text": {"type": "string"}}}), _say)
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, insend_distill_tokens=50,
                               insend_distill_batch=5, insend_distill_hot_tail=0,
                               insend_distill_keep_tools=("say",), **kwargs),
        tool_registry=reg,
    )
    sid = await loop.new_session()
    assert (await loop.send("hi", session_id=sid)).status == "completed"
    last = _args_of(llm.seen[-1])
    assert any(speech in a for a in last), "发言参数被瘦身了——模型会照着占位符再发一遍"
    assert any("⟨已移出上下文" in a for a in last), "write_file 的正文仍该瘦身"
    await loop.aclose()


def test_insend_distill_never_touches_rows_before_this_send():
    """6.27.0：保险丝只动本 send 追加的行；run 之前就在历史里的（上一 send 的逐字行）归跨 send 投影管。"""
    import json as _json
    from power_loop.core.pipeline import AgentPipeline

    json_dumps = _json.dumps
    p = AgentPipeline.__new__(AgentPipeline)
    rep_cls = getattr(power_loop, "ProjectedRepresentation", None)
    if rep_cls is None:
        pytest.skip("保险丝只在投影表示法下工作")
    p.config = AgentLoopConfig(system_prompt="t", insend_distill_tokens=1, insend_distill_batch=10,
                               insend_distill_hot_tail=0, representation=rep_cls())
    p.send_index = 3
    p.sink = None
    old_result = "O" * 500
    old_args = json_dumps({"path": "x", "content": "A" * 1000})
    new_result = "N" * 500
    p.history = [
        {"role": "user", "content": "earlier send"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "o1", "type": "function", "function": {"name": "write_file", "arguments": old_args}}]},
        {"role": "tool", "tool_call_id": "o1", "name": "write_file", "content": old_result},
        {"role": "user", "content": "this send"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "n1", "type": "function", "function": {"name": "write_file", "arguments": old_args}}]},
        {"role": "tool", "tool_call_id": "n1", "name": "write_file", "content": new_result},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "n2", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "n2", "name": "grep", "content": "x"},
    ]
    p._send_start_len = 4
    p._distill_oldest_tool_rows(10, 0)
    assert p.history[2]["content"] == old_result and p.history[1]["tool_calls"][0]["function"]["arguments"] == old_args
    assert p.history[5]["content"].startswith("[distilled #3") and "⟨已移出上下文" in p.history[4]["tool_calls"][0]["function"]["arguments"]
