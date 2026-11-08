"""design/188：pass_turn 内置进 power-loop（永远在）+ 结果里带每一步的文字。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from power_loop import AgentLoopConfig, SessionStore
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.tools import ToolDefinition
from power_loop.core import pass_turn as P
from power_loop.runtime.pricing import Free
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent


@dataclass
class _Script(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    calls: int = 0
    seen_tools: list[list[str]] = field(default_factory=list)

    async def complete(self, request: LLMRequest, **_kw) -> LLMResponse:
        self.calls += 1
        self.seen_tools.append(sorted(t["function"]["name"] for t in (request.tools or [])))
        return self.responses.pop(0) if self.responses else LLMResponse(raw_text="（多出来的一轮）")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _empty()

    async def close(self) -> None:
        return None


def _call(cid: str, name: str, args: dict, text: str = "") -> LLMResponse:
    return LLMResponse(raw_text=text, tool_calls=[{"id": cid, "type": "function",
                                                     "function": {"name": name, "arguments": json.dumps(args)}}])


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="lookup", description="查", input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), lambda **_k: "牛奶 12 元")
    return reg


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


async def _send(store, llm, prompt="算一下"):
    runner = await RunningAgent(store, llm, _registry(), AgentLoopConfig(max_rounds=10)).start()
    try:
        result = await runner.send(prompt)
        return result, await store.load_all_messages(runner.target.session_id)
    finally:
        await runner.aclose()


def steps(rows):
    return [{"step": row.round_index + 1, "text": row.content,
             "tools": [c["function"]["name"] for c in row.tool_calls or []]}
            for row in rows if row.role == "assistant"]


def pass_reasons(rows):
    return [json.loads(call['function']['arguments']).get('reason')
            for row in rows for call in row.tool_calls or []
            if call['function']['name'] == 'pass_turn']


async def test_pass_turn_is_always_offered_and_stops_before_the_next_model_call(store):
    llm = _Script(responses=[
        _call("c1", "lookup", {}, text="我先查一下价格。"),
        _call("c2", "pass_turn", {"reason": "算完了"}, text="牛奶 12 元，总共 12 元。"),
    ])
    r, rows = await _send(store, llm)
    assert llm.calls == 2                                     # 调了 pass_turn：不再调第三次模型
    assert all("pass_turn" in names for names in llm.seen_tools)
    assert r.status == "completed" and pass_reasons(rows) == ["算完了"]
    assert r.final_text == "牛奶 12 元，总共 12 元。"          # 明确结束：答案是最后写的那段
    assert steps(rows) == [
        {"step": 1, "text": "我先查一下价格。", "tools": ["lookup"]},
        {"step": 2, "text": "牛奶 12 元，总共 12 元。", "tools": ["pass_turn"]},
    ]


async def test_text_only_round_is_a_normal_end_and_texts_keep_every_step(store):
    llm = _Script(responses=[_call("c1", "lookup", {}, text="查一下。"), LLMResponse(raw_text="12 元。")])
    r, rows = await _send(store, llm)
    assert llm.calls == 2 and r.status == "completed" and not pass_reasons(rows)
    assert r.final_text == "12 元。"
    assert [t["text"] for t in steps(rows)] == ["查一下。", "12 元。"]


async def test_host_override_keeps_its_description_and_must_request_pass(store, runtime_agent):
    reg = _registry()
    seen = []

    def guarded(reason: str = "") -> str:
        seen.append(reason)
        if reason == "等后台":
            return "不能停：你等的任务已经结束了"              # 宿主拒绝：不调 request_pass，不停
        P.request_pass(reason)
        return "好"

    reg.register(ToolDefinition(name="pass_turn", description="宿主自己的说明", input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), FunctionRuntimeTool(guarded, ToolRuntimeContract("test.yield", capabilities=("task.yield.v1",))))
    llm = _Script(responses=[_call("c1", "pass_turn", {"reason": "等后台"}), _call("c2", "pass_turn", {"reason": "真做完了"})])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(max_rounds=10), store=store, registry=reg)
    r = await loop.send("x")
    assert seen == ["等后台", "真做完了"] and llm.calls == 2 and r.status == "completed"
    assert pass_reasons(await store.load_all_messages(loop.target.session_id)) == ["等后台", "真做完了"]
    assert reg.get("pass_turn").definition.description == "宿主自己的说明"   # 没被内置的覆盖


def test_request_pass_outside_a_send_is_a_no_op():
    assert P.request_pass("x") is False and not P.pass_requested()
    assert "没有在跑" in P.default_handler("x")


async def test_each_send_starts_unpassed(store, runtime_agent):
    llm = _Script(responses=[_call("c1", "pass_turn", {"reason": "一"}), LLMResponse(raw_text="第二次正常答")])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(max_rounds=10), store=store, registry=_registry())
    assert (await loop.send("a")).status == "completed"
    second = await loop.send("b")
    assert second.status == "completed" and second.final_text == "第二次正常答"
    assert llm.calls == 2
    assert pass_reasons(await store.load_all_messages(loop.target.session_id)) == ["一"]


async def test_pass_progress_counts_texts_and_work_tools_so_a_host_can_refuse_an_empty_pass(store, runtime_agent):
    reg = _registry()
    seen = []

    def guarded(reason: str = "") -> str:
        prog = P.pass_progress()
        seen.append(dict(prog))
        if not prog["texts"] and not prog["work_tools"] and len(seen) == 1:
            return "还没写出答案：先把答案写成正文，再结束"      # 宿主拒绝空着结束（只拒一次）
        P.request_pass(reason)
        return "好"

    reg.register(ToolDefinition(name="pass_turn", description="应用里的说明", input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), FunctionRuntimeTool(guarded, ToolRuntimeContract("test.yield", capabilities=("task.yield.v1",))))
    llm = _Script(responses=[_call("c1", "pass_turn", {"reason": "答完了"}),
                             _call("c2", "pass_turn", {"reason": "答完了"}, text="答案是 12 元。")])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(max_rounds=10), store=store, registry=reg)
    r = await loop.send("x")
    assert seen == [{"texts": 0, "work_tools": 0}, {"texts": 1, "work_tools": 0}]
    assert r.status == "completed" and r.final_text == "答案是 12 元。"


async def test_pass_progress_counts_other_tools_but_not_pass_turn(store, runtime_agent):
    got = []
    reg = _registry()

    def spy(reason: str = "") -> str:
        got.append(P.pass_progress())
        P.request_pass(reason)
        return "好"

    reg.register(ToolDefinition(name="pass_turn", description="d", input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), FunctionRuntimeTool(spy, ToolRuntimeContract("test.yield", capabilities=("task.yield.v1",))))
    llm = _Script(responses=[_call("c1", "lookup", {}), _call("c2", "pass_turn", {"reason": "存好了"})])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(max_rounds=10), store=store, registry=reg)
    await loop.send("x")
    assert got == [{"texts": 0, "work_tools": 1}]
    assert P.pass_progress() is None   # send 外没有状态
