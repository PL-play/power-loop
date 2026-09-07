"""截断提示的位置：必须补在**所有 tool 结果之后**，不能插在 tool_calls 与它的结果之间。

真实事故 conv-215：模型输出被 max_tokens 硬切，`send_message` 的 arguments JSON 断在半路
→ 降成 `{}` → 必填校验报「缺参数」。截断提示本身是对的（告诉模型该拆小而不是补参数），
但它被追加在 `assistant(tool_calls)` 之后、`tool(结果)` 之前，造出这样一段历史：

    assistant(tool_calls=[X]) → user(截断提示) → tool(tool_call_id=X)

下一次请求供应商直接 400：
"An assistant message with 'tool_calls' must be followed by tool messages responding to each
'tool_call_id'" → 重试耗尽 → 整个 run 降级。会话第一次发卡片就死在这。

同一条不变量在 TOOL_AFTER BREAK 那里本来就守着（跳过的工具也要补 tool 结果，
否则下一轮序列非法）——这里当初漏了。本测试钉住序列本身，而不是「提示在不在」。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.tools.registry import ToolRegistry
from power_loop import ToolDefinition


@dataclass
class _ScriptedLLM(LLMService):
    script: list[LLMResponse] = field(default_factory=list)
    calls: int = 0
    seen_requests: list[list[dict[str, Any]]] = field(default_factory=list)

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        self.seen_requests.append([dict(m) for m in (request.messages or [])])
        i = self.calls
        self.calls += 1
        if i < len(self.script):
            return self.script[i]
        return LLMResponse(raw_text="done", tool_calls=[])

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _cut_call(cid: str = "call_1") -> LLMResponse:
    """arguments 断在半路 —— 这正是 max_tokens 硬切在工具调用上的样子。"""
    return LLMResponse(
        raw_text="卡片已经发出，等你们选择。",
        tool_calls=[{"id": cid, "type": "function",
                     "function": {"name": "echo", "arguments": '{"text": "很长的卡片内容'}}],
    )


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def echo(**kwargs: Any) -> str:
        if "text" not in kwargs:
            return "Error: missing required parameter(s): text"
        return str(kwargs["text"])

    reg.register(
        ToolDefinition(
            name="echo", description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                          "required": ["text"]},
            required_params=("text",),
        ),
        echo,
    )
    return reg


def _assert_tool_calls_answered(messages: list[dict[str, Any]]) -> None:
    """每个 assistant.tool_calls 的 id，都必须由**紧随其后**的 tool 消息逐一应答。"""
    for i, m in enumerate(messages):
        ids = [str(c.get("id")) for c in (m.get("tool_calls") or [])]
        if not ids:
            continue
        answered: list[str] = []
        for nxt in messages[i + 1:]:
            if nxt.get("role") != "tool":
                break                       # 遇到非 tool 就停：中间插了别的角色 = 非法
            answered.append(str(nxt.get("tool_call_id")))
        missing = [x for x in ids if x not in answered]
        assert not missing, (
            f"assistant(tool_calls={ids}) 后面没有紧跟对应的 tool 结果，缺 {missing}；"
            f"实际紧随的是 {[n.get('role') for n in messages[i + 1:i + 3]]} —— "
            "这段历史会让供应商回 400"
        )


async def test_truncated_args_notice_does_not_break_the_tool_call_sequence(tmp_path) -> None:
    llm = _ScriptedLLM(script=[_cut_call()])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "a.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
        tool_registry=_registry(),
    )
    sid = await loop.new_session()
    await loop.send("画个卡片", sid)

    # 关键断言：发给供应商的每一次请求，序列都必须合法
    assert llm.seen_requests, "没有发出任何请求"
    for req in llm.seen_requests:
        _assert_tool_calls_answered(req)


async def test_truncation_notice_is_still_delivered(tmp_path) -> None:
    """位置挪了，提示本身不能丢——没有它，模型会以为自己忘填参数，原样重写再被截断。"""
    llm = _ScriptedLLM(script=[_cut_call()])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "b.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
        tool_registry=_registry(),
    )
    sid = await loop.new_session()
    await loop.send("画个卡片", sid)

    later = llm.seen_requests[-1]
    notice = [m for m in later
              if m.get("role") == "user" and "被从中间截断" in str(m.get("content") or "")]
    assert notice, "截断提示丢了——模型会当成「忘填参数」，原样重写、再被截断一次"
    # 且它必须在那条 tool 结果**之后**
    idx_notice = later.index(notice[0])
    idx_tool = max(i for i, m in enumerate(later) if m.get("role") == "tool")
    assert idx_notice > idx_tool, "提示又插到 tool 结果前面去了"


async def test_normal_tool_call_has_no_notice(tmp_path) -> None:
    """参数完整的正常调用不该收到截断提示——那会平白教模型「拆小」。"""
    llm = _ScriptedLLM(script=[LLMResponse(
        raw_text="", tool_calls=[{"id": "c1", "type": "function",
                                  "function": {"name": "echo",
                                               "arguments": json.dumps({"text": "hi"})}}])])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "c.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
        tool_registry=_registry(),
    )
    sid = await loop.new_session()
    await loop.send("说 hi", sid)
    for req in llm.seen_requests:
        _assert_tool_calls_answered(req)
        assert not [m for m in req if "被从中间截断" in str(m.get("content") or "")]
