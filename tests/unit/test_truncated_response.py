"""被 max_tokens 硬切的一轮，不能当成「供应商打嗝」原样重试。

真实事故（DeepTalk conv-213，glm-5.3-flash）：模型想在一轮里写一个 25KB 的 CSS 文件，
输出打到 max_tokens=20000 被切在工具调用的 JSON 中间 → 解析不出 tool_calls、正文也是空的
（内容全在那段 JSON 里）→ 被判成空响应打嗝 → **原样重试** → 同一个 prompt、同一个模型、
写出同样长的东西、同样被切断。两轮各约 8 分钟、产出为零，用户看到的是 16 分钟沉默。

区分两者的信号一直都在：provider 的 finish_reason（截断是 length / max_tokens）。
截断的正确处置不是重试，而是**改变输入**——告诉模型「你被截断了，拆小再来」。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.core.pipeline import (
    _TRUNCATION_FINISH_REASONS,
    _finish_reason,
    _sanitize_tool_calls,
)
from power_loop.tools.registry import ToolRegistry


class _Choice:
    def __init__(self, reason: str) -> None:
        self.finish_reason = reason


class _Completion:
    def __init__(self, reason: str) -> None:
        self.choices = [_Choice(reason)]


@dataclass
class _ScriptedLLM(LLMService):
    script: list[LLMResponse] = field(default_factory=list)
    calls: int = 0
    seen_prompts: list[str] = field(default_factory=list)

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        self.seen_prompts.append(
            " ".join(str(m.get("content") or "") for m in (request.messages or [])))
        i = self.calls
        self.calls += 1
        if i < len(self.script):
            return self.script[i]
        return LLMResponse(raw_text="写小一点之后写完了", tool_calls=[])

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _truncated() -> LLMResponse:
    r = LLMResponse(raw_text="", tool_calls=[])
    r.raw_completion = _Completion("length")
    return r


# ── finish_reason 提取（各家形状不同）──────────────────────────────────────────

def test_reads_openai_shape() -> None:
    assert _finish_reason(_truncated()) == "length"
    assert _finish_reason(_truncated()) in _TRUNCATION_FINISH_REASONS


def test_reads_anthropic_shape() -> None:
    r = LLMResponse(raw_text="", tool_calls=[])
    r.raw_message = type("M", (), {"stop_reason": "max_tokens"})()
    assert _finish_reason(r) == "max_tokens"
    assert _finish_reason(r) in _TRUNCATION_FINISH_REASONS


def test_unknown_shape_asserts_nothing() -> None:
    """取不到就返回空串——不猜。猜错的代价是把正常轮当成截断，比漏判更糟。"""
    assert _finish_reason(LLMResponse(raw_text="", tool_calls=[])) == ""


# ── 截断不重试，而是改变输入 ────────────────────────────────────────────────

async def test_truncated_round_nudges_instead_of_blind_retry(tmp_path) -> None:
    llm = _ScriptedLLM(script=[_truncated()])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "t.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, max_tokens=20000),
        tool_registry=ToolRegistry(),
    )
    sid = await loop.new_session()
    result = await loop.send("写个大文件", sid)
    assert result.final_text == "写小一点之后写完了"
    # 关键：下一轮的输入里必须多了那句实话——否则就是原样重试，必然再被截断。
    assert any("被从中间截断" in p for p in llm.seen_prompts[1:]), (
        "截断之后没有告诉模型，等于原样重试")


async def test_completion_tokens_hitting_the_cap_counts_as_truncation(tmp_path) -> None:
    """有的供应商不给 finish_reason——输出打满 max_tokens 本身就是够硬的信号。"""
    from power_loop._vendor.llm_client.interface import LLMTokenUsage

    r = LLMResponse(raw_text="", tool_calls=[])
    r.token_usage = LLMTokenUsage(prompt_tokens=1000, completion_tokens=20000)
    llm = _ScriptedLLM(script=[r])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "c.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=8, max_tokens=20000),
        tool_registry=ToolRegistry(),
    )
    sid = await loop.new_session()
    await loop.send("写个大文件", sid)
    assert any("被从中间截断" in p for p in llm.seen_prompts[1:])


# ── 截断的第二种表现：工具调用在，参数断在半路 ──────────────────────────────

def test_unparseable_args_are_reported_not_repaired() -> None:
    """🔴 **不做 json 修复**：补全出来的 content 就是那半个文件，write_file 会当成功写下去、
    agent 继续往前走，交付一份残缺的稿子——静默损坏比报错严重得多。
    降成 {} 让必填校验报，同时把标志带出去，管线据此说「你被截断了」而不是「你忘了填参数」。
    （conv-213 实测：一条 missing required parameter 背后是 completion_tokens=20000。）"""
    calls = [{"function": {"name": "write_file",
                           "arguments": '{"path":"a.css","content":"body{co'}}]
    out, cut = _sanitize_tool_calls(calls)
    assert cut is True
    assert out[0]["function"]["arguments"] == "{}"


def test_no_non_standard_key_leaks_into_the_wire_format() -> None:
    """标志走返回值：这些 dict 会原样进 assistant 消息、下一轮发回给供应商，
    多一个非标准字段可能直接把请求打挂。"""
    out, _ = _sanitize_tool_calls(
        [{"function": {"name": "w", "arguments": '{"path":"a","content":"x'}}])
    assert all(not k.startswith("_") for k in out[0])


def test_healthy_args_are_untouched() -> None:
    out, cut = _sanitize_tool_calls([{"function": {"name": "x", "arguments": '{"a":1}'}}])
    assert cut is False and out[0]["function"]["arguments"] == '{"a":1}'
