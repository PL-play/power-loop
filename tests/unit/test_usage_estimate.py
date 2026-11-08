"""Every LLM attempt is accounted for, including the ones that never reported usage (design/124
§10, U1/U2).

- a failed / timed-out / aborted attempt carries an ESTIMATE (prompt + what it streamed), flagged;
- a request rejected with an HTTP status before any output was never processed → prompt 0;
- a completed call without provider usage is estimated — never borrowed from the previous call;
- the run's usage totals include failed attempts, and say how much is estimated.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, SessionStore
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
    OpenAICompatibleChatConfig,
)
from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.usage_estimate import (
    IMAGE_BLOCK_TOKENS,
    estimate_completion_tokens,
    estimate_prompt_tokens,
    estimate_text_tokens,
)

# ── the estimator ─────────────────────────────────────────────────────────────────────────


def test_cjk_costs_more_per_char_than_latin() -> None:
    zh = estimate_text_tokens("你好世界" * 100)          # 400 CJK chars
    en = estimate_text_tokens("abcd" * 100)              # 400 latin chars
    assert zh == 280 and en == 111
    assert estimate_text_tokens("") == 0 and estimate_text_tokens(None) == 0


def test_prompt_estimate_counts_system_tools_images_and_tool_calls() -> None:
    base = estimate_prompt_tokens(messages=[{"role": "user", "content": "hi"}])
    with_sys = estimate_prompt_tokens(messages=[{"role": "user", "content": "hi"}],
                                      system_prompt="x" * 360)
    assert with_sys - base >= 100
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * 360}}]
    assert estimate_prompt_tokens(messages=[], tools=tools) >= 100
    img = [{"role": "user", "content": [{"type": "text", "text": "看"},
                                        {"type": "image_url", "image_url": {"url": "data:,"}}]}]
    assert estimate_prompt_tokens(messages=img) >= IMAGE_BLOCK_TOKENS
    tc = [{"role": "assistant", "content": "",
           "tool_calls": [{"function": {"name": "echo", "arguments": "a" * 360}}]}]
    assert estimate_prompt_tokens(messages=tc) >= 100


def test_completion_estimate_includes_thinking_and_tool_args() -> None:
    assert estimate_completion_tokens(text="a" * 36, think="b" * 36) == 20
    assert estimate_completion_tokens(
        tool_calls=[{"function": {"name": "x", "arguments": "c" * 36}}]) >= 10


# ── the pipeline ──────────────────────────────────────────────────────────────────────────


class _HTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status


class _Script(LLMService):
    """Per call: ('ok', usage|None) | ('stream_then_fail', text) | ('reject', status) | 'hang'."""

    def __init__(self, plan: list[Any]) -> None:
        self.plan = list(plan)

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, **kw: Any) -> LLMResponse:
        step = self.plan.pop(0)
        kind = step[0] if isinstance(step, tuple) else step
        if kind == "ok":
            r = LLMResponse(raw_text="answer")
            r.token_usage = step[1]
            return r
        if kind == "stream_then_fail":
            if on_chunk_delta_text:
                on_chunk_delta_text(step[1])
            raise ConnectionError("stream dropped")
        if kind == "reject":
            raise _HTTPError(step[1])
        if kind == "hang":
            if on_chunk_delta_text:
                on_chunk_delta_text("partial answer " * 20)
            await asyncio.sleep(3600)
        raise AssertionError(step)

    async def close(self) -> None:
        return None


async def _run(runtime_agent, plan: list[Any], *, attempts: int = 1) -> tuple[Any, list[dict]]:
    store = await SessionStore.open(":memory:")
    bus = AgentEventBus()
    events: list[dict] = []
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: events.append(dict(e.payload or {})))
    loop = await runtime_agent(model=_Script(plan), store=store, event_bus=bus, config=AgentLoopConfig(
        system_prompt="S" * 100, max_rounds=1, compactor=None,
        retry_policy=LLMRetryPolicy(max_attempts=attempts, backoff_initial=0, total_timeout=None)))
    try:
        res = await loop.send("用户的问题" * 20)
    finally:
        await loop.aclose()
        await store.close()
    return res, events


@pytest.mark.asyncio
async def test_failed_attempt_is_estimated_and_counted_in_run_totals(runtime_agent) -> None:
    res, events = await _run(runtime_agent, [("stream_then_fail", "写到一半" * 30)], attempts=2)
    assert res.status == "failed"
    assert len(events) == 1  # unknown external action is not automatically replayed
    failed = events[0]
    assert failed["success"] is False and failed["estimated"] is True
    assert failed["outcome"] == "error"
    assert failed["prompt_tokens"] > 50
    assert failed["completion_tokens"] == estimate_text_tokens("写到一半" * 30)
    totals = res.usage
    assert totals["calls"] == 1 and totals["failed_calls"] == 1
    assert totals["prompt_tokens"] == failed["prompt_tokens"]
    assert totals["estimated_calls"] == 1 and totals["estimated_tokens"] == failed["total_tokens"]


@pytest.mark.asyncio
async def test_rejected_request_costs_no_prompt(runtime_agent) -> None:
    res, events = await _run(runtime_agent, [("reject", 402)])
    assert res.status == "failed"
    assert events[0]["prompt_tokens"] == 0 and events[0]["completion_tokens"] == 0
    assert events[0]["estimated"] is True


@pytest.mark.asyncio
async def test_completed_call_without_usage_is_estimated_not_borrowed(runtime_agent) -> None:
    res, events = await _run(runtime_agent, [("ok", None)])
    assert res.status == "completed"
    assert events[0]["estimated"] is True and events[0]["success"] is True
    assert events[0]["prompt_tokens"] > 0
    assert res.usage["estimated_calls"] == 1
    assert res.usage["prompt_tokens"] == events[0]["prompt_tokens"]


@pytest.mark.asyncio
async def test_aborted_attempt_outcome_and_partial_output(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    bus = AgentEventBus()
    events: list[dict] = []
    streaming = asyncio.Event()
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: events.append(dict(e.payload or {})))
    bus.subscribe(AgentEventType.STREAM_DELTA, lambda e: streaming.set())
    loop = await runtime_agent(model=_Script(["hang"]), store=store, event_bus=bus,
                             config=AgentLoopConfig(system_prompt="S", max_rounds=1,
                                                    compactor=None, retry_policy=None))
    from power_loop.agent.runtime_store import now_ms
    receipt = await loop.runtime.client.send(loop.target, "hi", message_id="hang", actor=loop.actor)
    await asyncio.wait_for(streaming.wait(), timeout=5)
    await loop.runtime.client.cancel(receipt.input_ref, actor=loop.actor)
    result = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+5000)
    assert result.status == "cancelled"
    await loop.drain()
    await loop.aclose()
    await store.close()
    assert events and events[0]["outcome"] == "aborted"
    assert events[0]["estimated"] is True
    assert events[0]["completion_tokens"] == estimate_text_tokens("partial answer " * 20)
    assert events[0]["prompt_tokens"] > 0


# ── the transport (U1) ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transport_never_reports_a_previous_calls_usage(runtime_agent) -> None:
    svc = OpenAICompatibleChatLLMService(OpenAICompatibleChatConfig(
        base_url="http://unused", api_key="k", model="m"))
    svc._last_usage = {"prompt_tokens": 99999, "completion_tokens": 1, "total_tokens": 100000}

    async def fake_stream(request: LLMRequest):
        yield LLMStreamChunk(delta_text="hello")

    svc.stream = fake_stream  # type: ignore[method-assign]
    res = await svc.complete(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert res.raw_text == "hello"
    assert res.token_usage is None


@pytest.mark.asyncio
async def test_call_events_name_the_services_model_when_loop_sets_none(runtime_agent) -> None:
    from power_loop._vendor.llm_client.capabilities import ModelCapabilities

    class _Named(_Script):
        capabilities = ModelCapabilities(model="deepseek-flash")

    store = await SessionStore.open(":memory:")
    bus = AgentEventBus()
    events: list[dict] = []
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: events.append(dict(e.payload or {})))
    real = LLMTokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6)
    loop = await runtime_agent(model=_Named([("ok", real)]), store=store, event_bus=bus,
                             config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    try:
        await loop.send("hi")
    finally:
        await loop.aclose()
        await store.close()
    assert events[0]["model"] == "deepseek-flash"
