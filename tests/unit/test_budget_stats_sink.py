"""0.10.0 batch: per-run budget, session stats, threaded sync tools,
phase payload收口, logging sink."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    CompleteDecideCtx,
    HookDirective,
    HookPoint,
    StatefulAgentLoop,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk, LLMTokenUsage
from power_loop.contracts.tools import ToolDefinition
from power_loop.contrib.logging_sink import attach_logging_sink
from power_loop.tools.registry import ToolRegistry


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


def _resp(text: str, *, prompt: int = 10, completion: int = 2) -> LLMResponse:
    r = LLMResponse(raw_text=text, content_text=text)
    r.token_usage = LLMTokenUsage(prompt_tokens=prompt, completion_tokens=completion,
                                  total_tokens=prompt + completion)
    return r


def _tool_resp(call_id: str, name: str, args: str = "{}",
               *, prompt: int = 10, completion: int = 2) -> LLMResponse:
    r = LLMResponse(
        raw_text="",
        tool_calls=[{"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": args}}],
    )
    r.token_usage = LLMTokenUsage(prompt_tokens=prompt, completion_tokens=completion,
                                  total_tokens=prompt + completion)
    return r


def _echo_registry(handler=None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo", description="echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        ),
        handler or (lambda **kw: kw.get("text", "")),
    )
    return reg


# ── max_tokens_per_run ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_exceeded_stops_at_round_boundary(tmp_path):
    # Round 1 spends 1000+100 (over the 500 budget) but completes its tool
    # call; the loop then stops before paying for round 2.
    llm = _Scripted(responses=[
        _tool_resp("c1", "echo", prompt=1000, completion=100),
        _resp("should never be reached"),
    ])
    events: list = []
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.STATUS_CHANGED, events.append)
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=6, max_tokens_per_run=500),
        tool_registry=_echo_registry(), event_bus=bus,
    )
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "budget_exceeded"
    assert res.usage["prompt_tokens"] == 1000  # round 2 never ran
    # No dangling tool_calls: a follow-up send must work without healing.
    res2 = await loop.send("again", session_id=sid)
    assert res2.status in ("completed", "budget_exceeded")
    kinds = [(e.payload or {}).get("kind") for e in events]
    assert "budget_exceeded" in kinds
    await loop.aclose()


@pytest.mark.asyncio
async def test_budget_disabled_by_default(tmp_path):
    llm = _Scripted(responses=[_resp("ok", prompt=10**6, completion=10)])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t"),
    )
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "completed"
    await loop.aclose()


@pytest.mark.asyncio
async def test_zero_budget_means_unlimited(tmp_path):
    llm = _Scripted(responses=[
        _tool_resp("c1", "echo", prompt=10**6, completion=10),
        _resp("finished", prompt=10**6, completion=10),
    ])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4, max_tokens_per_run=0),
        tool_registry=_echo_registry(),
    )
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "completed"
    assert res.final_text == "finished"
    assert llm._idx == 2
    await loop.aclose()


@pytest.mark.asyncio
async def test_budget_exceeded_complete_decide_runs_bounded_finalize(tmp_path):
    # Round 0 crosses the normal budget. COMPLETE_DECIDE injects a durable
    # finalizer and grants exactly one extra round; that round may run despite
    # the exhausted budget, then the standard forced summary closes the send.
    llm = _Scripted(responses=[
        _tool_resp("c1", "echo", prompt=1000, completion=100),
        _tool_resp("c2", "echo", prompt=1000, completion=100),
        _resp("forced wrap-up", prompt=10, completion=2),
    ])
    reasons: list[str] = []

    def finalize(ctx: CompleteDecideCtx) -> None:
        reasons.append(ctx.reason)
        if ctx.reason != "budget_exceeded":
            return
        ctx.inject = "[system] budget exhausted — finalize now"
        ctx.extra_rounds = 1
        ctx.directive = HookDirective.SHORT_CIRCUIT

    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=10, max_tokens_per_run=500),
        tool_registry=_echo_registry(),
    )
    loop.hooks.register(HookPoint.COMPLETE_DECIDE, finalize)
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)

    assert reasons == ["budget_exceeded", "hit_round_limit"]
    assert res.status == "hit_round_limit"
    assert "forced wrap-up" in res.final_text
    # 1 normal round + exactly 1 injected round + the stock forced wrap-up.
    # If the old main-loop allowance leaked into finalize, this would complete
    # from _Scripted's fallback instead of reaching the forced wrap-up here.
    assert llm._idx == 3
    messages = await loop.get_messages(sid)
    assert any(m.get("content") == "[system] budget exhausted — finalize now" for m in messages)
    await loop.aclose()


# ── session stats ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_stats_accumulate_across_sends(tmp_path):
    llm = _Scripted(responses=[
        _resp("one", prompt=10, completion=1),
        _resp("two", prompt=20, completion=2),
    ])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t"),
    )
    sid = await loop.new_session()
    assert await loop.get_session_stats(sid) is None  # before first send
    await loop.send("a", session_id=sid)
    await loop.send("b", session_id=sid)
    stats = await loop.get_session_stats(sid)
    assert stats is not None
    assert (stats.sends, stats.llm_calls) == (2, 2)
    assert stats.prompt_tokens == 30
    assert stats.completion_tokens == 3
    assert stats.rounds == 2 and stats.tool_calls == 0
    assert stats.first_send_at is not None
    assert stats.first_send_at <= stats.last_send_at
    assert (await loop.list_session_stats())[0].session_id == sid
    # cascade delete with the session
    await loop.close_session(sid)
    assert await loop.store.get_session_stats(sid) is None
    await loop.aclose()


# ── sync tools run off the event loop ───────────────────────────────────


@pytest.mark.asyncio
async def test_sync_tool_does_not_block_event_loop(tmp_path):
    main_thread = threading.current_thread().name
    seen: dict[str, Any] = {}

    def slow_echo(**kw):
        seen["thread"] = threading.current_thread().name
        time.sleep(0.3)
        return "slow done"

    llm = _Scripted(responses=[
        _tool_resp("c1", "echo"),
        _resp("final"),
    ])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
        tool_registry=_echo_registry(slow_echo),
    )
    sid = await loop.new_session()

    ticks = 0

    async def _heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    hb = asyncio.create_task(_heartbeat())
    res = await loop.send("hi", session_id=sid)
    hb.cancel()
    assert res.status == "completed"
    assert seen["thread"] != main_thread  # ran in a worker thread
    assert ticks >= 4  # event loop kept ticking during the 0.3s sleep
    assert res.tool_calls == 1
    assert (await loop.get_session_stats(sid)).tool_calls == 1
    await loop.aclose()


# ── phase events always carry typed data ────────────────────────────────


@pytest.mark.asyncio
async def test_all_emitted_events_have_data(tmp_path):
    captured: list = []
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(None, captured.append)
    llm = _Scripted(responses=[_tool_resp("c1", "echo"), _resp("final")])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
        tool_registry=_echo_registry(), event_bus=bus,
    )
    sid = await loop.new_session()
    await loop.send("hi", session_id=sid)
    assert captured, "expected events"
    missing = [e.type for e in captured if e.data is None]
    assert not missing, f"events without typed data: {missing}"
    await loop.aclose()


# ── logging sink ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_logging_sink_emits_json_lines(tmp_path, caplog):
    bus = AgentEventBus(suppress_subscriber_errors=True)
    attach_logging_sink(bus, events={AgentEventType.USAGE_UPDATED})
    llm = _Scripted(responses=[_resp("ok", prompt=7, completion=3)])
    loop = StatefulAgentLoop(
        llm=llm, db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t"), event_bus=bus,
    )
    sid = await loop.new_session()
    with caplog.at_level(logging.INFO, logger="power_loop.events"):
        await loop.send("hi", session_id=sid)
    lines = [json.loads(r.message) for r in caplog.records
             if r.name == "power_loop.events"]
    assert lines and all(line["event"] == "usage_updated" for line in lines)
    assert lines[0]["payload"]["usage"]["prompt_tokens"] == 7
    assert lines[0]["session_id"] == sid
    await loop.aclose()


def test_logging_sink_truncates_long_fields():
    from power_loop.contrib.logging_sink import _sanitize

    out = _sanitize({"text": "x" * 1000, "n": 5}, 100, ())
    assert len(out["text"]) < 120 and out["n"] == 5


def test_logging_sink_redacts_secret_keys():
    """H4.5: values of secret-named keys are replaced with *** at any depth; the
    match is a case-insensitive substring on the key name."""
    from power_loop.contrib.logging_sink import _DEFAULT_REDACT_KEYS, _sanitize

    redact = tuple(k.lower() for k in _DEFAULT_REDACT_KEYS)
    payload = {
        "API_KEY": "sk-secret",
        "Authorization": "Bearer xyz",
        "tool_input": {"password": "hunter2", "path": "/ok"},
        "headers": [{"x-access_key": "leak"}],
        "model": "gpt-x",
    }
    out = _sanitize(payload, 500, redact)
    assert out["API_KEY"] == "***"
    assert out["Authorization"] == "***"
    assert out["tool_input"]["password"] == "***"
    assert out["tool_input"]["path"] == "/ok"      # non-secret preserved
    assert out["headers"][0]["x-access_key"] == "***"
    assert out["model"] == "gpt-x"
    # disabling redaction (empty tuple) leaves values intact
    assert _sanitize({"api_key": "k"}, 500, ())["api_key"] == "k"
