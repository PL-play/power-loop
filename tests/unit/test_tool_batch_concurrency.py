"""6.11.0 同轮并发：同一轮 ≥2 个 async_capable 工具调用并发执行，结果按原顺序回填。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, HookDirective, HookPoint, StatefulAgentLoop
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
        self.seen.append([dict(m) for m in (getattr(request, "messages", None) or []) if isinstance(m, dict)])
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


def _resp(text: str) -> LLMResponse:
    r = LLMResponse(raw_text=text, content_text=text)
    r.token_usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return r


def _batch(*calls: tuple[str, str, dict]) -> LLMResponse:
    r = LLMResponse(raw_text="", tool_calls=[
        {"id": cid, "type": "function", "function": {"name": name, "arguments": __import__("json").dumps(args)}}
        for cid, name, args in calls
    ])
    r.token_usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return r


def _registry(log: list[str], *, delay: float = 0.25) -> ToolRegistry:
    reg = ToolRegistry()

    async def slow(**kw):
        tag = str(kw.get("tag"))
        log.append(f"start:{tag}")
        await asyncio.sleep(delay)
        log.append(f"end:{tag}")
        if tag == "boom":
            raise RuntimeError("boom failed")
        return f"img:{tag}"

    async def serial(**kw):
        tag = str(kw.get("tag"))
        log.append(f"start:{tag}")
        await asyncio.sleep(delay)
        log.append(f"end:{tag}")
        return f"file:{tag}"

    schema = {"type": "object", "properties": {"tag": {"type": "string"}}}
    reg.register(ToolDefinition(name="gen_image", description="slow", input_schema=schema, async_capable=True), slow)
    reg.register(ToolDefinition(name="save_note", description="serial", input_schema=schema), serial)
    return reg


def _tool_rows(msgs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(str(m.get("tool_call_id")), str(m.get("content"))) for m in msgs if m.get("role") == "tool"]


@pytest.mark.asyncio
async def test_async_capable_calls_in_one_round_run_concurrently_and_keep_order(tmp_path):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "b"}), ("c3", "gen_image", {"tag": "c"})),
        _resp("done"),
    ])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=4),
                             tool_registry=_registry(log))
    sid = await loop.new_session()
    t0 = time.monotonic()
    res = await loop.send("go", session_id=sid)
    elapsed = time.monotonic() - t0
    assert res.status == "completed"
    assert elapsed < 0.6, f"3×0.25s ran serially? {elapsed:.2f}s"
    assert log[:3] == ["start:a", "start:b", "start:c"]  # 三个同时起
    assert _tool_rows(llm.seen[1]) == [("c1", "img:a"), ("c2", "img:b"), ("c3", "img:c")]
    await loop.aclose()


@pytest.mark.asyncio
async def test_non_async_tools_stay_serial_and_mixed_batch_keeps_order(tmp_path):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "save_note", {"tag": "f1"}), ("c2", "gen_image", {"tag": "a"}),
               ("c3", "save_note", {"tag": "f2"}), ("c4", "gen_image", {"tag": "b"})),
        _resp("done"),
    ])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=4),
                             tool_registry=_registry(log))
    sid = await loop.new_session()
    await loop.send("go", session_id=sid)
    # f1 串行先完成；a 与 b 一起起（在 f1 之后）；f2 串行
    assert log[0:2] == ["start:f1", "end:f1"]
    assert set(log[2:4]) == {"start:a", "start:b"}
    assert _tool_rows(llm.seen[1]) == [("c1", "file:f1"), ("c2", "img:a"), ("c3", "file:f2"), ("c4", "img:b")]
    await loop.aclose()


@pytest.mark.asyncio
async def test_tool_before_skip_still_wins_for_hoisted_calls(tmp_path):
    """闸类 hook 的语义不变：被 hoist 的调用也先过 TOOL_BEFORE，判 SKIP 就不执行。"""
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "blocked"}), ("c3", "gen_image", {"tag": "c"})),
        _resp("done"),
    ])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=4),
                             tool_registry=_registry(log))

    def gate(ctx):
        if ctx.tool_args.get("tag") == "blocked":
            ctx.output = "Error: gated"
            return HookDirective.SKIP
        return None

    loop.hooks.register(HookPoint.TOOL_BEFORE, gate, name="gate")
    sid = await loop.new_session()
    await loop.send("go", session_id=sid)
    assert "start:blocked" not in log
    assert _tool_rows(llm.seen[1]) == [("c1", "img:a"), ("c2", "Error: gated"), ("c3", "img:c")]
    await loop.aclose()


@pytest.mark.asyncio
async def test_one_failing_member_does_not_poison_the_batch(tmp_path):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "boom"}), ("c3", "gen_image", {"tag": "c"})),
        _resp("done"),
    ])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=4),
                             tool_registry=_registry(log))
    sid = await loop.new_session()
    res = await loop.send("go", session_id=sid)
    assert res.status == "completed"
    rows = _tool_rows(llm.seen[1])
    assert rows[0] == ("c1", "img:a") and rows[2] == ("c3", "img:c")
    assert rows[1][0] == "c2" and "boom failed" in rows[1][1]
    await loop.aclose()


@pytest.mark.asyncio
async def test_concurrency_off_runs_serially(tmp_path):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "b"})),
        _resp("done"),
    ])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=4, tool_batch_concurrency=0),
                             tool_registry=_registry(log, delay=0.05))
    sid = await loop.new_session()
    await loop.send("go", session_id=sid)
    assert log == ["start:a", "end:a", "start:b", "end:b"]
    await loop.aclose()


@pytest.mark.asyncio
async def test_started_events_fire_for_every_member_up_front(tmp_path):
    log: list[str] = []
    events: list = []
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.TOOL_CALL_STARTED, events.append)
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "b"})),
        _resp("done"),
    ])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=4),
                             tool_registry=_registry(log, delay=0.05), event_bus=bus)
    sid = await loop.new_session()
    await loop.send("go", session_id=sid)
    ids = [(e.payload or {}).get("tool_call_id") for e in events]
    assert ids == ["c1", "c2"]
    await loop.aclose()
