"""6.11.0 同轮并发：同一轮 ≥2 个 async_capable 工具调用并发执行，结果按原顺序回填。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, HookDirective, HookPoint
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_cancel_token
from power_loop.core.control import ControlledLoopStop
from power_loop.runtime.pricing import Free
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
    reg.register(ToolDefinition(name="gen_image", description="slow", input_schema=schema, async_capable=True, pricing=Free("test_fixture")), slow)
    reg.register(ToolDefinition(name="save_note", description="serial", input_schema=schema, pricing=Free("test_fixture")), serial)
    return reg


def _tool_rows(msgs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(str(m.get("tool_call_id")), str(m.get("content"))) for m in msgs if m.get("role") == "tool"]


async def _paired_durable_receipts(loop):
    records = await loop.runtime.store.load_active_messages(loop.target.session_id)
    announced = [call["id"] for row in records for call in row.tool_calls or []]
    rows = [(row.tool_call_id, row.content) for row in records if row.role == "tool"]
    assert announced == [call_id for call_id, _ in rows]
    return rows


@pytest.mark.asyncio
async def test_async_capable_calls_in_one_round_run_concurrently_and_keep_order(tmp_path, runtime_agent):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "b"}), ("c3", "gen_image", {"tag": "c"})),
        _resp("done"),
    ])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(system_prompt="t", max_rounds=4), registry=_registry(log))
    t0 = time.monotonic()
    res = await loop.send("go")
    elapsed = time.monotonic() - t0
    assert res.status == "completed"
    assert elapsed < 0.6, f"3×0.25s ran serially? {elapsed:.2f}s"
    assert log[:3] == ["start:a", "start:b", "start:c"]  # 三个同时起
    assert _tool_rows(llm.seen[1]) == [("c1", "img:a"), ("c2", "img:b"), ("c3", "img:c")]
    await loop.aclose()


@pytest.mark.asyncio
async def test_non_async_tools_stay_serial_and_mixed_batch_keeps_order(tmp_path, runtime_agent):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "save_note", {"tag": "f1"}), ("c2", "gen_image", {"tag": "a"}),
               ("c3", "save_note", {"tag": "f2"}), ("c4", "gen_image", {"tag": "b"})),
        _resp("done"),
    ])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(system_prompt="t", max_rounds=4), registry=_registry(log))
    await loop.send("go")
    # f1 串行先完成；a 与 b 一起起（在 f1 之后）；f2 串行
    assert log[0:2] == ["start:f1", "end:f1"]
    assert set(log[2:4]) == {"start:a", "start:b"}
    assert _tool_rows(llm.seen[1]) == [("c1", "file:f1"), ("c2", "img:a"), ("c3", "file:f2"), ("c4", "img:b")]
    await loop.aclose()


@pytest.mark.asyncio
async def test_tool_before_skip_still_wins_for_hoisted_calls(tmp_path, runtime_agent):
    """闸类 hook 的语义不变：被 hoist 的调用也先过 TOOL_BEFORE，判 SKIP 就不执行。"""
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "blocked"}), ("c3", "gen_image", {"tag": "c"})),
        _resp("done"),
    ])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(system_prompt="t", max_rounds=4), registry=_registry(log))

    def gate(ctx):
        if ctx.tool_args.get("tag") == "blocked":
            ctx.output = "Error: gated"
            return HookDirective.SKIP
        return None

    loop.hooks.register(HookPoint.TOOL_BEFORE, gate, name="gate")
    await loop.send("go")
    assert "start:blocked" not in log
    assert _tool_rows(llm.seen[1]) == [("c1", "img:a"), ("c2", "Error: gated"), ("c3", "img:c")]
    await loop.aclose()


@pytest.mark.asyncio
async def test_one_failing_member_does_not_poison_the_batch(tmp_path, runtime_agent):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "boom"}), ("c3", "gen_image", {"tag": "c"})),
        _resp("done"),
    ])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(system_prompt="t", max_rounds=4), registry=_registry(log))
    res = await loop.send("go")
    assert res.status == "completed"
    rows = _tool_rows(llm.seen[1])
    assert rows[0] == ("c1", "img:a") and rows[2] == ("c3", "img:c")
    assert rows[1][0] == "c2" and "boom failed" in rows[1][1]
    await loop.aclose()


@pytest.mark.asyncio
async def test_concurrency_off_runs_serially(tmp_path, runtime_agent):
    log: list[str] = []
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "b"})),
        _resp("done"),
    ])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(system_prompt="t", max_rounds=4, tool_batch_concurrency=0), registry=_registry(log, delay=0.05))
    await loop.send("go")
    assert log == ["start:a", "end:a", "start:b", "end:b"]
    await loop.aclose()


@pytest.mark.asyncio
async def test_started_events_fire_for_every_member_up_front(tmp_path, runtime_agent):
    log: list[str] = []
    events: list = []
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.TOOL_CALL_STARTED, events.append)
    llm = _Scripted(responses=[
        _batch(("c1", "gen_image", {"tag": "a"}), ("c2", "gen_image", {"tag": "b"})),
        _resp("done"),
    ])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(system_prompt="t", max_rounds=4), registry=_registry(log, delay=0.05), event_bus=bus)
    await loop.send("go")
    ids = [(e.payload or {}).get("tool_call_id") for e in events]
    assert ids == ["c1", "c2"]
    await loop.aclose()


async def test_later_controlled_stop_settles_running_tools_and_skips_waiters(runtime_agent):
    log = []
    finishing = asyncio.Event()
    aborting = asyncio.Event()

    async def finish():
        log.append("finish started")
        finishing.set()
        await get_current_cancel_token().wait()
        log.append("finish receipt retained")
        return "supplier receipt"

    async def reject():
        await finishing.wait()
        await aborting.wait()
        log.append("rejected")
        raise ControlledLoopStop("test_limit", message="No remaining budget.")

    async def abort():
        log.append("abort started")
        aborting.set()
        try:
            await asyncio.Future()
        finally:
            log.append("abort settled")

    def previous():
        log.append("previous")
        return "already completed"

    def unexpected():
        pytest.fail("Tool started after the controlled stop")

    reg = ToolRegistry()
    for name, handler, parallel, mode in [
        ("previous", previous, False, "finish"),
        ("finish", finish, True, "finish"),
        ("reject", reject, True, "finish"),
        ("abort", abort, True, "abort"),
        ("waiter", unexpected, True, "finish"),
        ("serial", unexpected, False, "finish"),
    ]:
        reg.register(ToolDefinition(name=name, description=name, input_schema={"type": "object"},
                     async_capable=parallel, interrupt=mode, pricing=Free("test_fixture")), handler)
    ids = ["previous", "finish", "serial", "reject", "abort", "waiter"]
    llm = _Scripted(responses=[_batch(*[(i, i, {}) for i in ids]), _resp("next send")])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(tool_batch_concurrency=3), registry=reg)
    processed = []
    loop.hooks.register(HookPoint.TOOL_AFTER, lambda ctx: processed.append(ctx.tool_name))

    res = await asyncio.wait_for(loop.send("go"), timeout=5)
    assert res.status == "failed" and res.error_code == "budget_exceeded"
    assert res.error == "No remaining budget."
    assert len(llm.seen) == 1
    assert log[:1] == ["previous"]
    assert set(log[1:3]) == {"finish started", "abort started"}
    assert log[3] == "rejected"
    assert sorted(log[4:]) == ["abort settled", "finish receipt retained"]
    rows = await _paired_durable_receipts(loop)
    assert [i for i, _ in rows] == ids
    assert dict(rows)["previous"] == "already completed"
    assert dict(rows)["finish"].endswith("supplier receipt")
    assert dict(rows)["reject"] == "[stopped: test_limit] No remaining budget."
    assert dict(rows)["serial"] == dict(rows)["waiter"] == "[skipped: test_limit]"
    assert "已中止" in dict(rows)["abort"]
    assert processed == ["previous", "finish"]
    assert (await loop.send("next")).status == "completed"
    projected = "\n".join(str(row.get("content")) for row in llm.seen[1])
    assert "supplier receipt" in projected and "<missing>" not in projected


@pytest.mark.parametrize("parallel", [False, True])
async def test_retry_can_stop_the_send_without_starting_later_tools(runtime_agent, parallel):
    attempts = []
    errors = []
    sibling_finished = asyncio.Event()

    async def retried():
        attempts.append("retried")
        if len(attempts) == 1:
            raise RuntimeError("retry once")
        raise ControlledLoopStop("test_limit", message="No remaining budget.")

    def unexpected():
        pytest.fail("Serial tool ran after the controlled stop")

    async def sibling():
        sibling_finished.set()
        return "sibling already completed"

    reg = ToolRegistry()
    for name, handler, eligible in [
        ("retry", retried, parallel),
        ("sibling", sibling, parallel),
        ("serial", unexpected, False),
    ]:
        reg.register(ToolDefinition(name=name, description=name, input_schema={"type": "object"},
                     async_capable=eligible, pricing=Free("test_fixture")), handler)
    calls = [("retry", "retry", {}), ("sibling", "sibling", {}), ("serial", "serial", {})]
    llm = _Scripted(responses=[_batch(*calls), _resp("next send")])
    loop = await runtime_agent(model=llm, config=AgentLoopConfig(), registry=reg)

    async def on_error(ctx):
        errors.append(str(ctx.error))
        if parallel:
            await sibling_finished.wait()
        return HookDirective.SHORT_CIRCUIT

    loop.hooks.register(HookPoint.TOOL_ERROR, on_error, name="test_retry")
    stopped = await loop.send("go")
    assert stopped.status == "failed" and stopped.error_code == "budget_exceeded"
    assert stopped.error == "No remaining budget."
    assert len(llm.seen) == 1
    assert attempts == ["retried", "retried"] and errors == ["retry once"]
    assert await _paired_durable_receipts(loop) == [
        ("retry", "[stopped: test_limit] No remaining budget."),
        ("sibling", "sibling already completed" if parallel else "[skipped: test_limit]"),
        ("serial", "[skipped: test_limit]"),
    ]
    assert (await loop.send("next")).status == "completed"


async def test_controlled_stop_during_parallel_admission_does_not_start_queued_tools(runtime_agent):
    log = []
    llm = _Scripted(responses=[_batch(
        ("first", "gen_image", {"tag": "first"}),
        ("denied", "gen_image", {"tag": "denied"}),
    )])
    loop = await runtime_agent(model=llm, registry=_registry(log))

    def before(ctx):
        if ctx.tool_args["tag"] == "denied":
            raise ControlledLoopStop("test_limit", message="No remaining budget.")

    loop.hooks.register(HookPoint.TOOL_BEFORE, before, name="test_limit")
    result = await loop.send("go")
    assert (result.status, result.error_code) == ("failed", "budget_exceeded")
    assert len(llm.seen) == 1 and not log
    assert await _paired_durable_receipts(loop) == [
        ("first", "[skipped: test_limit]"),
        ("denied", "[skipped: test_limit]"),
    ]


@pytest.mark.parametrize("timing", ["allowed", "returned_before_denial", "returned_during_finish"])
async def test_tool_after_processes_real_returns_once_and_in_call_order(runtime_agent, timing):
    raw = "draft supplier result"
    approved = "approved public result"
    arguments = {"key": "fixture"}
    denied = timing != "allowed"
    ready = asyncio.Event()
    timeline, processed, emitted = [], [], []

    async def decide():
        await ready.wait()
        timeline.append("decision")
        if denied:
            raise ControlledLoopStop("quota_exhausted", message="No remaining budget.")
        return "allowed"

    async def lookup(key):
        assert key == arguments["key"]
        ready.set()
        if timing == "returned_during_finish":
            await get_current_cancel_token().wait()
        timeline.append("returned")
        return raw

    def marker():
        timeline.append("marker")
        return "ran"

    def process_result(ctx):
        processed.append((ctx.tool_call["id"], ctx.tool_name, dict(ctx.tool_args), ctx.output, ctx.failed))
        if ctx.tool_name == "lookup":
            ctx.output = approved
            ctx.failed = True
            if denied:
                return HookDirective.BREAK
        return None

    registry = ToolRegistry()
    for name, handler, parallel in (("decide", decide, True), ("lookup", lookup, True), ("marker", marker, False)):
        registry.register(ToolDefinition(
            name=name, description=name, input_schema={"type": "object", "properties": {"key": {"type": "string"}}},
            async_capable=parallel, interrupt="finish", pricing=Free("test_fixture"),
        ), handler)
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.TOOL_CALL_FAILED, emitted.append)
    bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED, emitted.append)
    model = _Scripted(responses=[
        _batch(("gate", "decide", {}), ("body", "lookup", arguments), ("later", "marker", {})),
        _resp("done"), _resp("done"),
    ])
    loop = await runtime_agent(
        model=model, registry=registry, config=AgentLoopConfig(tool_batch_concurrency=2), event_bus=bus,
    )
    loop.hooks.register(HookPoint.TOOL_AFTER, process_result)

    result = await asyncio.wait_for(loop.send("first"), timeout=5)
    assert (result.status, result.error_code) == (("failed", "budget_exceeded") if denied else ("completed", None))
    assert len(model.seen) == (1 if denied else 2)
    assert timeline[:2] == (["decision", "returned"] if timing == "returned_during_finish"
                            else ["returned", "decision"])
    assert timeline[2:] == ([] if denied else ["marker"])
    lookup_hook = ("body", "lookup", arguments, raw, False)
    assert processed == ([lookup_hook] if denied else [
        ("gate", "decide", {}, "allowed", False), lookup_hook, ("later", "marker", {}, "ran", False),
    ])

    rows = await _paired_durable_receipts(loop)
    assert [call_id for call_id, _ in rows] == ["gate", "body", "later"]
    contents = dict(rows)
    assert contents["gate"] == ("[stopped: quota_exhausted] No remaining budget." if denied else "allowed")
    assert contents["later"] == ("[skipped: quota_exhausted]" if denied else "ran")
    if timing == "returned_during_finish":
        assert contents["body"].startswith("[stopped: 执行停止（quota_exhausted）")
        assert contents["body"].endswith("\n" + approved)
    else:
        assert contents["body"] == approved

    body_events = [event for event in emitted if event.payload["tool_call_id"] == "body"]
    assert [event.type for event in body_events] == [
        AgentEventType.TOOL_CALL_FAILED, AgentEventType.TOOL_CALL_COMPLETED,
    ]
    assert all(event.payload["output"] == contents["body"] for event in body_events)
    assert body_events[-1].payload["failed"] is True
    assert (await loop.send("next")).status == "completed"
    assert approved in str(model.seen[-1]) and raw not in str(model.seen[-1])
