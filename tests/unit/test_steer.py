"""Steering reaches the model while work is in flight (design/124 §7).

- a steer arriving during the model call aborts it; nothing it produced reaches history; the next
  round carries the steer; the aborted call is billed as an estimate; no round is spent (Z8 too);
- a queue-mode item does NOT interrupt anything;
- a flag that can never be cleared can't spin the loop;
- tools: ``finish`` (default) runs to completion, ``abort`` is cancelled, ``background`` is adopted
  into the background task table and delivered later; calls not yet started are skipped;
- ``wait_or_steer`` lets a waiting tool return early.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    AgentLoopConfig,
    InboxItem,
    SessionStore,
    StatefulAgentLoop,
    wait_or_steer,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
)
from power_loop.agent.follow_up import FOLLOW_UP_MESSAGE_NAME
from power_loop.contracts.hook_contexts import LlmBeforeCtx
from power_loop.contracts.hooks import HookPoint
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

USAGE = LLMTokenUsage(prompt_tokens=50, completion_tokens=5, total_tokens=55)


class _LLM(LLMService):
    """Scripted. A step ``("stream_then_hang", text)`` streams ``text`` then blocks until
    cancelled — a long generation a steer should cut."""

    def __init__(self, steps: list[Any]) -> None:
        self.steps = list(steps)
        self.calls: list[list[dict[str, Any]]] = []
        self.hanging = asyncio.Event()
        self.cancelled = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None, **kw: Any):
        self.calls.append([dict(m) for m in request.messages])
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, tuple) and step[0] == "stream_then_hang":
            if on_chunk_delta_text:
                on_chunk_delta_text(step[1])
            self.hanging.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if isinstance(step, LLMResponse):
            return step
        r = LLMResponse(raw_text=str(step))
        r.token_usage = USAGE
        return r

    async def close(self) -> None:
        return None


def _call(name: str, cid: str = "c1") -> LLMResponse:
    r = LLMResponse(raw_text="", tool_calls=[{"id": cid, "type": "function",
                                             "function": {"name": name, "arguments": "{}"}}])
    r.token_usage = USAGE
    return r


def _calls(*names: str) -> LLMResponse:
    r = LLMResponse(raw_text="", tool_calls=[
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
        for i, n in enumerate(names)])
    r.token_usage = USAGE
    return r


class _Probe:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def run(self, **kw: Any) -> str:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        return "slow tool real result"


def _registry(probe: _Probe, *, interrupt: str) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="slow", description="slow",
                                input_schema={"type": "object", "properties": {}},
                                interrupt=interrupt), probe.run)
    reg.register(ToolDefinition(name="fast", description="fast",
                                input_schema={"type": "object", "properties": {}}),
                 lambda **kw: "fast result")
    return reg


def _events(bus: AgentEventBus, kind: AgentEventType) -> list[dict]:
    out: list[dict] = []
    bus.subscribe(kind, lambda e: out.append(dict(e.payload or {})))
    return out


@pytest.fixture
async def store():
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


def _texts(rows: list[Any]) -> str:
    return "\n".join(str(r.content or "") for r in rows)


# ── the model call ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_steer_aborts_the_model_call_and_the_next_round_carries_it(store) -> None:
    bus = AgentEventBus()
    interrupted = _events(bus, AgentEventType.STEER_INTERRUPTED)
    completed = _events(bus, AgentEventType.LLM_CALL_COMPLETED)
    llm = _LLM([("stream_then_hang", "Once upon a time, a very long story began…"), "4"])
    loop = StatefulAgentLoop(llm=llm, store=store, event_bus=bus, config=AgentLoopConfig(
        system_prompt="S", max_rounds=1, compactor=None))
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("write me a long story", sid))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    await loop.deliver(InboxItem("stop — just tell me 2+2", mode="steer", item_id="s1"), sid)
    res = await asyncio.wait_for(run, 5)

    assert res.status == "completed", res.status   # max_rounds=1: the aborted round didn't count
    assert res.final_text == "4"
    assert llm.cancelled == 1
    second = llm.calls[1]
    assert any(m.get("name") == FOLLOW_UP_MESSAGE_NAME and "2+2" in m["content"] for m in second)
    rows = await store.load_active_messages(sid)
    assert "Once upon a time" not in _texts(rows), "aborted output leaked into history"
    assert interrupted and interrupted[0]["where"] == "llm" and interrupted[0]["action"] == "restart"
    assert completed[0]["outcome"] == "aborted" and completed[0]["estimated"] is True
    assert res.usage["failed_calls"] == 1 and res.usage["calls"] == 2


@pytest.mark.asyncio
async def test_queue_item_does_not_interrupt(store) -> None:
    llm = _LLM([_call("fast"), "done"])
    gate = asyncio.Event()
    orig = llm.complete

    async def slow_first(request: LLMRequest, **kw: Any):
        if not llm.calls:
            llm.hanging.set()
            await gate.wait()
        return await orig(request, **kw)

    llm.complete = slow_first  # type: ignore[method-assign]
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="fast", description="f",
                                input_schema={"type": "object", "properties": {}}),
                 lambda **kw: "ok")
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg, config=AgentLoopConfig(
        system_prompt="S", max_rounds=3, compactor=None))
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("go", sid))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    await loop.deliver(InboxItem("fyi", item_id="q1"), sid)    # queue mode
    gate.set()
    res = await run
    assert res.status == "completed"
    assert llm.cancelled == 0 and len(llm.calls) == 2


@pytest.mark.asyncio
async def test_a_steer_flag_that_cannot_clear_does_not_spin(store) -> None:
    llm = _LLM(["fine"])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(
        system_prompt="S", max_rounds=2, compactor=None))
    sid = await loop.new_session()
    loop.steer_event(sid).set()          # set with nothing in the inbox to clear it
    res = await asyncio.wait_for(loop.send("hi", sid), 5)
    assert res.status == "completed" and res.final_text == "fine"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_durable_injection_of_an_aborted_round_is_not_written_twice(store) -> None:
    hooks = AgentHooks()

    def remind(ctx: LlmBeforeCtx) -> None:
        # a hook that would inject on every round it sees — like a reminder whose condition
        # still holds when the aborted round is re-run
        ctx.persist_messages.append({"role": "user", "content": "REMINDER: keep notes"})

    hooks.register(HookPoint.LLM_BEFORE, remind, name="t.remind")
    llm = _LLM([("stream_then_hang", "partial"), "ok"])
    loop = StatefulAgentLoop(llm=llm, store=store, hooks=hooks, config=AgentLoopConfig(
        system_prompt="S", max_rounds=1, compactor=None))
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("go", sid))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    await loop.deliver(InboxItem("steer", mode="steer", item_id="s"), sid)
    await asyncio.wait_for(run, 5)
    rows = await store.load_active_messages(sid)
    assert _texts(rows).count("REMINDER: keep notes") == 1


# ── tools ─────────────────────────────────────────────────────────────────────────────────


async def _run_with_tool(store, interrupt: str, *, bus: AgentEventBus | None = None):
    probe = _Probe()
    llm = _LLM([_call("slow"), "answered the new message"])
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=_registry(probe,
                             interrupt=interrupt), event_bus=bus or AgentEventBus(),
                             config=AgentLoopConfig(system_prompt="S", max_rounds=3,
                                                    compactor=None))
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("do the slow thing", sid))
    await asyncio.wait_for(probe.started.wait(), 5)
    await loop.deliver(InboxItem("new question", mode="steer", item_id="s"), sid)
    return loop, sid, probe, llm, run


@pytest.mark.asyncio
async def test_finish_mode_tool_runs_to_completion(store) -> None:
    loop, sid, probe, llm, run = await _run_with_tool(store, "finish")
    await asyncio.sleep(0.05)
    assert not run.done(), "a finish-mode tool must not be interrupted"
    probe.release.set()
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed" and probe.finished
    rows = await store.load_active_messages(sid)
    assert "slow tool real result" in _texts(rows)
    assert any(m.get("name") == FOLLOW_UP_MESSAGE_NAME for m in llm.calls[1])


@pytest.mark.asyncio
async def test_abort_mode_tool_is_cancelled_and_says_so(store) -> None:
    bus = AgentEventBus()
    interrupted = _events(bus, AgentEventType.STEER_INTERRUPTED)
    loop, sid, probe, llm, run = await _run_with_tool(store, "abort", bus=bus)
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed"
    assert probe.cancelled and not probe.finished
    rows = await store.load_active_messages(sid)
    tool_rows = [r for r in rows if r.role == "tool"]
    assert "interrupted" in tool_rows[0].content and "不是用户拒绝" in tool_rows[0].content
    assert interrupted[0]["where"] == "tool" and interrupted[0]["action"] == "abort"
    assert any("new question" in str(m.get("content")) for m in llm.calls[1])


@pytest.mark.asyncio
async def test_background_mode_tool_keeps_running_and_is_delivered_later(store) -> None:
    from power_loop.tools.default_tools import register_tool_task_callback

    settled: list[tuple[str, str]] = []

    async def on_done(sid: str, task_id: str, status: str) -> None:
        settled.append((task_id, status))

    register_tool_task_callback(on_done)
    try:
        loop, sid, probe, llm, run = await _run_with_tool(store, "background")
        res = await asyncio.wait_for(run, 5)
        assert res.status == "completed"
        assert not probe.cancelled and not probe.finished, "still running in the background"
        rows = await store.load_active_messages(sid)
        tool_row = next(r for r in rows if r.role == "tool")
        assert "moved to background" in tool_row.content and "task_id=" in tool_row.content
        task_id = tool_row.content.split("task_id=")[1].split("。")[0]
        bg = await store.get_background_task(sid, task_id)
        assert bg is not None and bg.status == "running"
        probe.release.set()
        for _ in range(200):
            if settled:
                break
            await asyncio.sleep(0.01)
        assert settled == [(task_id, "completed")]
        bg = await store.get_background_task(sid, task_id)
        assert bg.status == "completed" and "slow tool real result" in (bg.output_tail or "")
    finally:
        register_tool_task_callback(None)


@pytest.mark.asyncio
async def test_calls_not_yet_started_are_skipped(store) -> None:
    probe = _Probe()
    llm = _LLM([_calls("slow", "fast"), "ok"])
    loop = StatefulAgentLoop(llm=llm, store=store,
                             tool_registry=_registry(probe, interrupt="finish"),
                             config=AgentLoopConfig(system_prompt="S", max_rounds=3,
                                                    compactor=None))
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("go", sid))
    await asyncio.wait_for(probe.started.wait(), 5)
    await loop.deliver(InboxItem("wait", mode="steer", item_id="s"), sid)
    probe.release.set()
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed"
    tool_rows = [r for r in await store.load_active_messages(sid) if r.role == "tool"]
    assert tool_rows[0].content == "slow tool real result"
    assert tool_rows[1].content.startswith("[skipped:") and "fast result" not in tool_rows[1].content


# ── waiting tools ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_or_steer_returns_early(store) -> None:
    started = asyncio.Event()
    outcome: dict[str, Any] = {}

    async def waiter(**kw: Any) -> str:
        started.set()
        _res, steered = await wait_or_steer(asyncio.sleep(3600))
        outcome["steered"] = steered
        return "stopped waiting — a new message arrived" if steered else "waited"

    reg = ToolRegistry()
    reg.register(ToolDefinition(name="wait", description="w",
                                input_schema={"type": "object", "properties": {}}), waiter)
    llm = _LLM([_call("wait"), "ok"])
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg, config=AgentLoopConfig(
        system_prompt="S", max_rounds=3, compactor=None))
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("go", sid))
    await asyncio.wait_for(started.wait(), 5)
    await loop.deliver(InboxItem("hey", mode="steer", item_id="s"), sid)
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed" and outcome["steered"] is True


@pytest.mark.asyncio
async def test_wait_or_steer_outside_a_loop_just_waits() -> None:
    res, steered = await wait_or_steer(asyncio.sleep(0, result="x"))
    assert (res, steered) == ("x", False)


def test_interrupt_value_is_validated() -> None:
    with pytest.raises(ValueError):
        ToolDefinition(name="x", description="x", interrupt="later")


@pytest.mark.asyncio
async def test_cancelling_the_run_waits_for_the_calls_own_cleanup(store) -> None:
    """A stop cancels the send while the (now separately-tasked) model call is streaming. The
    call's cleanup — closing its stream, recording its estimated usage — must finish BEFORE the
    send returns, as it did when the call ran inline. A transport whose cancel path awaits (httpx
    closing a connection) used to finish after the send was gone, and its usage event was lost."""
    bus = AgentEventBus()
    completed = _events(bus, AgentEventType.LLM_CALL_COMPLETED)
    streaming = asyncio.Event()

    class _SlowCleanup(LLMService):
        async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None, **kw: Any):
            if on_chunk_delta_text:
                on_chunk_delta_text("partial " * 20)
            streaming.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)       # async cleanup, like closing a stream
                raise

        async def close(self) -> None:
            return None

    loop = StatefulAgentLoop(llm=_SlowCleanup(), store=store, event_bus=bus,
                             config=AgentLoopConfig(system_prompt="S", max_rounds=1,
                                                    compactor=None, retry_policy=None))
    sid = await loop.new_session()
    task = asyncio.create_task(loop.send("hi", sid))
    await asyncio.wait_for(streaming.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed and completed[0]["outcome"] == "aborted", \
        "the call's accounting ran after the send was already gone"
