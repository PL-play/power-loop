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
    SessionStore,
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
from power_loop.runtime.pricing import Free
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
        self.release = asyncio.Event()

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None, **kw: Any):
        self.calls.append([dict(m) for m in request.messages])
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, tuple) and step[0] == "stream_then_gate":
            self.hanging.set()
            await self.release.wait()
            return LLMResponse(raw_text=step[1], token_usage=USAGE)
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
                                interrupt=interrupt, pricing=Free("test_fixture")), probe.run)
    reg.register(ToolDefinition(name="fast", description="fast",
                                input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")),
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


async def _steer(agent, body, message_id):
    from power_loop import InputRef
    current = await agent.runtime.client.inspect(InputRef(agent.target.store_id, agent.target.session_id, "m1"),
                                                  actor=agent.actor)
    return await agent.runtime.client.steer(current.job_ref, body, message_id=message_id, actor=agent.actor)


# ── the model call ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_steer_is_received_during_model_and_applies_at_next_safe_boundary(store, runtime_agent) -> None:
    bus = AgentEventBus()
    interrupted = _events(bus, AgentEventType.STEER_INTERRUPTED)
    completed = _events(bus, AgentEventType.LLM_CALL_COMPLETED)
    llm = _LLM([("stream_then_gate", "Original response"), "4"])
    loop = await runtime_agent(model=llm, store=store, event_bus=bus, config=AgentLoopConfig(
        system_prompt="S", max_rounds=3, compactor=None))
    sid = loop.target.session_id
    run = asyncio.create_task(loop.send("write me a long story"))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    await _steer(loop, "stop — just tell me 2+2", "s1")
    assert not run.done() and llm.cancelled == 0
    llm.release.set()
    res = await asyncio.wait_for(run, 5)

    assert res.status == "completed", res.status   # max_rounds=1: the aborted round didn't count
    assert res.final_text == "4"
    assert llm.cancelled == 0
    second = llm.calls[1]
    assert any(m.get("name") == FOLLOW_UP_MESSAGE_NAME and "2+2" in m["content"] for m in second)
    rows = await store.load_active_messages(sid)
    assert "Original response" in _texts(rows)
    assert not interrupted
    assert all(event["success"] for event in completed)
    assert res.usage.get("failed_calls", 0) == 0 and res.usage["calls"] == 2


@pytest.mark.asyncio
async def test_queue_item_does_not_interrupt(store, runtime_agent) -> None:
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
                                input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")),
                 lambda **kw: "ok")
    loop = await runtime_agent(model=llm, store=store, registry=reg, config=AgentLoopConfig(
        system_prompt="S", max_rounds=3, compactor=None))
    run = asyncio.create_task(loop.send("go"))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    await loop.runtime.client.append(loop.target, "fyi", message_id="q1", actor=loop.actor)    # queue mode
    gate.set()
    res = await run
    assert res.status == "completed"
    assert llm.cancelled == 0 and len(llm.calls) == 2


@pytest.mark.asyncio
async def test_session_append_joins_the_running_job_next_round(store, runtime_agent) -> None:
    from power_loop import InputRef
    llm = _LLM([("stream_then_gate", "first"), "done"])
    loop = await runtime_agent(model=llm, store=store, config=AgentLoopConfig(
        system_prompt="S", max_rounds=3, compactor=None))
    first = await loop.runtime.client.send(loop.target, "go", message_id="m1", actor=loop.actor)
    await asyncio.wait_for(llm.hanging.wait(), 5)
    await loop.runtime.client.append(loop.target, "one more thing", message_id="a1", actor=loop.actor)
    llm.release.set()
    await loop.runtime.client.wait(first.input_ref, actor=loop.actor, deadline_ms=None)
    added = await loop.runtime.client.inspect(InputRef(loop.target.store_id, loop.target.session_id, "a1"),
                                              actor=loop.actor)
    original = await loop.runtime.client.inspect(first.input_ref, actor=loop.actor)
    assert added.job_ref == original.job_ref
    assert any("one more thing" in str(m.get("content")) for m in llm.calls[1])


@pytest.mark.asyncio
async def test_session_steer_interrupts_the_running_job(store, runtime_agent) -> None:
    from power_loop.agent.runtime_types import Command
    probe = _Probe()
    llm = _LLM([_call("slow"), "stopped"])
    loop = await runtime_agent(model=llm, store=store, registry=_registry(probe, interrupt="abort"),
                               config=AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None))
    run = asyncio.create_task(loop.send("do the slow thing"))
    await asyncio.wait_for(probe.started.wait(), 5)
    await loop.runtime.client.submit(Command("s1", loop.target, "steer", {"body": "stop"}, "control"),
                                     actor=loop.actor)
    res = await asyncio.wait_for(run, 5)
    assert res.status == "failed" and probe.cancelled and not probe.finished
    assert (await loop.send("continue")).status == "completed"
    assert any("stop" in str(m.get("content")) for m in llm.calls[1])


@pytest.mark.asyncio
async def test_a_steer_flag_that_cannot_clear_does_not_spin(store, runtime_agent) -> None:
    llm = _LLM(["fine"])
    loop = await runtime_agent(model=llm, store=store, config=AgentLoopConfig(
        system_prompt="S", max_rounds=2, compactor=None))
    from power_loop.agent.runtime_transport import TransportPacket
    await loop.runtime.transport.publish(TransportPacket(42, loop.runtime.namespace, "stale",
                                                         loop.target, "activation"))
    res = await asyncio.wait_for(loop.send("hi"), 5)
    assert res.status == "completed" and res.final_text == "fine"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_durable_injection_occurs_once_per_model_round_with_steering(store, runtime_agent) -> None:
    hooks = AgentHooks()

    def remind(ctx: LlmBeforeCtx) -> None:
        # a hook that would inject on every round it sees — like a reminder whose condition
        # still holds when the aborted round is re-run
        ctx.persist_messages.append({"role": "user", "content": "REMINDER: keep notes"})

    hooks.register(HookPoint.LLM_BEFORE, remind, name="t.remind")
    llm = _LLM([("stream_then_gate", "first"), "ok"])
    loop = await runtime_agent(model=llm, store=store, hooks=hooks, config=AgentLoopConfig(
        system_prompt="S", max_rounds=3, compactor=None))
    sid = loop.target.session_id
    run = asyncio.create_task(loop.send("go"))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    await _steer(loop, "steer", "s")
    llm.release.set()
    await asyncio.wait_for(run, 5)
    rows = await store.load_active_messages(sid)
    assert _texts(rows).count("REMINDER: keep notes") == len(llm.calls) == 2


# ── tools ─────────────────────────────────────────────────────────────────────────────────


async def _run_with_tool(runtime_agent, store, interrupt: str, *, bus: AgentEventBus | None = None):
    probe = _Probe()
    llm = _LLM([_call("slow"), "answered the new message"])
    loop = await runtime_agent(model=llm, store=store, registry=_registry(probe,
                             interrupt=interrupt), event_bus=bus or AgentEventBus(), capacity=2 if interrupt == "background" else 1,
                             config=AgentLoopConfig(system_prompt="S", max_rounds=3,
                                                    compactor=None))
    sid = loop.target.session_id
    run = asyncio.create_task(loop.send("do the slow thing"))
    await asyncio.wait_for(probe.started.wait(), 5)
    await _steer(loop, "new question", "s")
    return loop, sid, probe, llm, run


@pytest.mark.asyncio
async def test_finish_mode_tool_runs_to_completion(store, runtime_agent) -> None:
    loop, sid, probe, llm, run = await _run_with_tool(runtime_agent, store, "finish")
    await asyncio.sleep(0.05)
    assert not run.done(), "a finish-mode tool must not be interrupted"
    probe.release.set()
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed" and probe.finished
    rows = await store.load_active_messages(sid)
    assert "slow tool real result" in _texts(rows)
    assert any(m.get("name") == FOLLOW_UP_MESSAGE_NAME for m in llm.calls[1])


@pytest.mark.asyncio
async def test_abort_mode_tool_is_cancelled_unknown_and_new_input_can_continue(store, runtime_agent) -> None:
    bus = AgentEventBus()
    interrupted = _events(bus, AgentEventType.STEER_INTERRUPTED)
    loop, sid, probe, llm, run = await _run_with_tool(runtime_agent, store, "abort", bus=bus)
    res = await asyncio.wait_for(run, 5)
    assert res.status == "failed" and "unresolved_prior_attempt" in res.error
    assert len(llm.calls) == 1
    assert (await loop.send("continue after cancellation")).status == "completed"
    assert probe.cancelled and not probe.finished
    rows = await store.load_active_messages(sid)
    tool_rows = [r for r in rows if r.role == "tool"]
    assert "interrupted" in tool_rows[0].content and "不是用户拒绝" in tool_rows[0].content
    assert interrupted[0]["where"] == "tool" and interrupted[0]["action"] == "abort"
    assert any("new question" in str(m.get("content")) for m in llm.calls[1])


@pytest.mark.asyncio
async def test_background_mode_tool_keeps_running_and_is_delivered_later(store, runtime_agent):
    import json

    from power_loop import JobRef
    from power_loop.agent.runtime_store import now_ms
    from power_loop.runtime.background import load_record
    loop, sid, probe, llm, run = await _run_with_tool(runtime_agent, store, "background")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed"
    assert not probe.cancelled and not probe.finished
    rows = await store.load_active_messages(sid)
    receipt = json.loads(next(r.content for r in rows if r.role == "tool"))
    assert "continues in the background" in receipt["message"]
    original = await loop.runtime.client.inspect(JobRef(**receipt["job_ref"]), actor=loop.actor)
    assert original.state == "running"
    assert await load_record(store, sid, receipt["task_id"]) is None
    probe.release.set()
    result = await loop.runtime.client.wait(JobRef(**receipt["job_ref"]), actor=loop.actor,
                                           deadline_ms=now_ms()+10000)
    assert result.status == "completed" and result.final_text == "slow tool real result"
    await loop.drain()
    original = await load_record(store, sid, receipt["task_id"])
    assert original["status"] == "completed" and original["result"] == "slow tool real result"


@pytest.mark.asyncio
async def test_calls_not_yet_started_are_skipped(store, runtime_agent) -> None:
    probe = _Probe()
    llm = _LLM([_calls("slow", "fast"), "ok"])
    loop = await runtime_agent(model=llm, store=store,
                             registry=_registry(probe, interrupt="finish"),
                             config=AgentLoopConfig(system_prompt="S", max_rounds=3,
                                                    compactor=None))
    sid = loop.target.session_id
    run = asyncio.create_task(loop.send("go"))
    await asyncio.wait_for(probe.started.wait(), 5)
    await _steer(loop, "wait", "s")
    probe.release.set()
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed"
    tool_rows = [r for r in await store.load_active_messages(sid) if r.role == "tool"]
    assert tool_rows[0].content == "slow tool real result"
    assert tool_rows[1].content.startswith("[skipped:") and "fast result" not in tool_rows[1].content


# ── waiting tools ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_or_steer_returns_early(store, runtime_agent) -> None:
    started = asyncio.Event()
    outcome: dict[str, Any] = {}

    async def waiter(**kw: Any) -> str:
        started.set()
        _res, steered = await wait_or_steer(asyncio.sleep(3600))
        outcome["steered"] = steered
        return "stopped waiting — a new message arrived" if steered else "waited"

    reg = ToolRegistry()
    reg.register(ToolDefinition(name="wait", description="w",
                                input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), waiter)
    llm = _LLM([_call("wait"), "ok"])
    loop = await runtime_agent(model=llm, store=store, registry=reg, config=AgentLoopConfig(
        system_prompt="S", max_rounds=3, compactor=None))
    run = asyncio.create_task(loop.send("go"))
    await asyncio.wait_for(started.wait(), 5)
    await _steer(loop, "hey", "s")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "completed" and outcome["steered"] is True


@pytest.mark.asyncio
async def test_wait_or_steer_outside_a_loop_just_waits() -> None:
    res, steered = await wait_or_steer(asyncio.sleep(0, result="x"))
    assert (res, steered) == ("x", False)


def test_interrupt_value_is_validated() -> None:
    with pytest.raises(ValueError):
        ToolDefinition(name="x", description="x", interrupt="later", pricing=Free("test_fixture"))


@pytest.mark.asyncio
async def test_cancelling_the_run_waits_for_the_calls_own_cleanup(store, runtime_agent) -> None:
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

    loop = await runtime_agent(model=_SlowCleanup(), store=store, event_bus=bus,
                             config=AgentLoopConfig(system_prompt="S", max_rounds=1,
                                                    compactor=None, retry_policy=None))
    sid = loop.target.session_id
    task = asyncio.create_task(loop.send("hi"))
    await asyncio.wait_for(streaming.wait(), 5)
    from power_loop import InputRef
    await loop.runtime.client.cancel(InputRef(store.store_id, sid, "m1"), actor=loop.actor)
    assert (await task).status == "cancelled"
    assert completed and completed[0]["outcome"] == "aborted", \
        "the call's accounting ran after the send was already gone"
