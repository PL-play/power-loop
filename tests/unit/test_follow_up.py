"""Unit tests for follow-up merge helpers and queue draining."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from power_loop import AgentLoopConfig, SessionStore
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry


class _GateLLM(LLMService):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []
        self._idx = 0
        self.release_first = asyncio.Event()

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        self.calls.append(list(request.messages))
        if self._idx == 0:
            await self.release_first.wait()
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        response = self.responses[self._idx]
        self._idx += 1
        return response

    async def close(self) -> None:
        return None


def _tool_resp(call_id: str, name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    )


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="Echo text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            required_params=("text",),
        pricing=Free("test_fixture")),
        lambda **kw: str(kw.get("text") or ""),
    )
    return reg


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_final_boundary_leaves_new_input_for_next_send_and_closes_both_rounds(store: SessionStore, runtime_agent) -> None:
    from power_loop import AgentEventBus, AgentEventType

    bus = AgentEventBus()
    completed: list = []
    bus.subscribe(AgentEventType.ROUND_COMPLETED, lambda e: completed.append((e.payload or {}).get("round_index")))
    llm = _GateLLM(responses=[
        LLMResponse(raw_text="thinking, no tools"),   # round 0: NO tool calls
        LLMResponse(raw_text="done after steering"),  # round 1: after the drained follow-up
    ])
    loop = await runtime_agent(
        model=llm, store=store, event_bus=bus,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4, context_budget_tokens=10**9),
    )
    send_task = asyncio.create_task(loop.send("start"))
    # Wait for round 0's LLM call to START, not merely for send() to take the session lock.
    # send() locks BEFORE the pipeline's round-0 follow-up drain, so gating on the lock leaves a
    # window where steering queued here is drained into round 0 instead of round 1 — delivered,
    # but not to the round these tests assert about. _GateLLM records `calls` from inside
    # complete(), which the pipeline only reaches after that drain.
    for _ in range(200):
        if llm.calls:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("expected round 0 to reach the LLM during send")

    receipt = await loop.runtime.client.send(loop.target, "steer me", message_id="steer me", actor=loop.actor)
    llm.release_first.set()
    result = await send_task
    assert result.status == "completed"
    from power_loop.agent.runtime_store import now_ms
    follow = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+5000)
    await loop.drain()
    assert follow.status == "completed" and follow.send_index == result.send_index + 1
    assert follow.final_text == "done after steering"
    assert completed == [0, 0]


@pytest.mark.asyncio
async def test_multiple_follow_ups_merge_at_next_round(store: SessionStore, runtime_agent) -> None:
    llm = _GateLLM(
        responses=[
            _tool_resp("c1", "echo", '{"text":"step1"}'),
            LLMResponse(raw_text="merged steering applied"),
        ]
    )
    loop = await runtime_agent(
        model=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        registry=_echo_registry(),
    )
    sid = loop.target.session_id
    send_task = asyncio.create_task(loop.send("start"))

    # Wait for round 0's LLM call to START, not merely for send() to take the session lock.
    # send() locks BEFORE the pipeline's round-0 follow-up drain, so gating on the lock leaves a
    # window where steering queued here is drained into round 0 instead of round 1 — delivered,
    # but not to the round these tests assert about. _GateLLM records `calls` from inside
    # complete(), which the pipeline only reaches after that drain.
    for _ in range(200):
        if llm.calls:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("expected round 0 to reach the LLM during send")

    first = await loop.runtime.client.send(loop.target, "focus on feelings", message_id="focus on feelings", actor=loop.actor)
    second = await loop.runtime.client.send(loop.target, "keep it short", message_id="keep it short", actor=loop.actor)
    assert first.state == second.state == "accepted"
    assert first.input_ref != second.input_ref

    llm.release_first.set()
    result = await send_task
    assert result.status == "completed"

    rows = await store.load_active_messages(sid)
    follow_rows = [r for r in rows if r.role == "user" and "focus on feelings" in (r.content or "")]
    assert len(follow_rows) == 1
    assert "focus on feelings" in follow_rows[0].content
    assert "keep it short" in follow_rows[0].content

    second_request = llm.calls[1]
    follow_contents = [
        str(m.get("content") or "")
        for m in second_request
        if m.get("role") == "user"
    ]
    assert any("focus on feelings" in c and "keep it short" in c for c in follow_contents)


# ── 3.18.0: drain-before-ROUND_START-hooks + stranded-steering flush ──


@pytest.mark.asyncio
async def test_round_start_break_hook_sees_drained_follow_ups(store: SessionStore, runtime_agent) -> None:
    """A break-deciding ROUND_START hook (host pass_turn pattern) must observe steering
    drained at the same boundary and be able to withdraw the break. Pre-3.18 the drain ran
    AFTER the hooks, so a BREAK silently stranded the queued input (conv-117 incident)."""
    from power_loop.contracts.hooks import HookDirective, HookPoint

    llm = _GateLLM(
        responses=[
            _tool_resp("c1", "echo", '{"text":"working"}'),  # round 0: tool call
            LLMResponse(raw_text="answered the steering"),   # round 1: runs only if un-broken
        ]
    )
    loop = await runtime_agent(
        model=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        registry=_echo_registry(),
    )
    seen: list[int] = []

    def _pass_turn_like(ctx: Any) -> None:
        # Break from round 1 on — UNLESS fresh steering was drained into this round.
        if ctx.round_index >= 1:
            seen.append(int(getattr(ctx, "drained_follow_ups", -1)))
            if not getattr(ctx, "drained_follow_ups", 0):
                ctx.reason = "pass_turn"
                ctx.directive = HookDirective.BREAK

    loop.hooks.register(HookPoint.ROUND_START, _pass_turn_like)
    sid = loop.target.session_id
    send_task = asyncio.create_task(loop.send("start"))
    # Wait for round 0's LLM call to START, not merely for send() to take the session lock.
    # send() locks BEFORE the pipeline's round-0 follow-up drain, so gating on the lock leaves a
    # window where steering queued here is drained into round 0 instead of round 1 — delivered,
    # but not to the round these tests assert about. _GateLLM records `calls` from inside
    # complete(), which the pipeline only reaches after that drain.
    for _ in range(200):
        if llm.calls:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("expected round 0 to reach the LLM during send")

    queued = await loop.runtime.client.send(loop.target, "please reply to the card", message_id="please reply to the card", actor=loop.actor)
    assert queued.state == "accepted"
    llm.release_first.set()
    result = await send_task
    # The hook saw the drained steering (1) and withdrew the break → round 1 ran.
    assert seen and seen[0] == 1, seen
    assert result.status == "completed"
    assert "answered the steering" in (result.final_text or "")
    # Nothing left stranded.
    assert await store._db.fetchall(f"SELECT id FROM {loop.runtime.worker.inputs.q} WHERE session_id=? AND status='pending'", (sid,)) == []
    # The steering text actually reached the LLM.
    assert any(
        "please reply to the card" in str(m.get("content") or "")
        for call in llm.calls for m in call
    )


@pytest.mark.asyncio
async def test_idle_append_is_driven_without_host_flush_and_original_id_deduplicates(store, runtime_agent):
    from power_loop.agent.runtime_store import now_ms
    llm = _GateLLM(responses=[LLMResponse(raw_text="handled stranded steering")])
    llm.release_first.set()
    loop = await runtime_agent(model=llm, store=store, config=AgentLoopConfig(context_budget_tokens=10**9))
    receipt = await loop.runtime.client.send(loop.target, "stranded card submission", message_id="card-1", actor=loop.actor)
    result = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+10000)
    assert result.status == "completed"
    await loop.drain()
    duplicate = await loop.runtime.client.send(loop.target, "stranded card submission", message_id="card-1", actor=loop.actor)
    assert not duplicate.newly_accepted and duplicate.input_ref == receipt.input_ref
    assert len(llm.calls) == 1 and "stranded card submission" in str(llm.calls[0])


@pytest.mark.asyncio
async def test_empty_activation_and_busy_append_do_not_start_duplicate_execution(store, runtime_agent):
    from power_loop.agent.runtime_transport import TransportPacket
    llm = _GateLLM(responses=[_tool_resp("c1", "echo", '{"text":"x"}'), LLMResponse(raw_text="ok")])
    loop = await runtime_agent(model=llm, store=store, registry=_echo_registry(), config=AgentLoopConfig(context_budget_tokens=10**9))
    await loop.runtime.worker.recover()
    assert not llm.calls
    send_task = asyncio.create_task(loop.send("start"))
    async with asyncio.timeout(5):
        while not llm.calls:
            await asyncio.sleep(.005)
    await loop.runtime.client.send(loop.target, "steer", message_id="steer", actor=loop.actor)
    for _ in range(3):
        await loop.runtime.transport.publish(TransportPacket(43, loop.runtime.namespace, "duplicate-wake",
                                                            loop.target, "activation"))
    assert len(llm.calls) == 1
    llm.release_first.set()
    assert (await send_task).status == "completed"
    assert len(llm.calls) == 2
