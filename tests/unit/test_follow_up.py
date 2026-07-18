"""Unit tests for follow-up merge helpers and queue draining."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from power_loop import AgentLoopConfig, FollowUpQueued, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.agent.follow_up import (
    FOLLOW_UP_MESSAGE_NAME,
    format_follow_up_user_message,
    merge_follow_up_inputs,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry


def test_format_follow_up_user_message_wraps_body() -> None:
    msg = format_follow_up_user_message("  focus on feelings  ")
    assert msg["role"] == "user"
    assert msg["name"] == FOLLOW_UP_MESSAGE_NAME
    assert msg["content"] == "<follow_up>\nfocus on feelings\n</follow_up>"


def test_merge_follow_up_inputs_empty_returns_none() -> None:
    assert merge_follow_up_inputs([]) is None
    assert merge_follow_up_inputs(["", "   "]) is None


def test_merge_follow_up_inputs_joins_multiple_strings() -> None:
    merged = merge_follow_up_inputs(["first", "second"])
    assert merged is not None
    assert merged["name"] == FOLLOW_UP_MESSAGE_NAME
    assert "first\n\nsecond" in merged["content"]


def test_merge_follow_up_inputs_accepts_loop_messages() -> None:
    merged = merge_follow_up_inputs(
        [
            {"role": "user", "content": "alpha"},
            format_follow_up_user_message("beta"),
        ]
    )
    assert merged is not None
    assert "alpha" in merged["content"]
    assert "beta" in merged["content"]


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
        ),
        lambda **kw: str(kw.get("text") or ""),
    )
    return reg


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_no_tools_drained_round_emits_round_completed(store: SessionStore) -> None:
    # pipeline-runner-3: a no-tools round that drains a queued follow-up (instead of completing)
    # must still close itself in the event stream — emit ROUND_COMPLETED + per-round usage — before
    # steering reopens the loop. Pre-fix the drained round was left unterminated.
    from power_loop import AgentEventBus, AgentEventType

    bus = AgentEventBus()
    completed: list = []
    bus.subscribe(AgentEventType.ROUND_COMPLETED, lambda e: completed.append((e.payload or {}).get("round_index")))
    llm = _GateLLM(responses=[
        LLMResponse(raw_text="thinking, no tools"),   # round 0: NO tool calls
        LLMResponse(raw_text="done after steering"),  # round 1: after the drained follow-up
    ])
    loop = StatefulAgentLoop(
        llm=llm, store=store, event_bus=bus,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4, compactor=None),
    )
    sid = await loop.new_session()
    send_task = asyncio.create_task(loop.send("start", sid))
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

    await loop.follow_up("steer me", sid)  # queued while round 0's LLM call is gated
    llm.release_first.set()
    result = await send_task
    assert result.status == "completed"
    # Both the drained no-tools round (0) and the final round (1) emitted ROUND_COMPLETED.
    assert 0 in completed and 1 in completed, completed


@pytest.mark.asyncio
async def test_multiple_follow_ups_merge_at_next_round(store: SessionStore) -> None:
    llm = _GateLLM(
        responses=[
            _tool_resp("c1", "echo", '{"text":"step1"}'),
            LLMResponse(raw_text="merged steering applied"),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        tool_registry=_echo_registry(),
    )
    sid = await loop.new_session()
    send_task = asyncio.create_task(loop.send("start", sid))

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

    first = await loop.follow_up("focus on feelings", sid)
    second = await loop.follow_up("keep it short", sid)
    assert isinstance(first, FollowUpQueued)
    assert isinstance(second, FollowUpQueued)
    assert first.queue_depth == 1
    assert second.queue_depth == 2

    llm.release_first.set()
    result = await send_task
    assert result.status == "completed"

    rows = await store.load_active_messages(sid)
    follow_rows = [r for r in rows if r.name == FOLLOW_UP_MESSAGE_NAME]
    assert len(follow_rows) == 1
    assert "focus on feelings" in follow_rows[0].content
    assert "keep it short" in follow_rows[0].content

    second_request = llm.calls[1]
    follow_contents = [
        str(m.get("content") or "")
        for m in second_request
        if m.get("name") == FOLLOW_UP_MESSAGE_NAME
        or "<follow_up>" in str(m.get("content") or "")
    ]
    assert any("focus on feelings" in c and "keep it short" in c for c in follow_contents)


# ── 3.18.0: drain-before-ROUND_START-hooks + stranded-steering flush ──


@pytest.mark.asyncio
async def test_round_start_break_hook_sees_drained_follow_ups(store: SessionStore) -> None:
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
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        tool_registry=_echo_registry(),
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
    sid = await loop.new_session()
    send_task = asyncio.create_task(loop.send("start", sid))
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

    queued = await loop.follow_up("please reply to the card", sid)
    assert isinstance(queued, FollowUpQueued)
    llm.release_first.set()
    result = await send_task
    # The hook saw the drained steering (1) and withdrew the break → round 1 ran.
    assert seen and seen[0] == 1, seen
    assert result.status == "completed"
    assert "answered the steering" in (result.final_text or "")
    # Nothing left stranded.
    assert loop.pending_follow_up_count(sid) == 0
    # The steering text actually reached the LLM.
    assert any(
        "please reply to the card" in str(m.get("content") or "")
        for call in llm.calls for m in call
    )


@pytest.mark.asyncio
async def test_flush_follow_ups_runs_stranded_queue(store: SessionStore) -> None:
    """Steering accepted in a run's terminal window stays queued on the idle session;
    the host detects it via pending_follow_up_count and flushes it as a fresh send."""
    llm = _GateLLM(responses=[LLMResponse(raw_text="handled stranded steering")])
    llm.release_first.set()  # no gating needed here
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4, compactor=None),
    )
    sid = await loop.new_session()
    # Simulate the terminal-window acceptance: enqueue directly onto the idle session.
    await loop._enqueue_follow_up(sid, "stranded card submission")
    assert loop.pending_follow_up_count(sid) == 1

    result = await loop.flush_follow_ups(sid)
    assert result is not None and result.status == "completed"
    assert loop.pending_follow_up_count(sid) == 0
    assert any(
        "stranded card submission" in str(m.get("content") or "")
        for call in llm.calls for m in call
    )
    # Idempotent: nothing left → None.
    assert await loop.flush_follow_ups(sid) is None


@pytest.mark.asyncio
async def test_flush_follow_ups_empty_or_busy_returns_none(store: SessionStore) -> None:
    llm = _GateLLM(responses=[_tool_resp("c1", "echo", '{"text":"x"}'), LLMResponse(raw_text="ok")])
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        tool_registry=_echo_registry(),
    )
    sid = await loop.new_session()
    assert await loop.flush_follow_ups(sid) is None  # empty queue
    send_task = asyncio.create_task(loop.send("start", sid))
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
    await loop.follow_up("steer", sid)  # queued on the busy session
    assert await loop.flush_follow_ups(sid) is None  # busy → owner drains, not us
    llm.release_first.set()
    await send_task
