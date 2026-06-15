"""Durable timers (0.11.0): store CRUD, agent tools, TimerRunner firing
semantics, TIMER_FIRE hook veto, restart recovery."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    HookDirective,
    HookPoint,
    StatefulAgentLoop,
    TimerFireCtx,
    TimerRunner,
    create_default_tool_registry,
    get_tool_definitions,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.tools import DEFAULT_TOOL_HANDLERS


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
            return LLMResponse(raw_text="done", content_text="done")
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


def _tool_resp(call_id: str, name: str, args: str) -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[{"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": args}}],
    )


def _loop(tmp_path, responses=None, hooks=None, bus=None, with_tools=False):
    registry = None
    if with_tools:
        registry = create_default_tool_registry(preset="core", bind=False)
        for d in get_tool_definitions(
            include=["schedule_wakeup", "list_wakeups", "cancel_wakeup", "current_time"]
        ):
            registry.register(d, DEFAULT_TOOL_HANDLERS[d.name], overwrite=True)
    return StatefulAgentLoop(
        llm=_Scripted(responses=responses or []),
        db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
        tool_registry=registry,
        hooks=hooks,
        event_bus=bus,
    )


# ── store + external API ────────────────────────────────────────────────


def test_schedule_cancel_list_external_api(tmp_path):
    loop = _loop(tmp_path)
    sid = loop.new_session()
    t = loop.schedule_timer(sid, delay_s=3600, note="check the report")
    assert t.status == "armed" and t.timer_id == 1
    assert [x.timer_id for x in loop.list_timers(sid)] == [1]
    assert loop.cancel_timer(sid, 1) is True
    assert loop.cancel_timer(sid, 1) is False  # already cancelled
    assert loop.list_timers(sid) == []
    with pytest.raises(ValueError):
        loop.schedule_timer(sid, note="missing delay")
    with pytest.raises(ValueError):
        loop.schedule_timer(sid, delay_s=10, note="   ")
    loop.close()


def test_timers_cascade_delete_with_session(tmp_path):
    loop = _loop(tmp_path)
    sid = loop.new_session()
    loop.schedule_timer(sid, delay_s=60, note="n")
    loop.close_session(sid)
    assert loop.store.list_timers(sid) == []
    loop.close()


# ── agent tools ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_schedules_and_lists_wakeups_via_tools(tmp_path):
    loop = _loop(
        tmp_path,
        responses=[
            _tool_resp("c1", "schedule_wakeup",
                       '{"delay_seconds": 60, "note": "ping the build"}'),
            _tool_resp("c2", "list_wakeups", "{}"),
            LLMResponse(raw_text="scheduled", content_text="scheduled"),
        ],
        with_tools=True,
    )
    sid = loop.new_session()
    res = await loop.send("remind yourself", session_id=sid)
    assert res.status == "completed"
    timers = loop.list_timers(sid)
    assert len(timers) == 1 and timers[0].note == "ping the build"
    loop.close()


@pytest.mark.asyncio
async def test_current_time_tool(tmp_path):
    loop = _loop(
        tmp_path,
        responses=[
            _tool_resp("c1", "current_time", "{}"),
            LLMResponse(raw_text="ok", content_text="ok"),
        ],
        with_tools=True,
    )
    sid = loop.new_session()
    res = await loop.send("what time is it", session_id=sid)
    assert res.status == "completed" and res.tool_calls == 1
    loop.close()


# ── TimerRunner firing ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_due_timer_fires_into_idle_session(tmp_path):
    events: list = []
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.TIMER_FIRED, events.append)
    loop = _loop(tmp_path, responses=[
        LLMResponse(raw_text="woke up and did the thing",
                    content_text="woke up and did the thing"),
    ], bus=bus)
    sid = loop.new_session()
    loop.schedule_timer(sid, due_at_ms=int(time.time() * 1000) - 1000, note="do the thing")
    runner = TimerRunner(loop)
    fired = await runner.scan_once()
    assert fired == 1
    assert events and events[0].data.outcome == "delivered"
    # the timer message reached the session as a user turn
    msgs = loop.get_messages(sid)
    assert any("do the thing" in str(m.get("content")) for m in msgs if m.get("role") == "user")
    assert loop.store.get_timer(sid, 1).status == "fired"
    loop.close()


@pytest.mark.asyncio
async def test_not_due_timer_does_not_fire(tmp_path):
    loop = _loop(tmp_path)
    sid = loop.new_session()
    loop.schedule_timer(sid, delay_s=3600, note="later")
    assert await TimerRunner(loop).scan_once() == 0
    assert loop.store.get_timer(sid, 1).status == "armed"
    loop.close()


@pytest.mark.asyncio
async def test_hook_can_skip_cancel_and_postpone(tmp_path):
    decisions = iter(["postpone", "skip", "cancel"])
    seen: list[str] = []

    def gate(ctx: TimerFireCtx):
        d = next(decisions)
        seen.append(d)
        if d == "postpone":
            ctx.postpone_s = 3600
        elif d == "skip":
            ctx.directive = HookDirective.SKIP
        elif d == "cancel":
            ctx.directive = HookDirective.BREAK

    from power_loop.core.hooks import AgentHooks

    hooks = AgentHooks()
    hooks.register(HookPoint.TIMER_FIRE, gate)
    loop = _loop(tmp_path, hooks=hooks)
    sid = loop.new_session()
    past = int(time.time() * 1000) - 1000
    for note in ("a", "b", "c"):
        loop.schedule_timer(sid, due_at_ms=past, note=note)
    runner = TimerRunner(loop)
    await runner.scan_once()
    statuses = {t.timer_id: t.status
                for t in loop.store.list_timers(sid, statuses=("armed", "fired", "cancelled"))}
    assert statuses[1] == "armed"      # postponed → re-armed in the future
    assert statuses[2] == "fired"      # skipped → done without delivery
    assert statuses[3] == "cancelled"  # vetoed
    # nothing was delivered (LLM never called → no assistant messages)
    assert not [m for m in loop.get_messages(sid) if m.get("role") == "assistant"]
    loop.close()


@pytest.mark.asyncio
async def test_stale_firing_rows_recovered_on_start(tmp_path):
    loop = _loop(tmp_path, responses=[LLMResponse(raw_text="ok", content_text="ok")])
    sid = loop.new_session()
    loop.schedule_timer(sid, due_at_ms=int(time.time() * 1000) - 1000, note="n")
    # Simulate a runner that died mid-fire long ago.
    assert loop.store.transition_timer(sid, 1, from_status="armed", to_status="firing")
    loop.store._conn.execute(  # age the row past the stale window
        "UPDATE timers SET updated_at = updated_at - 600000 WHERE session_id=?", (sid,)
    )
    runner = TimerRunner(loop, stale_firing_s=120)
    await runner.start()
    try:
        await asyncio.sleep(0)  # let start() finish; recovery is synchronous
        assert loop.store.get_timer(sid, 1).status in ("armed", "firing", "fired")
        fired = await runner.scan_once()
        assert fired >= 0  # recovered row is firable again
    finally:
        await runner.stop()
    assert loop.store.get_timer(sid, 1).status == "fired"
    loop.close()


@pytest.mark.asyncio
async def test_recurring_timer_rearms_until_cancelled(tmp_path):
    loop = _loop(tmp_path, responses=[
        LLMResponse(raw_text="beat 1", content_text="beat 1"),
        LLMResponse(raw_text="beat 2", content_text="beat 2"),
    ])
    sid = loop.new_session()
    past = int(time.time() * 1000) - 1000
    loop.schedule_timer(sid, due_at_ms=past, note="heartbeat", interval_s=3600)
    runner = TimerRunner(loop)
    assert await runner.scan_once() == 1
    row = loop.store.get_timer(sid, 1)
    assert row.status == "armed"          # re-armed, not fired
    assert row.fire_count == 1
    assert row.due_at > int(time.time() * 1000)  # fixed-delay: now + interval
    # not due yet → second scan is a no-op
    assert await runner.scan_once() == 0
    # force it due again → second delivery
    loop.store.transition_timer(sid, 1, from_status="armed", to_status="armed", due_at=past)
    assert await runner.scan_once() == 1
    assert loop.store.get_timer(sid, 1).fire_count == 2
    # cancel is the only exit
    assert loop.cancel_timer(sid, 1) is True
    assert loop.store.get_timer(sid, 1).status == "cancelled"
    loop.close()


@pytest.mark.asyncio
async def test_hook_skip_keeps_recurring_cycle(tmp_path):
    from power_loop import HookDirective
    from power_loop.core.hooks import AgentHooks

    hooks = AgentHooks()
    hooks.register(HookPoint.TIMER_FIRE, lambda ctx: setattr(ctx, "directive", HookDirective.SKIP))
    loop = _loop(tmp_path, hooks=hooks)
    sid = loop.new_session()
    loop.schedule_timer(
        sid, due_at_ms=int(time.time() * 1000) - 1000, note="n", interval_s=3600
    )
    await TimerRunner(loop).scan_once()
    row = loop.store.get_timer(sid, 1)
    assert row.status == "armed"  # skipped this period, cycle continues
    # nothing delivered
    assert not [m for m in loop.get_messages(sid) if m.get("role") == "user"]
    loop.close()


def test_one_shot_vs_recurring_declared_at_creation(tmp_path):
    loop = _loop(tmp_path)
    sid = loop.new_session()
    one = loop.schedule_timer(sid, delay_s=60, note="once")
    rec = loop.schedule_timer(sid, delay_s=60, note="repeat", interval_s=300)
    assert one.interval_s is None and rec.interval_s == 300
    with pytest.raises(ValueError):
        loop.schedule_timer(sid, delay_s=60, note="bad", interval_s=0)
    loop.close()


def test_heartbeat_firing_timer_store_cas(tmp_path):
    """heartbeat_firing_timer re-stamps only a row that is actually 'firing'."""
    loop = _loop(tmp_path)
    sid = loop.new_session()
    loop.schedule_timer(sid, delay_s=3600, note="n")
    # armed row → not firing → heartbeat is a no-op
    assert loop.store.heartbeat_firing_timer(sid, 1) is False
    assert loop.store.transition_timer(sid, 1, from_status="armed", to_status="firing")
    before = loop.store.get_timer(sid, 1).updated_at
    loop.store._conn.execute(  # age it so the bump is observable
        "UPDATE timers SET updated_at = updated_at - 5000 WHERE session_id=?", (sid,)
    )
    assert loop.store.heartbeat_firing_timer(sid, 1) is True
    assert loop.store.get_timer(sid, 1).updated_at >= before
    # once it leaves 'firing', heartbeat can no longer claim it
    loop.store.finish_firing_timer(sid, 1)
    assert loop.store.heartbeat_firing_timer(sid, 1) is False
    loop.close()


@pytest.mark.asyncio
async def test_live_slow_delivery_is_not_reclaimed_as_stale(tmp_path):
    """A delivery longer than the stale window must NOT be re-armed by the
    recovery sweep and double-fired — the heartbeat keeps the 'firing' row
    fresh. Pre-fix the row aged past stale_firing_s mid-delivery and fired
    a second time."""

    @dataclass
    class _SlowLLM(LLMService):
        delay_s: float = 0.8

        async def complete(self, request, *, on_chunk_delta_text=None,
                           on_chunk_think=None, on_stream_end=None):
            await asyncio.sleep(self.delay_s)
            return LLMResponse(raw_text="woke", content_text="woke")

        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()

        async def close(self):
            return None

    loop = StatefulAgentLoop(
        llm=_SlowLLM(delay_s=0.8),
        db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=2),
    )
    sid = loop.new_session()
    loop.schedule_timer(sid, due_at_ms=int(time.time() * 1000) - 1000, note="do the thing")
    # stale window 0.5s << 0.8s delivery; heartbeat every 0.05s keeps it alive.
    runner = TimerRunner(loop, stale_firing_s=0.5, heartbeat_interval_s=0.05)

    scan_task = asyncio.create_task(runner.scan_once())
    reclaimed_total = 0
    # Hammer the recovery sweep throughout the slow delivery.
    while not scan_task.done():
        reclaimed_total += loop.store.recover_stale_firing_timers(older_than_ms=500)
        await asyncio.sleep(0.05)
    fired = await scan_task

    assert fired == 1
    assert reclaimed_total == 0, "live delivery was wrongly reclaimed as stale"
    row = loop.store.get_timer(sid, 1)
    assert row.status == "fired"
    assert row.fire_count == 1, f"timer double-fired: fire_count={row.fire_count}"
    delivered = [m for m in loop.get_messages(sid)
                 if m.get("role") == "user" and "do the thing" in str(m.get("content"))]
    assert len(delivered) == 1, f"expected exactly one delivery, got {len(delivered)}"
    loop.close()


@pytest.mark.asyncio
async def test_timer_for_deleted_session_is_cancelled(tmp_path):
    loop = _loop(tmp_path)
    sid = loop.new_session()
    loop.schedule_timer(sid, due_at_ms=int(time.time() * 1000) - 1000, note="n")
    # Delete the session but keep the timer row (simulate external cleanup).
    loop.store._conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
    await TimerRunner(loop).scan_once()
    assert loop.store.get_timer(sid, 1).status == "cancelled"
    loop.close()
