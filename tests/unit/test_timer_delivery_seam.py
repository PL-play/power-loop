"""S5 (3.14): ``TimerRunner(delivery=...)`` host delivery seam +
``claim_wake`` / ``parse_workflow_wake`` public helpers."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    HookDirective,
    HookPoint,
    StatefulAgentLoop,
    TimerFireCtx,
    TimerRunner,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.workflow import claim_wake, parse_workflow_wake
from power_loop.workflow.journal import JOURNAL_PREFIX, seed
from power_loop.workflow.runner import _wake_note

pytestmark = pytest.mark.asyncio


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    calls: int = 0
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
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


def _loop(tmp_path, hooks=None) -> tuple[StatefulAgentLoop, _Scripted]:
    llm = _Scripted()
    return StatefulAgentLoop(
        llm=llm,
        db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=3),
        hooks=hooks,
    ), llm


async def _arm_due_timer(loop: StatefulAgentLoop, sid: str, note: str = "poke") -> None:
    await loop.schedule_timer(sid, due_at_ms=int(time.time() * 1000) - 1000, note=note)


async def test_custom_delivery_replaces_follow_up(tmp_path) -> None:
    loop, llm = _loop(tmp_path)
    sid = await loop.new_session()
    await _arm_due_timer(loop, sid, note="route me")
    delivered: list[TimerFireCtx] = []

    async def delivery(ctx: TimerFireCtx) -> None:
        delivered.append(ctx)

    runner = TimerRunner(loop, delivery=delivery)
    assert await runner.scan_once() == 1
    assert len(delivered) == 1
    ctx = delivered[0]
    assert ctx.session_id == sid and "route me" in ctx.message
    # The host took delivery: no follow_up happened (LLM never called) and the
    # one-shot row is done.
    assert llm.calls == 0
    assert (await loop.store.get_timer(sid, 1)).status == "fired"
    await loop.aclose()


async def test_custom_delivery_sees_hook_rewritten_message(tmp_path) -> None:
    from power_loop.core.hooks import AgentHooks

    hooks = AgentHooks()

    def rewrite(ctx: TimerFireCtx) -> None:
        ctx.message = "REWRITTEN: " + ctx.note

    hooks.register(HookPoint.TIMER_FIRE, rewrite)
    loop, _llm = _loop(tmp_path, hooks=hooks)
    sid = await loop.new_session()
    await _arm_due_timer(loop, sid, note="original")
    delivered: list[str] = []

    async def delivery(ctx: TimerFireCtx) -> None:
        delivered.append(ctx.message)

    await TimerRunner(loop, delivery=delivery).scan_once()
    assert delivered == ["REWRITTEN: original"]
    await loop.aclose()


async def test_hook_skip_suppresses_custom_delivery(tmp_path) -> None:
    from power_loop.core.hooks import AgentHooks

    hooks = AgentHooks()

    def veto(ctx: TimerFireCtx) -> None:
        ctx.directive = HookDirective.SKIP

    hooks.register(HookPoint.TIMER_FIRE, veto)
    loop, _llm = _loop(tmp_path, hooks=hooks)
    sid = await loop.new_session()
    await _arm_due_timer(loop, sid)
    delivered: list[Any] = []

    async def delivery(ctx: TimerFireCtx) -> None:
        delivered.append(ctx)

    await TimerRunner(loop, delivery=delivery).scan_once()
    assert delivered == []  # veto point runs BEFORE the host delivery
    assert (await loop.store.get_timer(sid, 1)).status == "fired"  # skipped → done
    await loop.aclose()


async def test_failing_delivery_rearms_and_refires(tmp_path) -> None:
    loop, _llm = _loop(tmp_path)
    sid = await loop.new_session()
    await _arm_due_timer(loop, sid)
    attempts = {"n": 0}

    async def delivery(ctx: TimerFireCtx) -> None:
        attempts["n"] += 1
        raise RuntimeError("host pipeline down")

    runner = TimerRunner(loop, delivery=delivery)
    await runner.scan_once()
    assert attempts["n"] == 1
    row = await loop.store.get_timer(sid, 1)
    assert row.status == "armed"  # re-armed (+30s) → re-fires: at-least-once
    assert row.due_at > int(time.time() * 1000)
    await loop.aclose()


async def test_default_delivery_unchanged(tmp_path) -> None:
    loop, llm = _loop(tmp_path)
    sid = await loop.new_session()
    await _arm_due_timer(loop, sid, note="classic")
    assert await TimerRunner(loop).scan_once() == 1
    assert llm.calls == 1  # built-in follow_up path ran the session
    await loop.aclose()


# ── claim_wake / parse_workflow_wake ─────────────────────────────────────


async def test_parse_workflow_wake() -> None:
    assert parse_workflow_wake(_wake_note("abc123", "completed")) == "abc123"
    assert parse_workflow_wake(f"{JOURNAL_PREFIX}xyz failed — boom") == "xyz"
    assert parse_workflow_wake("an ordinary reminder") is None
    assert parse_workflow_wake(None) is None
    assert parse_workflow_wake(JOURNAL_PREFIX) is None  # prefix but no run id


async def test_claim_wake_first_true_then_false(tmp_path) -> None:
    loop, _llm = _loop(tmp_path)
    sid = await loop.new_session()
    store = await loop.ensure_store()
    await seed(store, sid, "run1", "wf")
    assert await claim_wake(store, sid, "run1") is True
    assert await claim_wake(store, sid, "run1") is False  # second delivery deduped
    await loop.aclose()


async def test_claim_wake_no_journal_is_true(tmp_path) -> None:
    """Nothing to dedupe against → deliver (don't strand the parent)."""
    loop, _llm = _loop(tmp_path)
    sid = await loop.new_session()
    store = await loop.ensure_store()
    assert await claim_wake(store, sid, "ghost-run") is True
    await loop.aclose()
