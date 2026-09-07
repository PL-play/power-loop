"""H4.1: per-call LLM events (LLM_CALL_STARTED / LLM_CALL_COMPLETED).

A subscriber must see one STARTED + one COMPLETED per attempt (paired by call_id),
with per-call latency + per-call (not cumulative) token usage, so retries are
individually visible.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.retry import LLMRetryPolicy

pytestmark = pytest.mark.unit


@dataclass
class _UsageLLM(LLMService):
    """Returns a reply with per-call token usage; optionally fails the first N calls."""
    fail_first: int = 0
    _calls: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text: Callable | None = None,
                       on_chunk_think: Callable | None = None, on_stream_end: Callable | None = None) -> LLMResponse:
        self._calls += 1
        if self._calls <= self.fail_first:
            raise RuntimeError("transient")
        return LLMResponse(
            raw_text="ok",
            token_usage=LLMTokenUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _collect(bus: AgentEventBus) -> dict[AgentEventType, list]:
    seen: dict[AgentEventType, list] = {
        AgentEventType.LLM_CALL_STARTED: [], AgentEventType.LLM_CALL_COMPLETED: [],
    }
    for et in seen:
        bus.subscribe(et, lambda e: seen[e.type].append(e.data))
    return seen


@pytest.mark.asyncio
async def test_single_call_emits_paired_started_completed_with_usage() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = StatefulAgentLoop(
            llm=_UsageLLM(), store=store, event_bus=bus,
            config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
        )
        await loop.send("hi", session_id=await loop.new_session())

        started = seen[AgentEventType.LLM_CALL_STARTED]
        completed = seen[AgentEventType.LLM_CALL_COMPLETED]
        assert len(started) == 1 and len(completed) == 1
        assert started[0].call_id == completed[0].call_id  # paired
        c = completed[0]
        assert c.success is True
        assert c.duration_ms >= 0.0
        assert (c.prompt_tokens, c.completion_tokens, c.total_tokens) == (11, 7, 18)  # PER-CALL usage
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_stream_events_balanced_across_retries() -> None:
    """M-pipeline-runner-1: STREAM_STARTED/COMPLETED must stay balanced across retries. Pre-fix the
    terminal lived in the OUTER finally (once), so N attempts emitted N STARTED but 1 COMPLETED."""
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        counts = {AgentEventType.STREAM_STARTED: 0, AgentEventType.STREAM_COMPLETED: 0}
        for et in counts:
            bus.subscribe(et, lambda e: counts.__setitem__(e.type, counts[e.type] + 1))
        loop = StatefulAgentLoop(
            llm=_UsageLLM(fail_first=2), store=store, event_bus=bus,
            config=AgentLoopConfig(
                system_prompt="S", max_rounds=1, compactor=None,
                retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0.0, backoff_max=0.0),
            ),
        )
        await loop.send("hi", session_id=await loop.new_session())
        # 2 failures + 1 success = 3 attempts → one STARTED and one COMPLETED each (pre-fix: 3 vs 1).
        assert counts[AgentEventType.STREAM_STARTED] == 3
        assert counts[AgentEventType.STREAM_COMPLETED] == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_retries_are_individually_visible() -> None:
    """Two transient failures then success → three STARTED + three COMPLETED with
    distinct call_ids; the two failures are success=False."""
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = StatefulAgentLoop(
            llm=_UsageLLM(fail_first=2), store=store, event_bus=bus,
            config=AgentLoopConfig(
                system_prompt="S", max_rounds=1, compactor=None,
                retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0.0, backoff_max=0.0),
            ),
        )
        await loop.send("hi", session_id=await loop.new_session())

        completed = seen[AgentEventType.LLM_CALL_COMPLETED]
        assert len(seen[AgentEventType.LLM_CALL_STARTED]) == 3
        assert len(completed) == 3
        assert len({c.call_id for c in completed}) == 3       # distinct per attempt
        assert [c.success for c in completed] == [False, False, True]
        assert [c.error_type for c in completed[:2]] == ["RuntimeError", "RuntimeError"]
    finally:
        await store.close()


@dataclass
class _CacheUsageLLM(LLMService):
    """Reports the prompt-cache split the way DeepSeek/OpenAI do."""

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text: Callable | None = None,
                       on_chunk_think: Callable | None = None,
                       on_stream_end: Callable | None = None) -> LLMResponse:
        return LLMResponse(
            raw_text="ok",
            token_usage=LLMTokenUsage(
                prompt_tokens=5296, completion_tokens=7, total_tokens=5303,
                prompt_cached_tokens=5248, prompt_cache_miss_tokens=48,
            ),
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()


@pytest.mark.asyncio
async def test_completed_event_carries_the_prompt_cache_split() -> None:
    """Without this split a host sees "this round cost 5,296 prompt tokens" and cannot tell
    whether that was billed at full price or 99% served from cache at a tenth of it — which is
    the difference between "trimming history is a big win" and "it is a 20x loss" (editing
    history mid-prefix invalidates the cache from that point on)."""
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = StatefulAgentLoop(
            llm=_CacheUsageLLM(), store=store, event_bus=bus,
            config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
        )
        await loop.send("hi", session_id=await loop.new_session())

        done = seen[AgentEventType.LLM_CALL_COMPLETED][-1]
        assert done.prompt_tokens == 5296
        assert done.prompt_cached_tokens == 5248
        assert done.prompt_cache_miss_tokens == 48
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cache_fields_are_none_when_the_provider_omits_them() -> None:
    # None means "not reported", which must stay distinguishable from a real zero (= nothing
    # was cached) — otherwise a provider without cache reporting looks like a 0% hit rate.
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = StatefulAgentLoop(
            llm=_UsageLLM(), store=store, event_bus=bus,
            config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
        )
        await loop.send("hi", session_id=await loop.new_session())
        done = seen[AgentEventType.LLM_CALL_COMPLETED][-1]
        assert done.prompt_tokens == 11
        assert done.prompt_cached_tokens is None
        assert done.prompt_cache_miss_tokens is None
    finally:
        await store.close()
