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
async def test_single_call_emits_paired_started_completed_with_usage(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = await runtime_agent(model=_UsageLLM(), store=store, event_bus=bus, config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
        await loop.send("hi")

        started = seen[AgentEventType.LLM_CALL_STARTED]
        completed = seen[AgentEventType.LLM_CALL_COMPLETED]
        assert len(started) == 1 and len(completed) == 1
        assert started[0].call_id == completed[0].call_id  # paired
        c = completed[0]
        assert c.success is True
        assert c.duration_ms >= 0.0
        assert (c.prompt_tokens, c.completion_tokens, c.total_tokens) == (11, 7, 18)  # PER-CALL usage
    finally:
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_stream_events_balanced_across_unknown_result_and_new_requests(runtime_agent) -> None:
    """M-pipeline-runner-1: STREAM_STARTED/COMPLETED must stay balanced across retries. Pre-fix the
    terminal lived in the OUTER finally (once), so N attempts emitted N STARTED but 1 COMPLETED."""
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        counts = {AgentEventType.STREAM_STARTED: 0, AgentEventType.STREAM_COMPLETED: 0}
        for et in counts:
            bus.subscribe(et, lambda e: counts.__setitem__(e.type, counts[e.type] + 1))
        loop = await runtime_agent(model=_UsageLLM(fail_first=2), store=store, event_bus=bus, config=AgentLoopConfig(
                system_prompt="S", max_rounds=1, compactor=None,
                retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0.0, backoff_max=0.0),
            ))
        outcomes = [await loop.send("hi", message_id=f"new-request-{i}") for i in range(3)]
        assert [r.status for r in outcomes] == ['failed', 'failed', 'completed']
        # Unknown results are not retried. Three explicit inputs each emit a balanced pair.
        assert counts[AgentEventType.STREAM_STARTED] == 3
        assert counts[AgentEventType.STREAM_COMPLETED] == 3
    finally:
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_new_requests_after_unknown_result_are_individually_visible(runtime_agent) -> None:
    """Two transient failures then success → three STARTED + three COMPLETED with
    distinct call_ids; the two failures are success=False."""
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = await runtime_agent(model=_UsageLLM(fail_first=2), store=store, event_bus=bus, config=AgentLoopConfig(
                system_prompt="S", max_rounds=1, compactor=None,
                retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0.0, backoff_max=0.0),
            ))
        outcomes = [await loop.send("hi", message_id=f"new-request-{i}") for i in range(3)]
        assert [r.status for r in outcomes] == ['failed', 'failed', 'completed']

        completed = seen[AgentEventType.LLM_CALL_COMPLETED]
        assert len(seen[AgentEventType.LLM_CALL_STARTED]) == 3
        assert len(completed) == 3
        assert len({c.model_attempt_id for c in completed}) == 3       # distinct per attempt
        assert [c.success for c in completed] == [False, False, True]
        assert [c.error_type for c in completed[:2]] == ["RuntimeError", "RuntimeError"]
    finally:
        await loop.aclose()
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
async def test_completed_event_carries_the_prompt_cache_split(runtime_agent) -> None:
    """Without this split a host sees "this round cost 5,296 prompt tokens" and cannot tell
    whether that was billed at full price or 99% served from cache at a tenth of it — which is
    the difference between "trimming history is a big win" and "it is a 20x loss" (editing
    history mid-prefix invalidates the cache from that point on)."""
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = await runtime_agent(model=_CacheUsageLLM(), store=store, event_bus=bus, config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
        await loop.send("hi")

        done = seen[AgentEventType.LLM_CALL_COMPLETED][-1]
        assert done.prompt_tokens == 5296
        assert done.prompt_cached_tokens == 5248
        assert done.prompt_cache_miss_tokens == 48
    finally:
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_cache_fields_are_none_when_the_provider_omits_them(runtime_agent) -> None:
    # None means "not reported", which must stay distinguishable from a real zero (= nothing
    # was cached) — otherwise a provider without cache reporting looks like a 0% hit rate.
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen = _collect(bus)
        loop = await runtime_agent(model=_UsageLLM(), store=store, event_bus=bus, config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
        await loop.send("hi")
        done = seen[AgentEventType.LLM_CALL_COMPLETED][-1]
        assert done.prompt_tokens == 11
        assert done.prompt_cached_tokens is None
        assert done.prompt_cache_miss_tokens is None
    finally:
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_factory_event_bus_keeps_concurrent_session_output_separate(tmp_path):
    import asyncio

    from power_loop.agent.runtime_store import now_ms
    from tests.runtime_fixture import RunningAgent

    bus = AgentEventBus()
    delivered = []
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, delivered.append)
    entered, emit, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Model(_UsageLLM):
        async def complete(self, request, **callbacks):
            self._calls += 1
            if self._calls == 2:
                entered.set()
            await emit.wait()
            text = request.messages[-1]['content']
            callbacks['on_chunk_delta_text'](text)
            await finish.wait()
            return LLMResponse(raw_text=text)

    store = await SessionStore.open(str(tmp_path / 'bus.db'))
    runner = await RunningAgent(store, Model(), event_bus=bus, capacity=2).start()
    streams = []
    try:
        other = await runner.runtime.open_session(runner.descriptor, actor=runner.actor)
        receipts = [await runner.runtime.client.send(target, body, message_id=body, actor=runner.actor)
                    for target, body in [(runner.target, 'alpha'), (other, 'beta')]]
        await asyncio.wait_for(entered.wait(), 5)
        streams = [await runner.runtime.client.subscribe_output(r.input_ref, actor=runner.actor) for r in receipts]
        async def delta(stream):
            async for item in stream:
                if item.kind == 'delta':
                    return item.payload['text']
        emit.set()
        assert await asyncio.wait_for(asyncio.gather(*(delta(s) for s in streams)), 5) == ['alpha', 'beta']
        finish.set()
        for r in receipts:
            result = await runner.runtime.client.wait(r.input_ref, actor=runner.actor, deadline_ms=now_ms()+5000)
            assert result.status == 'completed'
        await runner.drain()
        assert len(delivered) == 2
        assert sum(len(entries) for entries in bus._handlers.values()) == 1  # only the host subscription survives
    finally:
        emit.set()
        finish.set()
        for stream in streams:
            await stream.aclose()
        await runner.aclose()
        await store.close()
