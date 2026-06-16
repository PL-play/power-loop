"""Real-LLM smoke for M1.1 retry / degraded paths.

We don't rely on the cloud LLM actually failing — that's flaky and slow.
Instead we wrap the real LLM service in a thin adapter that **forces**
``raise RuntimeError`` on the first N calls, then delegates to the real
service. This proves that:

1. The retry policy correctly counts attempts against a real ``await
   llm.complete(...)`` call (catches anything outside the framework's
   own exceptions and treats it as retryable).
2. The real LLM still returns a meaningful answer when retry eventually
   succeeds — i.e. our retry wrapper does not corrupt request args or
   the stream callback chain.
3. The degraded path returns the synthesized "[degraded: …]" message
   when every attempt fails, *without* calling the real LLM service.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    LLMRetryPolicy,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop._vendor.llm_client.interface import LLMService

from ._llm import make_llm


class _FlakyWrap(LLMService):
    """Wraps a real LLMService; raises ``RuntimeError`` for the first
    ``fail_first`` calls, then delegates."""

    def __init__(self, inner: LLMService, *, fail_first: int) -> None:
        self.inner = inner
        self.fail_first = fail_first
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(f"injected transient failure #{self.calls}")
        return await self.inner.complete(request, **kwargs)

    async def stream(self, request):
        return await self.inner.stream(request)

    async def close(self):
        await self.inner.close()


@pytest.mark.asyncio
async def test_real_llm_retries_through_transient_then_completes() -> None:
    """Two synthetic failures, third real attempt succeeds."""
    inner = make_llm(max_tokens=128, temperature=0.0)
    llm = _FlakyWrap(inner, fail_first=2)
    bus = AgentEventBus()
    events: list = []
    bus.subscribe(None, lambda e: events.append(e))

    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=llm, store=store, event_bus=bus,
            config=AgentLoopConfig(
                system_prompt="Reply in one short sentence.",
                max_rounds=1, max_tokens=128, temperature=0.0,
                compactor=None,
                retry_policy=LLMRetryPolicy(
                    max_attempts=4, backoff_initial=0.05, backoff_max=0.1,
                    total_timeout=30.0,
                ),
            ),
        )
        sid = await loop.new_session()
        r = await loop.send("Reply with the single word OK.", session_id=sid)
        assert r.status == "completed", r
        assert llm.calls == 3
        retry_events = [e for e in events if e.type == AgentEventType.LLM_RETRY_ATTEMPTED]
        assert len(retry_events) == 2
        # Sanity: the real LLM call returned non-empty text.
        assert r.final_text.strip()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_real_llm_path_degrades_when_all_attempts_fail() -> None:
    """All attempts fail synthetically — never hits the real network."""
    inner = make_llm(max_tokens=128, temperature=0.0)
    llm = _FlakyWrap(inner, fail_first=1_000_000)  # always fail
    bus = AgentEventBus()
    events: list = []
    bus.subscribe(None, lambda e: events.append(e))

    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=llm, store=store, event_bus=bus,
            config=AgentLoopConfig(
                system_prompt="x",
                max_rounds=1, compactor=None,
                retry_policy=LLMRetryPolicy(
                    max_attempts=2, backoff_initial=0.01, backoff_max=0.02,
                    total_timeout=5.0,
                ),
            ),
        )
        sid = await loop.new_session()
        r = await loop.send("anything", session_id=sid)
        assert r.status == "degraded"
        assert "degraded" in r.final_text
        assert any(e.type == AgentEventType.LLM_DEGRADED for e in events)
        # Real network was never reached past our wrap.
        assert llm.calls == 2
    finally:
        await store.close()
