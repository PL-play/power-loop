"""Coverage: TimerRunner recurring re-arm / due boundary / one-shot normalization + retry classification & backoff (probed clean)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_recurring_rearm_collapses_missed_periods(tmp_path):
    import time
    from dataclasses import dataclass, field

    from power_loop import AgentLoopConfig, StatefulAgentLoop, TimerRunner
    from power_loop._vendor.llm_client.interface import LLMResponse, LLMService, LLMStreamChunk

    @dataclass
    class _Scripted(LLMService):
        responses: list = field(default_factory=list)
        _idx: int = 0
        async def complete(self, request, *, on_chunk_delta_text=None, on_chunk_think=None, on_stream_end=None):
            if self._idx >= len(self.responses):
                return LLMResponse(raw_text="done", content_text="done")
            r = self.responses[self._idx]
            self._idx += 1
            return r
        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()
        async def close(self):
            return None

    loop = StatefulAgentLoop(
        llm=_Scripted(responses=[LLMResponse(raw_text="x", content_text="x")]),
        db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=4),
    )
    sid = loop.new_session()
    interval = 100
    way_past = int(time.time() * 1000) - 10 * 3600 * 1000  # overdue by 10h
    loop.schedule_timer(sid, due_at_ms=way_past, note="hb", interval_s=interval)
    runner = TimerRunner(loop)
    assert await runner.scan_once() == 1
    row = loop.store.get_timer(sid, 1)
    now = int(time.time() * 1000)
    assert row.status == "armed"
    assert row.due_at > now, "recurring re-arm landed in the past (catch-up storm)"
    assert abs(row.due_at - (now + interval * 1000)) < 5000
    assert await runner.scan_once() == 0  # not due again -> no storm
    loop.close()

def test_timer_due_at_boundary_is_inclusive(tmp_path):
    import time

    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import LLMResponse, LLMService, LLMStreamChunk

    class _NoLLM(LLMService):
        async def complete(self, request, **k):
            return LLMResponse(raw_text="x", content_text="x")
        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()
        async def close(self):
            return None

    loop = StatefulAgentLoop(
        llm=_NoLLM(), db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=1),
    )
    sid = loop.new_session()
    now = int(time.time() * 1000)
    loop.schedule_timer(sid, due_at_ms=now, note="exact")
    assert len(loop.store.due_timers(now=now)) == 1
    assert len(loop.store.due_timers(now=now - 1)) == 0  # 1ms early -> not due
    loop.close()

def test_interval_zero_normalizes_to_one_shot(tmp_path):
    import time

    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import LLMResponse, LLMService, LLMStreamChunk

    class _NoLLM(LLMService):
        async def complete(self, request, **k):
            return LLMResponse(raw_text="x", content_text="x")
        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()
        async def close(self):
            return None

    loop = StatefulAgentLoop(
        llm=_NoLLM(), db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=1),
    )
    sid = loop.new_session()
    row = loop.store.create_timer(sid, due_at=int(time.time() * 1000), note="z", interval_s=0)
    assert row.interval_s is None
    assert loop.store.get_timer(sid, row.timer_id).interval_s is None
    loop.close()

@pytest.mark.asyncio
async def test_non_retryable_error_not_retried_and_not_wrapped():
    from power_loop import CancellationToken, LLMRetryPolicy
    from power_loop.runtime.retry import with_retry
    calls = 0
    async def call():
        nonlocal calls
        calls += 1
        raise KeyError("not retryable")
    policy = LLMRetryPolicy(max_attempts=5, backoff_initial=0.001, backoff_max=0.001, retry_on=(ValueError,))
    with pytest.raises(KeyError):
        await with_retry(call, policy=policy, token=CancellationToken.never())
    assert calls == 1

def test_backoff_growth_and_cap():
    from power_loop import LLMRetryPolicy
    p = LLMRetryPolicy(max_attempts=100, backoff_initial=1.0, backoff_max=8.0)
    assert p.backoff_for_attempt(0) == 0.0
    assert p.backoff_for_attempt(1) == 1.0
    assert p.backoff_for_attempt(2) == 2.0
    assert p.backoff_for_attempt(3) == 4.0
    assert p.backoff_for_attempt(4) == 8.0
    for a in range(5, 60):
        assert p.backoff_for_attempt(a) == 8.0  # never exceeds cap
