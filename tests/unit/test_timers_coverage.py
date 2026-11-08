"""Due boundary and fixed-delay rearm through the current default timer driver."""
import time

import pytest

from .test_job_registry_192 import registry_store
from .test_timers import timers

__all__ = ["registry_store"]
pytestmark = pytest.mark.unit

async def test_recurring_rearm_collapses_missed_periods(registry_store):
    store, _ = registry_store
    async with timers(store) as (runtime, target, actor, source):
        interval = 100
        row = await runtime.timer_backend.create(source, due_at_ms=int(time.time()*1000)-36000000,
                                                  note="hb", interval_s=interval)
        assert await runtime.timer_driver.scan_once() == 1
        row = await store.get_timer(target.session_id, row.timer_id)
        now = int(time.time()*1000)
        assert row.status == "armed" and row.due_at > now
        assert abs(row.due_at-(now+interval*1000)) < 5000
        assert await runtime.timer_driver.scan_once() == 0

async def test_timer_due_at_boundary_is_inclusive(registry_store):
    store, _ = registry_store
    async with timers(store) as (runtime, target, actor, source):
        now = int(time.time()*1000)
        await runtime.timer_backend.create(source, due_at_ms=now, note="exact")
        assert len(await store.due_timers(now=now)) == 1
        assert len(await store.due_timers(now=now-1)) == 0

async def test_single_timer_and_invalid_period_have_explicit_contract(registry_store):
    from power_loop import RuntimeRequestError
    store, _ = registry_store
    async with timers(store) as (runtime, target, actor, source):
        for interval in (0, -1):
            with pytest.raises(RuntimeRequestError, match="invalid_options"):
                await runtime.timer_backend.create(source, delay_s=3600, note="bad", interval_s=interval)
        row = await runtime.timer_backend.create(source, delay_s=3600, note="single", interval_s=None)
        assert row.interval_s is None
        assert (await store.get_timer(target.session_id, row.timer_id)).interval_s is None

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
