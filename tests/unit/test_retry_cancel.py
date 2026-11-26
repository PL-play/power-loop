"""Unit tests for M1.1 — LLMRetryPolicy / CancellationToken / degraded path.

Uses a fake LLM whose ``complete`` is parameterized by a script of
``Exception | LLMResponse`` items, returning each in turn. Tests assert
on three control-flow paths:

1. Transient failures retried then succeed → status="completed".
2. Persistent failures exhaust retries → status="degraded", new events fire.
3. External cancel between attempts → status="cancelled", new events fire.

These are pure control-flow tests; no real LLM is used.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    CancellationToken,
    LLMNonRetryable,
    LLMRetryExhausted,
    LLMRetryPolicy,
    LLMTimeout,
)
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.contracts.errors import CancellationRequested
from power_loop.runtime.retry import with_retry

# ── Fake LLM ─────────────────────────────────────────────────────────────


@dataclass
class _ScriptedLLM:
    """Returns canned responses or raises canned exceptions, one per call."""

    script: list[object] = field(default_factory=list)
    calls: int = 0

    async def complete(self, request, **kwargs):
        idx = self.calls
        self.calls += 1
        if idx >= len(self.script):
            return LLMResponse(raw_text="fallback-done")
        item = self.script[idx]
        if isinstance(item, BaseException):
            raise item
        return item

    async def stream(self, request):
        raise NotImplementedError

    async def close(self):
        pass


# ── with_retry direct tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_with_retry_succeeds_after_transient() -> None:
    attempts = 0

    async def call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("boom")
        return "ok"

    policy = LLMRetryPolicy(max_attempts=5, backoff_initial=0.001, backoff_max=0.002)
    result = await with_retry(call, policy=policy, token=CancellationToken.never())
    assert result == "ok"
    assert attempts == 3


@pytest.mark.asyncio
async def test_with_retry_exhausted_raises_with_cause() -> None:
    async def call() -> str:
        raise ValueError("nope")

    policy = LLMRetryPolicy(max_attempts=2, backoff_initial=0.001, backoff_max=0.002)
    with pytest.raises(LLMRetryExhausted) as excinfo:
        await with_retry(call, policy=policy, token=CancellationToken.never())
    assert excinfo.value.attempts == 2
    assert isinstance(excinfo.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_with_retry_total_timeout_raises_llmtimeout() -> None:
    async def call() -> str:
        raise RuntimeError("slow")

    policy = LLMRetryPolicy(
        max_attempts=10, backoff_initial=0.05, backoff_max=0.05, total_timeout=0.02,
    )
    t0 = time.monotonic()
    with pytest.raises(LLMTimeout):
        await with_retry(call, policy=policy, token=CancellationToken.never())
    assert time.monotonic() - t0 < 0.5


@pytest.mark.asyncio
async def test_with_retry_cancel_between_attempts() -> None:
    token = CancellationToken()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        token.cancel("user_stop")  # cancel before the next retry sleep
        raise RuntimeError("boom")

    policy = LLMRetryPolicy(max_attempts=5, backoff_initial=0.001, backoff_max=0.001)
    with pytest.raises(CancellationRequested):
        await with_retry(call, policy=policy, token=token)
    assert calls == 1


# ── CancellationToken shape tests ────────────────────────────────────────


def test_token_from_threading_event() -> None:
    ev = threading.Event()
    tok = CancellationToken.from_any(ev)
    assert not tok.is_cancelled()
    ev.set()
    assert tok.is_cancelled()


def test_token_from_asyncio_event() -> None:
    ev = asyncio.Event()
    tok = CancellationToken.from_any(ev)
    assert not tok.is_cancelled()
    ev.set()
    assert tok.is_cancelled()


def test_token_from_callable() -> None:
    flag = {"v": False}
    tok = CancellationToken.from_any(lambda: flag["v"])
    assert not tok.is_cancelled()
    flag["v"] = True
    assert tok.is_cancelled()


def test_token_from_none_is_never() -> None:
    tok = CancellationToken.from_any(None)
    assert not tok.is_cancelled()
    tok.raise_if_cancelled()


def test_callable_raising_treated_as_not_cancelled() -> None:
    def boom() -> bool:
        raise RuntimeError("user fn broke")
    tok = CancellationToken.from_any(boom)
    assert tok.is_cancelled() is False


# With max_attempts=1, an unknown model failure is not retried and stays visible,
# while a subsequent user input can continue normally.
@pytest.mark.asyncio
async def test_pipeline_unknown_failure_is_visible_and_new_input_can_continue(runtime_agent):
    llm = _ScriptedLLM(script=[RuntimeError("unconfirmed call"), LLMResponse(raw_text="new input succeeded")])
    events = []
    bus = AgentEventBus()
    bus.subscribe(None, events.append)
    loop = await runtime_agent(model=llm, event_bus=bus, config=AgentLoopConfig(context_budget_tokens=10**9,
        retry_policy=LLMRetryPolicy(max_attempts=1, backoff_initial=0)))
    result = await loop.send("first input")
    assert result.status == "failed" and "unconfirmed call" in result.error
    assert llm.calls == 1
    assert not any(e.type == AgentEventType.LLM_RETRY_ATTEMPTED for e in events)
    assert (await loop.send("new input")).final_text == "new input succeeded"
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_pipeline_cancel_status_and_event(runtime_agent):
    from power_loop.agent.runtime_store import now_ms
    entered = asyncio.Event()
    class Blocking(_ScriptedLLM):
        async def complete(self, request, **kwargs):
            self.calls += 1
            entered.set()
            await asyncio.Event().wait()
    events = []
    bus = AgentEventBus()
    bus.subscribe(None, events.append)
    llm = Blocking()
    loop = await runtime_agent(model=llm, event_bus=bus)
    receipt = await loop.runtime.client.send(loop.target, "hi", message_id="cancel", actor=loop.actor)
    await asyncio.wait_for(entered.wait(), 5)
    status = await loop.runtime.client.inspect(receipt.input_ref, actor=loop.actor)
    await loop.runtime.client.cancel(status.target, actor=loop.actor)
    result = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+10000)
    await loop.finish(receipt.input_ref, failed=True)
    assert result.status == "cancelled" and llm.calls == 1
    assert any(e.type == AgentEventType.LOOP_CANCELLED for e in events)


@pytest.mark.asyncio
async def test_backoff_sleep_is_cancellable() -> None:
    """Cancelling during a long backoff raises promptly, not after the full sleep
    (the backoff used to be a plain asyncio.sleep that ignored the token)."""
    policy = LLMRetryPolicy(max_attempts=5, backoff_initial=10.0, backoff_max=30.0)
    tok = CancellationToken()

    async def call() -> str:
        raise RuntimeError("boom")  # always retryable → enters the backoff sleep

    task = asyncio.ensure_future(with_retry(call, policy=policy, token=tok))
    await asyncio.sleep(0.2)  # first attempt fails → now in the 10s backoff
    tok.cancel()
    t0 = time.monotonic()
    with pytest.raises(CancellationRequested):
        await task
    assert time.monotonic() - t0 < 2.0  # promptly cancelled, not ~10s later


def test_backoff_does_not_overflow_for_huge_attempt() -> None:
    """A pathological max_attempts must not raise OverflowError computing 2**exp."""
    policy = LLMRetryPolicy(max_attempts=5000, backoff_initial=1.0, backoff_max=8.0)
    assert policy.backoff_for_attempt(2000) == 8.0  # clamped, no OverflowError
    assert policy.backoff_for_attempt(1) == 1.0


# ── R08: inner TimeoutError vs outer total_timeout budget ────────────────


@pytest.mark.asyncio
async def test_with_retry_inner_timeout_retries_and_succeeds() -> None:
    """An inner TimeoutError raised by call() before total_timeout expires must be
    retried according to policy instead of immediately turning into LLMTimeout."""
    calls = 0
    retried: list[tuple[int, type[BaseException]]] = []

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("per-attempt timeout")
        return "ok"

    policy = LLMRetryPolicy(
        total_timeout=60.0,
        max_attempts=3,
        backoff_initial=0.001,
        backoff_max=0.001,
    )
    result = await with_retry(
        call,
        policy=policy,
        token=CancellationToken.never(),
        on_retry=lambda attempt, exc, _sleep: retried.append((attempt, type(exc))),
    )
    assert result == "ok"
    assert calls == 2
    assert retried == [(0, TimeoutError)]


@pytest.mark.asyncio
async def test_with_retry_real_total_timeout_raises_llmtimeout() -> None:
    """A hung call that outlasts total_timeout raises LLMTimeout."""
    calls = 0

    async def hang() -> str:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        return "unreachable"

    policy = LLMRetryPolicy(
        total_timeout=0.05,
        max_attempts=3,
        backoff_initial=0.001,
        backoff_max=0.001,
    )
    t0 = time.monotonic()
    with pytest.raises(LLMTimeout) as excinfo:
        await with_retry(hang, policy=policy, token=CancellationToken.never())
    assert time.monotonic() - t0 < 1.0
    assert calls == 1
    assert excinfo.value.attempts == 1
    assert excinfo.value.total_timeout == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_with_retry_direct_task_cancel_propagates_cancelled_error() -> None:
    """Direct task cancellation while awaiting call() under total_timeout stays CancelledError."""
    entered = asyncio.Event()
    calls = 0

    async def hang() -> str:
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    policy = LLMRetryPolicy(total_timeout=60.0, max_attempts=3)
    task = asyncio.create_task(with_retry(hang, policy=policy, token=CancellationToken.never()))
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == 1


@pytest.mark.asyncio
async def test_with_retry_inner_timeout_respects_retry_on_exclusion() -> None:
    """When retry_on excludes TimeoutError, an inner TimeoutError propagates directly,
    while actual outer total_timeout expiration still raises LLMTimeout."""
    calls = 0

    async def immediate_timeout() -> str:
        nonlocal calls
        calls += 1
        raise TimeoutError("inner transport timeout")

    policy = LLMRetryPolicy(
        total_timeout=60.0,
        max_attempts=3,
        backoff_initial=0.001,
        backoff_max=0.001,
        retry_on=(ValueError,),
    )
    with pytest.raises(TimeoutError, match="inner transport timeout"):
        await with_retry(immediate_timeout, policy=policy, token=CancellationToken.never())
    assert calls == 1

    async def hang() -> str:
        await asyncio.Event().wait()
        return "unreachable"

    short_policy = LLMRetryPolicy(
        total_timeout=0.05,
        max_attempts=3,
        retry_on=(ValueError,),
    )
    with pytest.raises(LLMTimeout):
        await with_retry(hang, policy=short_policy, token=CancellationToken.never())


@pytest.mark.asyncio
async def test_with_retry_inner_timeout_respects_give_up_on_and_max_attempts() -> None:
    """Inner TimeoutError obeys give_up_on and max_attempts exhaustion."""
    calls = 0

    async def always_timeout() -> str:
        nonlocal calls
        calls += 1
        raise TimeoutError("per-attempt timeout")

    give_up_policy = LLMRetryPolicy(
        total_timeout=60.0,
        max_attempts=3,
        backoff_initial=0.001,
        backoff_max=0.001,
        give_up_on=lambda exc: isinstance(exc, TimeoutError),
    )
    with pytest.raises(LLMNonRetryable) as non_retryable:
        await with_retry(always_timeout, policy=give_up_policy, token=CancellationToken.never())
    assert calls == 1
    assert non_retryable.value.attempts == 1
    assert isinstance(non_retryable.value.last_error, TimeoutError)

    calls = 0
    exhaust_policy = LLMRetryPolicy(
        total_timeout=60.0,
        max_attempts=3,
        backoff_initial=0.001,
        backoff_max=0.001,
    )
    with pytest.raises(LLMRetryExhausted) as exhausted:
        await with_retry(always_timeout, policy=exhaust_policy, token=CancellationToken.never())
    assert calls == 3
    assert exhausted.value.attempts == 3
    assert isinstance(exhausted.value.last_error, TimeoutError)
