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

from llm_client.interface import LLMResponse
from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    CancellationToken,
    LLMRetryExhausted,
    LLMRetryPolicy,
    LLMTimeout,
    SessionStore,
    StatefulAgentLoop,
)
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


def _new_loop(*, llm, store, bus, retry=None, max_rounds=1) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=llm, store=store, event_bus=bus,
        config=AgentLoopConfig(
            system_prompt="x", max_rounds=max_rounds, compactor=None,
            retry_policy=retry,
        ),
    )


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


# ── End-to-end through the pipeline (fake LLM) ──────────────────────────


@pytest.mark.asyncio
async def test_pipeline_retry_then_succeed_emits_retry_event() -> None:
    store = SessionStore.open(":memory:")
    try:
        events: list = []
        bus = AgentEventBus()
        bus.subscribe(None, lambda e: events.append(e))
        llm = _ScriptedLLM(script=[
            RuntimeError("first"),
            RuntimeError("second"),
            LLMResponse(raw_text="hello"),
        ])
        loop = _new_loop(
            llm=llm, store=store, bus=bus,
            retry=LLMRetryPolicy(max_attempts=3, backoff_initial=0.001, backoff_max=0.001),
        )
        sid = loop.new_session()
        r = await loop.send("hi", session_id=sid)
        assert r.status == "completed"
        assert r.final_text == "hello"
        assert llm.calls == 3
        retry_events = [e for e in events if e.type == AgentEventType.LLM_RETRY_ATTEMPTED]
        assert len(retry_events) == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_pipeline_retry_exhausted_degrades() -> None:
    store = SessionStore.open(":memory:")
    try:
        events: list = []
        bus = AgentEventBus()
        bus.subscribe(None, lambda e: events.append(e))
        llm = _ScriptedLLM(script=[RuntimeError("a"), RuntimeError("b")])
        loop = _new_loop(
            llm=llm, store=store, bus=bus, max_rounds=2,
            retry=LLMRetryPolicy(max_attempts=2, backoff_initial=0.001, backoff_max=0.001),
        )
        sid = loop.new_session()
        r = await loop.send("hi", session_id=sid)
        assert r.status == "degraded"
        assert "degraded" in r.final_text
        assert any(e.type == AgentEventType.LLM_DEGRADED for e in events)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_pipeline_cancel_status_and_event() -> None:
    store = SessionStore.open(":memory:")
    try:
        events: list = []
        bus = AgentEventBus()
        bus.subscribe(None, lambda e: events.append(e))
        token = CancellationToken()
        llm = _ScriptedLLM(script=[RuntimeError("boom")])
        loop = _new_loop(
            llm=llm, store=store, bus=bus,
            retry=LLMRetryPolicy(
                max_attempts=5, backoff_initial=0.2, backoff_max=0.2, total_timeout=10,
            ),
        )

        async def cancel_soon() -> None:
            await asyncio.sleep(0.05)
            token.cancel("external_stop")

        sid = loop.new_session()
        send_task = asyncio.create_task(loop.send("hi", session_id=sid, stop_event=token))
        await cancel_soon()
        result = await send_task
        assert result.status == "cancelled"
        assert any(e.type == AgentEventType.LOOP_CANCELLED for e in events)
    finally:
        store.close()
