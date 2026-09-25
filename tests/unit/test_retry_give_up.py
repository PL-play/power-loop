"""Errors a retry cannot fix are given up at once (design/124 Z11).

An exhausted account (HTTP 402) used to be hit ``pipeline attempts × transport retries`` times per
round — 3 × 4 = 12 requests, a minute of backoff — before the run degraded, and every sub-agent /
workflow leaf did the same again. Both retry layers now stop at the first permanent failure.
"""

from __future__ import annotations

from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    LLMNonRetryable,
    LLMRetryExhausted,
    SessionStore,
    StatefulAgentLoop,
    is_permanent_llm_error,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    OpenAICompatibleChatConfig,
)
from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.retry import LLMRetryPolicy, with_retry


class _HTTPError(Exception):
    def __init__(self, status: int | None, text: str = "boom") -> None:
        super().__init__(f"{status} {text}")
        if status is not None:
            self.status_code = status


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status


class _WrappedError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__("wrapped")
        self.response = _Resp(status)


@pytest.mark.parametrize(("exc", "permanent"), [
    (_HTTPError(400), True), (_HTTPError(401), True), (_HTTPError(402), True),
    (_HTTPError(403), True), (_HTTPError(404), True), (_HTTPError(422), True),
    (_WrappedError(402), True),
    (_HTTPError(408), False), (_HTTPError(409), False), (_HTTPError(429), False),
    (_HTTPError(500), False), (_HTTPError(503), False),
    (_HTTPError(None), False), (ConnectionError("reset"), False), (TimeoutError(), False),
])
def test_is_permanent_llm_error(exc: BaseException, permanent: bool) -> None:
    assert is_permanent_llm_error(exc) is permanent


def _policy(**kw: Any) -> LLMRetryPolicy:
    return LLMRetryPolicy(max_attempts=4, backoff_initial=0, backoff_max=0, total_timeout=None,
                          **kw)


async def _drive(exc: BaseException, policy: LLMRetryPolicy) -> tuple[int, BaseException]:
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise exc

    with pytest.raises(LLMRetryExhausted) as info:
        await with_retry(call, policy=policy, token=CancellationToken.never())
    return calls, info.value


@pytest.mark.asyncio
async def test_permanent_error_stops_after_one_attempt() -> None:
    calls, err = await _drive(_HTTPError(402, "Insufficient Balance"), _policy())
    assert calls == 1
    assert isinstance(err, LLMNonRetryable) and err.attempts == 1
    assert "Insufficient Balance" in str(err.last_error)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [_HTTPError(429), _HTTPError(503), ConnectionError("reset")])
async def test_transient_errors_still_use_every_attempt(exc: BaseException) -> None:
    calls, err = await _drive(exc, _policy())
    assert calls == 4
    assert not isinstance(err, LLMNonRetryable)


@pytest.mark.asyncio
async def test_give_up_on_none_restores_retry_everything() -> None:
    calls, err = await _drive(_HTTPError(402), _policy(give_up_on=None))
    assert calls == 4 and not isinstance(err, LLMNonRetryable)


@pytest.mark.asyncio
async def test_host_classifier_can_widen_the_rule() -> None:
    # a mid-stream failure has no status — the host recognizes it by its text
    def host(exc: BaseException) -> bool:
        return is_permanent_llm_error(exc) or "insufficient balance" in str(exc).lower()

    calls, err = await _drive(_HTTPError(None, "stream error: Insufficient Balance"),
                              _policy(give_up_on=host))
    assert calls == 1 and isinstance(err, LLMNonRetryable)


@pytest.mark.asyncio
async def test_broken_classifier_falls_back_to_retrying() -> None:
    def broken(exc: BaseException) -> bool:
        raise RuntimeError("classifier bug")

    calls, err = await _drive(_HTTPError(402), _policy(give_up_on=broken))
    assert calls == 4 and not isinstance(err, LLMNonRetryable)


@pytest.mark.asyncio
async def test_transport_layer_does_not_retry_a_permanent_status() -> None:
    svc = OpenAICompatibleChatLLMService(OpenAICompatibleChatConfig(
        base_url="http://unused", api_key="k", model="m", max_retries=3, retry_base_delay_s=0))
    calls = 0

    async def fn() -> None:
        nonlocal calls
        calls += 1
        raise _HTTPError(402)

    with pytest.raises(_HTTPError):
        await svc._with_retries(fn, method="t")
    assert calls == 1

    calls = 0

    async def flaky() -> None:
        nonlocal calls
        calls += 1
        raise _HTTPError(503)

    with pytest.raises(_HTTPError):
        await svc._with_retries(flaky, method="t")
    assert calls == 4


class _Broke(LLMService):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        raise _HTTPError(402, "Insufficient Balance")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_pipeline_degrades_once_with_reason_non_retryable() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        degraded: list[dict] = []
        bus.subscribe(AgentEventType.LLM_DEGRADED, lambda e: degraded.append(dict(e.payload or {})))
        llm = _Broke()
        loop = StatefulAgentLoop(llm=llm, store=store, event_bus=bus, config=AgentLoopConfig(
            system_prompt="S", max_rounds=2, compactor=None,
            retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0, total_timeout=None)))
        sid = await loop.new_session()
        res = await loop.send("hi", sid)
        assert res.status == "degraded"
        assert llm.calls == 1
        assert degraded and degraded[0]["reason"] == "non_retryable"
        assert degraded[0]["attempts"] == 1
    finally:
        await store.close()
