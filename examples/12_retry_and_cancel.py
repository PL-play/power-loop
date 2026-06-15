"""12 · LLM 重试 / 超时 / 取消 / Retry, timeout, and cancellation

What you learn / 你将学到
--------------------------
- 配 ``AgentLoopConfig.retry_policy=LLMRetryPolicy(...)`` 让 ``call_llm`` 在
  抛 ``retry_on`` 异常时自动重试，指数退避到 ``backoff_max`` 封顶。
  / configuring ``retry_policy=LLMRetryPolicy(...)`` makes ``call_llm`` auto-retry
  on ``retry_on`` exceptions, with exponential backoff capped at ``backoff_max``
- 总时长 ``total_timeout`` 跨所有 attempt 累计；超时直接报 ``LLMTimeout`` →
  pipeline 翻译成 ``status="degraded"``。
  / ``total_timeout`` accumulates across all attempts; timeout raises ``LLMTimeout`` →
  pipeline translates to ``status="degraded"``
- ``CancellationToken`` 是「一个形状统治所有 cancel」：``threading.Event`` /
  ``asyncio.Event`` / 任意 ``Callable[[], bool]`` / owned ``token.cancel()``
  全都兼容。Cancel 在 retry sleep 中也会立刻生效，不会等满 backoff。
  / ``CancellationToken`` unifies all cancel shapes: ``threading.Event`` /
  ``asyncio.Event`` / any ``Callable[[], bool]`` / owned ``token.cancel()`` —
  all compatible. Cancel takes effect even during retry backoff sleep.
- 三种 outcome / Three outcomes:
  ``completed``（重试后成功 / success after retries）/
  ``degraded``（重试耗尽或超时 / retries exhausted or timeout）/
  ``cancelled``（外部 cancel / external cancel）

为啥这是 must-have / Why this is a must-have
----------------------------------------------
云厂商 LLM 抖动是日常（429、连接重置、流被服务端中断）。没有 retry → 一次
小抖动就 hard-fail 整轮。这段代码用一个**注入失败的 LLM 包装**确定性演示
所有路径，不依赖真实网络真的抽风。
/ Cloud LLM hiccups are the norm (429s, connection resets, server-side stream
interruptions). Without retry → a single hiccup hard-fails the entire round.
This code deterministically demonstrates all paths using an **injected-failure
LLM wrapper**, no real network flakiness needed.

Run / 运行
----------
    python examples/12_retry_and_cancel.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    CancellationToken,
    LLMRetryPolicy,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop._vendor.llm_client.interface import LLMService


class FlakyWrap(LLMService):
    """包装真实 LLM；前 ``fail_first`` 次抛 RuntimeError 模拟抖动，之后透传。
    / Wraps real LLM; raises RuntimeError for first ``fail_first`` calls to simulate
    flakiness, then passes through."""

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


def _new_bus_with_audit() -> tuple[AgentEventBus, list]:
    """订阅 retry / degraded / cancel 三类事件，方便外部观察。
    / Subscribe to retry / degraded / cancel events for external observation."""
    bus = AgentEventBus()
    seen: list = []
    interesting = {
        AgentEventType.LLM_RETRY_ATTEMPTED,
        AgentEventType.LLM_DEGRADED,
        AgentEventType.LOOP_CANCELLED,
    }
    bus.subscribe(None, lambda e: seen.append(e) if e.type in interesting else None)
    return bus, seen


# ── Scenario 1: 抖两次后第三次成功 / Transient failures, eventually succeeds


async def scenario_completed_after_retries() -> None:
    print("\n── Scenario 1: transient failures, eventually completes ──")
    bus, seen = _new_bus_with_audit()
    inner = make_llm(max_tokens=64, temperature=0)
    llm = FlakyWrap(inner, fail_first=2)
    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=llm, store=store, event_bus=bus,
            config=AgentLoopConfig(
                system_prompt="Reply with the single word OK.",
                max_rounds=1, max_tokens=64, compactor=None,
                retry_policy=LLMRetryPolicy(
                    max_attempts=4, backoff_initial=0.1, backoff_max=0.3, total_timeout=15,
                ),
            ),
        )
        sid = loop.new_session()
        r = await loop.send("hi", session_id=sid)
        print(f"  status={r.status} llm_calls={llm.calls} text={r.final_text.strip()!r}")
        print(f"  events: {[e.type.value for e in seen]}")
    finally:
        store.close()


# ── Scenario 2: 永远失败 → degraded / All attempts fail → degraded


async def scenario_degraded() -> None:
    print("\n── Scenario 2: all attempts fail → degraded ──")
    bus, seen = _new_bus_with_audit()
    inner = make_llm(max_tokens=64, temperature=0)
    llm = FlakyWrap(inner, fail_first=1_000_000)  # always fail
    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=llm, store=store, event_bus=bus,
            config=AgentLoopConfig(
                system_prompt="x", max_rounds=1, compactor=None,
                retry_policy=LLMRetryPolicy(
                    max_attempts=2, backoff_initial=0.05, backoff_max=0.1, total_timeout=5,
                ),
            ),
        )
        sid = loop.new_session()
        r = await loop.send("hi", session_id=sid)
        print(f"  status={r.status} llm_calls={llm.calls}")
        print(f"  final_text={r.final_text!r}")
        print(f"  events: {[e.type.value for e in seen]}")
    finally:
        store.close()


# ── Scenario 3: 外部 cancel 在 retry sleep 中触发 / External cancel during retry backoff


async def scenario_cancelled() -> None:
    print("\n── Scenario 3: external cancel during retry backoff ──")
    bus, seen = _new_bus_with_audit()
    inner = make_llm(max_tokens=64, temperature=0)
    llm = FlakyWrap(inner, fail_first=1_000_000)  # always fail → will retry / sleep
    token = CancellationToken()
    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=llm, store=store, event_bus=bus,
            config=AgentLoopConfig(
                system_prompt="x", max_rounds=1, compactor=None,
                retry_policy=LLMRetryPolicy(
                    max_attempts=10, backoff_initial=1.0, backoff_max=2.0, total_timeout=30,
                ),
            ),
        )

        async def trip_cancel() -> None:
            await asyncio.sleep(0.2)              # 让 loop 进入第一次 retry sleep
                                                   # let the loop enter its first retry sleep
            token.cancel("user_pressed_stop")

        sid = loop.new_session()
        send_task = asyncio.create_task(loop.send("hi", session_id=sid, stop_event=token))
        await trip_cancel()
        r = await send_task
        print(f"  status={r.status} llm_calls={llm.calls}")
        print(f"  events: {[e.type.value for e in seen]}")
    finally:
        store.close()


async def main() -> None:
    await scenario_completed_after_retries()
    await scenario_degraded()
    await scenario_cancelled()


if __name__ == "__main__":
    asyncio.run(main())
