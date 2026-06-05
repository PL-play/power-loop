"""12 · LLM 重试 / 超时 / 取消（M1.1）

What you learn
--------------
- 配 ``AgentLoopConfig.retry_policy=LLMRetryPolicy(...)`` 让 ``call_llm`` 在
  抛 ``retry_on`` 异常时自动重试，指数退避到 ``backoff_max`` 封顶。
- 总时长 ``total_timeout`` 跨所有 attempt 累计；超时直接报 ``LLMTimeout`` →
  pipeline 翻译成 ``status="degraded"``。
- ``CancellationToken`` 是「一个形状统治所有 cancel」：``threading.Event`` /
  ``asyncio.Event`` / 任意 ``Callable[[], bool]`` / owned ``token.cancel()``
  全都兼容。Cancel 在 retry sleep 中也会立刻生效，不会等满 backoff。
- 三种 outcome：``completed``（重试后成功）/ ``degraded``（重试耗尽或超时）/
  ``cancelled``（外部 cancel）—— 全部通过 ``StatefulResult.status`` 一眼可读。

为啥这是 must-have
------------------
云厂商 LLM 抖动是日常（429、连接重置、流被服务端中断）。没有 retry → 一次
小抖动就 hard-fail 整轮。这段代码用一个**注入失败的 LLM 包装**确定性演示
所有路径，不依赖真实网络真的抽风。

Run
---
    python examples/12_retry_and_cancel.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from llm_client.interface import LLMService
from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    CancellationToken,
    LLMRetryPolicy,
    SessionStore,
    StatefulAgentLoop,
)


class FlakyWrap(LLMService):
    """包装真实 LLM；前 ``fail_first`` 次抛 RuntimeError 模拟抖动，之后透传。"""

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
    """订阅 retry / degraded / cancel 三类事件，方便外部观察。"""
    bus = AgentEventBus()
    seen: list = []
    interesting = {
        AgentEventType.LLM_RETRY_ATTEMPTED,
        AgentEventType.LLM_DEGRADED,
        AgentEventType.LOOP_CANCELLED,
    }
    bus.subscribe(None, lambda e: seen.append(e) if e.type in interesting else None)
    return bus, seen


# ── Scenario 1: 抖两次后第三次成功 ────────────────────────────────────────


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
        r = await loop.send("hi")
        print(f"  status={r.status} llm_calls={llm.calls} text={r.final_text.strip()!r}")
        print(f"  events: {[e.type.value for e in seen]}")
    finally:
        store.close()


# ── Scenario 2: 永远失败 → degraded ────────────────────────────────────────


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
        r = await loop.send("hi")
        print(f"  status={r.status} llm_calls={llm.calls}")
        print(f"  final_text={r.final_text!r}")
        print(f"  events: {[e.type.value for e in seen]}")
    finally:
        store.close()


# ── Scenario 3: 外部 cancel 在 retry sleep 中触发 ─────────────────────────


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
            token.cancel("user_pressed_stop")

        send_task = asyncio.create_task(loop.send("hi", stop_event=token))
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
