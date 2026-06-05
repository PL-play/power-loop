"""08 · 流式渲染：订阅 STREAM_DELTA 做打字机效果

What you learn
--------------
- ``AgentEventBus`` 是只读旁路通道——订阅事件不影响主循环
- ``STREAM_DELTA`` 在 LLM 每吐一片 token 时触发，``event.data.text`` 是这一片
- ``stream_id`` 区分流（同一会话理论可并发多个，主流默认 ``"main"``）
- ``STREAM_THINK_DELTA`` 是 reasoning/thinking 段，部分模型才有
- 订阅可以 sync 或 async；bus 自动判断并 await
- ``bus.subscribe(None, fn)`` 订阅**所有**事件（debug 用），单类型订阅传 enum

Run
---
    python examples/08_streaming.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    StatefulAgentLoop,
    StreamDeltaPayload,
)


def make_typewriter(bus: AgentEventBus) -> None:
    """Hook STREAM_DELTA into stdout as a typewriter."""

    chars_printed: list[int] = [0]

    def on_delta(event: AgentEvent) -> None:
        if not isinstance(event.data, StreamDeltaPayload):
            return
        text = event.data.text
        if event.data.is_think:
            return                          # 跳过 reasoning 流，只渲染最终回复
        print(text, end="", flush=True)
        chars_printed[0] += len(text)

    def on_start(event: AgentEvent) -> None:
        print(f"\n[stream {event.stream_id} starting...] ", end="", flush=True)

    def on_done(event: AgentEvent) -> None:
        print(f"\n[stream done — {chars_printed[0]} chars rendered]")

    bus.subscribe(AgentEventType.STREAM_STARTED, on_start)
    bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)
    bus.subscribe(AgentEventType.STREAM_COMPLETED, on_done)


async def main() -> str:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    make_typewriter(bus)

    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=512, temperature=0.3),
        db_path=":memory:",
        event_bus=bus,
        config=AgentLoopConfig(
            system_prompt="You are a helpful assistant. Reply in English.",
            max_rounds=1,
            compactor=None,
        ),
    )
    r = await loop.send(
        "Explain in 3 short sentences why HTTPS is more secure than HTTP."
    )
    # final_text 与流式拼出的内容应该一致
    print(f"\n[result] status={r.status}, final_text len={len(r.final_text)}")
    return r.final_text


if __name__ == "__main__":
    asyncio.run(main())
