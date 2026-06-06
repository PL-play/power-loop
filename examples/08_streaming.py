"""08 · 流式渲染 / Streaming: subscribe to STREAM_DELTA for typewriter effect

What you learn / 你将学到
--------------------------
- ``AgentEventBus`` 是只读旁路通道——订阅事件不影响主循环
  / ``AgentEventBus`` is a read-only side channel — subscribing does not affect the main loop
- ``STREAM_DELTA`` 在 LLM 每吐一片 token 时触发，``event.data.text`` 是这一片
  / ``STREAM_DELTA`` fires each time the LLM emits a token chunk; ``event.data.text`` is that chunk
- ``stream_id`` 区分流（同一会话理论可并发多个，主流默认 ``"main"``）
  / ``stream_id`` distinguishes streams (theoretically multiple per session; main defaults to ``"main"``)
- ``STREAM_THINK_DELTA`` 是 reasoning/thinking 段，部分模型才有
  / ``STREAM_THINK_DELTA`` is for reasoning/thinking segments, only some models emit it
- 订阅可以 sync 或 async；bus 自动判断并 await
  / subscribers can be sync or async; bus auto-detects and awaits
- ``bus.subscribe(None, fn)`` 订阅**所有**事件（debug 用），单类型订阅传 enum
  / ``bus.subscribe(None, fn)`` subscribes to **all** events (for debugging); pass enum for single type

Run / 运行
----------
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
                                             # skip reasoning stream, only render final reply
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
    sid = loop.new_session()
    r = await loop.send(
        "Explain in 3 short sentences why HTTPS is more secure than HTTP.",
        session_id=sid,
    )
    # final_text 与流式拼出的内容应该一致
    # final_text should match what was streamed
    print(f"\n[result] status={r.status}, final_text len={len(r.final_text)}")
    return r.final_text


if __name__ == "__main__":
    asyncio.run(main())
