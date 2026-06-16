"""25 · Token 用量记账 / Token usage accounting

What you learn / 你将学到
--------------------------
- ``result.usage``：一次 ``send`` 的累计 token 用量（对该 run 的全部 LLM 调用
  求和——工具循环一个 run 往往有多次调用）
  / per-send cumulative usage, summed over every LLM call of the run
- ``usage_updated`` 事件：单次 LLM 调用粒度的实时计量，带 ``session_id`` 可归属
  / per-call real-time metering via the event bus
- 两者的分工：**记账读返回值，监控订事件**
  / accounting reads the result; live metering subscribes

常见误区 / Pitfalls
-------------------
- ``ctx.token_usage`` / ``llm.get_last_token_usage()`` 都只是**末次调用**，
  覆盖式语义，不要当总量用 / last-call only, overwrite semantics
- 事件 handler 里 ``event.payload`` 是 dict（``event.data`` 才是 dataclass），
  按 dict 取值最稳 / treat ``event.payload`` as a dict

Run / 运行
----------
    python examples/25_token_usage.py        # 需要 .env 配置 POWER_LOOP_*
"""

from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

from _helpers import make_llm

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    StatefulAgentLoop,
    create_default_tool_registry,
)


async def main() -> None:
    llm = make_llm()

    # ── 可选：事件总线做实时计量（一次 LLM 调用一个事件） ────────────────
    bus = AgentEventBus(suppress_subscriber_errors=True)

    def on_usage(event) -> None:
        usage = (event.payload or {}).get("usage") or {}
        print(
            f"  [usage_updated] session={event.session_id} "
            f"call: in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}"
        )

    bus.subscribe(AgentEventType.USAGE_UPDATED, on_usage)

    with TemporaryDirectory() as tmp:
        loop = StatefulAgentLoop(
            llm=llm,
            db_path=f"{tmp}/sessions.db",
            config=AgentLoopConfig(
                system_prompt="你是一个简洁的助手。需要待办时使用 todo_write 工具。",
                max_rounds=6,
            ),
            tool_registry=create_default_tool_registry(preset="core", bind=False),
            event_bus=bus,
        )
        sid = await loop.new_session()

        # 让模型用一次工具，制造"一个 run 多次 LLM 调用"的真实形态。
        res = await loop.send(
            "把『写周报』和『回邮件』记成待办，然后告诉我你记了几条。",
            session_id=sid,
        )

        print(f"\nreply: {res.final_text[:80]}")
        # ── 记账：直接读返回值，整个 run 的累计 ──────────────────────────
        print(
            f"run total: calls={res.usage.get('calls')} "
            f"in={res.usage.get('prompt_tokens')} out={res.usage.get('completion_tokens')} "
            f"cache_read={res.usage.get('cache_read_tokens')}"
        )

        # 每次 send 各自独立计量，不跨 send 累计。
        res2 = await loop.send("谢谢，再见。", session_id=sid)
        print(
            f"second send: calls={res2.usage.get('calls')} "
            f"in={res2.usage.get('prompt_tokens')} out={res2.usage.get('completion_tokens')}"
        )

        # ── 累计记账：session_stats（store 持久化，跨 send 累加） ────────
        stats = await loop.get_session_stats(sid)
        print(
            f"session stats: sends={stats.sends} rounds={stats.rounds} "
            f"llm_calls={stats.llm_calls} tool_calls={stats.tool_calls} "
            f"in={stats.prompt_tokens} out={stats.completion_tokens}"
        )

        # ── 预算护栏：max_tokens_per_run（这里演示配置位置，真实值按需设） ──
        # AgentLoopConfig(max_tokens_per_run=50_000) → 越界后 status="budget_exceeded"

        await loop.aclose()


if __name__ == "__main__":
    asyncio.run(main())
