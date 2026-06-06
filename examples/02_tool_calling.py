"""02 · 工具调用：让模型调你写的 Python 函数

What you learn
--------------
- :class:`ToolDefinition` 声明工具名 / 描述 / JSON Schema / required 参数
- :class:`ToolRegistry` 注册定义 → handler
- 模型看到工具描述 → 自主决定调用 → power-loop 转发参数 → handler 返回字符串 → 模型继续

Why ``max_rounds > 1``
----------------------
工具调用是两步：
  Round 1: LLM 决定调用 `lookup_dish(city="Bangkok")`
  Round 2: 工具结果回灌，LLM 看到 "pad thai" 给出最终回答

所以 ``max_rounds=1`` 跑不通工具——必须 ≥ 2。

Run
---
    python examples/02_tool_use.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)

# ── 1. 定义工具 ─────────────────────────────────────────────────────────

DISHES = {
    "tokyo": "sushi",
    "rome": "cacio e pepe",
    "lima": "ceviche",
    "bangkok": "pad thai",
}


def lookup_dish(**kwargs) -> str:
    """Sync handler. handler 也可以是 ``async def``，ToolRegistry 自动适配。"""
    city = str(kwargs.get("city") or "").strip().lower()
    return DISHES.get(city, f"No data for {city!r}")


LOOKUP_TOOL = ToolDefinition(
    name="lookup_dish",
    description="Return the signature local dish for a given city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    required_params=("city",),   # 校验在 handler 之前，缺参直接报错给模型
)


# ── 2. 注册并跑循环 ─────────────────────────────────────────────────────


async def main() -> str:
    registry = ToolRegistry()
    registry.register(LOOKUP_TOOL, lookup_dish)

    loop = StatefulAgentLoop(
        llm=make_llm(),
        db_path=":memory:",
        tool_registry=registry,
        config=AgentLoopConfig(
            system_prompt=(
                "You answer questions about local cuisine. "
                "Use the `lookup_dish` tool — it is the only authoritative source."
            ),
            max_rounds=4,         # ≥ 2，留余地给可能的多次工具调用
            compactor=None,
        ),
    )
    sid = loop.new_session()
    result = await loop.send("What is Bangkok's signature dish?", session_id=sid)
    print(f"status: {result.status}, rounds: {result.rounds}")
    print(f"reply : {result.final_text}")
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
