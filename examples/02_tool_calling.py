"""02 · 工具调用 / Tool calling: let the model call your Python functions

What you learn / 你将学到
--------------------------
- :class:`ToolDefinition` 声明工具名 / 描述 / JSON Schema / required 参数
  / declares tool name / description / JSON Schema / required params
- :class:`ToolRegistry` 注册定义 → handler / registers definition → handler
- 模型看到工具描述 → 自主决定调用 → power-loop 转发参数 → handler 返回字符串 → 模型继续
  / model sees tool descriptions → decides to call → power-loop forwards args →
  handler returns string → model continues

Why ``max_rounds > 1`` / 为什么 max_rounds 必须大于 1
------------------------------------------------------
工具调用是两步 / Tool calling is two steps:
  Round 1: LLM 决定调用 `lookup_dish(city="Bangkok")` / decides to call
  Round 2: 工具结果回灌，LLM 看到 "pad thai" 给出最终回答 / tool result fed back

所以 ``max_rounds=1`` 跑不通工具——必须 ≥ 2
/ So ``max_rounds=1`` cannot complete a tool call — must be ≥ 2.

Run / 运行
----------
    python examples/02_tool_calling.py
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

# ── 1. 定义工具 / Define tools ──────────────────────────────────────────

DISHES = {
    "tokyo": "sushi",
    "rome": "cacio e pepe",
    "lima": "ceviche",
    "bangkok": "pad thai",
}


def lookup_dish(**kwargs) -> str:
    """Sync handler. handler 也可以是 ``async def``，ToolRegistry 自动适配。
    / Sync handler. Handlers can also be ``async def`` — ToolRegistry adapts automatically."""
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
                                  # validation runs before handler; missing params → error to model
)


# ── 2. 注册并跑循环 / Register and run the loop ─────────────────────────


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
            ),
            max_rounds=4,         # ≥ 2，留余地给可能的多次工具调用
                                   # ≥ 2, with headroom for multiple tool calls
            compactor=None,
        ),
    )
    sid = await loop.new_session()
    result = await loop.send("What is Bangkok's signature dish?", session_id=sid)
    print(f"status: {result.status}, rounds: {result.rounds}")
    print(f"reply : {result.final_text}")
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
