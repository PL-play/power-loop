"""01 · 工具调用：让 LLM 通过 ToolRegistry 调用自定义工具

What this example shows
-----------------------
- 定义一个业务工具（``ToolDefinition`` + 同步 handler）
- 注册到 ``ToolRegistry``
- 传给 ``StatefulAgentLoop``，模型按需调用
- 多轮（max_rounds > 1）：LLM 调工具 → 工具结果回灌 → LLM 总结

Key concepts (see README §核心概念)
----------------------------------
* ``ToolDefinition.input_schema`` 是 JSON Schema，权威；LLM 看到它来决定如何调用。
* ``required_params`` 在调用进入 handler 前做校验，缺参直接返回错误字符串给模型，不会抛。
* handler 可以是 sync 或 async；ToolRegistry 会自动适配。

How to run
----------
    python examples/01_tool_use.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop import (
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── 1. 业务工具定义 ──────────────────────────────────────────────────────

DISHES = {
    "tokyo": "sushi",
    "rome": "cacio e pepe",
    "lima": "ceviche",
    "bangkok": "pad thai",
}


def lookup_dish(**kwargs) -> str:
    """简单的同步工具 handler：从内置字典查菜名。"""
    city = str(kwargs.get("city") or "").strip().lower()
    return DISHES.get(city, f"No data for {city!r}")


LOOKUP_TOOL = ToolDefinition(
    name="lookup_dish",
    description="Return the signature local dish for the given city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "city name in English"}},
        "required": ["city"],
    },
    required_params=("city",),
)


def make_llm() -> OpenAICompatibleChatLLMService:
    cfg = OpenAICompatibleChatConfig(
        base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
        api_key=os.environ["OPENAI_COMPAT_API_KEY"],
        model=os.environ["OPENAI_COMPAT_MODEL"],
        max_tokens=512,
        temperature=0,
    )
    return OpenAICompatibleChatLLMService(cfg)


async def main() -> str:
    # ── 2. 注册工具 ──
    registry = ToolRegistry()
    registry.register(LOOKUP_TOOL, lookup_dish)

    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,
            tool_registry=registry,
            config=AgentLoopConfig(
                system_prompt=(
                    "You answer questions about local cuisine. "
                    "Use the `lookup_dish` tool — it is the only authoritative source. "
                    "Reply briefly in English."
                ),
                # max_rounds≥2：第 1 轮 LLM 调工具，第 2 轮 LLM 看到工具结果给最终答案。
                max_rounds=4,
                max_tokens=512,
                temperature=0,
                compactor=None,
            ),
        )
        result = await loop.send("What is Bangkok's signature dish?")
        print(f"status : {result.status}, rounds: {result.rounds}")
        print(f"reply  : {result.final_text}")
        return result.final_text
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
