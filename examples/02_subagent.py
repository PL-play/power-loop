"""02 · 子代理：父 agent 通过 ``spawn_agent`` 工具委托子任务

What this example shows
-----------------------
- ``register_spawn_agent`` 一次性注入 ``spawn_agent`` + ``run_agent`` 两个 meta-tool
- 父 LLM 自主决定调用 ``spawn_agent``，把任务交给一个独立子会话处理
- 子会话默认 ``EPHEMERAL`` 生命周期：成功完成后从 SessionStore 物理删除
- 父会话拿到子会话的 ``final_text`` 作为 tool 结果，继续推进

Key concepts (see README §子代理)
-------------------------------
* 子会话与父会话共享同一个 ``SessionStore``，但有独立的 ``session_id`` + 独立 history。
* ``parent_session_id`` / ``spawn_tool_call_id`` 在 sessions 表里建立父子链接，方便审计。
* 深度上限 ``MAX_SPAWN_DEPTH=3``，防递归爆栈；超限直接返回 rejected 字符串给父 LLM。
* 三种 ``SubagentLifecycle``：``EPHEMERAL`` / ``LINKED`` / ``DETACHED``，控制父 close 时是否级联删。

How to run
----------
    python examples/02_subagent.py
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
    ToolRegistry,
    register_spawn_agent,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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
    registry = ToolRegistry()
    # 一行注册 spawn_agent + run_agent 两个 meta-tool。
    register_spawn_agent(registry)

    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,
            tool_registry=registry,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are a delegating orchestrator. For any factual question, "
                    "call the `spawn_agent` tool with a clear `task` description; "
                    "do NOT answer from memory. After the sub-agent replies, "
                    "summarize its answer for the user in one sentence."
                ),
                max_rounds=5,
                max_tokens=512,
                temperature=0,
                compactor=None,
            ),
        )
        result = await loop.send(
            "Delegate this and report back: what is the capital of Japan?"
        )
        print(f"status        : {result.status}, rounds: {result.rounds}")
        print(f"reply         : {result.final_text}")
        # EPHEMERAL：子会话已删，list_children 返回空。
        print(f"surviving subs: {store.list_children(result.session_id)}")
        return result.final_text
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
