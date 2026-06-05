"""00 · 最小用法：单回合非流式回复（StatefulAgentLoop 入门）

What this example shows
-----------------------
power-loop 最简单的调用形态：
  - 用 SessionStore 持久化（这里用 ":memory:"，业务方应给一个文件路径）
  - 构造 StatefulAgentLoop
  - send(user_input) → 拿到 StatefulResult

When to copy this
-----------------
- 业务侧只想要 "传一段输入、拿一段回复" 的场景
- DeepTalk `agent` 服务的 MVP 用法
- 任何 "LLM 当函数用" 的地方

How to run
----------
    cd power-loop
    cp .env.example .env  # 填好 OPENAI_COMPAT_BASE_URL / API_KEY / MODEL
    python examples/00_minimal.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop

# 从项目根的 .env 读取 LLM 配置。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def make_llm() -> OpenAICompatibleChatLLMService:
    cfg = OpenAICompatibleChatConfig(
        base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
        api_key=os.environ["OPENAI_COMPAT_API_KEY"],
        model=os.environ["OPENAI_COMPAT_MODEL"],
        max_tokens=512,
        temperature=0.2,
    )
    return OpenAICompatibleChatLLMService(cfg)


async def main() -> str:
    llm = make_llm()
    store = SessionStore.open(":memory:")  # 业务方：换成文件路径以跨进程保留
    try:
        loop = StatefulAgentLoop(
            llm=llm,
            store=store,
            config=AgentLoopConfig(
                system_prompt="你是 DeepTalk 的关系协作者。回答简洁、温和、不替用户做决定。",
                max_rounds=1,
                max_tokens=512,
                temperature=0.2,
                compactor=None,
            ),
        )
        # 第一次发送：session_id=None → 自动创建并返回新 session。
        result = await loop.send("用一句话告诉我，今天怎么开始一段深度对话？")
        print(f"session : {result.session_id}")
        print(f"status  : {result.status}, rounds: {result.rounds}")
        print(f"reply   : {result.final_text}")
        return result.final_text
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
