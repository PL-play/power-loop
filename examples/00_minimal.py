"""00 · 最小用法：单回合非流式回复（DeepTalk Agent MVP 同款）

What this example shows
-----------------------
power-loop 最简单的调用形态：构造 LLMService → 装进 AgentLoop → 跑一回合 → 拿文本。
不带工具、不带 hooks、不带子代理。

When to copy this
-----------------
- 业务侧只想要"传 messages 进去、拿一段回复出来"的场景；
- DeepTalk `agent` 服务的 MVP 用法（@Agent 收到消息 → 单回合应答）；
- 任何"LLM 当函数用"的地方。

How to run
----------
    cd power-loop
    cp .env.example .env  # 填好 OPENAI_COMPAT_BASE_URL / API_KEY / MODEL
    python examples/00_minimal.py

Expected output
---------------
A short Chinese assistant reply, e.g. "你好！有什么我可以帮你的吗？"
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# ── 公开 API：业务侧只需要 import 这几样 ──
from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop import AgentLoop, AgentLoopConfig

# 从项目根的 .env 读取 LLM 配置；examples 跑通的前提。
# 放在 imports 之后是因为 llm_client 不在 import 时读 env，所以顺序安全。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def make_llm() -> OpenAICompatibleChatLLMService:
    """从环境变量构造 OpenAI 兼容的 LLM 客户端。

    M1.4 之后会有 ``create_llm_service_from_env(prefix="POWER_LOOP")`` 替代这段样板。
    """
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

    # 单回合配置：max_rounds=1 表示拿到 LLM 一次回复就结束，
    # 不进入"工具调用 → 再问 LLM → ..."的多轮循环。
    config = AgentLoopConfig(
        system_prompt="你是 DeepTalk 的关系协作者。回答简洁、温和、不替用户做决定。",
        max_rounds=1,
        max_tokens=512,
        temperature=0.2,
    )

    loop = AgentLoop(llm=llm, config=config)

    # messages 是 OpenAI ChatCompletions 风格的列表。业务侧的"历史 / 记忆"
    # 应该已经被组装好再传进来——power-loop 不感知"哪条是历史、哪条是当前输入"。
    result = await loop.run(
        messages=[
            {"role": "user", "content": "用一句话告诉我，今天怎么开始一段深度对话？"},
        ],
        session_id="example-00-minimal",
    )

    print(f"status: {result.status}, rounds: {result.rounds}")
    print(f"reply : {result.final_text}")
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
