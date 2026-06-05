"""18 · 多 Provider：同时使用三家 LLM，或按需切换

## What you'll learn
- 用 ``LLMProviderConfig`` 构造三家 provider（OpenAI / DashScope / DeepSeek）
- ``create_llm_service_from_config()`` 一行创建服务
- 同一个 ``StatefulAgentLoop`` 跑不同 model 的 send（通过重建 loop）

## Prerequisites
- 需要至少一个 provider 的凭证（``.env`` 中 ``POWER_LOOP_*``）

## Run
    python examples/18_multi_provider.py

## Key concepts
- ``LLMProviderConfig`` 的 ``provider`` 字段是标签，不是路由器——今天都走 OpenAI 兼容传输。
- ``create_llm_service_from_env(prefix=...)`` 支持自定义前缀，多服务场景好用。
- 切换 model 只需换 ``LLMProviderConfig.model``，无需改动业务代码。

## Next
看看 `19_full_chatbot.py` — 旗舰示例：所有功能全开
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from power_loop import (
    AgentLoopConfig,
    LLMProviderConfig,
    StatefulAgentLoop,
    create_llm_service_from_config,
)

load_dotenv()


def _cfg_from_env(prefix: str) -> LLMProviderConfig | None:
    """Read ``{prefix}_BASE_URL`` etc. from env. Falls back to OPENAI_COMPAT_*."""
    try:
        return LLMProviderConfig.from_env(prefix=prefix)
    except ValueError as exc:
        print(f"[{prefix}] skipped: {exc}")
        return None


async def run_with_provider(label: str, config: LLMProviderConfig, question: str) -> str:
    """Run one question against a specific provider configuration."""
    llm = create_llm_service_from_config(config)
    loop = StatefulAgentLoop(
        llm=llm,
        config=AgentLoopConfig(
            system_prompt="You are a concise assistant. Reply in English.",
            max_rounds=1, compactor=None,
        ),
    )
    try:
        r = await loop.send(question)
        print(f"[{label}] model={config.model}")
        print(f"[{label}] reply: {r.final_text[:120]}")
        return r.final_text
    finally:
        loop.close()


async def main() -> None:
    # ── 1. Primary provider (env) ────────────────────────────────────────
    primary = _cfg_from_env("POWER_LOOP")
    if primary is not None and primary.is_ready:
        await run_with_provider("Primary", primary,
                                "In one word: what color is the sky on a clear day?")

    # ── 2. Alternate provider (env with custom prefix) ───────────────────
    alt = _cfg_from_env("ALT_LLM")
    if alt is not None and alt.is_ready:
        await run_with_provider("Alternate", alt,
                                "In one word: what is the opposite of hot?")

    # ── 3. Programmatic (no env needed) ──────────────────────────────────
    # Demonstrate that you can build a config entirely in code.
    manual_cfg = LLMProviderConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("POWER_LOOP_API_KEY", "sk-placeholder"),
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100,
    )
    print(f"[Manual] cfg.provider={manual_cfg.provider}, "
          f"cfg.model={manual_cfg.model}, "
          f"is_ready={manual_cfg.is_ready}")


if __name__ == "__main__":
    asyncio.run(main())
