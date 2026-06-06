"""18 · 多 Provider / Multi-provider: use three LLM providers or switch on demand

## What you'll learn / 你将学到
- 用 ``LLMProviderConfig`` 构造三家 provider（OpenAI / DashScope / DeepSeek）
  / construct three providers with ``LLMProviderConfig`` (OpenAI / DashScope / DeepSeek)
- ``create_llm_service_from_config()`` 一行创建服务
  / create a service in one line with ``create_llm_service_from_config()``
- 同一个 ``StatefulAgentLoop`` 跑不同 model 的 send（通过重建 loop）
  / run different models with the same ``StatefulAgentLoop`` (by rebuilding the loop)

## Prerequisites / 前提
- 需要至少一个 provider 的凭证（``.env`` 中 ``POWER_LOOP_*``）
  / requires credentials for at least one provider (``POWER_LOOP_*`` in ``.env``)

## Run / 运行
    python examples/18_multi_provider.py

## Key concepts / 关键概念
- ``LLMProviderConfig`` 的 ``provider`` 字段是标签，不是路由器——今天都走 OpenAI 兼容传输。
  / ``LLMProviderConfig``'s ``provider`` field is a label, not a router — all go through OpenAI-compatible transport today
- ``create_llm_service_from_env(prefix=...)`` 支持自定义前缀，多服务场景好用。
  / supports custom prefixes, useful in multi-service scenarios
- 切换 model 只需换 ``LLMProviderConfig.model``，无需改动业务代码。
  / switching models only requires changing ``LLMProviderConfig.model`` — no business code changes

## Next / 下一步
看看 `19_full_chatbot.py` — 旗舰示例：所有功能全开
/ see `19_full_chatbot.py` — flagship example: all features enabled
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
        sid = loop.new_session()
        r = await loop.send(question, session_id=sid)
        print(f"[{label}] model={config.model}")
        print(f"[{label}] reply: {r.final_text[:120]}")
        return r.final_text
    finally:
        loop.close()


async def main() -> None:
    # ── 1. Primary provider (env) / 主 provider（环境变量）────────────────
    primary = _cfg_from_env("POWER_LOOP")
    if primary is not None and primary.is_ready:
        await run_with_provider("Primary", primary,
                                "In one word: what color is the sky on a clear day?")

    # ── 2. Alternate provider (env with custom prefix) / 备用 provider ─────
    alt = _cfg_from_env("ALT_LLM")
    if alt is not None and alt.is_ready:
        await run_with_provider("Alternate", alt,
                                "In one word: what is the opposite of hot?")

    # ── 3. Programmatic (no env needed) / 代码构建（无需 env）─────────────
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
