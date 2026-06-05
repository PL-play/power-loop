"""Shared real-LLM construction for ``tests/real/`` modules.

Centralizes env reading so every test gets the same OpenAI-compatible client
without duplicating boilerplate. Skipped automatically (via the project
``conftest.py``) when the required env vars are absent.
"""

from __future__ import annotations

import os

from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService


def make_llm(
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model: str | None = None,
) -> OpenAICompatibleChatLLMService:
    cfg = OpenAICompatibleChatConfig(
        base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
        api_key=os.environ["OPENAI_COMPAT_API_KEY"],
        model=model or os.environ["OPENAI_COMPAT_MODEL"],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return OpenAICompatibleChatLLMService(cfg)
