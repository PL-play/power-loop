"""Shared helpers for ``examples/``.

Centralizes ``.env`` loading and LLM client construction so each numbered
example file can focus on the ONE concept it teaches, not on boilerplate.

If you're copying an example into your own project, inline the two lines
from :func:`make_llm` and drop the import.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService


def load_env() -> None:
    """Read ``.env`` from the project root. Idempotent."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def make_llm(
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> OpenAICompatibleChatLLMService:
    """Build an OpenAI-compatible LLM client from environment variables.

    Expects ``OPENAI_COMPAT_BASE_URL`` / ``OPENAI_COMPAT_API_KEY`` /
    ``OPENAI_COMPAT_MODEL`` in the environment (loaded by :func:`load_env`).
    """
    load_env()
    return OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(
            base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
            api_key=os.environ["OPENAI_COMPAT_API_KEY"],
            model=os.environ["OPENAI_COMPAT_MODEL"],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )
