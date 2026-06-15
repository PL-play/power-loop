"""Shared helpers for ``examples/``.

Centralizes ``.env`` loading and LLM client construction so each numbered
example file can focus on the ONE concept it teaches, not on boilerplate.

If you're copying an example into your own project, inline the two lines
from :func:`make_llm` and drop the import.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from power_loop import LLMProviderConfig, create_llm_service_from_config
from power_loop._vendor.llm_client.interface import LLMService


def load_env() -> None:
    """Read ``.env`` from the project root. Idempotent."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def make_llm(
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> LLMService:
    """Build an LLM client from ``POWER_LOOP_*`` environment variables.

    Legacy ``OPENAI_COMPAT_*`` variables still work as fallback.
    """
    load_env()
    cfg = LLMProviderConfig.from_env()
    cfg.max_tokens = max_tokens
    cfg.temperature = temperature
    return create_llm_service_from_config(cfg)
