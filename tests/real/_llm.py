"""Shared real-LLM construction for ``tests/real/`` modules.

Centralizes env reading so every test gets the same configured client without
duplicating boilerplate. Skipped automatically (via the project ``conftest.py``)
when the required env vars are absent.
"""

from __future__ import annotations

from llm_client.interface import LLMService
from power_loop import LLMProviderConfig, create_llm_service_from_config


def make_llm(
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model: str | None = None,
) -> LLMService:
    cfg = LLMProviderConfig.from_env()
    if model is not None:
        cfg.model = model
    cfg.max_tokens = max_tokens
    cfg.temperature = temperature
    return create_llm_service_from_config(
        cfg,
    )
