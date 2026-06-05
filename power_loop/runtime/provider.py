"""Unified LLM provider configuration (M1.4).

Why
---
The library wraps a **single** transport today —
``OpenAICompatibleChatLLMService`` — but speaks to many actual providers
through it (OpenAI, DashScope/Qwen, DeepSeek, OpenRouter, Together,
Groq, local OpenAI-compatible servers). Each caller used to assemble an
``OpenAICompatibleChatConfig`` by hand and read env vars in its own way,
which made provider-swapping a per-call code change.

``LLMProviderConfig`` is the single config shape callers should target.
Two factories build an ``LLMService`` from it:

* :func:`create_llm_service_from_config` — given an explicit config.
* :func:`create_llm_service_from_env` — assembles from environment
  variables (``POWER_LOOP_*``), falling back to legacy
  ``OPENAI_COMPAT_*`` names so existing ``.env`` files keep working.

The ``provider`` field is currently informational (a string tag) — when
we add Anthropic-native transport in M3 it becomes the router key.
Callers that want to pin to a specific provider can set it; today it
does not affect the transport.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from llm_client.interface import LLMService, OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService

DEFAULT_PREFIX = "POWER_LOOP"
LEGACY_PREFIX = "OPENAI_COMPAT"


@dataclass
class LLMProviderConfig:
    """Unified, provider-agnostic LLM config.

    Required: ``base_url`` / ``api_key`` / ``model``. Everything else
    has sensible defaults that match :class:`OpenAICompatibleChatConfig`.

    ``provider`` is a free-form tag (``"openai"`` / ``"dashscope"`` /
    ``"deepseek"`` / …) used today only for telemetry and human
    readability; it becomes the routing key when multi-transport lands.
    """

    base_url: str
    api_key: str
    model: str
    provider: str = "openai"
    timeout_s: float = 180.0
    max_tokens: int = 8000
    temperature: float = 0.0
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fail fast at config build time, not on the first complete() call.
        missing = [
            name for name, val in (("base_url", self.base_url),
                                   ("api_key", self.api_key),
                                   ("model", self.model))
            if not val
        ]
        if missing:
            raise ValueError(
                f"LLMProviderConfig missing required field(s): {', '.join(missing)}"
            )

    @property
    def is_ready(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    # ── Factories ───────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = DEFAULT_PREFIX,
        fallback_prefix: str | None = LEGACY_PREFIX,
        env: dict[str, str] | None = None,
    ) -> LLMProviderConfig:
        """Build a config from ``{PREFIX}_*`` environment variables.

        Reads (in order of preference):

        * ``{prefix}_BASE_URL`` / ``{prefix}_API_KEY`` / ``{prefix}_MODEL``
          (required)
        * ``{prefix}_PROVIDER`` / ``{prefix}_TIMEOUT_S`` /
          ``{prefix}_MAX_TOKENS`` / ``{prefix}_TEMPERATURE`` /
          ``{prefix}_MAX_RETRIES`` (optional)

        If ``fallback_prefix`` is set and a primary var is missing, falls
        back to the same suffix under the fallback prefix. Default
        fallback is ``OPENAI_COMPAT`` so existing ``.env`` files
        (``OPENAI_COMPAT_BASE_URL`` etc.) keep working without edits.

        ``env`` argument is for tests; defaults to ``os.environ``.
        """
        src: dict[str, str] = dict(os.environ if env is None else env)

        def _get(name: str, default: str | None = None) -> str | None:
            primary = src.get(f"{prefix}_{name}")
            if primary:
                return primary
            if fallback_prefix:
                alt = src.get(f"{fallback_prefix}_{name}")
                if alt:
                    return alt
            return default

        base_url = _get("BASE_URL", "")
        api_key = _get("API_KEY", "")
        model = _get("MODEL", "")
        provider = _get("PROVIDER", "openai") or "openai"
        timeout_s = float(_get("TIMEOUT_S", "180") or 180)
        max_tokens = int(_get("MAX_TOKENS", "8000") or 8000)
        temperature = float(_get("TEMPERATURE", "0") or 0)
        max_retries = int(_get("MAX_RETRIES", "3") or 3)

        return cls(
            base_url=base_url or "",
            api_key=api_key or "",
            model=model or "",
            provider=provider,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )

    # ── Adaptation ──────────────────────────────────────────────────────

    def to_openai_compatible(self) -> OpenAICompatibleChatConfig:
        """Render into the transport-specific config the current backend
        expects. New transports (Anthropic-native in M3) will add their
        own adapter alongside this one.
        """
        return OpenAICompatibleChatConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout_s=self.timeout_s,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )


def create_llm_service_from_config(cfg: LLMProviderConfig) -> LLMService:
    """Build an ``LLMService`` from an :class:`LLMProviderConfig`.

    Today this always returns an ``OpenAICompatibleChatLLMService``;
    when a second transport lands it will dispatch on ``cfg.provider``.
    """
    return OpenAICompatibleChatLLMService(cfg.to_openai_compatible())


def create_llm_service_from_env(
    *,
    prefix: str = DEFAULT_PREFIX,
    fallback_prefix: str | None = LEGACY_PREFIX,
) -> LLMService:
    """One-liner for the common case: read env, build, return service."""
    cfg = LLMProviderConfig.from_env(prefix=prefix, fallback_prefix=fallback_prefix)
    return create_llm_service_from_config(cfg)


__all__ = [
    "DEFAULT_PREFIX",
    "LEGACY_PREFIX",
    "LLMProviderConfig",
    "create_llm_service_from_config",
    "create_llm_service_from_env",
]
