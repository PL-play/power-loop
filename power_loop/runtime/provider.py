"""Unified LLM provider configuration.

Why
---
The library exposes one config object while routing to provider-specific
transports. OpenAI-compatible providers use ``OpenAICompatibleChatLLMService``;
Anthropic-compatible endpoints use ``AnthropicMessagesLLMService``.

``LLMProviderConfig`` is the single config shape callers should target.
Two factories build an ``LLMService`` from it:

* :func:`create_llm_service_from_config` — given an explicit config.
* :func:`create_llm_service_from_env` — assembles from environment
  variables (``POWER_LOOP_*``), falling back to legacy
  ``OPENAI_COMPAT_*`` names so existing ``.env`` files keep working.

The ``provider`` field is the routing key. Use ``provider="anthropic"``
for Anthropic Messages API endpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Only the SDK-free interface (config dataclasses + the LLMService Protocol) is
# imported at module load. The concrete transports pull heavy vendor SDKs
# (``anthropic`` / ``openai``) and are imported lazily inside
# :func:`create_llm_service_from_config`, so ``import power_loop`` stays
# featherweight and works with neither SDK installed (H3.1).
from power_loop._vendor.llm_client.interface import AnthropicChatConfig, LLMService, OpenAICompatibleChatConfig

DEFAULT_PREFIX = "POWER_LOOP"
LEGACY_PREFIX = "OPENAI_COMPAT"


@dataclass
class LLMProviderConfig:
    """Unified, provider-agnostic LLM config.

    Required: ``base_url`` / ``api_key`` / ``model``. Everything else
    has sensible defaults that match :class:`OpenAICompatibleChatConfig`.

    ``provider`` is a routing key. ``"anthropic"`` selects the native
    Anthropic Messages transport; other values use the OpenAI-compatible
    transport.
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

        extra: dict[str, Any] = {}
        if provider.strip().lower() in {"echo", "stub"}:
            # The offline echo provider needs no real endpoint; supply placeholders
            # so the required-field check passes, and pass through an optional reply.
            base_url = base_url or "echo://local"
            api_key = api_key or "echo"
            model = model or "echo"
            reply = _get("ECHO_REPLY")
            if reply is not None:
                extra["echo_reply"] = reply
            sleep = _get("ECHO_SLEEP")
            if sleep is not None:
                extra["echo_sleep_s"] = float(sleep)

        return cls(
            base_url=base_url or "",
            api_key=api_key or "",
            model=model or "",
            provider=provider,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            extra=extra,
        )

    # ── Adaptation ──────────────────────────────────────────────────────

    def to_openai_compatible(self) -> OpenAICompatibleChatConfig:
        """Render into an OpenAI-compatible transport config."""
        return OpenAICompatibleChatConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout_s=self.timeout_s,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    def to_anthropic(self) -> AnthropicChatConfig:
        """Render into an Anthropic Messages API transport config."""
        return AnthropicChatConfig(
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

    ``provider="anthropic"`` routes to the Anthropic Messages API;
    everything else routes to the OpenAI-compatible chat-completions API.
    """
    provider = (cfg.provider or "").strip().lower()
    if provider in {"echo", "stub"}:
        from power_loop.runtime.stub_provider import EchoLLMService

        return EchoLLMService(
            reply=cfg.extra.get("echo_reply"),
            prefix=cfg.extra.get("echo_prefix", ""),
            sleep_s=float(cfg.extra.get("echo_sleep_s") or 0.0),
        )
    if provider in {"anthropic", "claude", "dashscope-anthropic"}:
        try:
            from power_loop._vendor.llm_client.anthropic_factory import AnthropicMessagesLLMService
        except ImportError as exc:  # pragma: no cover - exercised via H2.5 import-leg
            raise ImportError(
                f"provider={cfg.provider!r} needs the 'anthropic' SDK, which is not "
                "installed. Install it with:  pip install 'power-loop[anthropic]'"
            ) from exc
        return AnthropicMessagesLLMService(cfg.to_anthropic())
    try:
        from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService
    except ImportError as exc:  # pragma: no cover - exercised via H2.5 import-leg
        raise ImportError(
            f"provider={cfg.provider!r} needs the 'openai' SDK, which is not "
            "installed. Install it with:  pip install 'power-loop[openai]'"
        ) from exc
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
