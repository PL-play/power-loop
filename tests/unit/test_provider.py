"""Unit tests for M1.4 — LLMProviderConfig + env / factory wiring.

Covers:
- Required fields enforced at construction (no silent half-built config).
- ``from_env`` reads ``POWER_LOOP_*`` and falls back to legacy
  ``OPENAI_COMPAT_*``; the per-prefix selection is precise (don't pull
  ``BASE_URL`` from one prefix and ``API_KEY`` from the other).
- Three different providers can be built by env alone (openai /
  dashscope / deepseek style).
- ``create_llm_service_from_config`` instantiates a real transport
  (without sending a request).
"""

from __future__ import annotations

import pytest

from power_loop import (
    LLMProviderConfig,
    create_llm_service_from_config,
    create_llm_service_from_env,
)

# ── Construction guards ──────────────────────────────────────────────────


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValueError) as exc:
        LLMProviderConfig(base_url="", api_key="k", model="m")
    assert "base_url" in str(exc.value)


def test_all_required_present_constructs_ok() -> None:
    cfg = LLMProviderConfig(base_url="https://x", api_key="k", model="m")
    assert cfg.provider == "openai"          # default tag
    assert cfg.max_tokens == 8000            # default carried through


# ── from_env ────────────────────────────────────────────────────────────


def _env(**kw: str) -> dict[str, str]:
    return dict(kw)


def test_from_env_reads_primary_prefix() -> None:
    cfg = LLMProviderConfig.from_env(env=_env(
        POWER_LOOP_BASE_URL="https://api.openai.com/v1",
        POWER_LOOP_API_KEY="sk-test",
        POWER_LOOP_MODEL="gpt-4o-mini",
        POWER_LOOP_PROVIDER="openai",
        POWER_LOOP_MAX_TOKENS="4096",
        POWER_LOOP_TEMPERATURE="0.3",
    ))
    assert cfg.base_url.endswith("/v1")
    assert cfg.model == "gpt-4o-mini"
    assert cfg.provider == "openai"
    assert cfg.max_tokens == 4096
    assert cfg.temperature == 0.3


def test_from_env_falls_back_to_legacy_prefix() -> None:
    cfg = LLMProviderConfig.from_env(env=_env(
        OPENAI_COMPAT_BASE_URL="https://compat.example",
        OPENAI_COMPAT_API_KEY="legacy-key",
        OPENAI_COMPAT_MODEL="qwen-plus",
    ))
    assert cfg.base_url == "https://compat.example"
    assert cfg.api_key == "legacy-key"
    assert cfg.model == "qwen-plus"


def test_from_env_primary_overrides_fallback() -> None:
    cfg = LLMProviderConfig.from_env(env=_env(
        POWER_LOOP_BASE_URL="https://primary",
        POWER_LOOP_API_KEY="primary-key",
        POWER_LOOP_MODEL="primary-model",
        # legacy values that should be IGNORED
        OPENAI_COMPAT_BASE_URL="https://legacy",
        OPENAI_COMPAT_API_KEY="legacy",
        OPENAI_COMPAT_MODEL="legacy-model",
    ))
    assert cfg.base_url == "https://primary"
    assert cfg.api_key == "primary-key"


def test_from_env_missing_fields_raises_with_clear_message() -> None:
    # No env at all → required-field guard fires
    with pytest.raises(ValueError) as exc:
        LLMProviderConfig.from_env(env={})
    msg = str(exc.value)
    assert "base_url" in msg and "api_key" in msg and "model" in msg


# ── Three concrete provider styles by env only ──────────────────────────


@pytest.mark.parametrize("provider_env", [
    {
        "POWER_LOOP_PROVIDER": "openai",
        "POWER_LOOP_BASE_URL": "https://api.openai.com/v1",
        "POWER_LOOP_API_KEY": "sk-openai",
        "POWER_LOOP_MODEL": "gpt-4o-mini",
    },
    {
        "POWER_LOOP_PROVIDER": "dashscope",
        "POWER_LOOP_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "POWER_LOOP_API_KEY": "sk-ds",
        "POWER_LOOP_MODEL": "qwen-plus",
    },
    {
        "POWER_LOOP_PROVIDER": "deepseek",
        "POWER_LOOP_BASE_URL": "https://api.deepseek.com",
        "POWER_LOOP_API_KEY": "sk-ds",
        "POWER_LOOP_MODEL": "deepseek-chat",
    },
    {
        "POWER_LOOP_PROVIDER": "anthropic",
        "POWER_LOOP_BASE_URL": "https://dashscope.aliyuncs.com/apps/anthropic",
        "POWER_LOOP_API_KEY": "sk-anthropic",
        "POWER_LOOP_MODEL": "deepseek-v4-flash",
    },
])
def test_three_provider_styles_build_without_error(provider_env: dict[str, str]) -> None:
    cfg = LLMProviderConfig.from_env(env=provider_env)
    assert cfg.provider == provider_env["POWER_LOOP_PROVIDER"]
    assert cfg.model == provider_env["POWER_LOOP_MODEL"]
    # Factory creates a real LLMService object — no network call yet.
    svc = create_llm_service_from_config(cfg)
    assert svc is not None
    assert hasattr(svc, "complete") and hasattr(svc, "stream")


def test_anthropic_provider_routes_to_anthropic_transport() -> None:
    cfg = LLMProviderConfig(
        provider="anthropic",
        base_url="https://dashscope.aliyuncs.com/apps/anthropic",
        api_key="sk-test",
        model="deepseek-v4-flash",
    )
    svc = create_llm_service_from_config(cfg)
    assert svc.__class__.__name__ == "AnthropicMessagesLLMService"


def test_create_llm_service_from_env_one_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POWER_LOOP_BASE_URL", "https://x")
    monkeypatch.setenv("POWER_LOOP_API_KEY", "k")
    monkeypatch.setenv("POWER_LOOP_MODEL", "m")
    svc = create_llm_service_from_env()
    assert hasattr(svc, "complete")


# ── to_openai_compatible adapter ────────────────────────────────────────


def test_to_openai_compatible_round_trip() -> None:
    cfg = LLMProviderConfig(
        base_url="https://x", api_key="k", model="m",
        timeout_s=42, max_tokens=2048, temperature=0.7, max_retries=5,
    )
    inner = cfg.to_openai_compatible()
    assert inner.base_url == "https://x"
    assert inner.timeout_s == 42
    assert inner.max_tokens == 2048
    assert inner.max_retries == 5
    assert inner.is_ready
