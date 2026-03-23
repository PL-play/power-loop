from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model: str
    api_family: str = "chat"
    supports_image_input: bool = False
    supports_pdf_input_chat: bool = False
    supports_pdf_input_responses: bool = False
    supports_data_url: bool = True
    supports_tools: bool = True
    supports_stream: bool = True


def _caps(provider: str, model: str, **kwargs: Any) -> ModelCapabilities:
    return ModelCapabilities(provider=provider, model=model, **kwargs)


CAPABILITY_OVERRIDE_ENV_MAP: dict[str, str] = {
    "OPENAI_COMPAT_SUPPORTS_IMAGE_INPUT": "supports_image_input",
    "OPENAI_COMPAT_SUPPORTS_PDF_INPUT_CHAT": "supports_pdf_input_chat",
    "OPENAI_COMPAT_SUPPORTS_PDF_INPUT_RESPONSES": "supports_pdf_input_responses",
    "OPENAI_COMPAT_SUPPORTS_DATA_URL": "supports_data_url",
    "OPENAI_COMPAT_SUPPORTS_TOOLS": "supports_tools",
    "OPENAI_COMPAT_SUPPORTS_STREAM": "supports_stream",
}


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def capability_overrides_from_env(env: Mapping[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for env_name, field_name in CAPABILITY_OVERRIDE_ENV_MAP.items():
        parsed = _parse_optional_bool(env.get(env_name))
        if parsed is not None:
            overrides[field_name] = parsed
    return overrides


PROVIDER_DEFAULTS: dict[str, Dict[str, Any]] = {
    "openai": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "openai-compatible": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "deepseek": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "zhipu": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "moonshot": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "minimax": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "dashscope": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "volcengine": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "google": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
    "anthropic": {"api_family": "chat", "supports_data_url": True, "supports_tools": True, "supports_stream": True},
}


MODEL_PATTERNS: list[tuple[re.Pattern[str], Dict[str, Any]]] = [
    (re.compile(r"^(gpt-4o|gpt-4\.1|gpt-4\.5|o1|o3)", re.IGNORECASE), {
        "provider": "openai",
        "supports_image_input": True,
        "supports_pdf_input_responses": True,
    }),
    (re.compile(r"^(qwen3-vl|qwen-vl|qwen2(\.5)?-vl|qwen-vl-max|qwen-vl-plus|qvq|max-vl|internvl|qwen-omni)", re.IGNORECASE), {
        "provider": "dashscope",
        "supports_image_input": True,
    }),
    (re.compile(r"^(glm-4v|glm-4\.1v|glm-4\.5v|glm-4v-plus|glm-4v-thinking|cogvlm)", re.IGNORECASE), {
        "provider": "zhipu",
        "supports_image_input": True,
    }),
    (re.compile(r"^(glm-4|glm-4-plus|glm-4-air|glm-zero-preview)", re.IGNORECASE), {
        "provider": "zhipu",
    }),
    (re.compile(r"^(deepseek-vl|deepseek-vl2)", re.IGNORECASE), {
        "provider": "deepseek",
        "supports_image_input": True,
    }),
    (re.compile(r"^(deepseek-chat|deepseek-reasoner|deepseek-coder|deepseek-r1)", re.IGNORECASE), {
        "provider": "deepseek",
    }),
    (re.compile(r"^(kimi-vl|moonshot-v1-vision|moonshot-vision)", re.IGNORECASE), {
        "provider": "moonshot",
        "supports_image_input": True,
    }),
    (re.compile(r"^(kimi-k2|kimi-latest|moonshot-v1-(8k|32k|128k)|moonshot-kimi)", re.IGNORECASE), {
        "provider": "moonshot",
    }),
    (re.compile(r"^(minimax-vl|minimax-vl-01|abab[\-_]vision|minimax-vision)", re.IGNORECASE), {
        "provider": "minimax",
        "supports_image_input": True,
    }),
    (re.compile(r"^(abab6(\.5)?-chat|minimax-text|minimax-m1)", re.IGNORECASE), {
        "provider": "minimax",
    }),
    (re.compile(r"^(doubao-vision|doubao-1\.5-vision|doubao-seed-vision)", re.IGNORECASE), {
        "provider": "volcengine",
        "supports_image_input": True,
    }),
    (re.compile(r"^(doubao|doubao-1\.5|seed)", re.IGNORECASE), {
        "provider": "volcengine",
    }),
    (re.compile(r"^(gemini|gemini-1\.5|gemini-2\.0)", re.IGNORECASE), {
        "provider": "google",
        "supports_image_input": True,
        "supports_pdf_input_responses": True,
    }),
    (re.compile(r"^(claude-3|claude-3\.5|claude-3\.7|claude-sonnet|claude-opus|claude-4)", re.IGNORECASE), {
        "provider": "anthropic",
        "supports_image_input": True,
        "supports_pdf_input_responses": True,
    }),
]


def resolve_model_capabilities(model: str, base_url: str, overrides: Dict[str, Any] | None = None) -> ModelCapabilities:
    normalized_model = (model or "").strip()
    provider = "openai-compatible"
    caps = _caps(provider=provider, model=normalized_model, **PROVIDER_DEFAULTS.get(provider, {}))

    for pattern, payload in MODEL_PATTERNS:
        if pattern.search(normalized_model):
            merged = {
                "provider": payload.get("provider", provider),
                "model": normalized_model,
                "api_family": payload.get("api_family", caps.api_family),
                "supports_image_input": payload.get("supports_image_input", caps.supports_image_input),
                "supports_pdf_input_chat": payload.get("supports_pdf_input_chat", caps.supports_pdf_input_chat),
                "supports_pdf_input_responses": payload.get("supports_pdf_input_responses", caps.supports_pdf_input_responses),
                "supports_data_url": payload.get("supports_data_url", caps.supports_data_url),
                "supports_tools": payload.get("supports_tools", caps.supports_tools),
                "supports_stream": payload.get("supports_stream", caps.supports_stream),
            }
            caps = ModelCapabilities(**merged)
            break

    if overrides:
        merged = caps.__dict__ | dict(overrides)
        merged.setdefault("provider", caps.provider)
        merged.setdefault("model", normalized_model)
        caps = ModelCapabilities(**merged)

    return caps