# Providers

[中文](../../zh/user-guide/providers.md) | [User Guide](../index.md)

power-loop speaks to LLMs through `LLMProviderConfig` and `LLMService`. Any provider that exposes an OpenAI-compatible `chat/completions` endpoint works.

## Quick Start

```python
from power_loop import create_llm_service_from_env

llm = create_llm_service_from_env()  # reads POWER_LOOP_* env vars
```

Or programmatic:

```python
from power_loop import LLMProviderConfig, create_llm_service_from_config

cfg = LLMProviderConfig(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-…",
    model="gpt-4o-mini",
)
llm = create_llm_service_from_config(cfg)
```

## Environment Variables

| Var | Required | Default | Notes |
|---|---|---|---|
| `POWER_LOOP_BASE_URL` | Yes | — | Full chat-completions base, including `/v1` |
| `POWER_LOOP_API_KEY` | Yes | — | `Authorization: Bearer …` |
| `POWER_LOOP_MODEL` | Yes | — | Provider-specific model ID |
| `POWER_LOOP_PROVIDER` | No | `openai` | Tag for telemetry |
| `POWER_LOOP_TIMEOUT_S` | No | `180` | HTTP timeout |
| `POWER_LOOP_MAX_TOKENS` | No | `8000` | Per-request cap |
| `POWER_LOOP_TEMPERATURE` | No | `0.0` | |
| `POWER_LOOP_MAX_RETRIES` | No | `3` | Transport-level retry |

Legacy `OPENAI_COMPAT_*` names still work as a fallback.

## Provider Snippets

### OpenAI

```bash
export POWER_LOOP_PROVIDER=openai
export POWER_LOOP_BASE_URL=https://api.openai.com/v1
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=gpt-4o-mini
```

### DashScope (Alibaba Qwen)

```bash
export POWER_LOOP_PROVIDER=dashscope
export POWER_LOOP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=qwen-plus
```

### DeepSeek

```bash
export POWER_LOOP_PROVIDER=deepseek
export POWER_LOOP_BASE_URL=https://api.deepseek.com
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=deepseek-chat
```

### Local (Ollama / vLLM / LM Studio)

```bash
export POWER_LOOP_PROVIDER=local
export POWER_LOOP_BASE_URL=http://localhost:11434/v1
export POWER_LOOP_API_KEY=anything-non-empty
export POWER_LOOP_MODEL=llama3.1
```

## LLMProviderConfig Fields

```python
@dataclass
class LLMProviderConfig:
    base_url: str              # required
    api_key: str               # required
    model: str                 # required
    provider: str = "openai"   # informational tag
    timeout_s: float = 180.0
    max_tokens: int = 8000
    temperature: float = 0.0
    max_retries: int = 3
    extra: dict = {}           # passthrough to the provider
```

Missing required fields raise `ValueError` at construction — fail fast, not on the first `complete()`.

## Custom Prefix

```python
# Reads MY_APP_BASE_URL, MY_APP_API_KEY, MY_APP_MODEL
llm = create_llm_service_from_env(prefix="MY_APP")
```

Useful in multi-service setups where different services talk to different models.

## Next

- [Configuration](configuration.md) — all `AgentLoopConfig` fields
- [Architecture](../architecture.md) — how LLM calls fit into the pipeline