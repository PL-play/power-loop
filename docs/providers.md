# Providers

[English](en/user-guide/providers.md) | [回到文档站](README.md)

power-loop speaks to LLMs through `LLMProviderConfig` and `LLMService`.
OpenAI-compatible `chat.completions` endpoints use
`OpenAICompatibleChatLLMService`; Anthropic-compatible Messages API
endpoints use `AnthropicMessagesLLMService`.

Build the config one of three ways:

```python
from power_loop import LLMProviderConfig, create_llm_service_from_env

# 1. From env (most apps)
llm = create_llm_service_from_env()                       # reads POWER_LOOP_*

# 2. From env with a custom prefix
llm = create_llm_service_from_env(prefix="DEEPTALK_AGENT")  # reads DEEPTALK_AGENT_*

# 3. Programmatic
from power_loop import create_llm_service_from_config
cfg = LLMProviderConfig(
    provider="dashscope",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-…",
    model="qwen-plus",
    max_tokens=4096, temperature=0.0,
)
llm = create_llm_service_from_config(cfg)
```

The `provider` field is the transport router. Use `provider="anthropic"`
for Anthropic Messages API endpoints; other values route to the
OpenAI-compatible transport.

## Environment variables

Default prefix is `POWER_LOOP`. A legacy `OPENAI_COMPAT_*` fallback is
honoured so older `.env` files keep working without edits.

| Var (with prefix) | Required | Default | Notes |
|---|---|---|---|
| `{PREFIX}_BASE_URL` | ✅ | — | Full chat-completions base, including `/v1` if your provider needs it. |
| `{PREFIX}_API_KEY` | ✅ | — | Sent as `Authorization: Bearer …`. |
| `{PREFIX}_MODEL` | ✅ | — | Provider-specific model ID (e.g. `gpt-4o-mini`, `qwen-plus`, `deepseek-chat`). |
| `{PREFIX}_PROVIDER` | ⬜ | `openai` | Transport router: `anthropic` selects Anthropic Messages API; other values use OpenAI-compatible chat completions. |
| `{PREFIX}_TIMEOUT_S` | ⬜ | `180` | HTTP timeout (seconds, float). |
| `{PREFIX}_MAX_TOKENS` | ⬜ | `8000` | Per-request cap. |
| `{PREFIX}_TEMPERATURE` | ⬜ | `0.0` | |
| `{PREFIX}_MAX_RETRIES` | ⬜ | `3` | Transport-level retry (separate from `LLMRetryPolicy` in M1.1). |

Missing required vars raise `ValueError` at config build time, not on
the first `complete()` call — failing early is intentional so config
mistakes show up under `pytest`, not in production logs.

## Concrete provider snippets

### OpenAI

```bash
export POWER_LOOP_PROVIDER=openai
export POWER_LOOP_BASE_URL=https://api.openai.com/v1
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=gpt-4o-mini
```

### DashScope (Alibaba Qwen) — OpenAI-compatible mode

```bash
export POWER_LOOP_PROVIDER=dashscope
export POWER_LOOP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=qwen-plus
```

### DashScope — Anthropic-compatible mode

```bash
export POWER_LOOP_PROVIDER=anthropic
export POWER_LOOP_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=deepseek-v4-flash
```

### DeepSeek

```bash
export POWER_LOOP_PROVIDER=deepseek
export POWER_LOOP_BASE_URL=https://api.deepseek.com
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=deepseek-chat
```

### Local OpenAI-compatible server (Ollama / vLLM / LM Studio)

```bash
export POWER_LOOP_PROVIDER=local
export POWER_LOOP_BASE_URL=http://localhost:11434/v1
export POWER_LOOP_API_KEY=anything-non-empty
export POWER_LOOP_MODEL=llama3.1
```

## Migrating from the old per-vendor config

Before (each caller hand-rolled this):

```python
from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService

llm = OpenAICompatibleChatLLMService(OpenAICompatibleChatConfig(
    base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
    api_key=os.environ["OPENAI_COMPAT_API_KEY"],
    model=os.environ["OPENAI_COMPAT_MODEL"],
    max_tokens=4096, temperature=0.0,
))
```

After:

```python
from power_loop import create_llm_service_from_env

llm = create_llm_service_from_env()
```

The legacy `OPENAI_COMPAT_*` env names still resolve through the
fallback — no `.env` edits needed during migration. New code should
prefer `POWER_LOOP_*` (or a custom prefix per service via
`create_llm_service_from_env(prefix="MY_SERVICE")`).
