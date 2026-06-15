# Configuration

[中文](../../zh/user-guide/configuration.md) | [User Guide](../index.md)

Everything you can tweak — `AgentLoopConfig`, environment variables, and `LLMProviderConfig`.

## AgentLoopConfig

```python
from power_loop import AgentLoopConfig

config = AgentLoopConfig(
    system_prompt="You are a helpful assistant.",
    max_rounds=24,           # max LLM calls per send()
    temperature=0.0,         # 0 = deterministic
    max_tokens=8000,         # per-request token cap
    compactor=DefaultCompactor(),  # default-on; None to disable
    retry_policy=None,       # None = no retry (fail-fast)
    memory=None,             # None = no cross-session memory
    memory_budget_tokens=1500,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str \| None` | `None` | System message prepended to every LLM request |
| `max_rounds` | `int` | `24` | Max LLM + tool rounds per `send()`. 1 = single reply, no tools |
| `temperature` | `float \| None` | `0.0` | LLM temperature |
| `max_tokens` | `int \| None` | `8000` | Per-request token limit |
| `compactor` | `Compactor \| None` | `DefaultCompactor()` | Context compaction; `None` to disable |
| `retry_policy` | `LLMRetryPolicy \| None` | `None` | Retry on transient LLM errors |
| `memory` | `MemoryProvider \| None` | `None` | Cross-session memory provider |
| `memory_budget_tokens` | `int` | `1500` | Token budget passed to `memory.recall()` |

## Environment Variables

The preferred way to configure LLM credentials:

```bash
# Required
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-…
POWER_LOOP_MODEL=gpt-4o-mini

# Optional
POWER_LOOP_PROVIDER=openai          # tag for telemetry
POWER_LOOP_TIMEOUT_S=180            # HTTP timeout
POWER_LOOP_MAX_TOKENS=8000          # per-request cap
POWER_LOOP_TEMPERATURE=0.0
POWER_LOOP_MAX_RETRIES=3
```

Legacy `OPENAI_COMPAT_*` names still work as a fallback.

### One-liner service construction

```python
from power_loop import create_llm_service_from_env

llm = create_llm_service_from_env()
# or with a custom prefix
llm = create_llm_service_from_env(prefix="MY_APP")
```

## LLMProviderConfig (programmatic)

```python
from power_loop import LLMProviderConfig, create_llm_service_from_config

cfg = LLMProviderConfig(
    provider="dashscope",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-…",
    model="qwen-plus",
    max_tokens=4096,
    temperature=0.0,
)
llm = create_llm_service_from_config(cfg)
```

Missing required fields (`base_url`, `api_key`, `model`) raise `ValueError` at construction time — no silent failures on the first `complete()` call.

See [Providers](providers.md) for per-provider snippets (OpenAI, DashScope, DeepSeek, local).

## Compactor Tuning

```python
from power_loop.runtime.compact import DefaultCompactor

compactor = DefaultCompactor(
    trigger_ratio=0.75,       # compact when tokens > 75% of max_tokens
    keep_last_n=4,            # always keep the last 4 exchanges
    summary_max_tokens=512,   # max tokens for the summary LLM call
)
```

Or set an absolute threshold via env: `CONTEXT_COMPACT_THRESHOLD=6000`

Disable compaction: `AgentLoopConfig(compactor=None)`.

## Retry Policy

```python
from power_loop import LLMRetryPolicy

retry = LLMRetryPolicy(
    max_attempts=3,           # 1 initial + 2 retries
    backoff_initial=0.5,      # seconds before second attempt
    backoff_max=8.0,          # cap for exponential backoff
    total_timeout=60.0,       # wall-clock deadline across all attempts
    retry_on=(Exception,),    # default: all Exception subclasses
)

config = AgentLoopConfig(retry_policy=retry, ...)
```

See [Retry & Cancel](retry-cancel.md) for the full retry lifecycle.

## Logging hygiene

`import power_loop` attaches a `logging.NullHandler` to the `power_loop` root logger, so the
library stays silent until your app configures logging (all module loggers live under the
`power_loop.*` subtree).

For structured event logs, attach the JSON-lines sink — one line per event to the
`power_loop.events` logger:

```python
from power_loop.contrib.logging_sink import attach_logging_sink
attach_logging_sink(bus)                          # all events, INFO, secrets redacted
```

It **redacts secret-looking payload keys by default** (`api_key` / `authorization` / `secret` /
`password` / `*_token`, case-insensitive substring; bare `token` is intentionally *not* redacted so
`prompt_tokens`/`completion_tokens` counts survive). Override or disable:

```python
attach_logging_sink(bus, redact_keys=("api_key", "x-internal-secret"))  # custom denylist
attach_logging_sink(bus, redact_keys=())                                 # no redaction
```

## Next

- [Sessions](sessions.md) — understand the session lifecycle
- [Tools](tools.md) — give your agent abilities