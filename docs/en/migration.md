# Migration Guide

[中文](../zh/migration.md) | [Back to docs](../README.md)

## 0.13.x → 0.14.0

1. **`llm_client` is vendored.** It now lives at `power_loop._vendor.llm_client` and the
   bare top-level `llm_client` package is gone. Replace any direct import:
   ```python
   # before — no longer importable
   from llm_client.interface import LLMRequest, LLMResponse, LLMService
   # after — use the top-level re-exports
   from power_loop import LLMRequest, LLMResponse, LLMService, LLMStreamChunk, LLMTokenUsage, \
       OpenAICompatibleChatConfig, AnthropicChatConfig
   ```
   The concrete service classes are internal — build a service with the public factories
   `create_llm_service_from_env()` / `create_llm_service_from_config(LLMProviderConfig(...))`.
2. **Transports are extras.** The core trims to `certifi`; install a transport extra:
   `pip install 'power-loop[openai]'` (or `[anthropic]` / `[all]`; also `[skills]` / `[pdf]`).
   Constructing a provider without its SDK raises an `ImportError` with a `pip install` hint.
3. **`requirements.txt` was deleted** — `pyproject.toml` is the single source of truth. Pin via
   `power-loop[openai]==0.14.0`.
4. **`py.typed` is shipped** (PEP 561) — downstream mypy/pyright now see power-loop's types.
5. New surfaces you may want to adopt: per-call LLM events (`LLM_CALL_STARTED` / `LLM_CALL_COMPLETED`),
   the now-real `AGENT_ERROR` channel, the `ts`+`seq` event envelope, and stable `exc.code` error
   codes — see [Events](user-guide/events.md) and the API reference.

## 0.2.x → 0.3.0

power-loop v0.3.0 makes session creation explicit. `send()` no longer creates a session on the first call.

```python
# Before 0.3.0
r1 = await loop.send("My name is Alan.")
r2 = await loop.send("What is my name?", session_id=r1.session_id)

# 0.3.0+
sid = loop.new_session()
r1 = await loop.send("My name is Alan.", session_id=sid)
r2 = await loop.send("What is my name?", session_id=sid)
```

Move per-session metadata from `send(metadata=...)` to `new_session(metadata=...)`:

```python
sid = loop.new_session(metadata={"user_id": "alan"})
result = await loop.send("Hello", session_id=sid)
```

## 0.1.x → 0.2.0

power-loop v0.2.0 is a **hard break** from 0.1.x. The stateless `AgentLoop` is gone; everything revolves around `StatefulAgentLoop` and SQLite-backed `SessionStore`.

## Summary of Changes

| 0.1.x | 0.2.0 |
|---|---|
| `AgentLoop(llm, config).run(messages=[...])` | `sid = loop.new_session(); await loop.send(user_input, session_id=sid)` |
| Caller manages `messages` list | Library loads from `SessionStore` by `session_id` |
| No persistence | `db_path` (default `./power_loop_sessions.db`); `":memory:"` for tests |
| No pending detection | Crash mid-tool → next `send` raises `SessionPendingError` |
| `spawn_agent` with private `AgentLoop` | `register_spawn_agent(registry)`; shared `SessionStore` with parent linking |
| No declarative subagent | `AgentSpec` + `run_agent_spec(spec, input, parent_loop=...)` |
| Compaction via `ContextManager` | `AgentLoopConfig.compactor = DefaultCompactor()` (default-on) |
| `llm_factory.OpenAICompatibleChatConfig` | `LLMProviderConfig` + `create_llm_service_from_env()` |
| `OPENAI_COMPAT_*` env vars | `POWER_LOOP_*` (with legacy fallback) |

## Step-by-Step Migration

### 1. Replace AgentLoop with StatefulAgentLoop

**Before**:
```python
from power_loop import AgentLoop, AgentLoopConfig

loop = AgentLoop(llm=llm, config=AgentLoopConfig(...))
result = loop.run(messages=[{"role": "user", "content": "hello"}])
```

**After**:
```python
from power_loop import StatefulAgentLoop, AgentLoopConfig

loop = StatefulAgentLoop(llm=llm, config=AgentLoopConfig(...))
sid = loop.new_session()
result = await loop.send("hello", session_id=sid)
```

### 2. Manage Sessions via session_id

**Before**: you built `messages` lists manually.

**After**: pass `session_id` to continue a conversation:
```python
sid = loop.new_session()
r1 = await loop.send("My name is Alan.", session_id=sid)
r2 = await loop.send("What is my name?", session_id=sid)
```

### 3. Handle Pending State

**Before**: crashes during tool execution were silent data loss.

**After**: the next `send()` raises `SessionPendingError`:
```python
try:
    result = await loop.send("do something", session_id=sid)
except SessionPendingError:
    loop.abort_pending(sid, reason="user_cancelled")
    result = await loop.send("new input", session_id=sid)
```

### 4. Use LLMProviderConfig

**Before** (legacy external `llm_client` — removed; vendored as `power_loop._vendor.llm_client`
since 0.14.0, see the [0.13.x → 0.14.0](#013x--0140) section above):
```python
from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService

cfg = OpenAICompatibleChatConfig(base_url=..., api_key=..., model=...)
llm = OpenAICompatibleChatLLMService(cfg)
```

**After**:
```python
from power_loop import create_llm_service_from_env

llm = create_llm_service_from_env()
# or
from power_loop import LLMProviderConfig, create_llm_service_from_config
llm = create_llm_service_from_config(LLMProviderConfig(
    base_url="...", api_key="...", model="...",
))
```

### 5. Update Env Var Names

Old names still work (fallback), but prefer the new ones:

```bash
# Old (still works)
OPENAI_COMPAT_BASE_URL=...
OPENAI_COMPAT_API_KEY=...
OPENAI_COMPAT_MODEL=...

# New (preferred)
POWER_LOOP_BASE_URL=...
POWER_LOOP_API_KEY=...
POWER_LOOP_MODEL=...
```

## New Features in 0.2.0

- **LLMRetryPolicy** — retry transient failures with exponential backoff
- **CancellationToken** — unified cancel across threading/asyncio/callable
- **StructuredOutputSpec** — force JSON with schema + repair chain
- **MemoryProvider** — pluggable cross-session recall
- **trim_history** — pure-trim budget helper
- **ToolNotFound / ToolValidationError / SpecValidationError** — typed errors under `PowerLoopError`
- 18 hook points (+`memory.recalled`), 24 event types (+5)
- 15 examples (00–14), 151 tests

## Need Help?

- [Getting Started](getting-started.md)
- [Quickstart](user-guide/quickstart.md)
- [FAQs](faq.md)
