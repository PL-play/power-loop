# Migration Guide — 0.1.x → 0.2.0

[中文](../zh/migration.md) | [Back to docs](../README.md)

power-loop v0.2.0 is a **hard break** from 0.1.x. The stateless `AgentLoop` is gone; everything revolves around `StatefulAgentLoop` and SQLite-backed `SessionStore`.

## Summary of Changes

| 0.1.x | 0.2.0 |
|---|---|
| `AgentLoop(llm, config).run(messages=[...])` | `StatefulAgentLoop(llm=..., db_path=..., config=...).send(user_input)` |
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
result = await loop.send("hello")
```

### 2. Manage Sessions via session_id

**Before**: you built `messages` lists manually.

**After**: pass `session_id` to continue a conversation:
```python
r1 = await loop.send("My name is Alan.")
r2 = await loop.send("What is my name?", session_id=r1.session_id)
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

**Before**:
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