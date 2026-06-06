# Quickstart — Walk Through Every Feature

[中文](../../zh/user-guide/quickstart.md) | [User Guide](../index.md)

This single-page tutorial takes you from `new_session()` + `send(...)` all the way to sub-agents, compaction, and cross-process resume. Each section builds on the last. Run the code as you go.

> **Prerequisites**: Install power-loop and set `POWER_LOOP_*` env vars. See [Getting Started](../getting-started.md).

## 1. Minimal — One Message

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig,
    create_llm_service_from_env,
)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm,
        config=AgentLoopConfig(
            system_prompt="Reply concisely.",
            max_rounds=1,
        ),
    )
    sid = loop.new_session()
    result = await loop.send("Hello!", session_id=sid)
    print(result.final_text)
    # → "Hello! How can I help?"

asyncio.run(main())
```

**Key points**:
- `new_session()` creates the conversation; `send(..., session_id=sid)` appends the user turn and returns a `StatefulResult`.
- `max_rounds=1` means "one LLM call, no tools" — the simplest possible loop.

## 2. Multi-Turn — Keep the Conversation Going

```python
sid = loop.new_session()
r1 = await loop.send("My name is Alan.", session_id=sid)
print(sid)  # e.g., "sess_abc123..."

r2 = await loop.send("What is my name?", session_id=sid)
print(r2.final_text)  # → "Your name is Alan."
```

**Key points**:
- Pass the same `session_id` to continue the same conversation.
- The library loads the full history from SQLite — you never manage `messages` yourself.
- `loop.get_messages(sid)` returns the active history if you need it.

## 3. Tool Calling — Give the Agent Abilities

```python
from power_loop import ToolRegistry, ToolDefinition

def weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    weather,
)

loop = StatefulAgentLoop(
    llm=llm,
    tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt="You have a get_weather tool. Use it.",
        max_rounds=4,  # allow tool call + reply
    ),
)

sid = loop.new_session()
result = await loop.send("What's the weather in Tokyo?", session_id=sid)
# LLM calls get_weather(city="Tokyo") → result.final_text mentions "sunny, 22°C"
```

**Key points**:
- `max_rounds=4` gives the LLM room to call a tool and then respond.
- Tools auto-appear in the OpenAI-compatible `tools` field.
- Async handlers work too — `invoke_async()` detects `async def` at register time.

## 4. Sub-Agents — Delegate to Specialized Agents

```python
from power_loop import register_spawn_agent

# Register both meta-tools: spawn_agent (imperative) and run_agent (declarative)
register_spawn_agent(registry)

loop = StatefulAgentLoop(
    llm=llm,
    tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt=(
            "You can delegate research tasks to a sub-agent using "
            "the spawn_agent tool. For simple tasks, answer directly."
        ),
        max_rounds=6,
    ),
)

sid = loop.new_session()
result = await loop.send(
    "Research: find the population of Tokyo, then tell me if it's "
    "larger than London's population.",
    session_id=sid,
)
# LLM spawns a sub-agent → sub-agent runs its own loop → parent gets result
```

**Key points**:
- `spawn_agent` is a tool the LLM calls; the library runs a child `StatefulAgentLoop`.
- `AgentSpec` (declarative) gives you explicit control: tool whitelist, model, max_rounds.
- Sub-agents get their own SQLite rows, linked to the parent via `parent_session_id`.

## 5. Hooks — Intercept the Loop

```python
from power_loop import AgentHooks, HookPoint, HookDirective
from power_loop.contracts.hook_contexts import ToolBeforeCtx

hooks = AgentHooks()

def block_dangerous(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "bash" and "rm -rf" in str(ctx.tool_args):
        ctx.output = "[blocked]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, block_dangerous)

loop = StatefulAgentLoop(llm=llm, hooks=hooks, ...)
```

**Key points**:
- 18 hook points cover every phase: `session.start`, `round.start`, `llm.before`, `tool.before`, `compact.before`, …
- Hooks can modify messages, skip tools, short-circuit LLM calls, or end the loop.
- Async hooks work — `await` a user confirmation UI before running `bash`.

## 6. Events — Observe Without Interfering

```python
from power_loop import AgentEventBus, AgentEventType

bus = AgentEventBus()

def on_delta(event):
    print(event.data.text, end="", flush=True)  # typewriter effect

bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)

loop = StatefulAgentLoop(llm=llm, event_bus=bus, ...)
```

**Key points**:
- Events are read-only — they cannot change control flow (use hooks for that).
- `bus.subscribe(None, fn)` subscribes to ALL events (audit log, debug).
- 24 event types with typed payloads.

## 7. Persistence — Survive Process Restarts

```python
# Process 1
loop = StatefulAgentLoop(llm=llm, db_path="./chat.db", ...)
sid = loop.new_session()
r1 = await loop.send("Remember: my name is Alan.", session_id=sid)
loop.close()

# Process 2 (hours later, different Python process)
loop2 = StatefulAgentLoop(llm=llm, db_path="./chat.db", ...)
r2 = await loop2.send("What is my name?", session_id=sid)
print(r2.final_text)  # → "Your name is Alan."
```

**Key points**:
- `db_path` points to a real file (default: `./power_loop_sessions.db`); `":memory:"` for tests.
- The session lives in SQLite — open the same file in a new process, pass the same `session_id`, and the LLM sees the full history.
- Works across subprocesses, containers, and restarts.

## 8. What's Next?

| Feature | Where |
|---|---|
| Compaction | [User Guide: Compaction](compaction.md) — auto-summarize long sessions |
| Memory | [User Guide: Memory](memory.md) — cross-session recall via `MemoryProvider` |
| Retry & Cancel | [User Guide: Retry & Cancel](retry-cancel.md) — handle LLM failures gracefully |
| Structured Output | [User Guide: Structured Output](structured-output.md) — force JSON with schema validation |
| Full example list | [Examples](../../../examples/) — 20 runnable examples, each teaching one concept |
