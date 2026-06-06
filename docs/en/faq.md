# FAQ

[中文](../zh/faq.md) | [Back to docs](../README.md)

## General

### What is power-loop?

An embeddable Agent execution kernel for Python. It gives you the Agent loop (LLM → tools → hooks → events → persistence) so you can focus on your domain logic. It is not a framework or platform.

### How is it different from LangChain / CrewAI / AutoGen?

power-loop is a **kernel**, not a framework. It doesn't prescribe workflow DAGs, memory backends, or prompt templates. You bring your own LLM, tools, and memory — the library provides the execution loop, hooks, and persistence. It's ~5k lines of Python, not a 100k-line dependency tree.

### Does it support streaming?

Yes. Subscribe to `STREAM_DELTA` events for real-time token streaming. See [Events](user-guide/events.md).

### Does it work with Anthropic?

Yes. Set `POWER_LOOP_PROVIDER=anthropic` to use the native Anthropic Messages API transport. Anthropic-compatible endpoints such as DashScope's Anthropic app endpoint are supported.

## Configuration

### My LLM credentials aren't being picked up

Check that `POWER_LOOP_BASE_URL`, `POWER_LOOP_API_KEY`, and `POWER_LOOP_MODEL` are set. Legacy `OPENAI_COMPAT_*` names also work as fallback. Use `create_llm_service_from_env()` for the standard path.

### Can I use a local model (Ollama / vLLM)?

Yes. Set `POWER_LOOP_BASE_URL=http://localhost:11434/v1` and `POWER_LOOP_API_KEY=anything` (most local servers don't validate the key).

### How do I disable compaction?

```python
config = AgentLoopConfig(compactor=None)
```

## Sessions

### How do I share a session between processes?

Use the same `db_path` file. Open it in both processes, pass the same `session_id`. The session lives in SQLite. See [Sessions](user-guide/sessions.md).

### Can I delete a session?

```python
loop.close_session(sid, cascade=True)
```

This physically deletes the session and all its messages, compactions, and sub-agents.

### What happens if the process crashes during tool execution?

The session enters a "pending" state. The next `send()` raises `SessionPendingError`. Call `resume()` (finish the tools) or `abort_pending()` (cancel them). See [Sessions](user-guide/sessions.md).

## Tools

### Can I register tools after creating the loop?

No. `ToolRegistry` is set at `StatefulAgentLoop` construction. For dynamic tools, see M2.6 on the roadmap.

### How do I validate tool arguments?

`ToolDefinition.input_schema` (JSON Schema) and `required_params` are both enforced. Invalid args raise `ToolValidationError`.

### Can I block certain tools?

Use the `TOOL_BEFORE` hook:

```python
def block(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "dangerous_tool":
        ctx.output = "[blocked]"
        ctx.directive = HookDirective.SKIP
```

## Hooks vs Events

### When should I use a hook vs an event?

- **Hook**: to change behavior (block tools, modify requests, cache, moderate content).
- **Event**: to observe (streaming, audit, metrics, cost tracking).

Hooks can change control flow; events cannot. See [Hooks](user-guide/hooks.md) and [Events](user-guide/events.md).

## Memory

### Does power-loop include a memory backend?

No. It defines the `MemoryProvider` protocol. You implement `recall()` and `remember()`. See [Memory](user-guide/memory.md) and `examples/13_memory_sqlite.py`.

### Can I use a vector database for memory?

Yes. Implement `MemoryProvider` with your vector DB client in `recall()` and `remember()`. The library doesn't care what backend you use.

## Performance

### How many concurrent sessions can I run?

One `StatefulAgentLoop` instance can drive any number of sessions concurrently (each protected by its own `asyncio.Lock`). For multi-process, each process should have its own `StatefulAgentLoop` instance pointing to the same `db_path`.

### How large can a session get?

The compactor (default-on) keeps sessions within the token budget by summarizing old messages. Without compaction, sessions are limited by the LLM's context window. See [Compaction](user-guide/compaction.md).

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development setup, code style, and PR process.
