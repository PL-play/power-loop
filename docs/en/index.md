# power-loop — English Documentation

[中文](../zh/index.md) | [Back to docs](../README.md)

power-loop is an **embeddable Agent execution kernel** — not a framework, not a platform. It gives you the Agent loop so you can focus on your domain logic.

## Where to start

| Path | For |
|---|---|
| [Getting Started](getting-started.md) | First time? Start here. 5 min. |
| [User Guide](user-guide/index.md) | Deep-dive into each feature. |
| [Tutorials](tutorials/index.md) | Step-by-step project walkthroughs. |
| [API Reference](api/index.md) | Signature-level documentation. |
| [Architecture](architecture.md) | How the internals fit together. |
| [Migration Guide](migration.md) | Upgrading from 0.1.x. |
| [FAQ](faq.md) | Common questions. |

## What power-loop gives you

- **LLM abstraction** — one `LLMService` interface, multiple transports (OpenAI-compatible, Anthropic Messages API)
- **Multi-turn loop** — `new_session()` creates the conversation; `send(user_input, session_id=sid)` appends turns
- **Tool calling** — register tools with JSON Schema validation; sync and async handlers
- **Hooks** — 17 hook points to intercept every phase of the loop
- **Event bus** — 30 typed events for observability, audit, streaming
- **Declarative sub-agents** — `AgentSpec` → one-shot child agent with tool whitelist
- **Context compaction** — automatic LLM-summary compaction, default-on
- **Session persistence** — pluggable store: zero-infra SQLite by default, or PostgreSQL/MySQL by DSN ([Storage backends](user-guide/storage-backends.md)); the loop is stateless, so any session resumes cross-process from a `dsn` + `session_id`
- **Retry + cancel** — `LLMRetryPolicy` with exponential backoff; `CancellationToken` for any cancel shape
- **Structured output** — `StructuredOutputSpec` + JSON repair chain
- **Pluggable memory** — `MemoryProvider` protocol for cross-session recall
- **Runtime-bound tools** — durable tool state, projectors, hooks, and events for advanced flow control

## API stability

See the [API Reference](api/index.md) for the three-tier stability guarantee (Stable / Provisional / Internal).
