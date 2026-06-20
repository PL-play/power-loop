# User Guide

[中文](../../zh/user-guide/index.md) | [Back to docs](../../README.md)

Deep-dive into each feature. These are reference-style pages — for step-by-step walkthroughs, see [Tutorials](../tutorials/index.md).

## Core

| Page | What you'll learn |
|---|---|
| [Installation](installation.md) | pip install, dev setup, Python version requirements |
| [Quickstart](quickstart.md) | Walk through every major feature in one file |
| [Configuration](configuration.md) | `AgentLoopConfig` fields, env vars, `LLMProviderConfig` |

## Features

| Page | What you'll learn |
|---|---|
| [Sessions](sessions.md) | `SessionStore`, multi-turn, cross-process resume, session lifecycle |
| [Storage backends](storage-backends.md) | `dsn=` picks SQLite (default) / PostgreSQL / MySQL; `SchemaPolicy` provisioning, per-backend DDL, preconditions |
| [Tools](tools.md) | `ToolRegistry`, `ToolDefinition`, JSON Schema validation, sync vs async handlers |
| [Extending with tools](extending-tools.md) | The custom-tool recipe, per-call allowlisting, MCP (`contrib.mcp`), why no bundled connectors |
| [Sub-agents](subagents.md) | `spawn_agent`, `AgentSpec`, `run_agent_spec`, lifecycle (EPHEMERAL / LINKED / DETACHED) |
| [Hooks](hooks.md) | All 17 `HookPoint`s, typed Ctx, directives, common patterns |
| [Events](events.md) | All `AgentEventType`s, typed payloads, subscriber patterns |
| [Observability](observability.md) | Durable JSONL sink + `replay`, metrics (Prometheus/StatsD), OpenTelemetry spans, backpressure |
| [Compaction](compaction.md) | `DefaultCompactor`, `trigger_ratio`, keep-last-N, custom `Compactor` |
| [Send-context projection](send-context-projection.md) | Opt-in `HistoryProjector`; per-send plain-text projection (`pl_project_messages`); `ToolDefinition.project`; `recall_send` |
| [Memory](memory.md) | `MemoryProvider` protocol, `recall` / `remember`, injection position |
| [Retry & Cancel](retry-cancel.md) | `LLMRetryPolicy`, exponential backoff, `CancellationToken` |
| [Structured Output](structured-output.md) | `StructuredOutputSpec`, `parse_structured`, JSON repair chain |
| [Timers](timers.md) | Durable self-wake-ups, `TimerRunner`, recurring timers, `TIMER_FIRE` |
| [Advanced Runtime Tools](advanced-runtime-tools.md) | Runtime state, projectors, hooks/events, custom flow control |

## Orchestration & isolation

| Page | What you'll learn |
|---|---|
| [Dynamic Workflows](workflows.md) | `WorkflowSpec` DSL, deterministic engine, detached runs, cross-restart resume |
| [Shared Blackboard](blackboard.md) | `SqliteBlackboard`, `board_*` tools, multi-agent coordination |
| [Sandboxing & Isolation](sandboxing.md) | `ShellBackend` (sandbox bash), `WorkerLauncher` (sandbox a leaf process) |

## Performance & scale

| Page | What you'll learn |
|---|---|
| [Scaling](scaling.md) | Per-session writer model, picking SQLite vs a server backend, retention/VACUUM, the `bench/` harness, measured numbers, sharding & scale-out |

## Advanced

| Page | What you'll learn |
|---|---|
| [Architecture](../architecture.md) | Module boundaries, pipeline phases, invariants |
| [Providers](providers.md) | OpenAI, DashScope, DeepSeek, local — config snippets |
