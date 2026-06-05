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
| [Tools](tools.md) | `ToolRegistry`, `ToolDefinition`, JSON Schema validation, sync vs async handlers |
| [Sub-agents](subagents.md) | `spawn_agent`, `AgentSpec`, `run_agent_spec`, lifecycle (EPHEMERAL / LINKED / DETACHED) |
| [Hooks](hooks.md) | All 18 `HookPoint`s, typed Ctx, directives, common patterns |
| [Events](events.md) | All 24 `AgentEventType`s, typed payloads, subscriber patterns |
| [Compaction](compaction.md) | `DefaultCompactor`, `trigger_ratio`, keep-last-N, custom `Compactor` |
| [Memory](memory.md) | `MemoryProvider` protocol, `recall` / `remember`, injection position |
| [Retry & Cancel](retry-cancel.md) | `LLMRetryPolicy`, exponential backoff, `CancellationToken` |
| [Structured Output](structured-output.md) | `StructuredOutputSpec`, `parse_structured`, JSON repair chain |

## Advanced

| Page | What you'll learn |
|---|---|
| [Architecture](../architecture.md) | Module boundaries, pipeline phases, invariants |
| [Providers](providers.md) | OpenAI, DashScope, DeepSeek, local — config snippets |