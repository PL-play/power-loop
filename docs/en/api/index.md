# API Reference

[中文](../../zh/api/index.md) | [Back to docs](../../README.md)

This page tracks the public surface that is available from `import power_loop`.
For behavior and examples, use the linked user-guide pages. For exact signatures,
inspect the source modules linked below.

## Stability

| Tier | Meaning |
|---|---|
| Stable | Backward compatible across minor releases. See `power_loop.STABLE_API`. |
| Provisional | Re-exported from `power_loop` during 0.x, but may change. |
| Internal | Submodule imports such as `power_loop.core.*`; no compatibility guarantee. |

## Core

| Symbol | Covers | More |
|---|---|---|
| `StatefulAgentLoop` | `new_session`, `send`, `send_sync`, `resume`, `abort_pending`, `close_session`, `get_messages`, `get_pending` | [Sessions](../user-guide/sessions.md), [source](../../../power_loop/agent/stateful_loop.py) |
| `StatefulResult` | `session_id`, `status`, `final_text`, `rounds`, `pending_tool_calls` | [source](../../../power_loop/agent/stateful_loop.py) |
| `AgentLoopConfig` | loop limits, temperature, compaction, retry, memory | [Configuration](../user-guide/configuration.md), [source](../../../power_loop/agent/types.py) |
| `SessionStore` | SQLite sessions, messages, compactions, usage, pending state | [Sessions](../user-guide/sessions.md), [source](../../../power_loop/runtime/session_store.py) |

## Tools and Sub-Agents

| Symbol | Covers | More |
|---|---|---|
| `ToolRegistry` | register, invoke, validate, OpenAI tool conversion | [Tools](../user-guide/tools.md), [source](../../../power_loop/tools/registry.py) |
| `ToolDefinition` | name, description, JSON Schema, required params | [Tools](../user-guide/tools.md), [source](../../../power_loop/contracts/tools.py) |
| `AgentSpec` | declarative child-agent spec | [Sub-agents](../user-guide/subagents.md), [source](../../../power_loop/runtime/spec.py) |
| `run_agent_spec` | direct sub-agent execution | [Sub-agents](../user-guide/subagents.md), [source](../../../power_loop/runtime/spec.py) |
| `register_spawn_agent` | `spawn_agent` and `run_agent` meta-tools | [Sub-agents](../user-guide/subagents.md), [source](../../../power_loop/tools/spawn_agent.py) |

## Hooks and Events

| Symbol | Covers | More |
|---|---|---|
| `AgentHooks` | registering sync/async hooks | [Hooks](../user-guide/hooks.md), [full reference](../../hooks.md) |
| `HookPoint` | lifecycle hook enum | [Hooks](../user-guide/hooks.md), [source](../../../power_loop/contracts/hooks.py) |
| `HookDirective` | continue, skip, break, short-circuit | [Hooks](../user-guide/hooks.md), [source](../../../power_loop/contracts/hooks.py) |
| `AgentEventBus` | event subscriptions and publication | [Events](../user-guide/events.md), [full reference](../../events.md) |
| `AgentEventType` | typed event names | [Events](../user-guide/events.md), [source](../../../power_loop/contracts/events.py) |

## Runtime Helpers

| Symbol | Covers | More |
|---|---|---|
| `LLMRetryPolicy` | attempts, backoff, timeout, retry filter | [Retry & Cancel](../user-guide/retry-cancel.md), [source](../../../power_loop/runtime/retry.py) |
| `CancellationToken` | uniform cancellation shape | [Retry & Cancel](../user-guide/retry-cancel.md), [source](../../../power_loop/runtime/cancellation.py) |
| `StructuredOutputSpec` | response-format schema | [Structured Output](../user-guide/structured-output.md), [source](../../../power_loop/runtime/structured.py) |
| `MemoryProvider` | recall and remember protocol | [Memory](../user-guide/memory.md), [source](../../../power_loop/runtime/memory.py) |
| `LLMProviderConfig` | provider/env configuration | [Providers](../user-guide/providers.md), [source](../../../power_loop/runtime/provider.py) |
| `DefaultCompactor` | context summary compaction | [Compaction](../user-guide/compaction.md), [source](../../../power_loop/runtime/compact.py) |
| `PowerLoopError` and subclasses | common exception hierarchy | [source](../../../power_loop/contracts/errors.py) |
