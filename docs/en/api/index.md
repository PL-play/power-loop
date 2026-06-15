# API Reference

[中文](../../zh/api/index.md) | [Back to docs](../../README.md)

This page tracks the public surface that is available from `import power_loop`.
For behavior and examples, use the linked user-guide pages. For exact signatures,
inspect the source modules linked below.

## Stability

| Tier | Meaning |
|---|---|
| Stable | Backward compatible across minor releases. See `power_loop.STABLE_API`. |
| Provisional | Re-exported from `power_loop`, but may change in a future minor. |
| Internal | Submodule imports such as `power_loop.core.*`; no compatibility guarantee. |

## Core

| Symbol | Covers | More |
|---|---|---|
| `StatefulAgentLoop` | `new_session`, `send`, `send_sync`, `follow_up`, `follow_up_sync`, `resume`, `submit_input`, `abort_pending`, `close_session`, `get_messages`, `get_pending` | [Sessions](../user-guide/sessions.md), [source](../../../power_loop/agent/stateful_loop.py) |
| `StatefulResult` | `session_id`, `status`, `final_text`, `rounds`, `pending_tool_calls`, `pending_interactions` | [source](../../../power_loop/agent/stateful_loop.py) |
| `FollowUpQueued` | `session_id`, `queue_depth` — returned when `follow_up()` enqueues steering input for the next round | [Sessions](../user-guide/sessions.md), [source](../../../power_loop/agent/follow_up.py) |
| `AgentLoopConfig` | loop limits, temperature, compaction, retry, memory | [Configuration](../user-guide/configuration.md), [source](../../../power_loop/agent/types.py) |
| `SessionStore` | SQLite sessions, messages, compactions, usage, pending state | [Sessions](../user-guide/sessions.md), [source](../../../power_loop/runtime/session_store.py) |

## Tools and Sub-Agents

| Symbol | Covers | More |
|---|---|---|
| `ToolRegistry` | register, invoke, validate, OpenAI tool conversion, `names()`, restricted `subset()` registries | [Tools](../user-guide/tools.md), [source](../../../power_loop/tools/registry.py) |
| `ToolDefinition` | name, description, JSON Schema, required params | [Tools](../user-guide/tools.md), [source](../../../power_loop/contracts/tools.py) |
| `create_default_tool_registry` | bound or runtime-resolved built-in tool registry | [Tools](../user-guide/tools.md), [source](../../../power_loop/tools/__init__.py) |
| `DEFAULT_TOOL_HANDLERS` | public handler mapping for custom registry composition | [Tools](../user-guide/tools.md), [source](../../../power_loop/tools/default_tools.py) |
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
| `RuntimeEnv`, `runtime_env_context` | per-invocation workspace/home/skills and shell backend | [Tools](../user-guide/tools.md), [source](../../../power_loop/runtime/env.py) |
| `ShellBackend`, `LocalShellBackend` | persistent-shell launch and execution-target identity | [Tools](../user-guide/tools.md), [source](../../../power_loop/runtime/exec_backend.py) |
| `PowerLoopError` and subclasses | common exception hierarchy; every subclass carries a stable dotted `code` (class attribute) — branch on `exc.code`, not class identity | [Error codes](#error-codes), [source](../../../power_loop/contracts/errors.py) |

## LLM Contract

The provider-agnostic LLM types, re-exported from the top level (`from power_loop import …`) so
you don't reach into the vendored transport package. **STABLE as of 1.0** — `LLMService`,
`LLMRequest`, `LLMResponse`, `LLMStreamChunk`, `LLMProviderConfig`, and the
`create_llm_service_from_env`/`_config` factories are frozen so `StatefulAgentLoop` has
construction closure (you can build + use + implement a provider from STABLE symbols).

| Symbol | Covers | More |
|---|---|---|
| `LLMService` | the LLM Protocol you implement or wrap | [Providers](../user-guide/providers.md), [source](../../../power_loop/runtime/provider.py) |
| `LLMRequest`, `LLMResponse`, `LLMStreamChunk`, `LLMTokenUsage` | request/response/stream/usage shapes (used in `llm.*` hooks and custom services) | [Providers](../user-guide/providers.md) |
| `OpenAICompatibleChatConfig`, `AnthropicChatConfig` | per-transport config dataclasses | [Providers](../user-guide/providers.md) |
| `create_llm_service_from_env`, `create_llm_service_from_config`, `LLMProviderConfig` | build an `LLMService` from env/config | [Configuration](../user-guide/configuration.md) |

## Error codes

Every exception inherits from `PowerLoopError` and exposes a stable, machine-readable `code`
(a dotted string). Branch on `exc.code` rather than the class — it's robust across refactors.

| Exception | `code` |
|---|---|
| `PowerLoopError` (base) | `power_loop.error` |
| `SessionNotFoundError` | `session.not_found` |
| `SessionPendingError` | `session.pending` |
| `LLMTimeout` | `llm.timeout` |
| `LLMRetryExhausted` | `llm.retry_exhausted` |
| `CancellationRequested` | `cancelled` |
| `CompactionFailed` | `compaction.failed` |
| `ToolNotFound` | `tool.not_found` |
| `ToolValidationError` | `tool.invalid_args` |
| `SpecValidationError` | `spec.invalid` |
