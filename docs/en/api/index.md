# API Reference

[中文](../../zh/api/index.md) | [Back to docs](../../README.md)

Signature-level documentation for every public symbol. These are reference pages — for conceptual explanations, see the [User Guide](../user-guide/index.md).

## Core

| Page | Covers |
|---|---|
| [StatefulAgentLoop](stateful-loop.md) | `send()` / `send_sync()` / `resume()` / `abort_pending()` / `close_session()` / `close()` / `get_messages()` / `get_pending()` |
| [StatefulResult](stateful-result.md) | `session_id` / `status` / `final_text` / `rounds` / `pending_tool_calls` |
| [AgentLoopConfig](config.md) | `system_prompt` / `max_rounds` / `temperature` / `max_tokens` / `compactor` / `retry_policy` / `memory` / `memory_budget_tokens` |
| [SessionStore](session-store.md) | `open()` / `create_session()` / `append_message()` / `load_active_messages()` / `load_all_messages()` / `close()` |

## Tools

| Page | Covers |
|---|---|
| [ToolRegistry](tool-registry.md) | `register()` / `unregister()` / `invoke()` / `invoke_async()` / `validate()` / `to_openai_tools()` |
| [ToolDefinition](tool-definition.md) | `name` / `description` / `input_schema` / `required_params` |
| `AsyncToolInSyncContext` | Raised when sync `invoke()` called on async handler |

## Hooks & Events

| Page | Covers |
|---|---|
| [Hooks](hooks.md) | `AgentHooks` / `HookPoint` (18 values) / `HookDirective` (4 values) / all `*Ctx` dataclasses |
| [Events](events.md) | `AgentEventBus` / `AgentEventType` (24 values) / `AgentEvent` / all `*Payload` dataclasses |

## Runtime

| Page | Covers |
|---|---|
| [Errors](errors.md) | `PowerLoopError` + 11 subclasses (`SessionNotFoundError` / `LLMTimeout` / `ToolNotFound` / …) |
| [LLMRetryPolicy](retry-policy.md) | `max_attempts` / `backoff_initial` / `backoff_max` / `total_timeout` / `retry_on` |
| [CancellationToken](cancellation.md) | `from_any()` / `is_cancelled()` / `raise_if_cancelled()` / `cancel()` |
| [StructuredOutputSpec](structured-output.md) | `name` / `schema` / `strict` / `to_openai_response_format()` |
| [MemoryProvider](memory.md) | Protocol: `recall()` / `remember()`; `MemorySnapshot` |
| [LLMProviderConfig](provider-config.md) | `from_env()` / `to_openai_compatible()` / `create_llm_service_from_config()` |
| [Compactor](compactor.md) | `Compactor` Protocol / `DefaultCompactor` / `CompactionPlan` |

## Sub-agents

| Page | Covers |
|---|---|
| [AgentSpec](agent-spec.md) | `name` / `system_prompt` / `tools` / `max_rounds` / `model` / `lifecycle` |
| [run_agent_spec](run-agent-spec.md) | `run_agent_spec(spec, input, *, parent_loop)` / `filtered_registry()` |