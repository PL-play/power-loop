# API 参考

[English](../../en/api/index.md) | [回到文档站](../../README.md)

每个公开符号的签名级文档。这些是参考页面——概念解释见 [用户手册](../user-guide/index.md)。

## 核心

| 页面 | 覆盖 |
|---|---|
| [StatefulAgentLoop](stateful-loop.md) | `send()` / `send_sync()` / `resume()` / `abort_pending()` / `close_session()` / `close()` / `get_messages()` / `get_pending()` |
| [StatefulResult](stateful-result.md) | `session_id` / `status` / `final_text` / `rounds` / `pending_tool_calls` |
| [AgentLoopConfig](config.md) | `system_prompt` / `max_rounds` / `temperature` / `max_tokens` / `compactor` / `retry_policy` / `memory` / `memory_budget_tokens` |
| [SessionStore](session-store.md) | `open()` / `create_session()` / `append_message()` / `load_active_messages()` / `load_all_messages()` / `close()` |

## 工具

| 页面 | 覆盖 |
|---|---|
| [ToolRegistry](tool-registry.md) | `register()` / `unregister()` / `invoke()` / `invoke_async()` / `validate()` / `to_openai_tools()` |
| [ToolDefinition](tool-definition.md) | `name` / `description` / `input_schema` / `required_params` |
| `AsyncToolInSyncContext` | sync `invoke()` 对 async handler 调用时抛出 |

## Hooks & Events

| 页面 | 覆盖 |
|---|---|
| [Hooks](hooks.md) | `AgentHooks` / `HookPoint`（18 个值）/ `HookDirective`（4 个值）/ 全部 `*Ctx` dataclass |
| [Events](events.md) | `AgentEventBus` / `AgentEventType`（24 个值）/ `AgentEvent` / 全部 `*Payload` dataclass |

## 运行时

| 页面 | 覆盖 |
|---|---|
| [Errors](errors.md) | `PowerLoopError` + 11 个子类 |
| [LLMRetryPolicy](retry-policy.md) | `max_attempts` / `backoff_initial` / `backoff_max` / `total_timeout` / `retry_on` |
| [CancellationToken](cancellation.md) | `from_any()` / `is_cancelled()` / `raise_if_cancelled()` / `cancel()` |
| [StructuredOutputSpec](structured-output.md) | `name` / `schema` / `strict` / `to_openai_response_format()` |
| [MemoryProvider](memory.md) | 协议：`recall()` / `remember()`；`MemorySnapshot` |
| [LLMProviderConfig](provider-config.md) | `from_env()` / `to_openai_compatible()` / `create_llm_service_from_config()` |
| [Compactor](compactor.md) | `Compactor` 协议 / `DefaultCompactor` / `CompactionPlan` |

## 子代理

| 页面 | 覆盖 |
|---|---|
| [AgentSpec](agent-spec.md) | `name` / `system_prompt` / `tools` / `max_rounds` / `model` / `lifecycle` |
| [run_agent_spec](run-agent-spec.md) | `run_agent_spec(spec, input, *, parent_loop)` / `filtered_registry()` |