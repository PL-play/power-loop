# API 参考

[English](../../en/api/index.md) | [回到文档站](../../README.md)

本页跟踪 `import power_loop` 可用的公开表面。行为说明和示例请看关联的用户手册；精确签名以源码链接为准。

## 稳定性

| 层级 | 含义 |
|---|---|
| Stable | 跨 minor 版本保持向后兼容，列表见 `power_loop.STABLE_API`。 |
| Provisional | 从 `power_loop` 顶层导出，但未来 minor 可能调整。 |
| Internal | `power_loop.core.*` 等子模块导入，无兼容性承诺。 |

## 核心

| 符号 | 覆盖 | 更多 |
|---|---|---|
| `StatefulAgentLoop` | `new_session`、`send`、`send_sync`、`follow_up`、`follow_up_sync`、`resume`、`submit_input`、`abort_pending`、`close_session`、`get_messages`、`get_pending` | [会话](../user-guide/sessions.md)、[源码](../../../power_loop/agent/stateful_loop.py) |
| `StatefulResult` | `session_id`、`status`、`final_text`、`rounds`、`pending_tool_calls`、`pending_interactions` | [源码](../../../power_loop/agent/stateful_loop.py) |
| `FollowUpQueued` | `session_id`、`queue_depth` — `follow_up()` 在运行中入队时返回，表示下一轮会注入指引 | [会话](../user-guide/sessions.md)、[源码](../../../power_loop/agent/follow_up.py) |
| `AgentLoopConfig` | loop 限制、temperature、压缩、重试、记忆 | [配置](../user-guide/configuration.md)、[源码](../../../power_loop/agent/types.py) |
| `SessionStore` | SQLite 会话、消息、压缩、usage、pending 状态 | [会话](../user-guide/sessions.md)、[源码](../../../power_loop/runtime/session_store.py) |

## 工具与子代理

| 符号 | 覆盖 | 更多 |
|---|---|---|
| `ToolRegistry` | 注册、调用、校验、OpenAI tool 转换、`names()` 与受限 `subset()` registry | [工具](../user-guide/tools.md)、[源码](../../../power_loop/tools/registry.py) |
| `ToolDefinition` | 名称、描述、JSON Schema、必填参数 | [工具](../user-guide/tools.md)、[源码](../../../power_loop/contracts/tools.py) |
| `create_default_tool_registry` | 已绑定或运行时解析的内置工具 registry | [工具](../user-guide/tools.md)、[源码](../../../power_loop/tools/__init__.py) |
| `DEFAULT_TOOL_HANDLERS` | 用于自定义 registry 组合的公开 handler 映射 | [工具](../user-guide/tools.md)、[源码](../../../power_loop/tools/default_tools.py) |
| `AgentSpec` | 声明式子代理规格 | [子代理](../user-guide/subagents.md)、[源码](../../../power_loop/runtime/spec.py) |
| `run_agent_spec` | 直接执行子代理 | [子代理](../user-guide/subagents.md)、[源码](../../../power_loop/runtime/spec.py) |
| `register_spawn_agent` | `spawn_agent` 和 `run_agent` meta-tool | [子代理](../user-guide/subagents.md)、[源码](../../../power_loop/tools/spawn_agent.py) |

## Hooks 与 Events

| 符号 | 覆盖 | 更多 |
|---|---|---|
| `AgentHooks` | 注册同步/异步 hook | [Hooks](../user-guide/hooks.md)、[完整参考](../../hooks.md) |
| `HookPoint` | 生命周期 hook 枚举 | [Hooks](../user-guide/hooks.md)、[源码](../../../power_loop/contracts/hooks.py) |
| `HookDirective` | continue、skip、break、short-circuit | [Hooks](../user-guide/hooks.md)、[源码](../../../power_loop/contracts/hooks.py) |
| `AgentEventBus` | event 订阅和发布 | [Events](../user-guide/events.md)、[完整参考](../../events.md) |
| `AgentEventType` | 类型化 event 名称 | [Events](../user-guide/events.md)、[源码](../../../power_loop/contracts/events.py) |

## 运行时辅助

| 符号 | 覆盖 | 更多 |
|---|---|---|
| `LLMRetryPolicy` | 尝试次数、退避、超时、重试过滤 | [重试与取消](../user-guide/retry-cancel.md)、[源码](../../../power_loop/runtime/retry.py) |
| `CancellationToken` | 统一取消形状 | [重试与取消](../user-guide/retry-cancel.md)、[源码](../../../power_loop/runtime/cancellation.py) |
| `StructuredOutputSpec` | response-format schema | [结构化输出](../user-guide/structured-output.md)、[源码](../../../power_loop/runtime/structured.py) |
| `MemoryProvider` | `recall` / `remember` 协议 | [记忆](../user-guide/memory.md)、[源码](../../../power_loop/runtime/memory.py) |
| `LLMProviderConfig` | provider/env 配置 | [Providers](../user-guide/providers.md)、[源码](../../../power_loop/runtime/provider.py) |
| `DefaultCompactor` | 上下文摘要压缩 | [压缩](../user-guide/compaction.md)、[源码](../../../power_loop/runtime/compact.py) |
| `HistoryProjector`、`DefaultDeterministicProjector`、`IdentityProjector` | Send 上下文投影协议 + 内置实现(`trigger_ratio` token 折叠);把已结束 send 投影进派生表 `pl_project_messages`(`ProjectMessageRow`);`ToolDefinition.project` 钩子 + `recall_send` 工具取回明细 | [Send 上下文投影](../user-guide/send-context-projection.md)、[源码](../../../power_loop/runtime/history_projector.py) |
| `RuntimeEnv`, `runtime_env_context` | 每次调用的 workspace/home/skills 与 shell backend | [工具](../user-guide/tools.md)、[源码](../../../power_loop/runtime/env.py) |
| `ShellBackend`, `LocalShellBackend` | 持久 shell 启动方式与执行目标标识 | [工具](../user-guide/tools.md)、[源码](../../../power_loop/runtime/exec_backend.py) |
| `PowerLoopError` 及子类 | 通用异常层级；每个子类带稳定的点分 `code`（类属性）—— 按 `exc.code` 分支，而非类身份 | [错误码](#错误码), [源码](../../../power_loop/contracts/errors.py) |

## LLM 契约

provider 无关的 LLM 类型，从顶层 re-export（`from power_loop import …`），不必伸进 vendored
transport 包。**1.0 起 STABLE** —— `LLMService`、`LLMRequest`、`LLMResponse`、`LLMStreamChunk`、
`LLMProviderConfig` 与 `create_llm_service_from_env`/`_config` 工厂已冻结,使 `StatefulAgentLoop`
具备「构造闭包」(可仅用 STABLE 符号构造、使用、实现自定义 provider)。

| 符号 | 覆盖 | 更多 |
|---|---|---|
| `LLMService` | 你实现或包装的 LLM Protocol | [Providers](../user-guide/providers.md), [源码](../../../power_loop/runtime/provider.py) |
| `LLMRequest`, `LLMResponse`, `LLMStreamChunk`, `LLMTokenUsage` | 请求/响应/流/用量形状（用于 `llm.*` hook 和自定义 service） | [Providers](../user-guide/providers.md) |
| `OpenAICompatibleChatConfig`, `AnthropicChatConfig` | 各 transport 的配置 dataclass | [Providers](../user-guide/providers.md) |
| `create_llm_service_from_env`, `create_llm_service_from_config`, `LLMProviderConfig` | 从 env/config 构造 `LLMService` | [配置](../user-guide/configuration.md) |

## 错误码

每个异常都继承 `PowerLoopError` 并带稳定、机器可读的 `code`（点分串）。按 `exc.code` 分支
而非类身份 —— 重构友好。

| 异常 | `code` |
|---|---|
| `PowerLoopError`（基类） | `power_loop.error` |
| `SessionNotFoundError` | `session.not_found` |
| `SessionPendingError` | `session.pending` |
| `LLMTimeout` | `llm.timeout` |
| `LLMRetryExhausted` | `llm.retry_exhausted` |
| `CancellationRequested` | `cancelled` |
| `CompactionFailed` | `compaction.failed` |
| `ToolNotFound` | `tool.not_found` |
| `ToolValidationError` | `tool.invalid_args` |
| `SpecValidationError` | `spec.invalid` |
