# API 参考

[English](../../en/api/index.md) | [回到文档站](../../README.md)

本页跟踪 `import power_loop` 可用的公开表面。行为说明和示例请看关联的用户手册；精确签名以源码链接为准。

## 稳定性

| 层级 | 含义 |
|---|---|
| Stable | 跨 minor 版本保持向后兼容，列表见 `power_loop.STABLE_API`。 |
| Provisional | 从 `power_loop` 顶层导出，但未来 minor 可能调整。 |
| Internal | `power_loop.core.*` 等子模块导入，无兼容性承诺。 |

## 当前运行时（11.0.0）

见[运行时装配](../../runtime-11.md)和 `examples/design203_single_process.py`。
`StatefulAgentLoop(runtime=...)` 委托 RuntimeClient；旧直接构造、同步执行和进程启动接口已退出。
受理回执与持久结果分别查询。存储由 `SessionStore.open`（SQLite）或 `open_store`（DSN）装配。

## 工具与子代理

| 符号 | 覆盖 | 更多 |
|---|---|---|
| `ToolRegistry` | 注册、调用、校验、OpenAI tool 转换、`names()` 与受限 `subset()` registry | [工具](../user-guide/tools.md)、[源码](../../../power_loop/tools/registry.py) |
| `ToolDefinition` | 名称、描述、JSON Schema、必填参数 | [工具](../user-guide/tools.md)、[源码](../../../power_loop/contracts/tools.py) |
| `create_default_tool_registry` | 已绑定或运行时解析的内置工具 registry | [工具](../user-guide/tools.md)、[源码](../../../power_loop/tools/__init__.py) |
| `DEFAULT_TOOL_HANDLERS` | 用于自定义 registry 组合的公开 handler 映射 | [工具](../user-guide/tools.md)、[源码](../../../power_loop/tools/default_tools.py) |
| `AgentSpec` | 声明式子代理规格 | [子代理](../user-guide/subagents.md)、[源码](../../../power_loop/runtime/spec.py) |
| `run_agent_spec` | 可信宿主 command 受理子任务并等待持久结果 | [子代理](../user-guide/subagents.md)、[源码](../../../power_loop/runtime/spec.py) |
| `register_spawn_agent` | `spawn_agent` meta-tool（4.0 起 `run_agent` 已并入） | [子代理](../user-guide/subagents.md)、[源码](../../../power_loop/tools/spawn_agent.py) |

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
| `FoldStrategy`、`LLMSummaryFold`、`AgenticFold` | `fold_strategy` 轴：fold 协议 + 内置实现(`keep_last_sends`、`trigger_ratio`、async `fold`)；上下文摘要压缩 | [压缩](../user-guide/compaction.md)、[源码](../../../power_loop/runtime/fold.py) |
| `Representation`、`VerbatimRepresentation`、`ProjectedRepresentation` | `representation` 轴：representation 协议 + 内置实现(逐字默认 / `ProjectedRepresentation(max_chars=...)`);把已结束 send 投影进派生表 `pl_project_messages`(`ProjectMessageRow`);`ToolDefinition.project` 钩子 + `recall_send` 工具取回明细 | [Send 上下文投影](../user-guide/send-context-projection.md)、[源码](../../../power_loop/runtime/representation.py) |
| `RuntimeEnv`, `runtime_env_context` | 每次调用的 workspace/home/skills 与 shell backend | [工具](../user-guide/tools.md)、[源码](../../../power_loop/runtime/env.py) |
| `ShellBackend`, `LocalShellBackend` | 持久 shell 启动方式与执行目标标识 | [工具](../user-guide/tools.md)、[源码](../../../power_loop/runtime/exec_backend.py) |
| `PowerLoopError` 及子类 | 通用异常层级；每个子类带稳定的点分 `code`（类属性）—— 按 `exc.code` 分支，而非类身份 | [错误码](#错误码), [源码](../../../power_loop/contracts/errors.py) |

## LLM 契约

provider 无关的 LLM 类型，从顶层 re-export（`from power_loop import …`），不必伸进 vendored
transport 包。**1.0 起 STABLE** —— `LLMService`、`LLMRequest`、`LLMResponse`、`LLMStreamChunk`、
`LLMProviderConfig` 与 `create_llm_service_from_env`/`_config` 工厂供当前 worker 装配、持有模型连接。

| 符号 | 覆盖 | 更多 |
|---|---|---|
| `LLMService` | 你实现或包装的 LLM Protocol | [Providers](../user-guide/providers.md), [源码](../../../power_loop/runtime/provider.py) |
| `LLMRequest`, `LLMResponse`, `LLMStreamChunk`, `LLMTokenUsage` | 请求/响应/流/用量形状（用于 `llm.*` hook 和自定义 service） | [Providers](../user-guide/providers.md) |
| `OpenAICompatibleChatConfig`, `AnthropicChatConfig` | 各 transport 的配置 dataclass | [Providers](../user-guide/providers.md) |
| `create_llm_service_from_env`, `create_llm_service_from_config`, `LLMProviderConfig` | 从 env/config 构造 `LLMService` | [配置](../user-guide/configuration.md) |

## 错误码

`RuntimeRequestError` 继承 ValueError，实例带 `code`，包括 capability_unavailable、resource_unavailable、definition_unavailable、forbidden 和 invalid_options。持久 Result.error_code 和宿主查询保留原码。下表列出其它 SDK 错误的 PowerLoopError 层级。

| 异常 | `code` |
|---|---|
| `PowerLoopError`（基类） | `power_loop.error` |
| `SessionNotFoundError` | `session.not_found` |
| `LLMTimeout` | `llm.timeout` |
| `LLMRetryExhausted` | `llm.retry_exhausted` |
| `CancellationRequested` | `cancelled` |
| `CompactionFailed` | `compaction.failed` |
| `ToolNotFound` | `tool.not_found` |
| `ToolValidationError` | `tool.invalid_args` |
| `SpecValidationError` | `spec.invalid` |

## Timer, factory and resource ports (11.0.0)

| Public port | Contract |
|---|---|
| `TimerBackend.create/list/cancel` | Receives trusted `TimerContext`, including original actor/tool identity |
| `StoreTimerBackend` | Default SQLite/PostgreSQL/MySQL timer store |
| `TimerDriver`, `TimerDriverStatus` | start/stop/status; mode, readiness, recent scan/error, bounded pending page |
| `TimerRunner(store=..., client=..., hooks=..., event_bus=...)` | Bounded due scan and input acceptance; no model execution |
| `RegistryRuntimeFactory.register_profile(name, preflight=...)` | Versioned capability and async deployment check before accepting |
| `Runtime(required_profiles=..., timer_backend=..., timer_driver=..., timers_enabled=...)` | Explicit composition; external backend requires a ready driver |
| `ExecutionEnvironment(runtime_env=..., workflow_file_io=..., process_control=..., resource_profile_ref=..., resource_scope=...)` | Current worker rebuilds tools/resources; shared workspace requires matching stable reference and shared scope |
| `StoreBlackboard(store)` | Database-atomic post/update/remove, including concurrent first creation; `delete(board_id)` ends a host-owned scope |
| `Runtime.status()` | Local accepting, ports, driver status, bounded pending/active counts and last error; no health SQL per request |
| `Result.error_code` | Original machine-readable RuntimeRequestError code survives persistent result reads |

`SqliteBlackboard` and `TimerDelivery` are removed without aliases. Default process
coordination holds a native store lock; another independent process/domain is
rejected, while pure clients and workers sharing one coordinator/transport remain
supported. See [timers](../user-guide/timers.md), [blackboard](../user-guide/blackboard.md)
and [runtime setup](../../runtime-11.md).

## Runtime 宿主接口（11.0.0 / schema 15 / protocol 42 / descriptor 1）

| 公开接口 | 合同 |
|---|---|
| `Runtime`、`RuntimeClient`、`RuntimeWorker` | 显式启停、输入受理、查询/等待/输出、取消与有界关闭 |
| `StatefulAgentLoop(runtime=...)` | 客户端便利入口；send 显式传 target、message_id、可信 actor 和 deadline_ms，返回 Result |
| `RuntimeFactory`、`RegistryRuntimeFactory` | 声明必需 profiles 和启动检查，当前节点重新取得执行资源 |
| `ExecutionEnvironment` | 模型、工具、hooks、RuntimeEnv、workflow 文件接口、进程控制、稳定资源引用及作用范围 |
| `TimerBackend`、`TimerContext`、`timer_context()` | 一个定时工具，使用配置的 backend 及当前输入/原工具调用身份 |
| `TimerDriver`、`TimerDriverStatus`、`TimerRunner` | Runtime 管理启停和状态；索引分批扫描，只等待 RuntimeClient 受理 |
| `StoreTimerBackend` | 默认 SDK 定时表，无 Redis；宿主 backend 必须提供 driver |
| `StoreBlackboard` | 数据库内原子建行、修改、删除与清理，可用于独立连接并发 |

`RuntimeRequestError` 继承 ValueError，实例的 `code` 保留 capability_unavailable、resource_unavailable、definition_unavailable、forbidden 等具体原因。持久 Result.error_code 与宿主查询不能丢弃该码，也不能把所有错误改成 definition_unavailable。

`Runtime.status()` 读取内存中的 worker 和 driver 状态，不为每个请求新增健康查询。纯 Client 不启动定时驱动、不拿默认进程锁。默认内存协调的 worker 使用数据库或文件原生锁；共享模式必须配置共享适配器。

### 子任务定时等待

公开 `WaitPolicy(deadline_at_ms, max_wait_ms=1800000)` 和
`register_wait_handoff_tool(registry, policy=...)`。在原主 Agent 和子任务的工厂
工具集合中注册；可信截止时间由工厂提供，模型不能设置或延长。子任务请求定时复核，
释放执行容量；主 Agent 读取原续办内容，再明确继续、延后或取消原 JobRef。
继续使用 RuntimeClient.answer，取消使用 RuntimeClient.cancel，提醒使用已配置的
TimerBackend/driver。未知定时创建不在重开时重放；错误保存在 schedule_error，
后续新决定可继续或取消原任务。详细装配见 [运行时合同](../../runtime-11.md#finite-child-review)。
