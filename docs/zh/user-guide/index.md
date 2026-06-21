# 用户手册

[English](../../en/user-guide/index.md) | [回到文档站](../../README.md)

深入每个功能。这些是参考式文档——想一步步跟着构建项目，看 [教程](../tutorials/index.md)。

## 核心

| 页面 | 你会学到 |
|---|---|
| [安装](installation.md) | pip install、开发模式、Python 版本 |
| [快速入门](quickstart.md) | 在一个文件里走完所有主要功能 |
| [配置](configuration.md) | `AgentLoopConfig` 字段、环境变量、`LLMProviderConfig` |

## 功能

| 页面 | 你会学到 |
|---|---|
| [会话](sessions.md) | `SessionStore`、多轮对话、跨进程恢复、会话生命周期 |
| [存储后端](storage-backends.md) | `dsn=` 选 SQLite（默认）/ PostgreSQL / MySQL；`SchemaPolicy` 置备、各后端 DDL、前置条件 |
| [工具](tools.md) | `ToolRegistry`、`ToolDefinition`、JSON Schema 校验、sync vs async |
| [扩展工具](extending-tools.md) | 自定义工具配方、按调用白名单、MCP(`contrib.mcp`)、为什么不捆绑连接器 |
| [子代理](subagents.md) | `spawn_agent`、`AgentSpec`、`run_agent_spec`、生命周期 |
| [Hooks](hooks.md) | 17 个 `HookPoint`、类型化 Ctx、directive、常见模式 |
| [Events](events.md) | 各种 `AgentEventType`、类型化 payload、订阅模式 |
| [可观测性](observability.md) | 持久化 JSONL sink + `replay`、指标(Prometheus/StatsD)、OpenTelemetry span、背压 |
| [压缩](compaction.md) | `DefaultCompactor`、`trigger_ratio`、保留最后 N 轮、自定义 `Compactor` |
| [Send 上下文投影](send-context-projection.md) | 可选 `HistoryProjector`;每-send 纯文本投影(`pl_project_messages`);`ToolDefinition.project`;`recall_send` |
| [记忆](memory.md) | `MemoryProvider` 协议、`recall` / `remember`、注入位置 |
| [重试与取消](retry-cancel.md) | `LLMRetryPolicy`、指数退避、`CancellationToken` |
| [结构化输出](structured-output.md) | `StructuredOutputSpec`、`parse_structured`、JSON 修复链 |
| [定时器](timers.md) | 持久化自我唤醒、`TimerRunner`、循环定时器、`TIMER_FIRE` |
| [高级运行时工具](advanced-runtime-tools.md) | Runtime state、projector、hooks/events、自定义流程控制 |

## 编排与隔离

| 页面 | 你会学到 |
|---|---|
| [异步编排](async-orchestration.md) | **任何异步问题先看这里。** 宿主驱动模型（无守护进程）、`send`/`resume`/`submit_input`/`follow_up` 唤醒 API、各类异步结果如何回到循环、持久化与崩溃恢复、与投影/压缩的配合、如何写自定义异步唤醒工具、问题排查 |
| [动态工作流](workflows.md) | `WorkflowSpec` DSL、确定性引擎、detached 执行、跨重启 resume |
| [共享黑板](blackboard.md) | `SqliteBlackboard`、`board_*` 工具、多代理协作 |
| [沙箱与隔离](sandboxing.md) | `ShellBackend`（沙箱化 bash）、`WorkerLauncher`（沙箱化叶子进程） |

## 性能与扩展

| 页面 | 你会学到 |
|---|---|
| [扩展性](scaling.md) | 每会话写者模型、选 SQLite 还是服务器后端、保留/VACUUM、`bench/` 压测台、实测数据、分片与横向扩展 |

## 高级

| 页面 | 你会学到 |
|---|---|
| [架构设计](../../architecture.md) | 模块边界、Pipeline 阶段、关键不变量 |
| [Providers](providers.md) | OpenAI、DashScope、DeepSeek、本地 — 配置片段 |
