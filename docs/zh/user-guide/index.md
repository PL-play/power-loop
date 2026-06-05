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
| [工具](tools.md) | `ToolRegistry`、`ToolDefinition`、JSON Schema 校验、sync vs async |
| [子代理](subagents.md) | `spawn_agent`、`AgentSpec`、`run_agent_spec`、生命周期 |
| [Hooks](hooks.md) | 18 个 `HookPoint`、类型化 Ctx、directive、常见模式 |
| [Events](events.md) | 24 种 `AgentEventType`、类型化 payload、订阅模式 |
| [压缩](compaction.md) | `DefaultCompactor`、`trigger_ratio`、保留最后 N 轮、自定义 `Compactor` |
| [记忆](memory.md) | `MemoryProvider` 协议、`recall` / `remember`、注入位置 |
| [重试与取消](retry-cancel.md) | `LLMRetryPolicy`、指数退避、`CancellationToken` |
| [结构化输出](structured-output.md) | `StructuredOutputSpec`、`parse_structured`、JSON 修复链 |

## 高级

| 页面 | 你会学到 |
|---|---|
| [架构设计](../architecture.md) | 模块边界、Pipeline 阶段、关键不变量 |
| [Providers](providers.md) | OpenAI、DashScope、DeepSeek、本地 — 配置片段 |