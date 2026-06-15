# power-loop — 中文文档

[English](../en/index.md) | [回到文档站](../README.md)

power-loop 是一个**可嵌入的 Agent 执行内核**——不是框架，不是平台。它给你 Agent 循环，你专注领域逻辑。

## 从哪里开始

| 路径 | 适用 |
|---|---|
| [快速上手](getting-started.md) | 第一次？5 分钟跑通第一条回复。 |
| [用户手册](user-guide/index.md) | 深入每个功能。 |
| [教程](tutorials/index.md) | 从零构建项目。 |
| [API 参考](api/index.md) | 签名级文档。 |
| [架构设计](../architecture.md) | 内部机制和设计决策。 |
| [迁移指南](migration.md) | 从 0.1.x 升级。 |
| [常见问题](faq.md) | FAQ。 |

## power-loop 提供什么

- **LLM 抽象** — 单一 `LLMService` 接口，多 transport（OpenAI 兼容，Anthropic Messages API）
- **多轮循环** — `new_session()` 创建会话；`send(user_input, session_id=sid)` 追加轮次
- **工具调用** — JSON Schema 校验注册；sync + async 处理器
- **Hooks** — 17 个挂载点拦截循环每个阶段
- **事件总线** — 30 种类型化事件，覆盖观测/审计/流式
- **声明式子代理** — `AgentSpec` → 一次性子代理，工具白名单
- **上下文压缩** — LLM 摘要压缩，默认开启
- **会话持久化** — SQLite 存储，跨进程恢复
- **重试 + 取消** — `LLMRetryPolicy` 指数退避；`CancellationToken` 统一取消形状
- **结构化输出** — `StructuredOutputSpec` + 四级 JSON 修复链
- **可插拔记忆** — `MemoryProvider` 协议，跨会话召回
- **运行时绑定工具** — 持久工具状态、projector、hooks 和 events，用于高级流程控制

## API 稳定性

见 [API 参考](api/index.md) 的三层分级（Stable / Provisional / Internal）。
