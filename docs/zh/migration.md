# 迁移指南

[English](../en/migration.md) | [回到文档站](../README.md)

## 0.2.x → 0.3.0

power-loop v0.3.0 改为显式创建会话。`send()` 不再在第一次调用时自动创建 session。

```python
# 0.3.0 之前
r1 = await loop.send("我叫阿岚。")
r2 = await loop.send("我叫什么？", session_id=r1.session_id)

# 0.3.0+
sid = loop.new_session()
r1 = await loop.send("我叫阿岚。", session_id=sid)
r2 = await loop.send("我叫什么？", session_id=sid)
```

每个会话的 metadata 从 `send(metadata=...)` 移到 `new_session(metadata=...)`：

```python
sid = loop.new_session(metadata={"user_id": "alan"})
result = await loop.send("你好", session_id=sid)
```

## 0.1.x → 0.2.0

v0.2.0 与 0.1.x **不兼容**。无状态的 `AgentLoop` 已移除；一切围绕 `StatefulAgentLoop` 和 SQLite `SessionStore`。

## 变更总结

| 0.1.x | 0.2.0 |
|---|---|
| `AgentLoop(llm, config).run(messages=[...])` | `sid = loop.new_session(); await loop.send(user_input, session_id=sid)` |
| 调用方管理 `messages` 列表 | 库从 `SessionStore` 按 `session_id` 加载 |
| 无持久化 | `db_path`（默认 `./power_loop_sessions.db`） |
| 无悬挂检测 | 工具中崩溃 → 下次 `send` 抛 `SessionPendingError` |
| 老 `spawn_agent` | `register_spawn_agent(registry)` + 共享 `SessionStore` |
| 无声明式子代理 | `AgentSpec` + `run_agent_spec()` |
| 老 LLM 配置 | `LLMProviderConfig` + `create_llm_service_from_env()` |

## 逐步迁移

### 1. 替换入口

**前**：`AgentLoop(llm, config).run(messages=[...])`
**后**：先 `sid = loop.new_session()`，再 `await loop.send("hello", session_id=sid)`。

### 2. 用 session_id 管理会话

**前**：手动构建 `messages` 列表。**后**：`new_session()` 返回 `session_id`，后续 `send()` 传入它继续对话。

### 3. 处理悬挂态

**前**：工具执行中崩溃 = 静默数据丢失。**后**：下次 `send()` 抛 `SessionPendingError`。调用 `resume()` 或 `abort_pending()`。

### 4. 使用 LLMProviderConfig

**前**：`OpenAICompatibleChatConfig` + `OpenAICompatibleChatLLMService`。
**后**：`create_llm_service_from_env()` 一行。

### 5. 环境变量名

旧名仍可用（回退），推荐用新名 `POWER_LOOP_*`。

## 0.2.0 新功能

- `LLMRetryPolicy` — 指数退避重试
- `CancellationToken` — 统一取消形状
- `StructuredOutputSpec` — schema 校验 JSON
- `MemoryProvider` — 可插拔跨会话记忆
- `trim_history` — 纯裁剪 helper
- 类型化错误体系（`ToolNotFound` / `ToolValidationError` / `SpecValidationError`）
- 18 个 hook 点，24 种事件类型

## 需要帮助？

- [快速上手](getting-started.md)
- [快速入门](user-guide/quickstart.md)
- [常见问题](faq.md)
