# 常见问题

[English](../en/faq.md) | [回到文档站](../README.md)

## 通用

### power-loop 是什么？

可嵌入的 Agent 执行内核。它提供 Agent 循环（LLM → 工具 → hooks → events → 持久化），你专注领域逻辑。不是框架或平台。

### 和 LangChain / CrewAI / AutoGen 有什么区别？

power-loop 是**内核**，不是框架。不预设 workflow DAG、记忆后端或 prompt 模板。你自带 LLM、工具和记忆——库提供执行循环、hooks 和持久化。约 2.4 万行 Python（不含 vendored transport），不是 100k 行的依赖树。

### 支持流式输出吗？

支持。订阅 `STREAM_DELTA` 事件获取实时 token 流。见 [Events](user-guide/events.md)。

### 支持 Anthropic 吗？

支持。设置 `POWER_LOOP_PROVIDER=anthropic` 会使用原生 Anthropic Messages API transport；DashScope 的 Anthropic 兼容 app 端点也可以这样使用。

## 配置

### LLM 凭证没被读取

检查 `POWER_LOOP_BASE_URL`、`POWER_LOOP_API_KEY`、`POWER_LOOP_MODEL` 是否设置。旧 `OPENAI_COMPAT_*` 名称也支持回退。

### 能用本地模型吗（Ollama / vLLM）？

可以。设置 `POWER_LOOP_BASE_URL=http://localhost:11434/v1`，`POWER_LOOP_API_KEY=anything`。

### 怎么关闭压缩（折叠）？

```python
config = AgentLoopConfig(fold_strategy=None)
```

（旧的 `compactor=None` 参数仍兼容，会映射到 `fold_strategy=None` 并带 DeprecationWarning。）

### 怎么用程序区分不同错误？

每个异常都继承 `PowerLoopError`，并带稳定、机器可读的 `code`（点分串）。按 `exc.code` 分支，而非类身份：

```python
from power_loop import PowerLoopError
try:
    await loop.send("hi", session_id=sid)
except PowerLoopError as exc:
    if exc.code == "llm.timeout": ...
    elif exc.code == "session.pending": ...   # resume / abort_pending / heal_pending
```

常见 code：`llm.timeout`、`llm.retry_exhausted`、`session.pending`、`session.not_found`、
`tool.not_found`、`tool.invalid_args`、`spec.invalid`。详见 [API 参考](api/index.md#错误码)。

## 会话

### 怎么跨进程共享会话？

用同一个 `db_path` 文件。在两个进程中打开，传入相同 `session_id`。会话活在 SQLite 里。

### 能删除会话吗？

```python
loop.close_session(sid, cascade=True)
```
物理删除会话及其所有消息、压缩和子代理。

### 进程在工具执行中崩溃怎么办？

会话进入"悬挂"状态。下次 `send()` 抛 `SessionPendingError`。调用 `resume()` 或 `abort_pending()`。

## Hooks vs Events

### 什么时候用 hook，什么时候用 event？

- **Hook**：改变行为（拦截工具、修改请求、缓存、审核）。
- **Event**：观测（流式、审计、指标、成本追踪）。

Hook 能改控制流；event 不能。

## 记忆

### power-loop 包含记忆后端吗？

不。它定义 `MemoryProvider` 协议。你实现 `recall()` 和 `remember()`。见 `examples/13_memory_sqlite.py`。

### 能用向量数据库做记忆吗？

可以。在 `recall()` 和 `remember()` 中实现你的向量库客户端。库不关心你用什么后端。

## 性能

### 能跑多少并发会话？

一个 `StatefulAgentLoop` 实例可以驱动任意数量的并发会话（每个受 `asyncio.Lock` 保护）。多进程时每个进程需要自己的 `StatefulAgentLoop` 指向同一 `db_path`。

### 会话能多大？

压缩器（默认开启）通过摘要旧消息让会话保持在 token 预算内。无压缩时受 LLM 上下文窗口限制。

## 贡献

见 [CONTRIBUTING.md](../../CONTRIBUTING.md) 了解开发环境、代码规范和 PR 流程。
