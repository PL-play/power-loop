# 迁移指南

[English](../en/migration.md) | [回到文档站](../README.md)

## 1.x — 异步 API · 可插拔存储 · 无状态循环

任何现有代码都值得采纳的三处变化：

1. **公开 API 改为异步。** 原本同步的会话管理方法现在都是协程，必须 `await`：`new_session`、
   `close_session`、`get_messages`、`get_pending`、`resolve_system_prompt`、`abort_pending`、
   `prewarm`，定时器方法（`schedule_timer` / `cancel_timer` / `list_timers`），以及统计方法
   （`get_session_stats` / `list_session_stats`）。`SessionStore` 的每个方法也都是协程
   （`await store.open(...)`、`await store.append_message(...)`…）。`send` / `follow_up` /
   `resume` / `submit_input` 本来就是异步的；`send_sync` / `follow_up_sync` 包装器保持同步。
   ```python
   # 之前
   sid = loop.new_session()
   msgs = loop.get_messages(sid)
   # 之后
   sid = await loop.new_session()
   msgs = await loop.get_messages(sid)
   ```

2. **存储可插拔。** 存储按 DSN 打开：裸路径或 `sqlite://…` 选 SQLite（零基础设施默认），
   `postgresql://…` 选 PostgreSQL（`pip install 'power-loop[postgres]'`），`mysql://…` 选 MySQL
   （`pip install 'power-loop[mysql]'`）。循环接受 `dsn=`（别名 `db_path=` 仍可用）、`table_prefix=`
   （默认 `pl_`）和 `schema=SchemaPolicy.AUTO_CREATE | VERIFY`。`AUTO_CREATE`（默认）在表缺失时创建；
   `VERIFY` 只检查，schema 缺失时抛 `StoreSchemaError`（`.ddl` 里带着完整建表脚本）。存储版本存放在可移植、
   后端中立的 `pl_schema_migrations` 表里，三种后端表现一致，并拒绝打开比代码更新的数据库。
   `create_schema: bool` 是已弃用的别名（`True → AUTO_CREATE`，`False → VERIFY`）。见
   [存储后端](user-guide/storage-backends.md)。

3. **循环无状态 / 可恢复。** `StatefulAgentLoop` 不持有权威会话状态——状态全在存储里——所以冷启动 /
   全新的循环凭 id 即可恢复任意会话：用相同的 `dsn` + `session_id` 构造并 `send()` 即可。可选的
   `await loop.prewarm(session_id)` 预加载活动窗口。循环为每个会话维护一份活动窗口缓存
   （`session_cache_size`，默认 256，`0` 关闭；用 `loop.cache_stats` 查看），它只缓存持久化投影——
   一个纯加速器，绝不改变模型看到的内容。

## 0.13.x → 0.14.0

1. **`llm_client` 收编进库内。** 现位于 `power_loop._vendor.llm_client`，裸的顶层 `llm_client`
   包已不存在。直接 import 的都要改：
   ```python
   # 之前 —— 已无法 import
   from llm_client.interface import LLMRequest, LLMResponse, LLMService
   # 之后 —— 用顶层 re-export
   from power_loop import LLMRequest, LLMResponse, LLMService, LLMStreamChunk, LLMTokenUsage, \
       OpenAICompatibleChatConfig, AnthropicChatConfig
   ```
   具体 service 类属内部实现 —— 用公开工厂 `create_llm_service_from_env()` /
   `create_llm_service_from_config(LLMProviderConfig(...))` 构造。
2. **Transport 成为 extras。** 核心瘦身为仅 `certifi`；装对应 extra：
   `pip install 'power-loop[openai]'`（或 `[anthropic]` / `[all]`；另有 `[skills]` / `[pdf]`）。
   不装对应 SDK 就构造 provider 会抛带 `pip install` 提示的 `ImportError`。
3. **删除 `requirements.txt`** —— `pyproject.toml` 为单一事实源，按 `power-loop[openai]==0.14.0` 锁版。
4. **发布 `py.typed`**（PEP 561）—— 下游 mypy/pyright 现在能看到 power-loop 的类型。
5. 可选采纳的新面：每调用 LLM 事件（`LLM_CALL_STARTED` / `LLM_CALL_COMPLETED`）、现已真实发射的
   `AGENT_ERROR`、`ts`+`seq` 事件信封、以及稳定的 `exc.code` 错误码 —— 见[事件](user-guide/events.md)
   与 API 参考。

## 0.2.x → 0.3.0

power-loop v0.3.0 改为显式创建会话。`send()` 不再在第一次调用时自动创建 session。

```python
# 0.3.0 之前
r1 = await loop.send("我叫阿岚。")
r2 = await loop.send("我叫什么？", session_id=r1.session_id)

# 0.3.0+
sid = await loop.new_session()
r1 = await loop.send("我叫阿岚。", session_id=sid)
r2 = await loop.send("我叫什么？", session_id=sid)
```

每个会话的 metadata 从 `send(metadata=...)` 移到 `new_session(metadata=...)`：

```python
sid = await loop.new_session(metadata={"user_id": "alan"})
result = await loop.send("你好", session_id=sid)
```

## 0.1.x → 0.2.0

v0.2.0 与 0.1.x **不兼容**。无状态的 `AgentLoop` 已移除；一切围绕 `StatefulAgentLoop` 和 SQLite `SessionStore`。

## 变更总结

| 0.1.x | 0.2.0 |
|---|---|
| `AgentLoop(llm, config).run(messages=[...])` | `sid = await loop.new_session(); await loop.send(user_input, session_id=sid)` |
| 调用方管理 `messages` 列表 | 库从 `SessionStore` 按 `session_id` 加载 |
| 无持久化 | `db_path`（默认 `./power_loop_sessions.db`） |
| 无悬挂检测 | 工具中崩溃 → 下次 `send` 抛 `SessionPendingError` |
| 老 `spawn_agent` | `register_spawn_agent(registry)` + 共享 `SessionStore` |
| 无声明式子代理 | `AgentSpec` + `run_agent_spec()` |
| 老 LLM 配置 | `LLMProviderConfig` + `create_llm_service_from_env()` |

## 逐步迁移

### 1. 替换入口

**前**：`AgentLoop(llm, config).run(messages=[...])`
**后**：先 `sid = await loop.new_session()`，再 `await loop.send("hello", session_id=sid)`。

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

## 任务协议升级的真实旧版演练（design/192）

`test_old_protocol_requires_explicit_upgrade_and_keeps_original_identity` 等普通回归验证协议分支；发布前还须用实际旧版生成数据，不能只修改新库的版本号当作升级证明。

独立验收入口为 `scripts/acceptance/test_schema_bridge.py`（显式执行，不纳入普通 `tests/` 自动发现）。准备一个干净的 protocol 6 SDK checkout；本轮基线为 `7003235a8eb307ed02f2a08e98c868dcfd38a932`。设置：

- `POWER_LOOP_BRIDGE_LEGACY_SDK`：该旧版 checkout 的绝对路径，必须含其原始测试辅助模块。
- `POWER_LOOP_TASK_PG_DSN`：仅用于验收的 PostgreSQL 数据库，数据库名必须以 `_test` 结尾。
- 原真实 LLM 测试配置 `POWER_LOOP_*` 或 `OPENAI_COMPAT_*`。没有模型配置时真实验收应失败，不能标为通过。

从候选 SDK 根目录执行（Python 环境须包含本仓库测试依赖）：

```bash
python -m pytest scripts/acceptance/test_schema_bridge.py -v
```

共四个组合：SQLite/PostgreSQL × 程序/真实模型。原始数据由单独旧版解释器创建：实际 loop send、未提交成功的原预算用量、排队任务及原调用/回执。新版显式迁移后逐表比对消息、任务、事件、预算和执行者记录；除协议字段外迁移不改这些原件。测试还注入 DDL 后失败验证事务回滚，重复迁移验证幂等，并验证旧版重新打开被拒绝、迁移前已打开的旧 JobManager 写操作被拒绝。最后通过新接口幂等取消原排队任务、补交原用量，保持原 scope 计数、消息和 boot 不变。

原模型请求最多每个真实用例一次；预算 authority 是确定性测试端口，此演练不代表供应商最终发票或业务 API 全链路验收。测试保留运行报告中的原标识与用量；服务端表使用随机前缀并清理，但整个专用测试库仍应由外层运行器建立和销毁。

**部署仍须停止并隔离全部旧写者。** 此故障用例刻意保留一个空闲旧连接，证明 JobManager 协议闸有效；不证明旧版所有 SessionStore/业务 SQL 都具有同样的写保护。不能据此在真实迁移期间继续运行旧服务。迁移后的回滚应采用已验证的新协议控制版本，禁止降版本字段或用旧备份覆盖新增数据。生产迁移、备份恢复演练与控制宿主/API 的完整组合须单独放行。
