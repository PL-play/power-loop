# 会话

[English](../../en/user-guide/sessions.md) | [用户手册](../index.md)

会话是 power-loop 中的对话单位。先用 `new_session()` 显式创建会话，再把返回的 `session_id` 传给每次 `send()`。库按这个 id 管理历史、持久化和恢复。

## 会话生命周期

```mermaid
stateDiagram-v2
    [*] --> Active: new_session()
    Active --> Active: send(user_input, session_id=sid)
    Active --> Closed: close_session(sid)
    Active --> Pending: 工具调用中崩溃
    Pending --> Active: resume(sid) 或 abort_pending(sid)
    Active --> WaitingForInput: request_user_input
    WaitingForInput --> Active: submit_input(sid, interaction_id, value)
    Closed --> [*]
```

1. **创建**——调用 `new_session()`。
2. **延续**——后续 `send()` 传入相同 `session_id`。
3. **悬挂**——进程在 `assistant(tool_calls)` 和最后一个 `tool` 消息之间崩溃。
4. **等待输入**——`request_user_input` 要求调用方/UI 收集外部输入。
5. **关闭**——显式调用 `close_session(sid, cascade=True)`。

## 基本用法

```python
loop = StatefulAgentLoop(llm=llm, config=config)

# 新会话
sid = await loop.new_session()  # → "sess_abc123..."
r1 = await loop.send("你好，我叫阿岚。", session_id=sid)

# 继续
r2 = await loop.send("我叫什么？", session_id=sid)
print(r2.final_text)  # → "你叫阿岚。"

# 查看历史
messages = await loop.get_messages(sid)
for m in messages:
    print(m["role"], m.get("content", "")[:60])
```

**关键**：你永远不需要手动构建 `messages` 列表。库通过 `session_id` 从 store 加载历史。

## SessionStore

`SessionStore` 是后端中立的持久化层。通常不需要直接交互——`StatefulAgentLoop` 管理它。但你可以自己打开一个用于检查、跨 loop 共享或高级用法。后端由 DSN 选择：裸路径或 `sqlite://…` 是 SQLite（零基础设施的默认值），`postgresql://…` / `mysql://…` 是真正的多写者服务器，藏在可选的 driver extra 后面。store 的每个方法都是协程。

```python
from power_loop import SessionStore, open_store

# SQLite（默认后端）：按路径打开。
store = await SessionStore.open("./my_sessions.db")
# 或按 DSN 选任意后端（例如跨多个 loop 共享一个 store）：
# store = await open_store("postgresql://u:p@host/app", table_prefix="pl_")

# 读取特定会话
session = await store.get_session(sid)
print(session.status)     # "active" | "closed"
print(session.created_at)

# 读取消息
active = await store.load_active_messages(sid)     # 未压缩的
all_msgs = await store.load_all_messages(sid)       # 含已压缩的

# 读取某会话的直接子代理
children = await store.list_children(sid)           # SessionRow 列表

# 关闭
await store.close()
```

选 SQLite 还是 PostgreSQL/MySQL、各后端的 DDL、以及 schema 配置（`SchemaPolicy`），见 [存储后端](storage-backends.md)。

### 表结构

store 把所有东西放在 12 张表加一张版本表里，全部带 `table_prefix`（默认 `pl_`）。核心几张：

| 表 | 用途 |
|---|---|
| `pl_sessions` | 每个会话一行：`session_id`、`status`、`kind`、`parent_session_id`、时间戳 |
| `pl_messages` | 按 `(session_id, seq)` 排序的消息日志，含 `state`（`active` / `compacted_out`） |
| `pl_compactions` | 每次压缩日志：`(session_id, compact_seq)`，折叠内容 |
| `pl_usage_rounds` | 每轮 token 用量：`(session_id, round_index)`，prompt/completion tokens |
| `pl_session_state` | 可变状态：`next_seq`、`round_index`、当前 `pending` tool_calls |
| `pl_timers` / `pl_notes` / `pl_session_stats` / … | 持久化定时器、笔记、每会话统计、runtime/shared state、后台任务 |
| `pl_schema_migrations` | 可移植的版本表——在每种后端上行为一致，并拒绝比代码更新的数据库 |

各后端的精确 DDL 见 [存储后端](storage-backends.md#各后端的-ddl)。

### 存储配置

- **后端中立。** 同一个 store 跑在 SQLite（默认）、PostgreSQL 或 MySQL 上——由 DSN 选择。PostgreSQL/MySQL 原生异步；其 driver 是可选 extra。
- **SQLite：WAL + `busy_timeout`** —— 开启 WAL，读不会阻塞写者；busy timeout 吸收短暂的写入争用。
- **每会话一个写者。** 异步 store 把阻塞的 SQLite I/O 卸载到 worker 线程，在单一写者锁下执行，从而保证 `next_seq` 不碰撞；PostgreSQL/MySQL 用 `SELECT … FOR UPDATE` 行锁分配每会话序号。多个 `StatefulAgentLoop` 实例在同一 store 上安全，只要每个 session 同一时刻只由一个写者驱动。见 [扩展性](scaling.md)。

## 跨进程恢复

store 是持久化锚点；loop **不持有任何权威状态**。在任意进程中用相同的 `dsn` + `session_id` 重建 loop 即可继续——一个全新的冷 loop 不需要别的就能恢复：

```python
# 进程 1
loop = StatefulAgentLoop(llm=llm, dsn="./chat.db", config=config)
sid = await loop.new_session()
r1 = await loop.send("记住：我喜欢的颜色是蓝色。", session_id=sid)
loop.close()

# 进程 2 —— 几小时后，不同的 Python 进程
loop2 = StatefulAgentLoop(llm=llm, dsn="./chat.db", config=config)
await loop2.prewarm(sid)  # 可选：预加载活动窗口
r2 = await loop2.send("我喜欢什么颜色？", session_id=sid)
print(r2.final_text)  # → "你喜欢蓝色。"
```

换成服务器后端也一样——把两个进程都指向 `postgresql://…` / `mysql://…`（见 [存储后端](storage-backends.md)）。

> **注意**：同一个 session 同一时刻必须只由一个写者驱动。`asyncio.Lock` 只在单个 `StatefulAgentLoop` 实例内保护，因此当多个进程共享一个 store 时，请在你的 dispatcher/queue 层把同一 session 的 send 串行化。用 SQLite 时，每个文件跑一个写者进程（把 session 按文件分片）；见 [扩展性](scaling.md)。

## 悬挂恢复

如果进程在执行工具调用时崩溃，会话进入"悬挂"状态。下次 `send()` 抛出 `SessionPendingError`：

```python
try:
    result = await loop.send("do something", session_id=sid)
except SessionPendingError as exc:
    print(f"未解决的 tool calls: {exc.pending_tool_calls}")
    # 选项 A：完成执行悬挂的工具
    result = await loop.resume(sid)
    # 选项 B：放弃并继续
    await loop.abort_pending(sid, reason="user_cancelled")
    result = await loop.send("new input", session_id=sid)
```

## 可恢复用户输入

`request_user_input` 会有意暂停会话，但不会阻塞 Python 进程：

```python
waiting = await loop.send("needs confirmation", session_id=sid)
interaction = waiting.pending_interactions[0]

# 在产品 UI 里展示 interaction["prompt"] 和 interaction["options"]。
result = await loop.submit_input(sid, interaction["interaction_id"], {"choice": "yes"})
```

待输入项会存进 SQLite，因此另一个进程之后重新打开同一个数据库，也可以调用 `submit_input()` 继续。

## 每次调用覆盖

`send()` 和 `send_sync()` 接受 `tools=` 与 `system_prompt=`，且不会修改 loop 或已存储的
session：

```python
result = await loop.send(
    "Summarize the repository",
    session_id=sid,
    tools=["read_file", "glob", "grep"],
    system_prompt="Be concise and cite file paths.",
)
```

prompt 优先级为：每次调用覆盖 > session prompt > loop config。模型只能看到选中工具的
definition。当 session 空闲、`follow_up()` / `follow_up_sync()` 降级为新 send 时，也会透传这些参数。

## 运行中追加指引（`follow_up`）

当会话已在运行（`send` / `resume` / `submit_input` 持有 per-session 锁）时，对同一会话再调用 `send()` 会阻塞到当前 run 结束。若要在不等待的情况下注入补充指引，使用 `follow_up()`：

```python
send_task = asyncio.create_task(loop.send("long task", session_id=sid))

# 等待 session 锁被占用（同一进程内）。
while not loop._lock_for(sid).locked():
    await asyncio.sleep(0.01)

queued = await loop.follow_up("Also mention the budget constraint", sid)
assert isinstance(queued, FollowUpQueued)
assert queued.queue_depth == 1

result = await send_task
```

Pipeline 会在每个**轮次边界**（`ROUND_START` 之后、`prepare_round` 之前）排空 per-session 队列，将多条 follow-up 合并为一条 user 消息并写入 transcript：

```xml
<follow_up>
Also mention the budget constraint
</follow_up>
```

当会话空闲（锁未被占用）时，`follow_up()` 会降级为 `send()`。
当 follow-up 被排入正在运行的 run 时，该 run 会继续使用启动它的调用所选定的 `tools` 与
`system_prompt` 策略；follow-up 文本会引导下一轮，但不会在 run 中途替换安全边界。

| API | 适用场景 |
|---|---|
| `submit_input()` | loop 因 `request_user_input` 暂停；你有 `interaction_id`，可跨进程稍后恢复。 |
| `follow_up()` | loop 仍在同一进程内运行；你想在**下一轮** LLM 调用前注入指引，而不阻塞当前 run。 |

详见 [示例 22](../../../examples/22_follow_up_steering.py) 与 [示例指南 §22](../tutorials/examples-guide.md#22--运行中追加指引)。

## 关闭会话

```python
# 关闭一个会话（cascade=True 时一并关闭所有子代理会话）
await loop.close_session(sid, cascade=True)

# 关闭整个 store（所有会话）
loop.close()
```

关闭的会话**物理删除**——所有消息、压缩和使用记录一并删除。

## 下一步

- [工具](tools.md) — 注册工具，给 Agent 能力
- [子代理](subagents.md) — 用 `spawn_agent` 和 `AgentSpec` 创建子代理
