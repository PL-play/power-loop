# 共享黑板

[English](../../en/user-guide/blackboard.md) | [用户手册](../index.md)

**带作用域的共享黑板**是一块小型、结构化、可变的空间，供多个 agent 读写以进行协作——它独立于每个 agent 私有的消息历史。规划者（planner）留下任务；工作者（worker）认领并完成任务；双方都能看到对方的更新——*而无需*把一个 agent 的整段对话记录倒进另一个 agent 的上下文里。

无论作用域是一次聊天会话还是一次工作流运行，这都是同一套抽象；区别只在于 `blackboard_id` 与生命周期。DeepTalk 用它（作用域 = `conversation_id`）来实现房间内的 agent 协作板。

## 组成部分

| 组成部分 | 角色 |
|---|---|
| `Blackboard` (Protocol) | 异步 `read` / `post` / `update` / `remove`——按条目（per-entry）合并，绝不整文档覆盖 |
| `SqliteBlackboard` | 默认实现，持久化在 `SessionStore` 的 sqlite 中（黑板*并不*绑定到某个 session） |
| `register_blackboard_tools(registry)` | 注册面向 agent 的 `board_read` / `board_post` / `board_update` / `board_remove` 工具 |
| `RuntimeEnv(blackboard=, blackboard_id=)` | 每次运行时注入实时的黑板 + 该 agent 的 board id |
| `render_entries(entries, header=)` | 将黑板快照格式化以注入到 prompt 中 |

拥有**相同** `blackboard_id` 的 agent 共享同一块黑板；不同 id 之间相互隔离。没有黑板的 agent 直接不注册这些工具（默认 = 隔离）。

## 配置

```python
from power_loop import (
    SqliteBlackboard, ToolRegistry, register_blackboard_tools,
    RuntimeEnv, runtime_env_context, render_entries,
)

registry = ToolRegistry()
# The kind/status vocabularies are YOUR policy (they shape the tool schemas).
register_blackboard_tools(registry, kinds=("note", "task"), statuses=("open", "doing", "done"))

board = SqliteBlackboard(loop.store)
BOARD_ID = "project-x"
```

## 让 agent 在黑板上运行

按发送（per send）注入黑板（与 `shell_backend` 是同一个接缝）。**作者由 session 元数据加盖**（`spec_name`），而非由模型提供——因此署名无法被伪造。宿主通常会把当前黑板*投影*进每一轮的 prompt（即“拉取”一侧）：

```python
sid = loop.new_session(metadata={"spec_name": "planner"})
snapshot = render_entries(await board.read(BOARD_ID), header="Shared board:", empty="(empty)")

with runtime_env_context(RuntimeEnv(blackboard=board, blackboard_id=BOARD_ID)):
    await loop.send(f"{snapshot}\n\nPost two tasks for your teammate.", session_id=sid)
```

此时，第二个拥有*相同* `BOARD_ID` 的 agent 就能看到这些条目并据此行动：

```python
sid2 = loop.new_session(metadata={"spec_name": "worker"})
with runtime_env_context(RuntimeEnv(blackboard=board, blackboard_id=BOARD_ID)):
    await loop.send("Mark the first task done and leave a note.", session_id=sid2)

for e in await board.read(BOARD_ID):
    print(f"#{e.id} [{e.kind}·{e.status}] ({e.author}) {e.text}")
```

完整的 planner/worker 运行示例见 [example 29](../../../examples/29_shared_blackboard.py)。

## agent 看到的工具

| 工具 | 动作 |
|---|---|
| `board_read` | 拍下黑板快照（通常在每轮开始时自动展示；调用它可重新检查） |
| `board_post` | 添加一条条目（`text`，可选 `kind`、`status`） |
| `board_update` | 按 `entry_id` 编辑某条条目的 `text` / `status` |
| `board_remove` | 按 `entry_id` 删除某条条目 |

条目是仅追加（append-only）的，使用单调递增的整数 id。写入按条目进行（post / update 单行），而非整文档的“最后写入者胜出”——因此并发的作者不会相互覆盖。`SqliteBlackboard` 强制执行 `max_entries` 与 `max_text_len` 上限，违反时抛出 `BlackboardError`。

## 直接 API（无 agent）

黑板就是一个普通的异步对象，你可以自行驱动它——这对于预填数据、测试或宿主侧的审计视图很有用：

```python
e = await board.post(BOARD_ID, text="claim the plan", kind="task", status="open", author="alice")
await board.update(BOARD_ID, e.id, status="done")
await board.remove(BOARD_ID, e.id)
```

## 自定义实现

`Blackboard` 是一个 `Protocol`，因此宿主可以用任意后端来支撑它（一个 HTTP API、一个共享数据库），只要它保持按条目合并的语义即可。DeepTalk 的 `DeeptalkBlackboard` 把每一次写入都路由经其 REST API（遵守其单写入者不变量），同时原样复用 power-loop 的工具与投影器（projector）。

## 另请参阅

- [Tools](tools.md) — 注册与预设工具
- [Configuration](configuration.md) — `RuntimeEnv` 与按发送注入
- [Workflows](workflows.md) — 编排共享同一块黑板的多个 agent
