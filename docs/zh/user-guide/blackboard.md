# 共享黑板

`StoreBlackboard(store)` 是 `Blackboard` 协议的默认数据库实现，支持 SQLite、PostgreSQL、MySQL。宿主决定 `board_id` 和使用期限；黑板独立于消息会话。

每次新增、修改、删除都在数据库的同一个短事务内完成读取和修改，包括第一次建行。不同实例和进程同时写同一个黑板，不会丢条目或分配重复 ID。SQLite 使用已有写事务；PostgreSQL/MySQL 先建立 owner/key 唯一行且不覆盖内容，再锁行更新。条数和文本长度限制保留，条目不存在或操作不合法时抛出 `BlackboardError`。

```python
from power_loop import StoreBlackboard

board = StoreBlackboard(store)
entry = await board.post("project-37", text="评审计划", kind="task", status="open")
await board.update("project-37", entry.id, status="done")
entries = await board.read("project-37")
await board.remove("project-37", entry.id)
await board.delete("project-37")  # 宿主结束整个黑板的使用范围。
```

模型工具通过 `register_blackboard_tools(registry)` 注册。当前执行节点的工厂返回 `ExecutionEnvironment` 时，提供 `runtime_env=RuntimeEnv(blackboard=board, blackboard_id="project-37")`。模型只看到一个 `board` 工具，操作为 `read/post/update/remove`，不能自行指定其它黑板 ID。宿主只向已授权的 Agent 注入同一黑板，并负责最终清理。

执行迁移到另一个节点时，由那个节点的工厂重新取得相同的黑板后端与范围。提交端的上下文变量不会跨进程传递。DeepTalk 继续使用 `DeeptalkBlackboard`，通过 API 修改业务黑板，不写 SDK 黑板表。

无需模型凭据的可运行示例：

```bash
python examples/29_shared_blackboard.py
```

旧公开名 `SqliteBlackboard` 已删除，不提供兼容别名。装配和生命周期见 [Runtime 11](../../runtime-11.md)。
