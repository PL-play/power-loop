# Hook 注入上下文审计（`pl_hook_events`）

`LLM_BEFORE` hook 可以向单次 LLM 调用注入**临时**上下文——最典型的是内置的
[记忆召回](memory.md) hook，它把召回的笔记追加到请求尾部。这段被注入的上下文只在那一次调用里发给模型，之后就丢弃：它**从不**写入 `self.history` 或存储（这是刻意设计——让 prompt 前缀保持字节稳定，利于厂商 prefix 缓存）。代价是它**没有被记录在任何地方**，因此你事后无法审计某一轮模型实际看到了哪些额外上下文。

可选的 **hook 事件审计日志** 在不改变上述任何行为的前提下解决这个问题。

## 启用

```python
from power_loop import AgentLoopConfig

config = AgentLoopConfig(
    memory=my_memory_provider,
    record_hook_events="full",   # "off"（默认）| "metadata" | "full"
)
```

| 模式 | 记录内容 |
|------|----------|
| `"off"`（默认） | 不记录——零开销 |
| `"metadata"` | 每个注入项的 `role`、`name`、`source`、`chars` 以及 `position`（tail/front）——**不含**正文 |
| `"full"` | 以上**再加**注入的 `content` 正文 |

`"full"` 按原样存储注入正文、无单项上限，因此大的 RAG/记忆块会让表变大——在意体量就用 `"metadata"`。

## 它**不会**做什么（保证）

审计**仅用于可观测性**。它的写入方式与 `send_index` 完全一致——挂在该轮 assistant 消息的 *sink 副本* 上，绝不挂在 `self.history` 里的那条消息上。所以它**永远不会**重新进入对话、到达 LLM 请求、或扰动 prefix 缓存。打开它不会以任何方式改变模型行为。

## 读取

```python
events = await store.list_hook_events(session_id)                   # 全部，按时间正序
events = await store.list_hook_events(session_id, message_seq=seq)  # 某条消息的
```

每个 [`HookEventRow`](../api/index.md) 链接到它喂入的 assistant 消息（`message_seq`），以及定位用的
`round_index` / `send_index`。`payload` 为
`{v, items: [{role, name, source, chars, content?}], item_count, total_chars}`。

**每轮**写一行（`LLM_BEFORE` hook 每轮都跑；记忆块在一个 send 内只召回一次但每轮都重新注入），所以多轮 send 会产生每轮一行的审计。

## 存储与生命周期

- 独立表 `{prefix}hook_events`（schema **v3**；迁移是幂等的 `CREATE TABLE`——绝不改动热点 `messages` 表）。任何通用的 `pl_*` 表查看器都会自动显示它。
- 行随会话一起删除（`close_session_tree`）。
- 该表**仅用于审计**，刻意**不**纳入 `export_session` / `import_session`（审计数据存活在实时存储里）。

## 注意 / 边界情况

- 捕获是对 `LLM_BEFORE` 之后的消息列表与 hook 前快照做**身份差集**，因此 tail 与 front 位置的注入都能记录。
- 它假设 `LLM_BEFORE` handler **原地修改 `ctx.messages`**（内置约定）。若某个 handler 把 `ctx.messages` 整个**替换**为新副本，则逐项差集无法解析；此时该行退化为一个小的 `kind="inject_unresolved"` 标记（仍然绝不影响上下文或缓存）。
