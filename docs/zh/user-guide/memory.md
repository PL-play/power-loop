# 记忆

[English](../../en/user-guide/memory.md) | [用户手册](../index.md)

`MemoryProvider` 是一个可插拔的**跨会话召回**协议。库本身不实现记忆后端——你提供自己的（SQLite 事实库、HTTP API 日记、向量库）。协议告诉 pipeline **何时**召回、**何时**持久化。

## 工作原理

1. **Recall** — `session.start` 时，第一轮之前。召回的消息注入到 leading `role=system` 块之后（与 `compact_note` 同区，压缩器会保留）。
2. **Remember** — `session.end` 时。收到 `MemorySnapshot`，含完整最终历史、final_text、状态和轮数。

## 快速开始

```python
from power_loop import MemorySnapshot, AgentLoopConfig

class MyMemory:
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [{"content": "用户偏好Python。喜欢的颜色：蓝色。"}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        pass  # 持久化你需要的内容

config = AgentLoopConfig(memory=MyMemory())
```

## MemorySnapshot

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | `str` | 会话 ID |
| `messages` | `list[dict]` | 完整最终历史（压缩后） |
| `final_text` | `str` | 最后一条回复 |
| `rounds` | `int` | 完成的总轮数 |
| `status` | `str` | `"completed"` / `"cancelled"` / `"degraded"` / `"hit_round_limit"` |

## 注入位置

```
[system_prompt]          ← 来自 AgentLoopConfig
[compact_note]           ← 来自压缩器（如有）
[memory_0]               ← 来自 recall
[memory_1]               ← 来自 recall
[user msg 1]             ← 对话从这里开始
```

记忆消息共享压缩器的系统区保护——永不折叠。

## 失败模式

记忆是 best-effort。失败绝不阻塞用户获取回复：

| 失败 | 行为 |
|---|---|
| `recall()` 抛异常 | 返回 `[]`。发 `MEMORY_FAILED(phase="recall")`。循环继续。 |
| `remember()` 抛异常 | 发 `MEMORY_FAILED(phase="remember")`。`StatefulResult` 原样返回。 |

## MEMORY_RECALLED Hook

注入前过滤或丢弃召回消息：

```python
hooks = AgentHooks()

async def gate_memory(ctx: MemoryRecalledCtx) -> None:
    if not user_has_consented(ctx.session_id):
        ctx.directive = HookDirective.SKIP  # 丢弃所有记忆
```

## 示例

完整可运行版见 [`examples/13_memory_sqlite.py`](../../examples/13_memory_sqlite.py)（SQLite 事实库跨会话召回）。

## 下一步

- [重试与取消](retry-cancel.md) — 处理 LLM 失败
- [Events](events.md) — 观测记忆事件