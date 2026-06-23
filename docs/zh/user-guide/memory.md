# 记忆

[English](../../en/user-guide/memory.md) | [用户手册](../index.md)

`MemoryProvider` 是一个可插拔的**跨会话召回**协议。库本身不实现记忆后端——你提供自己的（SQLite 事实库、HTTP API 日记、向量库）。协议告诉 pipeline **何时**召回、**何时**持久化。

## 工作原理

1. **Recall** — 由内置的 **`MemoryRecallHook`**（一个 `HookPoint.LLM_BEFORE` hook）执行。召回块**每次 send 计算一次**（首轮 / 会话切换时记忆化），并**临时地（ephemeral）**追加到本次调用的消息列表。它**不会**进入 `self.history` 或 store——每轮重新追加，run 结束即丢弃。
2. **Remember** — `session.end` 时。收到 `MemorySnapshot`，含完整最终历史、final_text、状态和轮数。

当设置了 `AgentLoopConfig.memory` 时该 hook **自动注册**。设 `builtin_memory_hook=False` 可改为自己注入记忆（见 [内置 Hook 与覆盖](#内置-hook-与覆盖)）。

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

召回的消息被标记为 `role=system, name=memory_*`，默认（`memory_position="tail"`）追加到本次调用请求的**尾部**——在全部历史之后：

```
[system_prompt]          ← 来自 AgentLoopConfig
[compact_note]           ← 来自压缩器（如有）
[user msg 1]             ← 对话从这里开始
[assistant msg 1]
...
[memory_0]               ← 来自 recall（临时，请求尾部）
[memory_1]               ← 来自 recall
```

尾部位置让**之前历史前缀在多次 send 间逐字节保持不变**，即便召回的记忆发生变化，从而保持厂商前缀缓存（prefix cache）命中。由于该块只追加到本次调用的消息列表（绝不进入 `self.history` / store），它对压缩不可见、每轮重置——不存在系统区折叠问题。

设 `memory_position="front"` 可恢复旧位置（在 leading system 块之后、对话之前）。一旦召回记忆变化便会破坏前缀缓存，因此推荐 `"tail"`。

为让 `history + memory` 处于模型窗口内，折叠/压缩触发会通过 `config.effective_context_budget()` 为尾部记忆预留余量——当设置了 `memory` 时为 `max_tokens − memory_budget_tokens`——使折叠足够早触发。

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

## 内置 Hook 与覆盖

Recall 由内置的 `MemoryRecallHook`（一个 `LLM_BEFORE` hook）实现，并以稳定名 `MemoryRecallHook.NAME == "builtin.memory_recall"` 注册。`LLM_BEFORE` hook 收到的 `LlmBeforeCtx` 带有 `session_id`。

host 可在不禁用记忆的前提下接管注入：

```python
from power_loop import MemoryRecallHook
from power_loop.contracts.hooks import HookPoint

# 覆盖：用自己的 LLM_BEFORE handler 替换内置 hook
hooks.replace(HookPoint.LLM_BEFORE, my_handler, name=MemoryRecallHook.NAME)

# 禁用：彻底移除（recall 不再运行）
hooks.remove(MemoryRecallHook.NAME)
```

在构造 loop 之前先以该名字注册一个 handler 也可以——loop 不会覆盖已存在于 `MemoryRecallHook.NAME` 下的条目。或者在 `AgentLoopConfig` 上设 `builtin_memory_hook=False` 完全关闭自动注册，自行接上 `LLM_BEFORE` 注入。

## 会话笔记：NoteMemory

`NoteMemory` 是内置的 `MemoryProvider`，召回会话**自身的笔记**（由 agent 的笔记工具写入）。它与后端无关——从你配置的任意 `SessionStore` 读取（SQLite / Postgres / MySQL）：

```python
from power_loop import NoteMemory, AgentLoopConfig

config = AgentLoopConfig(memory=NoteMemory(store))
```

> `NoteMemory` 原名 `SQLiteNoteMemory`；该名作为向后兼容别名保留，因此既有的 `from power_loop import SQLiteNoteMemory` 导入仍可用。

## 示例

完整可运行版见 [`examples/13_memory_sqlite.py`](../../../examples/13_memory_sqlite.py)（SQLite 事实库跨会话召回）。

## 下一步

- [重试与取消](retry-cancel.md) — 处理 LLM 失败
- [Events](events.md) — 观测记忆事件
