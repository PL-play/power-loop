# 压缩

[English](../../en/user-guide/compaction.md) | [用户手册](../index.md)

上下文压缩防止长会话超出 LLM 上下文窗口限制。它把旧消息摘要为一条紧凑的系统注释——**默认开启**。

## 工作原理

```mermaid
flowchart TD
    A[轮次开始] --> B{tokens > 阈值?}
    B -->|否| E[正常运行]
    B -->|是| C[LLM 摘要旧消息]
    C --> D[替换为 compact_note]
    D --> E
```

1. 每轮开始前，`estimate_tokens(messages)` 与 `max_tokens × trigger_ratio`（默认 0.75）比较。
2. 超阈值时，压缩器识别最旧可安全折叠的消息。
3. 摘要 LLM 调用生成压缩注释（`role=system, name=compact_note`）。
4. 旧消息标记 `compacted_out`；注释插入。

## 配置

```python
from power_loop.runtime.compact import DefaultCompactor
from power_loop import AgentLoopConfig

compactor = DefaultCompactor(
    trigger_ratio=0.75,        # token > max_tokens 的 75% 时触发
    keep_last_n=4,             # 始终保留最后 4 轮
    summary_max_tokens=512,    # 摘要的最大 token 数
)

config = AgentLoopConfig(compactor=compactor)

# 关闭压缩
config_no = AgentLoopConfig(compactor=None)
```

### 绝对阈值

环境变量 `CONTEXT_COMPACT_THRESHOLD=6000` 设置绝对 token 数。当模型有已知上下文窗口时有用。

## 不变量

| 规则 | 原因 |
|---|---|
| **系统消息保留** | `role=system` 消息（含先前的 `compact_note`）永不折叠 |
| **最后 N 轮保留** | 最近的 `keep_last_n` 轮始终保留 |
| **工具调用对原子性** | `assistant(tool_calls) ↔ tool` 永不拆分 |
| **每轮最多一次** | `round_compacted=True` 防重复 |
| **软失败** | 摘要 LLM 调用失败 → 用原（未压缩）历史继续 |

## 自定义压缩器

实现 `Compactor` 协议：

```python
from power_loop.runtime.compact import Compactor, CompactionPlan

class MyCompactor:
    async def maybe_compact(self, messages, *, llm, max_tokens, round_index) -> CompactionPlan | None:
        # 你的逻辑
        return None  # None = 跳过

config = AgentLoopConfig(compactor=MyCompactor())
```

## 下一步

- [记忆](memory.md) — 通过 `MemoryProvider` 跨会话召回
- [会话](sessions.md) — 理解会话生命周期