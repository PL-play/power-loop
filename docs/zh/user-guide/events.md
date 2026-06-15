# Events

[English](../../en/user-guide/events.md) | [用户手册](../index.md)

Event 是**只读观测**通道。订阅者可以看到所有事情——但不能改变控制流（控制流用 [Hooks](hooks.md)）。

## 快速示例

```python
from power_loop import AgentEventBus, AgentEventType

bus = AgentEventBus()

# 流式打字机
def on_delta(event):
    print(event.data.text, end="", flush=True)

bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)

# 订阅所有事件
def audit(event):
    log.write(f"[{event.type.value}] sid={event.session_id}\n")

bus.subscribe(None, audit)  # None = 所有事件

loop = StatefulAgentLoop(llm=llm, event_bus=bus, config=config)
```

## 事件类型分类

| 类别 | 事件数 | 关键事件 |
|---|---|---|
| 会话 | 2 | `session_started`, `session_ended` |
| 轮次 | 3 | `round_started`, `round_completed`, `round_tools_present` |
| 流式 | 4 | `stream_started`, `stream_delta`, `stream_think_delta`, `stream_completed` |
| 工具 | 3 | `tool_call_started`, `tool_call_completed`, `tool_call_failed` |
| 状态/用量 | 3 | `status_changed`（4 种子类型）, `usage_updated`, `timer_fired` |
| 子代理 | 4 | `subagent_task_start`, `subagent_text`, `subagent_limit`, `subagent_completed` |
| 每次LLM调用 | 2 | `llm_call_started`, `llm_call_completed`（按 `call_id` 配对，含本次调用延迟/用量；0.14.0） |
| 重试/取消 | 3 | `llm_retry_attempted`, `llm_degraded`, `loop_cancelled` |
| 记忆 | 2 | `memory_recalled`, `memory_failed` |
| 其他 | 4 | `todo_updated`, `user_notification`, `agent_error`（异常逃逸 `run()` 时真实发射，随后补发 `session_ended(reason="error")`）, `system_log` |

**共 30 种事件**，每种带类型化 payload。

## 常见模式

### 流式打字机（CLI/UI）

```python
def typewriter(event):
    if event.data.is_think:
        return
    print(event.data.text, end="", flush=True)

bus.subscribe(AgentEventType.STREAM_DELTA, typewriter)
```

### 成本追踪

```python
def track_cost(event):
    if event.type == AgentEventType.STATUS_CHANGED:
        d = event.data
        if hasattr(d, "prompt_tokens"):
            statsd.gauge("agent.tokens", d.prompt_tokens + (d.completion_tokens or 0))
```

### 审计日志

```python
import json

def write_audit(event):
    log.write(json.dumps({
        "ts": time.time(),
        "type": event.type.value,
        "session_id": event.session_id,
        "payload": event.payload,
    }) + "\n")

bus.subscribe(None, write_audit)  # 所有事件
```

## 下一步

- [Hooks](hooks.md) — 改变控制流
- [完整 Event 参考](../../events.md) — 每个 payload 字段