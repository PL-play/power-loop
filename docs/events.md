# Events 完整参考

Event 是 power-loop 的**旁路观测**通道。订阅者**只读**——event 不能改控制流。
所有事件都流过同一个 `AgentEventBus`；订阅者错误不会污染主循环（默认
`suppress_subscriber_errors=True`）。

> 需要改 messages / 改 LLM 请求 / 终止 loop？用 [Hooks](hooks.md)，不是 event。

## 目录

- [1. 概念速览](#1-概念速览)
- [2. 完整 AgentEventType 列表](#2-完整-agenteventtype-列表)
  - [2.1 Session lifecycle](#21-session-lifecycle)
  - [2.2 Round lifecycle](#22-round-lifecycle)
  - [2.3 Streaming lifecycle](#23-streaming-lifecycle)
  - [2.4 Tool lifecycle](#24-tool-lifecycle)
  - [2.5 Status / usage](#25-status--usage)
  - [2.6 Todo](#26-todo)
  - [2.7 通知 / 日志 / 错误](#27-通知--日志--错误)
  - [2.8 Subagent](#28-subagent)
- [3. 订阅 event](#3-订阅-event)
- [4. 常见模式](#4-常见模式)

---

## 1. 概念速览

每个 `AgentEventType` 对应一个 **typed payload dataclass**
（`power_loop.contracts.event_payloads`）。`AgentEvent` 还携带四个公共
路由字段：

| 字段 | 说明 |
|---|---|
| `type` | `AgentEventType` |
| `data` | typed payload（推荐访问） |
| `payload` | 同 payload 的 dict 形式（legacy 兼容） |
| `session_id` | 触发事件的 session |
| `round_index` | 触发时所在的轮次（部分事件） |
| `stream_id` | 流式事件的流 id（同一回合可能并行多个） |
| `source` | 事件来源标签，常用于子代理冒泡：`"subagent:<sid>"` |

```python
def on_delta(event: AgentEvent) -> None:
    delta: StreamDeltaPayload = event.data
    print(delta.text, end="")
```

## 2. 完整 AgentEventType 列表

### 2.1 Session lifecycle

#### `session_started`

Payload — `SessionStartedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `scope` | `str` | `"main"` / `"subagent"` |

**何时触发**：`session.start` hook 之后。
**典型订阅者**：metrics 系统打开始时间戳；UI 显示 "thinking..."。

#### `session_ended`

Payload — `SessionEndedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `reason` | `str` | `"completed"` / `"hit_round_limit"` / `"cancelled"` / `"hook_break"` |

**何时触发**：所有 round 走完或被打断后。
**典型订阅者**：UI 收尾、关闭后台 streaming 渲染、上报 latency。

### 2.2 Round lifecycle

#### `round_started`

Payload — `RoundStartedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `round_index` | `int` | 0-based |

**何时触发**：每轮开始（compact / todo 提示之后）。

#### `round_completed`

Payload — `RoundCompletedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `round_index` | `int` | |
| `has_tools` | `bool` | 本轮 LLM 是否调用了工具 |
| `used_todo` | `bool` | 是否用了 todo 工具 |

#### `round_tools_present`

Payload — `RoundToolsPresentPayload`：`has_tools: bool`。

LLM 返回后立刻发，给 UI 提示 "agent is about to call tools"。

### 2.3 Streaming lifecycle

> 仅当 LLM 客户端实际产生流式 chunk 时才会发。

#### `stream_started`

Payload — `StreamStartedPayload`：`stream_id: str`（默认 `"main"`）。

#### `stream_delta`

Payload — `StreamDeltaPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `stream_id` | `str` | |
| `text` | `str` | 这一片文本 |
| `is_think` | `bool` | 是否是 reasoning/thinking 段（部分模型） |

**典型订阅者**：UI 增量渲染、CLI 打字机效果。

#### `stream_think_delta`

同 `stream_delta`，但 `is_think=True`。

#### `stream_completed`

Payload — `StreamCompletedPayload`：`stream_id: str`。

### 2.4 Tool lifecycle

#### `tool_call_started`

Payload — `ToolCallStartedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | tool 名 |
| `tool_input` | `dict` | 工具参数 |
| `tool_call_id` | `str` | OpenAI tool_call_id |

#### `tool_call_completed`

Payload — `ToolCallCompletedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | |
| `output` | `str` | 工具结果 |
| `tool_input` | `dict` | |
| `tool_call_id` | `str` | |

**典型订阅者**：UI 工具调用面板、审计日志。

#### `tool_call_failed`

同 `tool_call_completed`，但 `output` 是错误描述。在 `tool.error` hook 没把
failed 改成 false 时触发。

### 2.5 Status / usage

#### `status_changed`

`StatusChangedPayload` 是个抽象基类，`kind` 字段做 discriminator。当前有三个子类型：

##### `kind="auto_compact"` — `AutoCompactStatusPayload`

| 字段 | 类型 | 说明 |
|---|---|---|
| `phase` | `str` | `"started"` |
| `round_index` | `int` | |
| `trigger` | `str` | `"compactor_plan_emitted"` |
| `before_tokens` | `int` | |
| `after_tokens` | `int` | |

##### `kind="round_usage"` — `RoundUsageStatusPayload`

| 字段 | 类型 | 说明 |
|---|---|---|
| `time_iso` | `str` | |
| `round_index` | `int` | |
| `round_number` | `int` | 1-based |
| `max_rounds` | `int` | |
| `prompt_tokens` | `int \| None` | |
| `completion_tokens` | `int \| None` | |
| `cache_read_tokens` | `int \| None` | |
| `reasoning_tokens` | `int \| None` | |

每轮在 `round.end` 之后发。**用作 cost / token 仪表盘的主数据源**。

##### `kind="hit_round_limit"` — `HitRoundLimitStatusPayload`

| 字段 | 类型 | 说明 |
|---|---|---|
| `max_rounds` | `int` | |

#### `usage_updated`

Payload — `UsageUpdatedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `usage` | `dict` | 原始 LLM usage dict |

### 2.6 Todo

#### `todo_updated`

Payload — `TodoUpdatedPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `items` | `list[dict]` | 每条 todo（`text` / `status` / `id`） |
| `counts` | `dict[str, int]` | `{"total": N, "completed": M}` |
| `rendered` | `str` | 给 LLM 看的 ASCII 渲染 |
| `text` | `str` | 同 rendered，向后兼容字段 |

**典型订阅者**：UI 任务清单面板。

### 2.7 通知 / 日志 / 错误

#### `user_notification`

Payload — `UserNotificationPayload`：`message: str`。
power-loop 主动给用户的提示（例如 "update your todos"）。

#### `agent_error`

Payload — `AgentErrorPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `error` | `str` | |
| `error_type` | `str` | |
| `details` | `str` | |

#### `system_log`

Payload — `SystemLogPayload`：`message: str` / `level: str`（`"info"` / `"warn"` / `"error"`）。

### 2.8 Subagent

#### `subagent_task_start`

Payload — `SubagentTaskStartPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task` | `str` | 截断后的任务描述（≤500 chars） |
| `preset` | `str` | |
| `sub_session_id` | `str` | |
| `depth` | `int` | 嵌套深度 |

#### `subagent_text`

Payload — `SubagentTextPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sub_session_id` | `str` | |
| `status` | `str` | `"completed"` / `"hit_round_limit"` / ... |
| `rounds` | `int` | |
| `final_text` | `str` | 截断后子代理回复（≤2000 chars） |

#### `subagent_limit`

子代理 hit round limit 时单独冒一次，方便 UI 区分。

Payload — `SubagentLimitPayload`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sub_session_id` | `str` | |
| `max_rounds` | `int` | |

#### `subagent_completed`

Payload — `SubagentCompletedPayload`：同 `SubagentTextPayload` 字段。

---

## 3. 订阅 event

```python
from power_loop import StatefulAgentLoop, AgentEvent, AgentEventBus, AgentEventType

bus = AgentEventBus(suppress_subscriber_errors=True)

def on_stream_delta(event: AgentEvent) -> None:
    print(event.data.text, end="", flush=True)

def on_round_usage(event: AgentEvent) -> None:
    if event.data.kind == "round_usage":
        print(f"\n[round {event.data.round_number}] "
              f"tokens={event.data.prompt_tokens}+{event.data.completion_tokens}")

bus.subscribe(AgentEventType.STREAM_DELTA, on_stream_delta)
bus.subscribe(AgentEventType.STATUS_CHANGED, on_round_usage)

loop = StatefulAgentLoop(llm=..., db_path="...", event_bus=bus, ...)
```

订阅可以 sync 或 async；bus 会自动判断。
`bus.subscribe(None, fn)` 订阅**所有**事件（用于 logging / 调试）。

## 4. 常见模式

### CLI 打字机

```python
def typewriter(event: AgentEvent) -> None:
    if event.data.is_think:
        return  # 跳过 reasoning chunk
    print(event.data.text, end="", flush=True)

bus.subscribe(AgentEventType.STREAM_DELTA, typewriter)
```

### Token / 成本上报

```python
def on_status(event: AgentEvent) -> None:
    if event.data.kind == "round_usage":
        statsd.gauge("agent.prompt_tokens", event.data.prompt_tokens or 0)
        statsd.gauge("agent.completion_tokens", event.data.completion_tokens or 0)
```

### 子代理冒泡到父 UI

```python
def bubble_subagent(event: AgentEvent) -> None:
    if event.source and event.source.startswith("subagent:"):
        ui.show_subagent_progress(event.source, event.data)

bus.subscribe(None, bubble_subagent)
```

### 全量审计日志

```python
audit_log = open("session.jsonl", "a")

def write_audit(event: AgentEvent) -> None:
    audit_log.write(json.dumps({
        "ts": time.time(),
        "type": event.type.value,
        "session_id": event.session_id,
        "round_index": event.round_index,
        "payload": event.payload,
    }, default=str) + "\n")

bus.subscribe(None, write_audit)
```

### Push 进度到 WebSocket

```python
async def push_to_ws(event: AgentEvent) -> None:
    if event.type in {AgentEventType.STREAM_DELTA,
                       AgentEventType.TOOL_CALL_STARTED,
                       AgentEventType.TOOL_CALL_COMPLETED,
                       AgentEventType.ROUND_COMPLETED}:
        await websocket.send_json({
            "type": event.type.value,
            **event.payload,
        })

bus.subscribe(None, push_to_ws)
```

---

需要改控制流（请求 / 工具 / 终止）？看 [`docs/hooks.md`](hooks.md)。
