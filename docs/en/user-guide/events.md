# Events

[中文](../../zh/user-guide/events.md) | [User Guide](../index.md)

Events are the **read-only observation** channel. Subscribers see everything that happens — but cannot change control flow (use [Hooks](hooks.md) for that).

> **Full reference**: [docs/events.md](../../events.md) — all 30 event types with typed payload fields.

## Quick Example

```python
from power_loop import AgentEventBus, AgentEventType

bus = AgentEventBus()

# Streaming typewriter
def on_delta(event):
    print(event.data.text, end="", flush=True)

bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)

# Subscribe to everything
def audit(event):
    log.write(f"[{event.type.value}] sid={event.session_id}\n")

bus.subscribe(None, audit)  # None = all events

loop = StatefulAgentLoop(llm=llm, event_bus=bus, config=config)
```

## Event Anatomy

```python
@dataclass
class AgentEvent:
    type: AgentEventType          # enum value
    data: BaseEventPayload        # typed payload (preferred access)
    payload: dict                 # same data as dict (legacy)
    session_id: str | None
    round_index: int | None
    stream_id: str | None
    source: str | None            # "subagent:<sid>" for child agents
    ts: float                     # epoch-seconds creation time (auto-filled)
    seq: int                      # process-wide monotonic order (H4.2)
```

Access payload fields with IDE autocomplete via `event.data.field_name`.

## Event Types by Category

### Session

| Event | Payload | When |
|---|---|---|
| `session_started` | `scope` | After `session.start` hook |
| `session_ended` | `reason` | After the loop terminates |

### Round

| Event | Payload | When |
|---|---|---|
| `round_started` | `round_index` | Start of each round |
| `round_completed` | `round_index`, `has_tools` | End of each round |
| `round_tools_present` | `has_tools` | LLM returned tool calls |

### Streaming

| Event | Payload | When |
|---|---|---|
| `stream_started` | — | LLM streaming begins |
| `stream_delta` | `text`, `is_think` | Each token chunk |
| `stream_think_delta` | `text`, `is_think=True` | Reasoning chunk |
| `stream_completed` | — | LLM streaming ends |

### Tools

| Event | Payload | When |
|---|---|---|
| `tool_call_started` | `name`, `tool_input`, `tool_call_id` | Before tool execution |
| `tool_call_completed` | `name`, `output`, `tool_input` | After successful tool |
| `tool_call_failed` | `name`, `output` (error) | Tool raised an exception |

### Status / Usage

| Event | Payload | When |
|---|---|---|
| `status_changed` | Varied by `kind` | Compaction, usage, round limit |
| `usage_updated` | `usage` dict | Token usage of ONE LLM call (per-call, not cumulative). For whole-run totals just read `result.usage` on the `send()` return value instead of subscribing. |

### Sub-agent

| Event | Payload | When |
|---|---|---|
| `subagent_task_start` | `task`, `sub_session_id`, `depth` | Child agent begins |
| `subagent_text` | `sub_session_id`, `final_text`, `rounds` | Child agent completes |
| `subagent_limit` | `sub_session_id`, `max_rounds` | Child hit round limit |
| `subagent_completed` | Same as `subagent_text` | Child finished normally |

### Per-call LLM (0.14.0)

Each LLM call attempt emits a pair (paired by `call_id`), exposing per-call latency
and usage — unlike `usage_updated`'s cumulative totals — so retries are individually visible.

| Event | Payload | When |
|---|---|---|
| `llm_call_started` | `call_id`, `round_index`, `attempt`, `model` | Start of each LLM call attempt |
| `llm_call_completed` | `call_id`, `duration_ms`, `success`, `error_type`, per-call `prompt_/completion_/total_tokens` | Each attempt finishes (success or error) |

### Retry / Cancel (M1.1)

| Event | Payload | When |
|---|---|---|
| `llm_retry_attempted` | `attempt`, `error_type`, `next_sleep_seconds` | After each failed LLM attempt |
| `llm_degraded` | `reason`, `attempts`, `error_type` | Retry exhausted or timeout |
| `loop_cancelled` | `reason`, `round_index` | CancellationToken fired |

### Memory (M1.9)

| Event | Payload | When |
|---|---|---|
| `memory_recalled` | `returned`, `injected`, `budget_tokens` | After memory recall + injection |
| `memory_failed` | `phase`, `error_type` | `recall()` or `remember()` raised |

### Other

| Event | Payload | When |
|---|---|---|
| `todo_updated` | `items`, `counts` | Todo list changed |
| `user_notification` | `message` | Library wants to tell the user something |
| `agent_error` | `error`, `error_type` | An unexpected exception escaped `run()` (raising hook/sink/store); now actually emitted (H1.5), followed by `session_ended(reason="error")` |
| `system_log` | `message`, `level` | Internal log message |

## Common Patterns

### Token Streaming (CLI/UI)

```python
def typewriter(event):
    if event.data.is_think:
        return
    print(event.data.text, end="", flush=True)

bus.subscribe(AgentEventType.STREAM_DELTA, typewriter)
```

### Cost Tracking

```python
def track_cost(event):
    if event.type == AgentEventType.STATUS_CHANGED:
        d = event.data
        if hasattr(d, "prompt_tokens"):
            statsd.gauge("agent.tokens", d.prompt_tokens + (d.completion_tokens or 0))
```

### Audit Log

```python
import json

def write_audit(event):
    log.write(json.dumps({
        "ts": time.time(),
        "type": event.type.value,
        "session_id": event.session_id,
        "payload": event.payload,
    }) + "\n")

bus.subscribe(None, write_audit)  # all events
```

## Next

- [Hooks](hooks.md) — change control flow (the other channel)
- [Full Event Reference](../../events.md) — every payload field