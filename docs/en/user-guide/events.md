# Events and output

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Attach an EventBus through ExecutionEnvironment.event_bus. Subscribe before execution; use session and execution identities to correlate model, tool, child, usage and lifecycle events. A shared bus must filter by the intended session. Runtime removes its own streaming subscriptions and drains bounded async subscribers when execution ends; the host owns its subscribers and bus.

Events observe activity. Hooks may change decisions. Neither is a second accounting ledger: use persisted task results and accounting ports for fees. Streaming deltas are live notifications; reconnecting clients query the original input result instead of assuming a lost delta can be replayed.

Output subscriptions have bounded buffers. Slow consumers must handle gaps and fetch persisted state. Do not perform blocking network or database work on every token callback. Event payloads can include user/tool content, so hosts decide retention and access.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [08_streaming.py](../../../examples/08_streaming.py) | Live output |
| [09_audit_log.py](../../../examples/09_audit_log.py) | Event observations |
| [25_token_usage.py](../../../examples/25_token_usage.py) | Usage |
| [36_observability.py](../../../examples/36_observability.py) | Metrics |
| [44_hook_events_audit.py](../../../examples/44_hook_events_audit.py) | Hooks and events |

```bash
.venv/bin/python examples/08_streaming.py
```
