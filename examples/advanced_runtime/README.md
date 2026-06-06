# Advanced Runtime Examples

[Back to examples](../README.md)

These examples show how to build runtime-bound tools from public power-loop primitives. They use a scripted fake LLM, so they run without API credentials.

| File | Scenario | Key APIs |
|---|---|---|
| [`01_runtime_projector.py`](01_runtime_projector.py) | Incident commander state projected into the next LLM round | `SessionStore.set_runtime_state`, `RuntimeProjector` |
| [`02_tool_runtime_context.py`](02_tool_runtime_context.py) | Tool queries the current session and writes runtime state | `get_tool_runtime_context`, `loop.get_messages()` |
| [`03_hooks_control_flow.py`](03_hooks_control_flow.py) | Approval hook rewrites a deployment tool call | `TOOL_BEFORE`, `TOOL_AFTER` |
| [`04_events_observability.py`](04_events_observability.py) | Event subscriber builds an audit trail for custom tools | `AgentEventBus`, `TOOL_CALL_*` |

Run:

```bash
python examples/advanced_runtime/01_runtime_projector.py
python examples/advanced_runtime/02_tool_runtime_context.py
python examples/advanced_runtime/03_hooks_control_flow.py
python examples/advanced_runtime/04_events_observability.py
```
