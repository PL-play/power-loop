# Advanced Runtime Tools

[中文](../../zh/user-guide/advanced-runtime-tools.md) | [User Guide](index.md)

Some tools are more than request/response functions. They need durable state, loop awareness, policy gates, UI events, or state projected into the next LLM round. power-loop exposes these pieces as public primitives so you can build your own advanced tools without relying on private internals.

## Mental Model

Use four layers:

| Layer | Primitive | Purpose |
|---|---|---|
| Tool handler | `ToolRegistry`, `ToolDefinition`, `get_tool_runtime_context()` | Execute domain logic with access to the current session/store. |
| Durable state | `SessionStore.get_runtime_state()` / `set_runtime_state()` | Persist tool state beside the conversation log. |
| Prompt projection | `RuntimeProjector`, `AgentLoopConfig.runtime_projectors` | Convert durable state into transient LLM messages. |
| Control/observability | `AgentHooks`, `AgentEventBus` | Intercept decisions and observe lifecycle events. |

The conversation `messages` table remains the protocol log. Runtime state lives beside it and is projected only when needed, so compaction does not duplicate or corrupt it.

## Tool Runtime Context

Inside a tool handler:

```python
from power_loop import get_tool_runtime_context

def save_marker(marker: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    ctx.store.set_runtime_state(ctx.session_id, "marker", {"value": marker})
    return "saved"
```

`ctx` includes:

| Field | Use |
|---|---|
| `session_id` | Current session id. |
| `store` | Current `SessionStore`; use it for messages, runtime state, background tasks, session rows. |
| `loop` | Current `StatefulAgentLoop`; useful for `get_messages()` or higher-level loop APIs. |
| `config` | Current `AgentLoopConfig`. |

## Runtime Projectors

Projectors turn durable runtime state into temporary LLM-visible messages:

```python
from power_loop import RuntimeProjector

class MarkerProjector(RuntimeProjector):
    def project(self, *, store, session_id, round_index, context):
        state = store.get_runtime_state(session_id, "marker", default={}) or {}
        if not state:
            return []
        return [{"role": "user", "name": "marker_state", "content": str(state)}]
```

Register it:

```python
config = AgentLoopConfig(runtime_projectors=(MarkerProjector(),))
```

Pass `runtime_projectors=()` to disable all default projections.

## Hooks

Hooks are the policy and control plane:

```python
def before_tool(ctx):
    if ctx.tool_name == "deploy" and ctx.tool_args["target"] == "production":
        ctx.tool_args["target"] = "staging"

hooks.register(HookPoint.TOOL_BEFORE, before_tool)
```

Common patterns:

- Rewrite unsafe tool arguments.
- Skip a tool call and return an approval message.
- Persist derived state in `TOOL_AFTER`.
- Short-circuit an LLM call in `LLM_BEFORE`.

## Events

Events are the observability plane:

```python
def on_event(event):
    if event.type is AgentEventType.TOOL_CALL_COMPLETED:
        print(event.data.name, event.data.output)

bus.subscribe(None, on_event)
```

Use events for UI updates, audit logs, metrics, and external schedulers. Events do not replace durable state; they are live notifications.

## Examples

Runnable examples live in [`examples/advanced_runtime/`](../../../examples/advanced_runtime/):

| Example | Shows |
|---|---|
| `01_runtime_projector.py` | Project incident state into the next LLM round. |
| `02_tool_runtime_context.py` | Tool queries current session metadata/messages and writes runtime state. |
| `03_hooks_control_flow.py` | Hooks rewrite deployment arguments and record audit state. |
| `04_events_observability.py` | Event bus collects custom tool lifecycle audit entries. |

## Design Rule

If a tool needs state that should survive compaction, process restarts, or future turns, store it in `SessionStore` runtime state. If the LLM needs to see that state, expose it with a `RuntimeProjector`. If execution needs policy or monitoring, use hooks and events.
