# Tools and built-in capabilities

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

The factory supplies ToolRegistry and explicit RuntimeEnv. create_default_tool_registry assembles selected built-ins; it does not authorize every resource by itself. Use RequestOptions.tools to restrict a call, and declare deployment-required capabilities in required_profiles. Validation errors and missing capabilities stay visible.

Workspace tools require the intended workspace/skills profile and shell backend. background_run uses the current Runtime job ports and the original process controller. workflow uses explicit workflow_file_io. StoreBlackboard performs atomic database updates; a business API backend remains authoritative when supplied. schedule_wakeup is the single SDK timer tool, backed by TimerBackend and a ready driver. Runtime starts the default store driver; pure clients do not. Explicit custom timer backends require explicit drivers and never fall back to local tables.

Synchronous functions, async functions and callable objects are supported by the registry. Direct registry calls are for standalone ordinary tools; Runtime-bound tools require their execution context. Tool results preserve text/image/error formatting. Normal before/after hooks, actual tool pricing and allowlists still apply.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [20_default_tools.py](../../../examples/20_default_tools.py) | Default tools |
| [26_timers.py](../../../examples/26_timers.py) | Timers |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | Workflow |
| [29_shared_blackboard.py](../../../examples/29_shared_blackboard.py) | Blackboard |
| [38_mcp_tools.py](../../../examples/38_mcp_tools.py) | MCP |
| [42_build_your_own_tools.py](../../../examples/42_build_your_own_tools.py) | Tool contracts |

```bash
.venv/bin/python examples/20_default_tools.py
```

The workflow `logs` action reads the original node history. Use `source="messages"` for stored messages or `source="projection"` for the model view; narrow by `message_id`/`message_ids`, `seq`, or `send_index`. `cursor`, `limit`, `char_offset`, and `max_chars` bound the result. `output_file` exports JSONL through the factory-provided `workflow_file_io`, within its authorized workspace; it does not rerun the node.
