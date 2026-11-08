# Hooks

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Supply AgentHooks in ExecutionEnvironment.hooks. Register handlers against HookPoint with the documented HookDirective for that point. Returning an arbitrary replacement object is not an instruction. Hooks run on the execution node, so factories must reconstruct them from the declared hook profile.

Use TOOL_BEFORE for permission and argument decisions, model hooks for short-circuit/cache decisions, and memory hooks for injection/compaction policy. A rewritten ordinary tool call is priced from the actual arguments. A hook must not replace cancellation, ownership loss or unknown external work with synthetic success.

TIMER_FIRE preserves continue, skip, cancel and postpone decisions. TimerFireCtx carries trusted timer/session provenance; the model cannot forge it. Runtime's timer driver submits accepted work and does not wait for model completion.

Named hooks support register/replace/remove/has. Built-in memory hooks can be replaced explicitly. EventBus is for observations; hooks participate in execution and obey configured timeouts. Host policies must run at the authoritative admission or write port when a decision must also cover non-model callers.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [07_human_approval.py](../../../examples/07_human_approval.py) | Approval policy |
| [26_timers.py](../../../examples/26_timers.py) | Timer hooks and driver |
| [43_memory_hook_override.py](../../../examples/43_memory_hook_override.py) | Memory overrides |
| [44_hook_events_audit.py](../../../examples/44_hook_events_audit.py) | Hook directives |

```bash
.venv/bin/python examples/07_human_approval.py
```
