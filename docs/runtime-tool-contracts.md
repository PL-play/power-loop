# Runtime tool implementations

The runtime does not grant execution permission from a tool's name, a host
module path, a concrete business class, or an unregistered handler attribute.
The trusted Python host implements `RuntimeTool` and registers that instance
with its actual `ToolDefinition`. `FunctionRuntimeTool` is a convenience wrapper
for the same interface. Model arguments cannot declare runtime capabilities.

```python
from power_loop.runtime.tool_contracts import (
    FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract,
)

handler = FunctionRuntimeTool(
    read_original,
    ToolRuntimeContract(
        "example.original-reader.v1",
        allowed_roles=("main", "workflow_node"),
        background_allowed=False,
    ),
    policy=lambda arguments: ToolDispatchPolicy(
        native_admitted=True,
        control_only=True,
    ),
)
registry.register(definition, handler)
```

The example assumes a host-implemented, read-only control action. Declaring
`control_only=True` for an action that starts new work is an implementation
error. A declaration cannot supply missing cancellation, external resource
exit, model budget, accounting, or outcome-recovery support. Such work must use
the SDK lifecycle interfaces for its actual resource type.

## Contracts and responsibilities

| Interface | Responsibility |
| --- | --- |
| `runtime_contract` | Stable implementation key, allowed execution roles, capabilities and background policy. |
| `dispatch_policy(arguments, runtime=...)` | Classify the actual call for native admission and the control channel. Default grants neither. |
| `format_result(result, max_chars)` | Optional structured result view. Default preserves normal formatting. |
| `context_spec(spec)` | Optional model-facing schema when tool-context preparation is enabled. It may adapt its own copied spec and must retain the registered name. |
| `background_hint(definition, scheduler_name=...)` | An implemented scheduler describes how to submit a target using its actual registered name. |
| `result_sources(source, store=..., registry=...)` | A declared result container reads original target results; it must not start tasks or make paid requests. |

`background_allowed=False` forbids placing this entry inside another background
tool call. Orchestration implementations declare this explicitly, including
SDK workflow, SDK subagent, the platform's subagent wrapper and the SDK
background scheduler. This restriction survives aliases and ordinary wrappers.
An ordinary tool called `workflow` is still governed by its own definition and
contract. `async_capable` remains a separate prerequisite for background
submission; the new flag never grants it.

`result_container=True` declares a composite receipt. The SDK background tool
implements its own persisted-result reader. Context preparation and archiving
use that implementation, including under aliases. Missing or unfinished target
results stay uncompressed. Containers do not recursively expand other
containers. The legacy original-message reader recognizes the stored receipt
format without requiring a still-registered implementation; it only reads
original records in the same session and grants no execution permission.

A scheduler advertises `tool.background-submit.v1` to offer background hints.
The registry emits hints only when there is one such implementation, avoiding
an arbitrary choice between schedulers. Names and lookalike attributes alone
supply no hints, schemas or result-container behavior.

## Binding, aliases and compatibility

Registration snapshots the contract, actual schema, required parameters,
concurrency and interruption policy, and the actual implementation instance.
A subset preserves that binding. Ordinary wrappers must retain `__wrapped__`
(for example with `functools.wraps`) or explicitly implement the interface.
Changing the implementation or frozen declaration requires a fresh registration.
Invocation rechecks the binding after admission waits. Background queues also
recheck immediately before starting a target, preventing a queued call from
silently invoking a replacement handler.

Plain handlers remain supported through their registered definition. They gain
no native/control capability merely by sharing a built-in tool name. Public
legacy argument-validation helpers and built-in catalog names remain available
for compatibility; runtime validation uses the actual registered schema.

Product routing, message normalization, API clients, user allowlists and host
presentation remain in the host. The SDK may implement its own generic resource
kinds and built-in data formats, but must not import a platform's business code
to decide execution authority.
