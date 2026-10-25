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

## Progress and presentation

Progress tracking uses the implemented `task.progress.v1` capability. Only a
successful call through its registered `RuntimeTool` resets progress reminders
and sets the existing `used_todo` event field. That field keeps its public name
for compatibility. A plain handler named `todo` supplies no progress capability;
a registered implementation keeps the capability under an alias. The built-in
progress handler declares this capability, without granting native admission.

`ToolDefinition.usage_hint` holds optional model-facing guidance (up to 4096
characters). `background_hint=False` suppresses background submission guidance
for a quick tool while preserving `async_capable` and all execution policies.
Both are presentation metadata, not permissions. An actual registry's guide
reads its own definitions; it never substitutes a same-named built-in. Built-in
catalog previews without a registry remain available to existing prompt-builder
callers. The default runtime prompt uses the actual registry and includes the
tool catalog once when catalog injection is enabled.

```python
from dataclasses import replace

# Preserve the implemented handler and metadata when giving a built-in an alias.
source = defaults.get("read_file")
registry.register(replace(source.definition, name="read_notes"), source.handler)

# A host-defined quick tool supplies its own guidance and presentation policy.
registry.register(
    ToolDefinition(
        "read_status",
        "Read current status.",
        async_capable=True,
        usage_hint="Read once before requesting a change.",
        background_hint=False,
    ),
    read_status,
)
```

Ordinary host wrappers should use `functools.wraps` to preserve the implemented
contract. The DeepTalk progress and timer guards do this while retaining their
product-specific validations in the host. Contracts, including presentation
metadata, remain bound to the actual registration; an alias, wrapper or subset
cannot authorize a changed implementation.

## Context presentation

Context preparation also consumes the actual registered implementation:

- `context_presentation(source)` returns `ToolContextPresentation`, with an
  optional archival label, JSON-object argument view and strict boolean
  `keep_in_send`. The default is a generic view without in-send preservation.
- `truncate_context(source, policy, raw_arguments=...)` returns an optional
  `(arguments_json, result_text)` pair. The default uses generic bounded
  previews; `None` keeps the original pair.

These methods are pure views of an existing call. They must not execute tools,
start jobs, call a provider or acknowledge results. They grant no execution,
role, native admission, background or recovery capabilities. Context readers
revalidate the actual registration before using the methods. Invalid return
shapes are rejected.

The SDK shell owns its command/output truncation, purpose/exit-code labels and
`context_policy=keep` interpretation. The SDK skill reader owns its skill/doc
label. Both work under aliases and `functools.wraps` wrappers. A plain handler
called `bash` or `load_skill` receives the generic presentation. The explicit
`truncate_bash` format helper remains callable, while legacy `truncate_view`
and archival helpers without a registered implementation use generic views.

In-send preservation applies to both fuse levels and to targets of result
containers. Between sends, normal `general` policy still applies. Group archives
use implemented labels and keep exact original recall coordinates. Persisted
patches remain idempotent across restarts; the projection cache version changes
for this presentation protocol. Original message rows remain untouched and
readable even when a current implementation is absent.

Hosts can subclass `RuntimeTool` to provide the same presentation extension
without inheriting a shell, skill reader or platform-specific class. Business
message routing and conversation authorization remain host responsibilities.
SDK job kinds, execution roles, built-in catalogs, persisted state keys and
historical named configuration are data contracts, not tool-name admission.

Skill-reader guidance uses `skill.read.v1` on the actual registry and respects
selected tool names. The default skill reader declares it; an ordinary handler
named `load_skill` supplies no implemented reader guidance. Runtime prompt
assembly passes its registry to the skill section rather than advertising a
fixed entry name. Standalone SDK catalog helpers keep their catalog-preview role.

Yield statistics use `task.yield.v1`. An implemented yield call is excluded from
`pass_progress().work_tools` under aliases and wrappers. A plain handler called
`pass_turn` counts as a work call; its name cannot stop the loop. Actual yielding
still requires the existing `request_pass()` effect. Declaring the capability
only describes statistics and grants no native admission or control permission.
The host's guarded yield tool explicitly declares the capability while retaining
its own business checks. Without a registry, the statistics helper cannot infer
semantic yield from a name.

Image placeholders retain the original filename and opaque host recall reference.
Their wording does not assume any platform-specific image tool is installed.
