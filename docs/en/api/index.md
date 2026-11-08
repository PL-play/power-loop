# API Reference

[中文](../../zh/api/index.md) | [Back to docs](../../README.md)

This page tracks the public surface that is available from `import power_loop`.
For behavior and examples, use the linked user-guide pages. For exact signatures,
inspect the source modules linked below.

## Stability

| Tier | Meaning |
|---|---|
| Stable | Backward compatible across minor releases. See `power_loop.STABLE_API`. |
| Provisional | Re-exported from `power_loop`, but may change in a future minor. |
| Internal | Submodule imports such as `power_loop.core.*`; no compatibility guarantee. |

## Current runtime (11.0.0)

See the [current runtime contract](../../runtime-11.md) and the executable
`examples/design203_single_process.py`. `StatefulAgentLoop(runtime=...)` delegates
to `RuntimeClient`; the old direct constructor, sync methods and process worker
bootstrap are retired. Acceptance receipts and persisted results are distinct.


## Tools and Sub-Agents

| Symbol | Covers | More |
|---|---|---|
| `ToolRegistry` | register, invoke, validate, OpenAI tool conversion, `names()`, restricted `subset()` registries | [Tools](../user-guide/tools.md), [source](../../../power_loop/tools/registry.py) |
| `ToolDefinition` | name, description, JSON Schema, required params | [Tools](../user-guide/tools.md), [source](../../../power_loop/contracts/tools.py) |
| `create_default_tool_registry` | bound or runtime-resolved built-in tool registry | [Tools](../user-guide/tools.md), [source](../../../power_loop/tools/__init__.py) |
| `DEFAULT_TOOL_HANDLERS` | public handler mapping for custom registry composition | [Tools](../user-guide/tools.md), [source](../../../power_loop/tools/default_tools.py) |
| `AgentSpec` | declarative child-agent spec | [Sub-agents](../user-guide/subagents.md), [source](../../../power_loop/runtime/spec.py) |
| `run_agent_spec` | admit a child and await its durable result from a trusted host command | [Sub-agents](../user-guide/subagents.md), [source](../../../power_loop/runtime/spec.py) |
| `register_spawn_agent` | `spawn_agent` meta-tool (the former `run_agent` is merged in since 4.0) | [Sub-agents](../user-guide/subagents.md), [source](../../../power_loop/tools/spawn_agent.py) |

## Hooks and Events

| Symbol | Covers | More |
|---|---|---|
| `AgentHooks` | registering sync/async hooks | [Hooks](../user-guide/hooks.md), [full reference](../../hooks.md) |
| `HookPoint` | lifecycle hook enum | [Hooks](../user-guide/hooks.md), [source](../../../power_loop/contracts/hooks.py) |
| `HookDirective` | continue, skip, break, short-circuit | [Hooks](../user-guide/hooks.md), [source](../../../power_loop/contracts/hooks.py) |
| `AgentEventBus` | event subscriptions and publication | [Events](../user-guide/events.md), [full reference](../../events.md) |
| `AgentEventType` | typed event names | [Events](../user-guide/events.md), [source](../../../power_loop/contracts/events.py) |

## Runtime Helpers

| Symbol | Covers | More |
|---|---|---|
| `LLMRetryPolicy` | attempts, backoff, timeout, retry filter | [Retry & Cancel](../user-guide/retry-cancel.md), [source](../../../power_loop/runtime/retry.py) |
| `CancellationToken` | uniform cancellation shape | [Retry & Cancel](../user-guide/retry-cancel.md), [source](../../../power_loop/runtime/cancellation.py) |
| `StructuredOutputSpec` | response-format schema | [Structured Output](../user-guide/structured-output.md), [source](../../../power_loop/runtime/structured.py) |
| `MemoryProvider` | recall and remember protocol | [Memory](../user-guide/memory.md), [source](../../../power_loop/runtime/memory.py) |
| `LLMProviderConfig` | provider/env configuration | [Providers](../user-guide/providers.md), [source](../../../power_loop/runtime/provider.py) |
| `FoldStrategy`, `LLMSummaryFold`, `AgenticFold` | the `fold_strategy` axis: fold Protocol + built-ins (`keep_last_sends`, `trigger_ratio`, async `fold`); context summary compaction | [Compaction](../user-guide/compaction.md), [source](../../../power_loop/runtime/fold.py) |
| `Representation`, `VerbatimRepresentation`, `ProjectedRepresentation` | the `representation` axis: representation Protocol + built-ins (verbatim default / `ProjectedRepresentation(max_chars=...)`); per-send projection of finished sends into the derived `pl_project_messages` (`ProjectMessageRow`); `ToolDefinition.project` hook + the `recall_send` tool re-expand detail | [Projection](../user-guide/send-context-projection.md), [source](../../../power_loop/runtime/representation.py) |
| `RuntimeEnv`, `runtime_env_context` | per-invocation workspace/home/skills and shell backend | [Tools](../user-guide/tools.md), [source](../../../power_loop/runtime/env.py) |
| `ShellBackend`, `LocalShellBackend` | persistent-shell launch and execution-target identity | [Tools](../user-guide/tools.md), [source](../../../power_loop/runtime/exec_backend.py) |
| `PowerLoopError` and subclasses | common exception hierarchy; every subclass carries a stable dotted `code` (class attribute) — branch on `exc.code`, not class identity | [Error codes](#error-codes), [source](../../../power_loop/contracts/errors.py) |

## LLM Contract

The provider-agnostic LLM types, re-exported from the top level (`from power_loop import …`) so
you don't reach into the vendored transport package. **STABLE as of 1.0** — `LLMService`,
`LLMRequest`, `LLMResponse`, `LLMStreamChunk`, `LLMProviderConfig`, and the
`create_llm_service_from_env`/`_config` factories are frozen for factories to build and own the current worker's provider.

| Symbol | Covers | More |
|---|---|---|
| `LLMService` | the LLM Protocol you implement or wrap | [Providers](../user-guide/providers.md), [source](../../../power_loop/runtime/provider.py) |
| `LLMRequest`, `LLMResponse`, `LLMStreamChunk`, `LLMTokenUsage` | request/response/stream/usage shapes (used in `llm.*` hooks and custom services) | [Providers](../user-guide/providers.md) |
| `OpenAICompatibleChatConfig`, `AnthropicChatConfig` | per-transport config dataclasses | [Providers](../user-guide/providers.md) |
| `create_llm_service_from_env`, `create_llm_service_from_config`, `LLMProviderConfig` | build an `LLMService` from env/config | [Configuration](../user-guide/configuration.md) |

## Error codes

`RuntimeRequestError` is a `ValueError` with a per-instance `code`, including `capability_unavailable`, `resource_unavailable`, `definition_unavailable`, `forbidden` and `invalid_options`. Failed durable results retain this as `Result.error_code`; host query responses must preserve it. Other SDK errors below use the `PowerLoopError` hierarchy.

| Exception | `code` |
|---|---|
| `PowerLoopError` (base) | `power_loop.error` |
| `SessionNotFoundError` | `session.not_found` |
| `LLMTimeout` | `llm.timeout` |
| `LLMRetryExhausted` | `llm.retry_exhausted` |
| `CancellationRequested` | `cancelled` |
| `CompactionFailed` | `compaction.failed` |
| `ToolNotFound` | `tool.not_found` |
| `ToolValidationError` | `tool.invalid_args` |
| `SpecValidationError` | `spec.invalid` |

## Timer, factory and resource ports (11.0.0)

| Public port | Contract |
|---|---|
| `TimerBackend.create/list/cancel` | Receives trusted `TimerContext`, including original actor/tool identity |
| `StoreTimerBackend` | Default SQLite/PostgreSQL/MySQL timer store |
| `TimerDriver`, `TimerDriverStatus` | start/stop/status; mode, readiness, recent scan/error, bounded pending page |
| `TimerRunner(store=..., client=..., hooks=..., event_bus=...)` | Bounded due scan and input acceptance; no model execution |
| `RegistryRuntimeFactory.register_profile(name, preflight=...)` | Versioned capability and async deployment check before accepting |
| `Runtime(required_profiles=..., timer_backend=..., timer_driver=..., timers_enabled=...)` | Explicit composition; external backend requires a ready driver |
| `ExecutionEnvironment(runtime_env=..., workflow_file_io=..., process_control=..., resource_profile_ref=..., resource_scope=...)` | Current worker rebuilds tools/resources; shared workspace requires matching stable reference and shared scope |
| `StoreBlackboard(store)` | Database-atomic post/update/remove, including concurrent first creation; `delete(board_id)` ends a host-owned scope |
| `Runtime.status()` | Local accepting, ports, driver status, bounded pending/active counts and last error; no health SQL per request |
| `Result.error_code` | Original machine-readable RuntimeRequestError code survives persistent result reads |

`SqliteBlackboard` and `TimerDelivery` are removed without aliases. Default process
coordination holds a native store lock; another independent process/domain is
rejected, while pure clients and workers sharing one coordinator/transport remain
supported. See [timers](../user-guide/timers.md), [blackboard](../user-guide/blackboard.md)
and [runtime setup](../../runtime-11.md).

## Runtime host ports (11.0.0 / schema 15 / protocol 42 / descriptor 1)

| Public symbol | Contract |
|---|---|
| `Runtime`, `RuntimeClient`, `RuntimeWorker` | Explicit lifecycle, accepted inputs, query/wait/output, cancellation and bounded shutdown |
| `StatefulAgentLoop(runtime=...)` | Convenience client facade; `send(target, body, message_id=..., actor=..., deadline_ms=...)` returns `Result` |
| `RuntimeFactory`, `RegistryRuntimeFactory` | Declare required profiles and startup checks; rebuild the current execution environment on each node |
| `ExecutionEnvironment` | Model, registry, hooks, RuntimeEnv, workflow file IO, process control, stable resource reference and scope |
| `TimerBackend`, `TimerContext`, `timer_context()` | One timer tool with a configured backend and trusted input/call identity |
| `TimerDriver`, `TimerDriverStatus`, `TimerRunner` | Runtime-owned start/stop/status; bounded indexed scans and RuntimeClient acceptance |
| `StoreTimerBackend` | Default SDK timer table, no Redis; custom host backend requires an explicit driver |
| `StoreBlackboard` | Database-atomic scoped board; first creation, update, remove and delete work across independent connections |

`Runtime.status()` reads in-memory driver/worker state; it does not issue a health query for each request. Pure clients do not start timers or acquire a process coordination lock. A worker using default memory coordination takes a native store lock; shared mode requires explicit shared adapters.

### Finite child wait

`WaitPolicy(deadline_at_ms, max_wait_ms=1800000)` and
`register_wait_handoff_tool(registry, policy=...)` are public. Register the tool
in the original main and child profiles. A child requests a timed review; the
main inspects the original continuation and explicitly continues, defers or
cancels the same JobRef. This uses RuntimeClient answers/cancellation and the
configured TimerBackend. The factory supplies the trusted deadline. See
[runtime contract](../../runtime-11.md#finite-child-review) for assembly and
failure behavior.
