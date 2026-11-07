# Power Loop 11 runtime

Source delivery: SDK 11.0.0, task schema 15, protocol 42, descriptor schema 1.
This branch is not a published package. The authoritative stable exports are
`power_loop.STABLE_API`; public signatures are in `agent/runtime*.py` and the
`StatefulAgentLoop` facade at the end of `agent/stateful_loop.py`.

## Start locally

Run `python examples/design203_single_process.py` from an empty test directory
using an interpreter with this checkout installed. It creates a new SQLite file,
starts a worker, submits input by ID, waits for its persisted result, and closes
all owned resources. No Redis, broker or model credential is required by the demo.
Replace `FixedModel` in the registered factory with your provider for real requests.

1. `await SessionStore.open(path)` opens or creates the current structure.
2. Register a versioned builder with `RegistryRuntimeFactory.register(name, build)`.
   The builder returns `ExecutionEnvironment(llm, tool_registry, config, ...)`.
3. Construct `Runtime(store=..., factory=..., node_id=...)` and start its worker.
4. Create a `TrustedActor` from authenticated host identity, then call
   `runtime.open_session(descriptor, actor=actor)` to obtain a `TargetRef`.
5. `runtime.client.send(target, body, message_id=..., actor=..., deadline_ms=...)`
   returns a receipt. Acceptance is not completion. Wait on `receipt.input_ref`.
6. `await runtime.aclose(timeout_s=...)` drains and closes the owned store/transport.

A retry uses the same message ID, target and frozen request options. Host identity
is not supplied by model-generated tool arguments. `TargetRef` includes store ID;
`InputRef` identifies the input; `JobRef` identifies an execution and optionally a
generation. Store/session/job/generation identities are not interchangeable.

## Current operations

| Port | Behavior |
|---|---|
| `RuntimeClient.send` | Start a new turn with explicit actor and request ID |
| `append` | Add input to a session, preserving the requested input semantics |
| `inspect`, `wait` | Query original identity and generation; wait deadlines are explicit |
| `subscribe_output` | Live output; a gap is not a replay guarantee |
| `cancel`, `steer`, `nudge` | Address original tasks and sessions without a local loop handle |
| `answer`, `resume` | Continue the matching interaction or original execution |
| `StatefulAgentLoop(runtime=...)` | Convenience facade; no direct LLM/DSN constructor |

Consult the source signatures for required arguments. Model/tool configuration,
representations, folds, hooks and resources are resolved by the factory. Per-input
`RequestOptions` are frozen at acceptance. Register the same versioned definitions
on all workers; an unavailable definition fails explicitly.

## Shared execution

Use `shared_execution=True`, a shared `SessionOwnershipCoordinator`, and a
`MessageTransport` whose capabilities declare shared scope. Process-only adapters
are rejected. The default memory composition supports one process only; changing
the database DSN alone does not make it distributed.

DeepTalk provides real Redis ownership and Kafka transport in
`agent/app/runtime_coordination.py` and `agent/app/runtime_transport.py`.
`agent/examples/design203_shared.py` is an executable two-process example: the
publisher only runs `RuntimeClient`; models execute in the separate worker.
All participants share a store ID and deployment namespace, while node IDs differ
and boot IDs are generated on each startup. Use `RuntimeNode` for the full host's
registration, health, factory and lifecycle wiring.

## Fresh store boundary

Use a new database or table prefix. The store creates its new identity; configure
all hosts with that exact ID. Current stores reopen; any other schema/protocol
version is rejected, never upgraded, translated or read through a compatibility
mode. Historical tasks and SDK bindings are not imported. Business users, ordinary
chat, installations and shared workspaces are not SDK migration inputs.

Normal authorization, accounting, tool-call pairing and resource cleanup remain
required. Failures can leave interrupted/unknown results; unknown external actions
are not automatically replayed. This is not exact recovery of every previous step.
