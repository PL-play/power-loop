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

## Timers, readiness and resource identity

Runtime starts the default StoreTimerBackend/TimerRunner lifecycle automatically.
The driver only submits through RuntimeClient.append; it does not hold model work.
Use `timers_enabled=False` for deployments without timers. A host backend must
supply a ready driver. See `examples/26_timers.py` for an executable default.

Declare every deployment-required profile in `required_profiles`. Register static
ports using `factory.register_profile("workspace@1", preflight=check_workspace)`;
checks are async and run at startup, before accepting, never per request. A missing
profile/port is `capability_unavailable`; unavailable storage is
`resource_unavailable`. Preserve these RuntimeRequestError codes in custom checks.

Factories return explicit `runtime_env`, `workflow_file_io` and `process_control`.
For a shared workspace they also return its stable `resource_profile_ref` and
`resource_scope="shared"`. Matching path strings are insufficient: deployment
configuration maps the reference to the actual shared storage, and deployment
probes must verify shared contents. A resource's per-task credentials and target
are resolved by the execution node. Process control binds the original shell
target once; failures remain unconfirmed.

Default memory coordination acquires a native startup lock on the SQLite database
file or server store prefix. A second independent process/domain is rejected.
Workers explicitly sharing one memory coordinator and transport share a reference
counted lock. Last exit releases it; connection loss stops admission/execution.
RuntimeClient alone does not acquire this lock. Shared execution still requires
explicit shared adapters.

DeepTalk shared nodes additionally configure `RUNTIME_WORKSPACE_REF` and
`RUNTIME_SKILLS_REF`, with matching logical identities across nodes and accessible
workspace/skills roots. Sandbox mode requires the executor and host mount mapping.
Dispatcher publishes timer readiness through its existing Redis connection; its
absence prevents startup. No configuration change here deploys or enables it.

A factory can supply `ExecutionEnvironment.event_bus` to observe the current
execution's model and tool events. The worker removes its own streaming
subscriptions when the execution ends and filters shared-bus deltas by session.
The host owns the bus and its subscribers; the worker drains pending async
subscribers up to `Runtime.hook_timeout_s` before closing execution resources.
This port adds no database queries.

## Finite child review

Register `register_wait_handoff_tool(registry, policy=WaitPolicy(deadline_at_ms))`
in each worker factory that supports finite child waits. The trusted policy may
also be a callable of `ExecutionContext`; model arguments cannot set or extend
the deadline. Configure the same TimerBackend/driver as `schedule_wakeup`.
The original main must expose `wait_handoff` to inspect and decide.

A subagent or workflow leaf requests `action="request"`, `due_at_ms`, `note` and
`continuation`. This persists the original tool interaction and releases worker
capacity. Its timer wakes the original main, which supplies the original
`job_id`, `generation` and `event_id` to `action="decide"`: `continue` answers the
same interaction, `cancel` cancels that task, and `defer` schedules
`next_review_at_ms` before the unchanged deadline. Inspection pages the full
note/continuation with `field`, `char_offset`, and `max_chars`. Old notices cannot
authorize a later review. Runtime shutdown releases local deadline tasks; reopen
reinstates the deadlines but does not repeat an unknown external timer creation.
A failed schedule remains visible as `schedule_error`; an explicit new decision
may continue or cancel the original task. No required timer job or recovery
proof ledger is created.

## Original managed outcome notifications

Hosts publish a known original outcome with `managed_delivery.enqueue_runtime`
using the original task runtime and authenticated main actor. Supply `client=`
to signal accepted work immediately. The notification reuses the original
runtime descriptor and request options; it does not reset model budgets or
execute the external action again. Unknown outcomes are queried through the
original ManagedTool adapter or remain unknown. Several notifications accepted
before an idle worker starts can be consumed in the same model round. Queue
acceptance alone does not acknowledge the outcome: a successful execution must
actually see it. Original suppressed/not-dispatched results remain suppressed.
