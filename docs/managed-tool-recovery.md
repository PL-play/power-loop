# Managed finite tool recovery (development)

This prototype is not connected to the default native host. The candidate task
protocol is 9; production activation, package publication and migrations are
separate gates. The supported resource is one finite callback with no detached
work or untracked paid requests. Other resource models need their actual
lifecycle interfaces and tests.

## Original execution and writes

`ManagedTool` implementations provide preparation, execution and query-only
recovery. The SDK binds the actual instance and schema through `RuntimeTool`;
tool names do not grant admission. The SDK handler declares
`background_allowed=False` and does not allow another background wrapper.

```mermaid
flowchart LR
  A[Original raw tool call] --> B[Required child and original lease]
  B --> C[Prepare and dispatch evidence]
  C --> D[Actual callback exits]
  D --> E[Retain exact local completion]
  E --> F[Transaction 1: exit checkpoint and anchor]
  F --> G[Transaction 2: canonical result and publication trace]
  G --> H[Original tool response and later supervision closure]
```

The completion contains the original execution lease, callback exit kind,
whether dispatch started, and the exact requested result. Exit evidence v2 and
its independent anchor are committed atomically. The anchor binds the original
creation, execution scope and completion hash. The original result is then
published in a second transaction, retaining a committed exit checkpoint if the
result write fails.

The SDK retains actual unsaved completions in a bounded, runtime-bound cache.
At most 1000 active or completed-but-unsaved entries are admitted; exhaustion
rejects a new call before registration, preparation or dispatch. It never drops
an unresolved entry to make room. Each completion document has the same 64 KiB
bound as the persisted checkpoint. An ordinary known result leaves no retained
entry after successful publication.

## Repair ports

- `flush_local(runtime, actor, job_id)` retries the exact SDK-held completion.
  It requires the same actual runtime authority and original execution. A new
  runtime cannot create a completion from an absent handle or remote lookup.
- `restore_result(manager, actor, job_id)` requires no runtime, provider or
  effect callback. In one transaction it validates the original call, creation,
  scope, lease, preparation, dispatch, exit and independent anchor, then publishes
  the original result and its trace. It does not claim a new generation.

Both public ports require trusted host context, outside a model/child execution
scope. Authorization is checked before reading the selected original job.
Concurrent repairs and commit-ACK loss recall the same result; conflicting,
missing, oversized, corrupt or wrong-generation evidence fails without guessing.

A successful response identifies the original job, generation and result event.
`owned_callback_exited=True` refers only to this supported finite callback.
`execution_authority`, `billing_settled`, `acknowledgement_created` and
`tool_receipt_written` remain false. Repair does not write or rewrite transcript
rows, dispatch an effect, call a model, grant a lease, acknowledge the outcome,
or finish its owner.

`ResultPublicationProof` is a generic, trusted SDK interface for publishing an
already-exited execution's result after its boot/lease expires. The managed
implementation rechecks the exact stored checkpoint and requested result in the
same transaction. It does not relax ordinary executor admission or ordinary
`record_execution_result` behavior. It cannot authorize new work, downstream
completion or billing settlement.

## Cancellation and shutdown

If cancellation precedes result publication, canonical result handling retains
the requested result while recording the effective cancelled outcome. A known
business result is preserved; cancellation cannot erase or resend the effect.

`TaskRuntime.shutdown` and `executor_status` both report
`managed_completion_pending`. Any retained unsaved completion forces
`store_safe_to_close=False`, including after a later status query. Once a repair
in another process succeeds, the original runtime still confirms its own exact
copy with `flush_local`; absence of activity alone is not confirmation.

## History and remaining obligations

Legacy exit evidence v1 remains readable without being rewritten. It has no
exact result checkpoint, so `restore_result` refuses to fabricate one. Protocol
upgrade requires explicit maintenance and retains the original store/message
identities. Old writers are rejected; opening a compatible store is not a
production migration policy.

If the process dies before any exit checkpoint is committed, its local cache is
lost. Remote business success or an expired lease does not prove the owned
callback's exit. That case still requires a trusted physical-stop/recovery
adapter. Claim-ACK loss before callback entry has the same need for independent
negative or physical evidence.

The original error tool row also does not acknowledge a later recovered success.
Repair leaves the result pending delivery unless an existing valid original ACK
already exists. Canonical processing responsibility, cancellation/inspection
integration and formal business-host wiring remain separate unfinished parts of
the overall async lifecycle work. Default native entry stays disabled.
