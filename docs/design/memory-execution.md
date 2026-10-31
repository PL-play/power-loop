# Memory-owned execution and recoverable checkpoints

Status: implementation in progress. This is the single task-execution contract;
there is no legacy execution switch.

## Contract

An execution has one owner. It loads its immutable identity and current control
state at admission, then advances model/tool attempts in memory. Intermediate
inspection state is eventually consistent. Only bounded new facts are appended
at checkpoints; a checkpoint never rewrites completed history.

Control records are not a financial ledger. Actual new usage must remain
accurate, idempotent and reliably reported. Existing response capture records
contain cost/usage and response identifiers, **not** model response bodies.
Recovery must not invent a lost response or automatically repeat an unknown
external effect. Valid originals can rebuild derived state; missing/corrupt
originals require explicit reconciliation or termination of recovery.

Cancellation has requested and confirmed states. A local owner observes signals
immediately; renewal/checkpoint validates durable revocation and ownership.
Expired or superseded owners cannot commit new admission. Result/usage evidence
may still be accepted for an already admitted original attempt after cancellation.

Shared limits remain authoritative. A bounded allocation can be consumed locally;
unknown consumption is retained across restart, and unused capacity is returned
only after confirmation. Independent child jobs retain their own ownership and
lifecycle. There is no global lock for unrelated execution checkpoints.

## Ownership and transaction boundaries

A live send owns model/tool attempts in memory. Admission joins registration,
claim and control opening in one transaction. Closing joins the final checkpoint,
model exit and eligible finalization in one transaction. The existing global
control lock still serializes admission/close; incremental execution checkpoints
have their own owner/version fence.

`TRUST_OWN_WRITES` is scoped to a live owned send. Transaction-local immutable
facts are staged in an overlay; commit publishes them to process caches and
rollback discards them. Checkpoint positions also advance only after commit.
Cold inspection and recovery verify original executor, membership and event
proofs. Recovery entry points force stored verification even inside a send.
Cancellation, leases and other writers' mutable control state remain fenced.

An append publishes its sequence and pending caches only after the transaction
context exits successfully. A failed or unknown commit invalidates the sequence
hint and retains a deferred round index for retry.

## Tool result and receipt ordering

Within an owned send, the frozen return records the task operations associated
with each `(assistant_seq, call_id)` in `_control_tool_operations`.

- A known empty list has no task receipt obligation. Its tool-result append and
  reduction of pending calls commit atomically in the same storage transaction.
- A nonempty list has a task operation. The result append preserves pending;
  only successful receipt binding and its required settlement permit reducing
  or clearing pending in a subsequent transaction.
- An unknown operation list is conservative: receipt lookup must finish before
  pending can be cleared.

This ordering does not depend on exception compensation. Process death after
result persistence, receipt-binding failure, or a failed pending-clear write
leaves a durable recovery barrier. Death after binding but before clearing also
leaves a conservative barrier. The real tool result is retained throughout.
Cold recovery may bind that original result idempotently; it cannot redispatch a
non-idempotent callback merely because the original send ended unexpectedly.
Pending interactions belonging to still-unresolved calls survive reductions.

The single-writer session lease is a precondition for sequence/pending hints.
Recovery uses fresh durable state rather than the dead writer's memory caches.

## Session obligation invalidation

`session_controls` is an invalidation counter, not an authority ledger. A send
may reuse a clean-session discovery result only when its version is unchanged.
Relevant event insertions collect affected sessions in `RecordingTransaction`;
their versions advance before the enclosing transaction commits.

A notification can be inserted under a child job and routed later. Every such
writer calls `RecordingTransaction.set_event_recipient(event_id, session_id)`.
The method writes the route and explicitly records its recipient for version
invalidation in that same transaction. Correctness does not depend on matching
the SQL prefix. Rollback discards both the route and the version advancement.
A source-level test covers all seven current routing calls in six modules and
rejects direct recipient UPDATEs outside this method.

## Incremental evidence and storage limits

The hash-linked execution journal remains the control authority. Inspection
projections retain stable references and may be rebuilt from valid original
records. Financial outboxes and actual usage have their own exact, idempotent
reporting obligations; a control cache never substitutes for them.

Task-event identifiers fit the common `VARCHAR(64)` storage contract. Completion
window IDs use `cwin_` plus a 48-character digest on every database. Managed
effect IDs bound their digest length by the event-type prefix so resource-exit
IDs also fit 64 characters. Writers and readers share each generator. No runtime fallback reads the previous longer
identifier format. Changing retained development data is an explicit maintenance
operation outside the live execution path.

## Validation requirements

Fault tests must cross actual commit boundaries, including lost acknowledgements
and process death. Transaction proxies delegate the complete transaction port
while injecting their faults. Observers inside joined admission/close reuse the
current transaction; they do not open another SQLite read/write transaction.

Growth measurements distinguish ordinary message storage, control and billing.
Server validation uses isolated databases and must enter workflow/model assertions
rather than count initialization failures as exercised behavior. Real process
exit tests reopen storage without owner caches and assert that results/pending
survive and neither model calls nor external effects are repeated.
