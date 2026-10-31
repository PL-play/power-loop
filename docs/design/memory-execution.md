# Memory-owned execution and recoverable checkpoints

This is the single task-execution contract. Execution, recovery and inspection
share its current schema and protocol.

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

An eligible root close stages its model-exit row in the current transaction's
read view. Successful finalization publishes model exit and supervision closure
in one conditional row update. An unfinished child permits committing the model
exit with supervision still open. Both paths retain their original events and
versions; rollback or an unknown commit retries the original closing evidence.

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

## Entry reads and round change signals

Each send has a bounded lifetime for its entry session/state/runtime snapshots.
Committed appends and mutations refresh or invalidate those snapshots. Failed or
unknown commits invalidate mutable hints. An inherited background context cannot
reuse an expired read scope. Public getters and `@verify_stored` recovery always
read storage. The final assistant reference follows the same commit boundary;
closing reuses that reference, while recovery reads the actual stored tail.

`session_state` holds `inbox_version`, `background_version` and
`runtime_version`. Every producer advances the appropriate counter in the same
transaction as its payload. A normal append returns these counters with its
existing sequence/identity result. The live round reads detailed queues and
runtime projections only after a committed counter change. Claimed-input retry
deadlines permit reclaiming work from a failed consumer without a new producer
write.

Natural completion and `pass_turn` perform a fresh final signal check. Input or
background results committed during the last model round remain discoverable;
explicit hard-stop semantics still apply to `pass_turn`.

The current base store schema is 11; task schema is 12 with task protocol 23.
Explicit maintenance adds and backfills the counters, including partially
completed DDL. Old task stores require maintenance before current writers open
them. Runtime execution reads only the current format.

## Shared budget checkpoints and round accounting

An open execution retains its full bounded grant in shared scope counters.
Actual consumption is included in the execution checkpoint summary and records;
ordinary open checkpoints do not rewrite shared counters. Close applies the
cumulative difference under the original scope locks. Report versions make
unknown-commit retries and repeated projection repair idempotent. Actual usage
above the entire grant is booked immediately and closes new admission.

Expired-owner retirement preserves its unknown suffix and reserved capacity.
Actual supplier capture and financial submissions remain authoritative for paid
usage, including a response received after the last control checkpoint. Recovery
requires original evidence before correcting or releasing that capacity.

The pipeline passes per-response usage only on the message copy sent to the
sink. Hooks, history and later model requests do not receive that private field.
The first response message writes `usage_rounds` in its own append transaction,
using `(session_id, send_index, round_index)` as the idempotency key. A committed
append marks the round locally so repeated round-end callbacks do no further
accounting SQL. Empty response rounds persist usage immediately at round end.

If the append transaction rolls back, both the message and its local usage row
roll back. Supplier capture and financial reporting retain their separate
originals; recovery preserves unknown grants and never repeats a model call to
manufacture a lost response. The usage INSERT count is unchanged; joining the
append removes one transaction per persisted response.

## Cross-process session turns

When processes share a session, `distributed_sessions` acquires its durable turn
lease. Owned message allocation, pending clears and explicit round writes check
the lease holder and validity in their existing conditional UPDATEs. PostgreSQL
and MySQL lock the matching lease row in that statement; SQLite serializes the
writer transaction. This adds a lease-row read/lock without a separate SQL
round trip. A stale owner raises `SessionLeaseLost` (`session.lease_lost`).

The write fence remains attached to a detached child context after its original
turn ends. A newly acquired lease discards mutable hints from the previous
holding period. Writers check that lifetime before touching hints and again
after waiting for a transaction. A delayed successful commit ACK returns its
original durable sequence without publishing mutable hints into a new turn.
An old failed/unknown commit also cannot restore its deferred round over that
turn. These checks cover reacquisition by the same loop owner ID.

Renewal uses the database port's integer matched-row count;
MySQL already enables `CLIENT.FOUND_ROWS`, including unchanged timestamps.
There is no follow-up confirmation SELECT.

## Remaining control costs

Admission and close retain the global `task_store` lock. Executor drain and
root-control cancellation can touch multiple roots, so replacing it requires a
proven common lock order across those transactions. Tests cover real-process
session contention, independent concurrent models and same-tree stop/result
races with the existing lock.

`session_controls` skips unchanged clean-session discovery. A changed or cold
session can still scan processed historical obligations; a durable pending-only
index has not been implemented. Unified task renewal also retains ancestor,
deadline and executor-admission checks. Finite durable leases require ongoing
renewal writes; total control SQL is not independent of elapsed time.

`tests/unit/test_send_sql_profile.py` is the SQL regression authority. It covers
three engines, budgeted/unbudgeted sends with 1–32 model responses, and 0 versus
20 historical sends. Its current fixed control bounds are 19 unbudgeted and 29
budgeted, with bounded projection/checkpoint growth. Ordinary storage has fixed
bounds 16/18 and tool-round slopes 5/7 for SQLite/PostgreSQL versus MySQL, plus
the existing elapsed-second activity-touch allowance. Billing bounds remain
2/13. These limits are never increased to accommodate a regression.
