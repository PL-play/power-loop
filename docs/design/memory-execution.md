# Memory-owned execution and root-scoped control

This is the single execution contract. Runtime execution, recovery and inspection
use the same final format: ordinary schema 12, task schema 14, task protocol 41.
The only supported offline upgrade source is deployed task schema 10 / protocol 21.
No intermediate development protocol has a runtime or maintenance bridge.

## Ownership and authority

A live execution owns model and tool attempts in memory. Messages, original task
events and supplier captures retain their respective original evidence. Sequence
counters, pending-only indexes, views, invalidation versions and inspection
projections are derived state; none independently grants execution or billing
permission. Recovery verifies originals and can rebuild valid projections. Missing
or conflicting originals stop recovery rather than authorize another external action.

Actual new usage stays accurate, idempotent and reportable after a restart.
Control checkpoints are not a replacement financial ledger. Supplier captures
retain response identifiers and usage, not a recoverable model-response body.
An unknown external effect or lost model response is never repeated to invent
missing evidence. Cancellation distinguishes a request from confirmed resource
exit; an already admitted attempt can still publish its original result and usage.

## One root per control writer

`ControlTransaction` acquires at most one root job row. The target-to-root lookup
also proves immutable membership; full child state is fetched only when needed.
PostgreSQL/MySQL lock that root with `FOR UPDATE`; SQLite uses its writer
transaction. Unrelated root trees do not share a task-store control lock.
`task_store` identity/protocol verification happens at startup, not each control
transaction. Metadata transactions that do not publish job facts need no root.

Root locks remain necessary for concurrent child completion and stop operations
within one process. Session ownership excludes other containers; it does not
serialize those local control operations. Ordinary message/state writes use
the active ownership handle and `next_seq` condition, without ancestor joins.
A miss raises an ownership error and never falls back to an unfenced append.

Cross-tree operations publish one tree at a time. Discovery uses a coherent read
snapshot; immutable source evidence and the original permission vector travel to
a separately fenced target publication, followed by source settlement when
required. Retrying an interrupted step uses original deterministic identities.
Executor drain and context shutdown first close their existing admission gates,
then settle each tree. No writer acquires a second root.

Root locks precede message-state/checkpoint locks. Admission joins registration,
claim and control opening; close joins its final checkpoint, original model exit
and eligible supervision finalization. An unfinished child permits model exit
with supervision still open. Lost commit acknowledgements retain the exact
original closing evidence for reconciliation.

## Explicit write intent and pending discovery

Business writers use explicit ports such as `insert_job`, `write_job`,
`append_event`, `update_event`, `set_event_recipient` and `handled`. The
transaction records affected jobs, sessions and obligations. Before commit it
refreshes pending entries. After commit, bounded process-local cache versions
advance for the affected recipients, shared by managers of the same store.
The database session row remains only a registration/cancellation lock. Neither
INSERT parsing nor SQL-prefix matching determines those effects.

`session_obligations` contains only outstanding work, keyed by
`(session_id, kind, source_id)`. Discovery joins it to original jobs/events for
results, abort reviews, managed effects, controller reviews, wait notifications
and queued inputs. Completion removes the derived entry; historical completed
work does not enter the hot discovery scan. Cold verification compares the index
with validated originals; explicit rebuilding rejects missing originals.
The control test fixtures compare pending projections with originals at teardown.

A handling decision alone does not erase a review barrier. The original handler's
clean model close must commit as well. Negative generation-zero outcomes remain
visible until the original main reviews them. A consumed input retains its actual
processing proof after a later owner failure. Notification route and version
advancement commit together. Concurrent receipt lookup reads both mutable
delivery flags in one statement instead of mixing a cached flag with a new row.

## Session turns and input attribution

A `SessionHolding` owns a turn's entry snapshots, sequence hint, pending calls,
round index, change signals, activity timestamp and committed assistant reference.
Commit publishes its hints; rollback or an unknown acknowledgement invalidates
mutable hints and retains the original deferred round. Exiting the turn retires
the holding, including contexts inherited by background tasks. A delayed old
acknowledgement cannot publish into a newly acquired turn.

`session_state` contains inbox, background and runtime version counters.
Producers advance the corresponding counter with their payload. Ordinary append
returns those counters; a live round reads detailed state only after a change.
Final completion performs a fresh signal check. Public getters and stored
verification read durable state rather than reuse a live turn's snapshot.

Session ownership uses `(store_id, session_id)` and an application-provided
`SessionOwnershipCoordinator`, with memory as the single-process default.
Shared adapters pass `power_loop.testing.SessionOwnershipContractTests`.
Inputs are accepted without owning the target; the unified worker acquires
before executing. Each independent session has its own grant and context.
Only the session grant is renewed; jobs retain generation/version attribution,
logical parentage, explicit cancellation and cumulative budgets.

Expected owner epoch is read before acquire and installed conditionally.
Existing message writes combine owner epoch/token with next_seq protection;
result publication uses its existing state/job transaction. Locks are followed
by local deadline checks. Loss stops new dispatch; takeover interrupts only the
acquired session's unfinished execution and repairs pending message format.
The original main send receives an interrupted result at its original generation.
See [the scaling guide](../en/user-guide/scaling.md) for context and hook APIs.

## Message usage and crash boundaries

A model response carries private per-round usage only on its storage copy.
The first persisted response message embeds `_round_usage` in its existing
INSERT. Decoded message metadata, hooks, history and future model requests do
not expose that private field. `usage_rounds` is a derived view combining these
message originals with `usage_observations` for imported/empty-response usage.
The key remains `(session_id, send_index, round_index)`; repeated round-end
callbacks do not add another charge. Financial capture/reporting retains its
own exact evidence and idempotent outbox.

Current message allocation and INSERT share one transaction. Process death after
allocation but before the message rolls both back. There is no multi-message slot
reservation in this release. Readers tolerate committed unused sequence gaps.
The ordinary per-tool slope stops at four statements on SQLite/PostgreSQL and
six on MySQL: reaching three would require changing slot allocation, pending
reduction and recovery together, outside the agreed completion scope.

Three-engine process tests cover allocation-before-INSERT
(`test_usage_append_atomic.py`), death during the actual tool-result INSERT
(`test_tool_receipt_crash.py`), and old appenders after ownership takeover (`test_session_ownership_processes.py`,
`test_owned_lease_writes.py`).
Rollback and lost-acknowledgement tests keep exact original message/usage checks.

## Tool results and the receipt barrier

The frozen return maps each `(assistant_seq, call_id)` to its task operations in
`_control_tool_operations`.

- A known empty list needs no task receipt; result append and pending reduction
  commit in the same ordinary storage transaction.
- A nonempty list preserves pending when the result is appended. Only successful
  receipt binding and required settlement allow a later pending reduction.
- An unknown list requires receipt discovery before clearing the barrier.

This ordering survives process death between append and binding without
exception-compensation writes. Death after binding but before clear retains a
conservative barrier. Cold recovery can bind the real original result
idempotently and cannot redispatch a non-idempotent callback. Still-unresolved
calls retain their pending entries.

## Checkpoints, budget and renewal

Bounded grants can be consumed in memory. Open checkpoints append only new
hash-linked facts, not completed history. A close applies the cumulative
difference under the original budget scope locks. Multiple staged batches retain
ordered hashes and publish their positions only after commit. Unknown
consumption and reserved capacity survive retirement until original evidence
permits settlement. Actual usage beyond the grant books immediately and stops
new admission.

An unbudgeted send uses the existing admission policy gate and reads no budget
tables. A concurrent first policy activation and unbudgeted admission serialize
on that gate. Paid sends retain exact capture and re-reporting behavior.

A live task renewal uses one root lock plus one conditional UPDATE. The UPDATE
checks generation, executor/boot, original ancestry, cancellation, lease expiry,
completion deadline and executor admission. SQLite/PostgreSQL return the flag
with RETURNING; MySQL uses its matched-row count and OK-packet value, without an
extra confirmation SELECT. A child reuses the membership proven by root lookup.
Cancellation rolls back renewal rather than extend the lease.

Execution renewal runs at TTL/3, independently of the checkpoint interval
(default 30-second lease: every 10 seconds instead of every 5 seconds).
Finite leases require continuing renewals and executor heartbeats during long
sends. SQL is not independent of elapsed time. Control lock timing adds no SQL
and records acquisition wait and holding duration, including outcome.

## Finalization retention

Finalization deletes the job's execution checkpoints, execution allocations,
budget attempts, and `execution_scope`/`execution_membership_seal` events in the
same transaction that closes supervision. The allocation table is indexed by
`job_id`. Cleanup uses four fixed DELETE statements for the planned closures;
control SQL limits are 18 unbudgeted / 25 budgeted (approved for D1), with the
same 2/3 control transactions and 0/12 billing statements.

Closing still requires stopped work, acknowledged accounting and completed
children. A failed commit rolls back cleanup with the closure; unsettled fees
cannot pass the gate. Durable policy/scope balances and model usage remain.

The existing `supervision_closed` record retains the execution identity and
owner/session information needed by historical usage, parent completion and
workflow restore. Closed generations read that record; live generations use
their active scope. Retained results no longer reference deleted scope/seal
proofs. Controller exits, review handoffs and wait completion use the retained
identity/results, without preserving a duplicate chain of scope/seal events.

## Host discovery and executor heartbeat

`task_inspection.list_open_roots(manager, after_job_id='', limit=100)` pages only
roots with open supervision. `model_budgets.list_unsettled(manager,
after_attempt_id='', limit=100)` pages pending fees and reservations/dispatches
that have not reached an acknowledged terminal phase. Both are trusted host
queries across the store, return `next_cursor`, and accept at most 1,000 rows.
They do not grant task or billing authority; the existing actor checks still
apply when inspecting or reconciling each returned task. The fee query returns
identities and routing metadata, without loading settled accounting payloads.

Executor activation reads only `starting`/`accepting` boots. `draining` boots
already have admission closed; prior running/unknown jobs are checked separately
with pending-work indexes. Heartbeat updates only `task_executors.lease_until_ms`
and `updated_at`, using the current admission version and a live-lease condition.
It adds no event and does not advance that version. `heartbeat` no longer takes
`system_key`; a failed heartbeat stops local supervision instead of recalling a
historical heartbeat receipt. State changes retain their existing events.
The one-minute live-heartbeat and 10,000-row history tests run on all three stores.

## Offline protocol 21 to 41 upgrade

`scripts/upgrade_protocol21.py` is the single maintenance entry. It accepts
schema 10/protocol 21 and an idempotent rerun on final schema 14/protocol 41.
Ordinary opening refuses old formats and performs no conversion.

Before any business application: record the installed package/import/source
identity, stop all writers, take a native backup, and apply the exact upgrade to
an isolated restored database ending in `_test`. The current rehearsals use the
actual deployed protocol-21 DDL with representative originals, not a business
database snapshot. Business upgrade/restart/deployment requires separate user
authorization and was not executed here.

From the SDK checkout, dry-run is the default:

```bash
.venv/bin/python -m scripts.upgrade_protocol21 --dsn "$POWER_LOOP_MAINTENANCE_DSN"
```

It reports source structure/version/counts; it does not prove every original can
be converted. With writers stopped and backup/rehearsal complete, the explicit
application command is:

```bash
.venv/bin/python -m scripts.upgrade_protocol21 --dsn "$POWER_LOOP_MAINTENANCE_DSN" --apply --writers-stopped
```

The migration converts original model-operation lookup keys and workflow storage
keys, bounds old event IDs, updates structured references and derivable hashes,
imports existing accounting/context evidence, and rebuilds admission and
pending-only indexes. Original messages, message coordinates, operation IDs and
historical fee amounts remain unchanged. New version stamps are written last.
Final event IDs fit the common VARCHAR(64) contract; runtime readers compute one
key and never try an old alias.

Unattributable old system keys, missing original caller permission, conflicting
originals or opaque changed fingerprints stop migration. Resolve their attribution
offline in a backup clone rather than invent authority. Deployed MySQL cannot
store the old 65/66-character IDs; its rehearsal covers representable protocol-21
data, while SQLite/PostgreSQL exercise those long originals.

MySQL DDL can commit independently. After a failed upgrade, keep writers stopped
and restore the native backup or finish that same migration. Changing only the
version stamp does not restore the old format. No historical fees are repriced
and no provider, executor or external action runs during migration.

## Host-invoked retention

`await store.purge_expired(now_ms, limit=1000, retention_days={...})` performs one
maintenance pass. `now_ms` is UTC epoch milliseconds. Each delete statement
selects at most `limit` indexed keys, and `limit` must be between 1 and 1,000.
The return value contains actual deletion counts by table, plus `task_trees`
and distinct `usage_rounds`. Large trees are removed in bounded pages within
their existing root-row transaction. Concurrent callers can repeat the pass.

Default retention, in days:

| Parameter | Days | Eligibility |
| --- | --- | --- |
| `task_trees` | 30 | Every task closed/stopped; no obligations, pending alerts or unconfirmed fees |
| `executors` | 1 | Draining or expired, with no current job reference |
| `follow_ups` | 7 | Ordinary delivered, void or transferred inputs; claimed/pending items remain |
| `timers` | 7 | Fired or cancelled |
| `hooks` | 30 | Record age |
| `usage` | 90 | Observation age, including details embedded in response metadata |
| `runtime` | 30 | Owning root closed; both root and runtime record past the window |

Applications can override individual parameters. Voided/transferred inbox rows
use their creation date; delivered rows must also have an old delivery date.
Task-associated notifications remain until their task tree is removed, because
normal completion and workflow continuation still read them after delivery.
The existing event ID, `nudge:<operation_id>` and `subagent:<run_id>` inbox keys
identify the association. No new table or event is used. This retention change
was explicitly approved on 2026-11-05.
There are no retained plan/callback events after A2, so no separate seven-day
cleanup is needed for those event types. The SDK adds no background task.

The existing timer/usage prune methods accept `now` and `limit`; a `None`
session selects across sessions. Message bodies and cumulative session usage
remain. Runtime cleanup removes task-owned subagent/workflow keys and removes
the run from the shared workflow index. A recent runtime key retains its task
tree until both retention windows end. Keys without a task owner are removed
with their session.

`close_session_tree` also removes roots owned by each deleted session and their
task descendants. Deleting an ephemeral child session keeps the jobs belonging
to its still-existing parent root, so parent completion can read them.

Budget manifests and scope balances live solely in `task_budget_policies` and
`task_budget_scopes`. Policy preparation, admission reconstruction and fee
reconciliation read those rows. The SDK no longer writes policy/scope events
under the first task; tree retention and session deletion preserve these rows.
The first-policy admission lock still prevents an unbudgeted send from crossing
mandatory policy activation. New indexes apply to fresh schemas; the D3 offline
upgrade and version change remain a separate task.
That upgrade must also adjust executor-admission reconstruction: the current
offline rebuilder requires historical registration events, which this retention
pass can remove. Normal runtime registration keeps the existing admission gate;
do not use the old offline reconstruction on a purged store before D3.

## SQL and validation contract

`tests/unit/test_send_sql_profile.py` is the hard SQL authority. It covers three
engines, paid/unpaid sends, 1–32 model responses and 0 versus 20 historical sends.
Let T be tool rounds (model responses minus one), and L be elapsed whole seconds.

| Category | Short-send bound |
| --- | --- |
| Ordinary SQLite/PostgreSQL | 15 + 4T + L |
| Ordinary MySQL | 17 + 6T + L |
| Billing, unpaid / paid | 0 / 12 |
| Control, unpaid | 18 |
| Control, paid | 25 |
| Control transactions, unpaid / paid | 2 / 3 |

Activity touching contributes at most L ordinary statements. Long-lived renewal
costs are additional and explicitly measured; these short-send bounds are never
raised to absorb them. Control/billing SQL does not grow with completed session
history. The MySQL ordinary path retains its locking read.

Fault validation crosses real transaction/process boundaries: rollback,
acknowledgement loss, unknown external effects, lease takeover, ancestor
revocation and concurrent stop/result publication. Test proxies inject faults
through the current explicit transaction ports. Damage tests restore only their
deliberately removed originals after the rejection assertions so teardown can
verify the ordinary projection invariant.

Final acceptance runs the whole test collection with SQLite, isolated PostgreSQL
and isolated MySQL configured; external-provider tests are explicitly skipped.
Failure fixes receive only affected-case reruns before the last complete run.
The ops result records exact source commit, engine counts and the one final
eight-session PostgreSQL benchmark. SDK source changes are not deployment or
enablement; local commits have not been pushed or released.
