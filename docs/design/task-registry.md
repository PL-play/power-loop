# Experimental task registry: B1 storage and identity ports

Implementation: `runtime/jobs.py`, `runtime/message_ref.py`, `runtime/store/task_schema.py`.
This is an unreleased development interface, not part of `STABLE_API`.

## Activation and compatibility

Default `AUTO_CREATE` and `VERIFY` remain schema v9. A fresh store explicitly opened
with `schema=SchemaPolicy.TASKS` uses experimental schema v10: raw message identity
primary key, unique `(session_id,seq)`, `task_store`, `jobs`, and `job_events`.
The current experimental task protocol is **2**, with persisted execution roles
and separate control-owner bindings. Protocol-1 experimental stores are refused
without modification; they were never a serving-mode migration target. No
implicit v10/protocol-1 upgrade is performed.
`COMPATIBLE` verifies an existing v9/v10 store without creating or migrating it.
Normal old-mode entry refuses v10. The original 8.0 binary also refuses its stamp.

An existing v9 store is never upgraded by `TASKS`. An operator can call
`task_schema_preflight(db,prefix)` to inspect version and counts, then explicitly
`migrate_task_schema(db,prefix,maintenance=True)` after stopping/closing writers.
SQLite/PostgreSQL migration is transactional; stamp changes last. SQLite custom
indexes/triggers, foreign keys, extra columns, stale migration tables, and
unexpected pre-existing IDs require a reviewed migration rather than destructive
guesswork. All messages, seqs, metadata and next_seq remain available.

MySQL implementation and DDL exist but are **unverified and testing paused at the
user's request**. No production migration is authorized by these APIs. A compatible
store reader is not the complete bridge image: runtime adapters, recovery,
budget/usage, old-scanner exclusion and rollback acceptance remain outstanding.

## Message identity

`append_message()` still returns session seq. `MessageRow.id` is optional and null
on v9. `await store.message_ref(sid,seq)` returns a `MessageRef`; `to_dict()` emits
version/store/session, decimal-string id/seq/send_index, round_index, role and
tool_call_id. Old zero send_index is preserved, never reconstructed as a real send.
Legacy refs report `id=null,id_status=unsupported`. IDs are store-wide, do not
define causal order and are not reused after deletion. `message_by_id(sid,id)`
always restricts by SID. It accepts an exact integer/decimal string, not floats.

Session exports deliberately omit raw IDs; import gets new identities rather
than copying references into another store. File backup/restore is not a new
store-identity allocation mechanism: restore/fork procedures and mapping are
still release prerequisites; do not clone a store into concurrent production use.

`get_tool_runtime_context()` adds `assistant_seq` and `tool_call_id`. The pipeline
sets these from the persisted sink and actual invocation, including concurrent
batch tasks and resume. Outside an invocation they are null; child sessions do
not inherit a parent's tool identity. A handler obtains its exact call reference
with `await ctx.store.message_ref(ctx.session_id,ctx.assistant_seq)`.

## Registry ports

All actor, parent/owner execution and independence choices come from the trusted
host, not model arguments. `JobActor(tenant_id,principal_id,session_id)` is frozen
attribution; `JobLease(job_id,executor_id,boot_id,generation)` is an execution proof.

| API | Guarantees |
|---|---|
| `JobManager(store)` | Requires explicit v10 identity/protocol; no legacy sidecar registry |
| `register_job(actor,kind,...,parent=lease,call_ref=ref,tool_call_id=...)` | Job and create operation commit atomically before dispatch; same operation returns same job |
| `register_job(...,system_key=...)` | Non-model origin requires a host-persisted key; root session is explicit |
| `claim_job(actor,job_id,executor_id,boot_id,expected_version,lease_ms)` | Exactly one queued claimant; no automatic replay of expired executions |
| `register_operation(actor,target_lease,caller=lease,action,call_ref,tool_call_id,request,result)` | Separate nudge/steer operation identity on the same target; records acceptance metadata, does not itself enqueue steering or execute it |
| `accept_nudge(actor,target_lease,caller=lease,call_ref,tool_call_id,session_id,message,node=None)` | Atomic active-child/node operation and inbox; original creator/SID, live caller/target proof and actual caller-send identity required; no terminal replay |
| `bind_receipt(actor,operation_id,receipt_ref,caller=lease)` | Binds an actually persisted tool row; rejects foreign, mismatched, reused-call or replacement receipts |
| `record_result(actor,lease,outcome,result,resource_state='unknown')` | Durable execution evidence/event; status remains settling and supervision open; **not** B2 final success |
| `close_model_execution(actor,lease)` | Ends only a send's model phase; retains supervision, cancellation and descendant budget obligations |
| `request_cancel(actor,job_id,system_key,reason)` | Persistent cancellation/intake barrier; queued cannot start; running/unknown only reports cancelling, not confirmed stop |
| `resume_subagent(actor,job_id,caller,call_ref,tool_call_id,session_id,message,expected_generation,executor_id,boot_id,contract,lease_ms)` | Atomic success-only standalone new generation, immutable result history, operation and original inbox; trusted host arguments |
| `claim_resume_dispatch(actor,operation_id,caller,token)` | Persistent one-dispatch claim; same-token recall is identity evidence, never permission to bypass execution fencing |
| `get_result_revision(actor,job_id,generation)` | Authorized immutable result event, revision and delivery state |
| `execution_usage(actor,job_id,generation=None)` | Lifetime model-use evidence by default, optional generation-only rollup; not API billing |
| `get_job/get_operation` | Persistent authorized self/descendant reads, including after message/session deletion |
| `prepare_processing(actor,caller,round_index,send_index,messages)` | Freeze exact visible inbox results/revisions before an actual model request; prepared is not handled |
| `commit_processing_decision(actor,caller,snapshot_id,response_ref,decision,reason)` | Fenced, atomic decision and handled batch with exact response and resolved tool/async receipts |
| `recall_processing_decision(...)` | Read the identical committed decision after a lost ACK without running work or acknowledging a prepared request |

Operation keys include store, caller principal/SID, actual call seq, tool_call_id
and action. Request fingerprint also includes the full call reference. Two
assistant rows reusing a tool_call_id remain separate. Replaying an admitted
operation is a read of the original acceptance, not permission to dispatch again;
changed requests conflict. Receipt persistence failure must retry the original
operation, never create replacement work. These ports validate actual stored
assistant/tool messages and do not predict future row IDs.

Parent/root supervision edges and completion scopes are separate. A subagent,
workflow or node execution owns the scope of its internal work while itself
blocks its parent's scope. Explicit host-selected independent work retains its
supervision parent but owns a separate scope. No model tool exposes this escape.

Short registry writes serialize on the store control row (SQLite writer lock /
server row lock). No transaction spans LLM calls. Caller proof and the actual
control-owner chain validate generations/executor/boot/leases/intake. Historical
supervision ancestors retain cancellation/unknown barriers without requiring
their model leases. Model sends under a node inherit that node's control owner;
host-selected independent jobs keep their supervision parent and have no live
creator-lease dependency. Unknown resources block new descendants. Result evidence
remains separate from resource-stop claims and completion decisions.

## Explicit loop adapter: sends, fresh workflows and subagent creates

`runtime.task_runtime.TaskRuntime` is an additional, trusted host opt-in. Opening
a v10 store alone does not enable it. Example for isolated development stores:

```python
from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

runtime = TaskRuntime(JobManager(store), tenant_id="tenant", principal_id="host",
                      executor_id="executor", boot_id="unique-process-boot")
loop = StatefulAgentLoop(llm=llm, store=store, task_runtime=runtime,
    tool_registry=registry, config=AgentLoopConfig(delegation_policy="single_level"))
```

This automatically registers/claims each model send, renews its execution lease,
and saves its model-phase exit and usage evidence. A standalone linked subagent
create is admitted from the actual stored assistant/tool-call identity BEFORE
its task is started. Child session binding/claim is required and fail-closed,
separate from the existing best-effort observation callback. Child model sends
inherit their durable subagent owner. A creator ending naturally, passing or
being cancelled does not implicitly stop independently accepted subagents.
Explicit persistent cancellation of their supervision tree still blocks them.

Receipts include job/operation/run/session identities. The sink binds the actual
tool row before clearing pending state. Receipt evidence can be reconciled after
caller closure without reopening execution. A receipt-binding failure preserves
pending state and the already accepted task; automatic `resume` is rejected in
this increment, never blindly dispatching that task again. Replaying an actual
admitted create returns its existing identity without starting another task.

Subagent results commit to the canonical ledger before compatibility projection
and owner inbox/wake. A lost result-commit acknowledgement remains queryable for
reconciliation; no replacement failed result or fake success notification is
invented. Cancellation winning the result transaction yields a cancelled outcome
while retaining actual output, usage and execution status. Finished execution
closes intake and lease but stays `settling/pending_delivery/supervision=open`;
this is NOT the final completion gate or durable delivery acknowledgement.
Expired/cancelled current attempts may still write evidence, not dispatch work;
stale generations cannot overwrite it. An unconfirmed stop retains `unknown`.

Fresh detached workflows are now supported as described below. Until their
adapters are implemented, this experimental runtime explicitly rejects background
create/adopt, timer creation, unsupported idle subagent resume and finished workflow/node resume, loop continuation
that rebinds an old session, ephemeral lifecycle, eager workflow wake and
interaction continuation. Legacy default runtime behavior is unchanged. Do not
enable this partial adapter in serving Agent processes.

### Fresh workflow and node adapters

`Workflow.start(detached=True)` from an actual persisted main-model tool call
registers the workflow before journal/task creation. The run ID derives from
its operation ID; the handle exposes `job_id` and `operation_id`. An in-process
replay returns the original handle without executing again; without a confirmed
handle after process loss it requires reconciliation, never replays work.
The admitted serialized spec/tools are frozen before waiting on DB transactions
and are the actual execution contract, even if the original mutable metadata
changes later. A request hash alone is insufficient recovery evidence.

The synthetic workflow driver binds once to the workflow execution. Logical
nodes own their execution scopes and can have multiple fresh retry sessions;
each physical session is admitted before its first model call. Driver/role/job/
run/node/iteration/attempt bindings are checked and retained as events when the
driver is deleted. Full nested foreach/loop coordinates distinguish repeated
IDs. Actual model send keys include the child SID when a shared driver registers
them. The child model's actor and message provenance remain its own SID.

Required subagents continue under the node scope after the node model phase
ends. The default executor now collects them, consumes their durable inbox on
the original node SID, and handles pass-boundary input before retrying, validating
or advancing. Waiting releases leaf admission slots; collection has bounded
rounds/deadline. Step journal publication is deferred until the canonical node
result commits. Workflow results commit before journal/inbox/callback. Cancelled
results remain in the owner's inbox but do not trigger an external completion
callback. With no callback, the durable inbox stays available without implicitly
starting another main-model call or creating a timer in this backend.

`record_execution_result(...,require_children_stopped=True)` supplies a limited
result-transaction gate, enabled by TaskRuntime: required running/unknown/queued
work blocks success. For node-owned subagents, an exact request snapshot, a
persisted processing decision for the result revision, and its matching successful
original-node model exit are checked; merely copying an inbox item into history
or producing an unrelated later assistant is insufficient. Admission and result writes
serialize on the same control lock. Usage sums each persisted model-send exit
once rather than summing already aggregated child/node totals. These are execution
evidence, not billing settlement or the complete B2 workflow-success/delivery
gate. Recorded executions remain settling with supervision open; observed results
move from pending_delivery to handled only through the decision protocol below.

### Request snapshots and durable processing decisions

`runtime/processing.py` uses the existing generic event ledger, with no schema
migration or serving activation. Each model send freezes its actual send index.
After request hooks and capacity checks, and before calling the provider, the
pipeline snapshots result events carried by actual original inbox message rows.
Snapshots bind store/send/generation/round, event ID/revision/payload hash and
delivered seq. Omitted, copied without identity, rewritten, projected-only or
late-arriving rows receive no credit. Unsupported projection transformations
therefore remain unhandled rather than guessing visibility from raw history.
Prepared snapshots alone are not successful model observations or ACKs.

At a completed send boundary without unprocessed pass input, TaskRuntime commits
a mechanical `consumed` decision. The exact persisted last assistant, round,
send and original SID must match. The caller and control-owner leases must still
be valid. Pending interactions/tools, missing raw tool receipts, and missing
bound async operation receipts prevent commitment. The decision retains reason,
response MessageRef, actual async operation IDs and follow-up job IDs; acknowledging
one result never closes those follow-up jobs or releases their completion scope.
Trusted hosts can explicitly record consumed/rejected/escalated decisions; these
describe processing, not whether an artifact is semantically correct.

All decisions for a snapshot and result handled transitions commit in one store
transaction. Revision changes, cancelled/expired/stale callers and changed replay
arguments fail closed. Exact committed replays can be recalled after the handler
closes or stops. On a lost commit ACK the runtime performs a read-only lookup;
it never re-runs a model/tool to repair acknowledgement. A missing/unreadable
decision blocks downstream success, retaining canonical outputs and usage.
The node success guard also requires the matching model_closed evidence; a
committed decision followed by a failed send cannot alone advance the workflow.

This is a partial B2 processing protocol. Final workflow closure/supervision,
outbox delivery/recovery, finished-task/workflow continuation and cross-process continuation still need
their own acceptance. No generic tools or timer delegation become enabled here.

### Active subagent nudge

The experimental backend supports `spawn_agent(action="nudge",sub_session=...,
message=...)` only while the original linked child executor remains active in
this process. Main and workflow nodes can steer their own direct children;
delegated children cannot delegate, and another main SID cannot borrow an
executor. The original job/run/SID, frozen tools/role and owner remain unchanged.
Queued children and the original finalizer's settling window are included; no
second task is started. Already closed/unknown/unowned work requires reconciliation
or a future continuation protocol and is explicitly rejected.

The admission transaction stores the operation, full unmodified message, target
proof and actual call reference together with `nudge:<operation_id>` in the target
inbox. The caller must own a live model send whose frozen send index matches
the persisted call; an earlier assistant from that SID cannot authorize new work.
Exact replays return the original acceptance, including after the child finishes.
Lost commit acknowledgements use a read-only lookup, not another acceptance call.
The local steer event is a post-commit optimization: signal failure cannot erase
accepted input, which the original executor drains at boundaries and finalization.
Caller/target cancellation and intake closing serialize with admission.

Input operations join the same exact request/response decision protocol as
results. Only the original receiver SID/control owner/generation can handle them.
Acknowledging input does not set the running child's result_status to handled;
that field belongs to the later completion result consumed by its parent.
Successful scope submission requires every admitted nudge to have a durable
decision and the matching successful model exit. Late input, a raw inbox copy,
missing response/decision, or an unrelated later assistant cannot release the gate.
Failed/cancelled execution retains pending inputs and evidence for later supervision.
The in-process finalization gate prevents losing an accepted input at cleanup.

The success-only standalone continuation below is a separate admission protocol;
other finished-task restart, node rebinding, timer delegation, external control,
budgets and generic tool migration remain unavailable.

### Active workflow-node nudge

`workflow.runner.steer_node(run_id,node_id,sid,message)` now supports the original
active engine in the experimental runtime. The trusted caller must be a live main
send in the original workflow creator's SID with an actual persisted tool call.
The engine maps physical SID to its admitted node scope; the operation freezes the
durable run/node/full route/attempt/SID binding and target lease. A logical node has
no SID and its driver never substitutes for a model receiver. Concurrent foreach
iterations sharing a node name remain independently addressable.

The admission and `node_input_closed` record serialize on the task control lock.
Required-child collection keeps input open while waiting; accepted input is processed
at its post-child boundary, without preempting the wait or starting a parallel model.
Before a fresh retry or
final schema/journal/downstream publication, close the current attempt's input and
drain any accepted race input in that original SID. Execution stays open for this
drain; input closing does not grant new execution. Old retry sessions reject new
input. A failed attempt with unprocessed input cannot start another business
attempt; preserve its original output/usage and unresolved input instead.

Node input decisions require the actual receiver send, control owner, durable
physical descriptor, complete target execution proof and original assistant/model
exit. No broad lineage access is granted to a physical node. Exact admissions can
be recalled after engine cleanup; new calls cannot reopen completed nodes. Lost
ACK uses read-only lookup. Signal failure does not roll back durable admission.
This does not enable finished continuation, result revision/rebinding, process
recovery, generic workflow/jobs tools or timer delegation. Serving stays disabled.

### Success-only standalone continuation

After the original main has actually handled a successfully settled independent
linked child's result, a later `spawn_agent(action="nudge",sub_session=...,message=...)`
may admit a new execution generation on the same job/SID. It preserves the original
role, persisted child contract, historical supervision and tools clamped by current
parent policy. Each generation has its own run ID, result event and owner inbox
identity. Workflow-owned children/nodes, cancelled/unknown/failed work, unfinished
old results, pending tools/input and missing or changed contracts remain rejected.
A main cannot start this continuation in the tool phase of processing the old
result before that processing decision has actually committed.

Admission stores the operation/input, frozen caller and target proof/contract hash,
and generation transition in one control transaction. Old execution credentials
immediately stop working. A second persistent unique dispatch claim precedes local
task creation; a lost commit ACK reads back the same claim token without repeating
admission/claim or launching replacement work. Exact replays only return the original
receipt, including when another owner accepted but dispatch is uncertain. Missing
claims, crashes or lost executor state require reconciliation, not blind replay;
durable cancellation remains available. This is not automatic cross-process recovery.

Generation one retains `result_<job_id>`; subsequent results use a generation suffix.
Old snapshots/decisions cannot acknowledge the current result. Current `get_job`
shows the new execution; `get_result_revision` retains old output/processing evidence.
Scope usage includes only that execution's model sends; lifetime usage is separately
queryable. Compatible parent totals add each execution once. Known dispatch failure
preserves a negative new result, and result delivery failure cannot overwrite a
successful result or rerun business work.

A completed child turn with empty final text cannot silently become a successful
continuation. One original-session final-answer supplement uses remaining configured
rounds and no business tools, under the same host guards. Empty/exhausted/failed
supplements produce a failed revision. The core may additionally make its existing
read-only round-limit finalization request, and every actual send remains in usage
and history. This is not a B4 hard provider-budget guarantee. Final supervision,
finished workflow-node dependency repair, generic tools/timers and recovery remain
closed. Serving defaults are unchanged.

An unclaimed queued admission can record a negative result atomically through
`record_unstarted_result`; it can never claim success or prove a claimed resource
stopped. A claim commit with a lost acknowledgement remains unresolved. Result
commit acknowledgement or compatibility delivery failures retain canonical data
for reconciliation, without rewriting success or sending a fake success wake.

## What is not wired yet

This module does not automatically replace all existing delegation/background/
workflow registries, authorize/charge provider calls, run external cancellation adapters,
recover processes, finish reliable result delivery/recovery, or publish final workflow
success. Complete runtime adapter integration,
atomic success gate, tree UI/jobs/recall facade, retention/privacy purge, and
complete bridge rollout remain subsequent work. Never run old/new controllers
against the same execution, or enable this schema on the serving Agent yet.

The subsequent owner/role increment implements those distinct checks in the
experimental manager. Jobs persist execution_role/control_owner_job_id and frozen
owner execution proof. Model closure is idempotent and is not final task success.
Registration and steering reject forbidden recursive delegation by the actual
caller role. The explicit adapter above covers sends, standalone creates and fresh
workflow/node executions; background and full continuation adapters remain pending.

Separately, `AgentLoopConfig(delegation_policy='single_level')` enables the host
runtime role policy; generic library callers retain the legacy default. The host
stamps child roles, filters tools on create/restore, preserves the saved stricter
policy, and checks actual dispatch even when a wrapper retains a full registry.
Workflow nodes can delegate but cannot start workflows; delegated agents cannot
delegate or use workflow tools. Driver cleanup does not promote retained leaves.
DeepTalk opts in and makes its workflow model tool asynchronous only. Internal
host await APIs remain available. See DeepTalk design/192 §§14–15 for the local
implementation and the still-missing complete admission/delivery/recovery gates.

## Verification

From the library's venv:

```bash
.venv/bin/python scripts/test_task_registry.py
.venv/bin/python scripts/test_task_registry.py --full
```

The script creates a fresh random PostgreSQL database ending `_test`, runs the
SQLite/PostgreSQL tests plus twenty-one actual configured LLM cases, sanitizes artifacts,
and drops the DB in finally. MySQL tests are explicitly excluded. Missing/skipped
real tests make the runner fail. It never restarts the existing Postgres service.


## Dependency planning before finished-node admission

`workflow.continuation.plan_continuation` produces an immutable, inspectable
invalidation closure without accessing the database, running a model, changing a
journal or authorizing dispatch. It includes sequence tails, ancestor aggregates,
explicit data/control references, `node.key` input templates, known explicit
`output_file`/`FILEREF` dependencies, and fallback ownership. It cannot infer
arbitrary external file/tool dependencies. Foreach bodies require iteration-aware
targets; fallback repair targets its owning agent. Changed foreach input/body
invalidates the entire aggregate, and an iteration result never replays alone.

The existing non-TaskRuntime live engine freezes this plan before registering a
repair. Active consumers, affected earlier work with backward dependencies, and settled affected results outside its selected branch
are refused before mutation. A parallel join closes all branch continuation
intake synchronously before awaiting repairs. Run exit closes intake and cancels
and drains owned repair tasks before driver deletion. Unrelated terminal failures
remain evidence, rather than being implicitly retried. Replaying a foreach prefix
in the same run does not charge its tokens twice.

This is a prerequisite, not a finished-node adapter. The experimental runtime
still refuses finished workflow/node continuation: owner/controller execution
rebinding, durable invalidation, execution/result revisions and restart recovery
must be implemented together. The new real-model test explicitly exercises the
legacy owning engine; it does not claim those new protocols are enabled.


First-generation managed children also receive one bounded original-session final
answer supplement if they settle with completed but empty text. The original
configured remaining rounds, owner proof, cancellation, guards and finalizer apply;
no business tools are enabled (the builtin pass_turn control tool may remain).
No remaining rounds, repeated empty/pass, or failure cannot publish success.
Subsequent generations retain their existing continuation-specific handling.
The same bounded repair applies to legacy asynchronous managed children; it does not activate TaskRuntime or finished-node continuation.
A real model case deliberately passes empty on its first child turn and must then
produce the answer through the same session; a parent's guessed answer is not
accepted as a substitute. Actual provider calls, including the core read-only
limit finalizer, remain in usage and are not a hard billing guarantee.

## Generation-qualified physical node evidence

Node-session bindings and input-closure markers retain their original ID/payload
for generation one. Later generations have separate IDs and explicit generation
in their payloads. Model-send registration, input admission/processing and repeated
binding check the complete physical descriptor against both the session and the
original frozen workflow/node admission. Matching edits to a session and its
binding cannot change the admitted run, node, route or attempt range. Numeric JSON
types are compared exactly; booleans/floats cannot stand in for integer coordinates.

An existing corrupt binding or closure is rejected, never accepted as an idempotent
success or overwritten. Input admission and closure use the same control lock;
new-generation records cannot borrow old-generation bindings or closure markers.
This provides storage isolation: `bind_node_session` rejects later generations
without the owning-engine admission/dispatch proof described below; it only verifies
an already-admitted binding. Synthetic storage tests alone do not enable runtime resume.

## Original active-workflow leaf resume

`runner.steer_node` can now admit a successful, settled single agent branch while
its original parallel workflow and driver remain alive. Only the default in-process
executor is supported. The original engine registers and owns its repair task before
any admission await; parallel join and shutdown wait/stop that task. Main receives an
acceptance receipt, not completion. Other completed consumers, a closing branch,
foreach/loop/fallback scopes and finished workflows are refused.

The manager transaction checks the actual main call and current workflow lease,
exact prior canonical and journal result, physical binding/closure, saved contract,
frozen dependency plan and settled resources. It preserves the original job/SID and
historical controller/creator, admitting a new node generation, immutable execution,
physical binding and queued input together. Persistent dispatch claims precede any
new model call. Lost ACKs are reconciled by reads. An unowned/ambiguous dispatch
retains an unresolved cancellable ledger and blocks workflow success; it cannot be
rewritten as stopped execution. A positively owned failure before model dispatch can
save a negative revision.

Resumed execution reinstalls the original workflow and new node owner context,
flushes the input once on the original session, and uses the saved persona/tool
contract. No implicit fresh attempt or empty successful answer. Old decisions check
their own generation bindings; new inputs require new owner/send proof. A new
canonical node result precedes journal/downstream; workflow/node usage and old
revisions remain separate. This is still opt-in, with no whole-workflow resume,
cross-process stop/recovery or serving activation claim.

## Immutable execution membership and historical usage

Every experimental claim or admitted resumed generation writes an `execution_scope`
event in the same transaction. It freezes the execution lease, original control-owner
proof, subject, role and historical creator. A controlled execution also writes an
`execution_member` under that exact owner generation. Independent asynchronous jobs
retain their supervision parent without inventing a control-owner edge.

Model close and scope result persist an `execution_membership_seal` atomically with
exit evidence. The seal freezes direct membership and the model-close evidence hash;
it does not assert that supervision is closed or all provider charges are known.
Already-admitted descendants can exit later, contributing observed usage to their
original owner without rewriting the owner's old result revision.

`JobManager.get_execution_membership(actor, job_id, generation)` authorizes the root
and returns its immutable scope, direct member proofs, seal status and model-close
evidence. `execution_usage(..., generation=n)` traverses only those frozen edges and
counts each actual model exit once, without adding scope rollups. It holds the store
control lock for a consistent read and bounds traversal at 64 levels. Lifetime
`execution_usage(..., generation=None)` keeps the historical supervision-tree
semantics, including independent asynchronous children and all their generations.

Live job owner/executor/generation changes cannot reassign historical usage. Invalid
or missing scopes, contradictory leases/subjects, malformed edges/seals/model
evidence, duplicate membership and cycles fail closed. A completed execution missing
its atomic seal is incomplete history, not an empty execution. Resume additionally
requires the old sealed graph to validate before admitting any new generation.

An old experimental store without scope anchors can still save actual exit/stop
evidence and query lifetime usage. It cannot silently manufacture complete historical
membership or resume through that missing history. No ordinary v9 store is migrated
or activated. Tests that construct future owner generations in storage do not grant
finished-workflow restore; controller rebinding, dependent recomputation, restart
recovery and final supervision closure remain separate work.

## Generation-qualified workflow controllers

`bind_workflow_driver` now commits a `workflow_controller` event together with the
job SID update. The event freezes the full execution lease, original creation
operation/request hash, run, driver SID and historical creator against the claimed
execution scope. Repeating the same call revalidates both saved evidence and actual
driver metadata; it cannot rebuild a missing event from the mutable job row. A
different SID is not the same call. Two managers compete under the same control lock,
and an existing driver cannot be adopted by another workflow.

`get_workflow_controller(actor, job_id, generation)` is an authorized historical
read. It does not require the ephemeral physical driver to survive run cleanup and
does not grant dispatch. Actual new node/send admission, physical binding, claims,
nudges/leaf resume and provider/tool dispatch require the current frozen controller
and its real root driver, exact role/owner/run and integer generation. Required
subagent dispatch checks its controlling node's workflow too. Claims check again
after registration; dispatch checks again after claims. Missing/corrupt creation,
scope/controller evidence or changed physical metadata cannot start replacement
work. Historical model exit/stop/result evidence remains recordable after driver
deletion; it is not permission to issue another request.

Pure logical registry scopes can exist before a driver is bound. That path does
not authorize a physical node session or provider dispatch. First-generation driver
metadata may omit `task_workflow_generation` for compatibility; new engines write
it explicitly. Booleans/floats/strings are never accepted as integer generation.

Controller IDs include execution generation and historical records are independent
of current job generation. This stage intentionally permits ordinary binding only
for generation one. Storage-only fixtures with coherent later-generation records
still cannot admit execution. Whole-workflow generation admission, new-controller
authority for retained node sessions, dependent recomputation and new result
publication remain closed pending their atomic resume/dispatch protocol.

## Empty workflow text output is not delivery

Real acceptance exposed a native first-generation workflow node passing an answer
only through `pass_turn.reason`, leaving empty text and starting review with no
context. For experimental TaskRuntime/default InProcessExecutor text nodes, after
required child/input collection and closure, an empty completed result receives at
most one original-session final-answer supplement. It reuses managed-child answer
completion, original saved contract/remaining rounds, owner/controller and child-run
guards with business tools disabled. The builtin pass_turn control tool may remain;
its reason is never copied as output. Empty, exhausted, cancelled or failed output
cannot publish successful text delivery. The core's existing read-only limit
finalizer can still add provider calls; this is not a hard billing guarantee.

The original engine marks the supplement before awaiting it, preventing re-entry
from granting another attempt. Failure/cancellation preserves actual model exit
evidence and usage. Existing later node generations retain their adapter's answer
checks. Explicit/computed output files and custom executors preserve their previous
delivery contracts; they do not receive forced native-text supplementation.

### Workflow result delivery revisions

Experimental `runtime.workflow_results` freezes a generation-qualified delivery
revision from the committed canonical result and effective cancellation state,
not from a caller's engine rollup. It anchors the original creation, full current
lease, sealed membership and (for success) historical controller; delivery never
needs a live ephemeral driver or grants execution authority. Unstarted failure
is explicitly generation zero without inventing a claimed execution scope.

The envelope identifies the canonical event, revision and hash. Generation one
keeps its existing inbox ID; later generations use `result_<job>_gN`. Journal
generation defaults strictly to integer one; another generation needs an explicit
matching marker. Frozen/corrupt evidence, terminal journal conflicts and inbox
content/type/mode conflicts fail closed. Journal and inbox writes remain separate,
idempotent compatibility projections. The frozen event is preparation evidence,
not delivery or main-consumption acknowledgement. Explicit projection repair
does not run a provider/tool, callback, new execution or processing decision.

Scope result usage always comes from that generation's execution ledger, even
when empty; replayed historical rollups cannot become newly incurred cost. Both
workflow and subagent result snapshots filter current generation before the
request; commit rechecks it. This does not enable whole-workflow restore, strong
budget guarantees, automatic startup recovery or cross-generation takeover.
The future resume transaction must fence unfinished projections as well as node
dispatch; a one-time projection read cannot solve that race by itself.

### Atomic projection and callback exits

The current experimental `project` now holds task control, then the ordinary
session-state row lock, through preparation, journal finalization, inbox insertion
and immutable `workflow_projected` receipt. All reads/writes use the same tx;
public store methods that open another connection/transaction are forbidden inside
that scope. Rollback/CancelledError undo the whole projection; commit-ACK loss
leaves a complete, idempotently readable projection. The canonical result stays
separate and immutable. Existing prepared-only/legacy view writes can be explicitly
projected; a closed receipt with missing preparation cannot recreate history.

Post-commit memory hints never dispatch a model or enqueue again. Completed inbox
items do not gain a fresh hint on retry. `require_closed(..., consumed=True)` is a
transaction-local future-admission prerequisite: verify the current revision,
closed receipt, actual views, callback exit and persisted main processing decision.
It does not admit another generation or authorize retained-node execution.

New workflow creation records a host-supplied `completion_callback` bool. Missing
historical fields are unknown, not false. Actual callbacks separately claim one
generation/token and record immutable completed/failed/cancelled exit evidence;
prepared/start receipts do not prove external effects. Lost callback claim ACKs,
missing exits and failures remain unresolved and never trigger automatic callback
replay. A callback failure does not overwrite a finished business result. This is
library callback settlement, not host timer/wake consumption, final supervision
closure, reliable startup recovery or whole-workflow restore. The old paragraph
about separately written compatibility views describes the preceding increment.


### Workflow creation profile and restore preview

Experimental creation freezes the complete request plus `execution_profile` before the first await. The original operation fingerprint covers executor, soft-budget origin/balance, file port and completion capability. Native means an exact InProcessExecutor, never an inferred subclass. Host objects are classified, not serialized; external ports and opaque callbacks require adapters before restoration. A runtime SharedBudget snapshot is not a transferable reservation. Legacy missing profiles remain unknown.

`get_workflow_profile` validates the original creation checksum and authorization without requiring the ephemeral driver. `get_workflow_restore_plan` checks closed projection/real consumption, stopped and sealed membership, historical controller, canonical node results, detached physical bindings, original contracts/tools/metadata, pending tools/inbox and dependency invalidation. It returns evidence only, creates no operation or execution and grants no permission. Admission must recompute under the control transaction. Historical descriptor reads permit detached parent=NULL; execution descriptors retain their existing strict parent check. Whole-workflow resume remains closed until atomic admission and cross-driver authority are implemented.


### Storage-level workflow continuation admission

`runtime.workflow_resume.admit` atomically reserves a later workflow execution from a real active main call. It recomputes the original recovery plan, freezes separate creation/current-operation identities, reserves a unique driver SID and affected node generations, and resets the current journal from canonical output. Old node jobs, physical parents, results/controllers/memberships are retained. Silent writes and partial commits cannot produce acceptance. `claim_dispatch` preserves a single token-qualified responsibility; replay never runs an engine. Node reservations are not execution membership.

This is not yet a running restore adapter. Public resume/force and later-generation binding remain closed. The future adapter must retain owned work before admission, fence driver/node authority and ensure pending reservations block final success. Callback/wake settlement, crash recovery and hard budgets retain their separate outstanding requirements.

### Native asynchronous workflow restoration

The subsequent experimental TaskRuntime adapter now connects `resume_detached`
to the admission protocol. It retains an owned task before admission awaits and
claims exactly one dispatch before running a reserved driver. Synchronous resume,
unadmitted engine replay and force bypasses remain closed. An originating main
send may end while the accepted workflow continues independently.

The same workflow job/run and original node jobs/SIDs are retained. A new driver
does not become the old sessions' physical parent or creator. Frozen admission,
dispatch, controller, reservation and token evidence grants generation-qualified
node authority; actual sends and provider dispatch recheck it. Affected nodes
receive updated upstream inputs and rerun. Unaffected branches replay frozen
canonical results without new membership, provider requests or charges. A
subsequent restore follows immutable replay sources across older generations.

Success requires a typed settlement of every affected reservation, real input
consumption and canonical stopped-node results. Required child tasks must settle
and their actual output be consumed before an empty pass can be supplemented or
downstream can run. Ambiguous dispatch keeps cancellation/supervision open;
missing settlement cannot become success. Lost acknowledgement is reconciled
against the same identity/token at admission, driver/node/input and finalization
boundaries. A committed result is reprojected rather than reexecuted.

New envelopes use the restoration operation ID and separately retain the creation
operation ID. Per-generation usage remains incremental; native spec soft budgets
account for prior actual generation usage. Old handle cleanup cannot erase a new
generation's controls. These are not hard reservations, supplier reconciliation,
automatic process recovery or final supervision closure.

The first native slice supports successful, consumed, fully recorded text
sequence/parallel workflows with unique agent nodes. External executor/file IO,
callbacks, explicit shared host budgets and unsupported control structures need
adapters and remain rejected. Generic jobs/workflow tools, root wake handoff and
production activation remain outstanding. Design/192 §25.8 and the development
record track the current test evidence and remaining scope.

### Bounded persistent directory queries

The experimental `runtime.task_query.list_jobs/get_job` read model is independent
of model handlers and process-local handles. It reads the actor's authorized task
forest, or a specifically authorized root, retaining connecting ancestors as
`context_only` rows. A leaf cannot expand the scope to siblings or ancestors;
invalid/cross-tenant/cyclic lineage fails closed. Scan, queue and depth bounds
remain explicit; a truncated tree is never called complete.

Lists filter business outcomes separately from active supervision/resource/result
obligations. Thus a terminal outcome can still be active. Billing is explicitly
`unavailable` until the billing adapter exists; queries do not invent settlement.
Cursor pages bind the store, actor, filters, creation upper bound and authorized
job/parent membership digest. A same-millisecond insert below the random-ID bound
or a changed retained scope requires a fresh snapshot; later inserts do not enter
the old page. Status is read again on each page, not frozen across requests.

Responses obey a whole-JSON character budget, including context and cursor, and
return original call/receipt raw message IDs plus canonical result pointers.
Saved excerpts come from the immutable result event rather than a mutable cache.
Unvalidated reference fields and control credentials are never included. Queries
do not dispatch, acknowledge, cancel, heal history or execute returned content.
The generic tool registration, durable stop port and task-qualified recall/export
are subsequent work; this query module does not replace those deliverables.

### Library jobs facade and durable stop acceptance

`JOBS_DEFINITION`, `register_jobs_tools` and explicit
`create_default_tool_registry(include=["jobs"])` select the same library-owned
schema/handler. Existing presets and role allowlists do not expand. Native reads
use the persistent directory. Native stop selects one job and its supervised
subtree; disjoint roots use separate actual calls. Legacy `ids` remains a limited
v9 compatibility input, not permission to synthesize multiple operations from a
single native call. The host may format legacy output and localize the definition,
but cannot replace the library schema or duplicate its control state.

`task_control.request_stop` atomically validates the actual source send/call,
current owner proofs and optional expected target generation, saves cancellation
and closes intake with the operation. Expired leases do not prevent safe control;
closed/superseded callers cannot issue new actions. A stop of the invoking send or
its ancestors is rejected; trusted mechanical host cancellation is separate.
Stop receipts are not resource-stop or billing confirmations. Independent
`recall_stop` reads the same accepted intent after ACK loss without writing or
re-dispatching. The standard raw receipt binder now includes stop operations.

TaskRuntime retains current execution/task handles until finally. A committed
intent signals only current authorized local descendants and never a different
generation or job. Nested scopes share one cancellation, and the heartbeat avoids
recancelling a task that is saving actual exit/usage. Missing handles or signals
leave the persistent responsibility open; another process and remote providers
still need their explicit control/recovery adapters.

Only the exact library jobs handler, including trusted control-only
`functools.wraps` host formatters, may use the expired-lease control lane. A tool
name is not authority. Actual raw source/owner proofs remain required; normal
business tools and every provider call retain their full execution fences.
Known cancellation/supersession suppresses external subagent completion callbacks
at their pre-call check, preserving the owner's durable internal result.

Legacy views explicitly expose best-effort control and unavailable history/tree
capabilities. Native backend failures report unknown reconciliation rather than
inventing empty history or confirmed stop. Workflow facade, task-qualified
recall/export, root wake/default English constraints, final supervision closure,
cross-process recovery, hard billing and serving activation remain outstanding.

### Shared workflow submission and initial facade

`workflow.WorkflowToolPorts` contains host-only source, policy, executor, budget,
file, effective-tool and presentation ports. Model kwargs cannot replace them.
`register_workflow_tool` explicitly installs the library schema/router for the
initial help/create/validate/status/list actions. It is not added to presets.
Compatibility create/validate/status names share preparation/submission or
inspection; model create never waits for completion, while direct SDK awaited
execution remains available under its existing runtime rules.

Preparation freezes model arguments before awaiting a loader and round-trips the
compiled/transformed spec. An empty input and empty calling tool allowlist remain
empty. Host whitelists intersect the actual calling send's captured tools.
Default YAML sources/includes stay within the workspace, resolve symlinks, and
read at most 1 MiB per file. A host supplies explicit skill/charter permissions.
Validation never instantiates executors or budgets. Compiled/accepted presentation
callbacks receive isolated snapshots; async waits have a bounded timeout. Hosts
must keep synchronous observers nonblocking. Their errors cannot undo admission
or trigger replacement execution; completion callbacks retain the engine's
actual-exit protocol.

Admission returns run/job/operation IDs and the actual delivery mechanism:
native owner inbox, explicit host callback, or legacy timer requiring a runner.
No guaranteed wake or completed result is invented. Native status/list reads
authorized canonical task data; legacy status identifies its journal view and
does not claim physical liveness. Arbitrary host exceptions remain private.

DeepTalk create/validate use this boundary; its other action routing, local
compatibility handles and domain views remain until the next migration. This is
not the completed workflow facade: cancel/nudge/resume/logs/leaves/pause/extend/
delete, mixed-action control dispatch, host restoration resolvers, generic wait,
root wake, cross-process recovery and billing remain required outstanding work.

### Persistent task/session read authority

`runtime.task_sessions.list_sessions/authorize_session` resolves only a task's
actual physical bindings after persistent actor/lineage and execution-snapshot
checks. It grants reads, never execution. Logical workflow/background/timer SIDs
are not their own model transcripts. Send bindings retain the original send_index
and subsequent readers must apply that restriction. Node descriptors verify the
original frozen route/attempt and survive driver cleanup and retained parent=NULL.
The same SID can have several generation bindings without making the whole
transcript belong to one generation. Deleted sessions are explicitly unavailable;
readers must not substitute a new SID. More than 100 node bindings is an explicit
scope-size error, not silent truncation. This mechanical port does not yet route
logs, raw/projection recall or exports; those adapters remain required work.

### Workflow whole-run control and action-qualified dispatch

The unified workflow cancel action resolves and authorizes the original creation
operation, then shares `task_control.stop_from_current_call` with jobs. No tool
recursively invokes another model tool. Actual call/receipt identity, durable
acceptance, read-only ACK recall and pending physical control retain their
existing semantics. Cancellation never confirms remote stop or billing closure.

Only the owned workflow handler's help/status/list/cancel actions use the safe
control lane; create/validate and normal provider/business dispatch remain fully
fenced. Submission/validation also recheck dispatch inside the boundary so a
faulty wrapper cannot change a read into paid preparation. Fixed create aliases
cannot borrow a supplied action=status. Trusted platform wrappers preserve the
handler identity and route native control to the library.

Legacy whole-run cancellation checks the owning journal plus exact local
store/parent handle, returns best-effort asynchronous acceptance, retains the
bounded stop task and suppresses completion callbacks. Missing handles remain
unknown without journal reaping. Forced local task cancellation records the
actual cancelled exit while retaining steps/usage. SDK stop_job also scopes its
workflow handle; it cannot stop another parent's run by its global ID.

Native node-only cancellation now uses one logical node token before admission
and carries its operation through the model send. A send heartbeat stops its
own token; it cannot cancel a task hosting its parent. Local control prefers
operation tokens and refuses Task.cancel when the task also hosts an unselected
scope. A supervision/backend failure still fails the workflow, rather than
becoming a tolerated user cancellation. Input closure is exit bookkeeping: only
the current exact proof may write its closure marker after durable cancellation;
it grants no model dispatch authority.

workflow cancel accepts either an unambiguous existing node name or an exact
node_job_id belonging to the original workflow. Reused foreach IDs require the
job selector; malformed/missing/foreign/ambiguous selectors never become a
whole-run cancellation. jobs.stop(node) uses the same durable operation and
signal path. A durable node stop winning just before provider admission is
reconciled from its exact lease, not mistaken for an ancestor stop. Cancelled
nodes are recorded as cancelled and tolerated for the engine's continuation;
they do not retry or start fallback work. This does not close supervision or
confirm remote resource/billing settlement.

Legacy node cancellation proves the journal/store/parent handle, then signals
the engine retained by that handle, never a global engine lookup by run ID.
Missing engines/handles remain control_unavailable/unknown; native generation
or exact job selection is explicitly unavailable in legacy mode. The platform
delegates both run and node cancellation to the common library adapter.

### Jobs error boundary

Only typed library JobError codes become public jobs errors. Input validators
use that type explicitly. Arbitrary host/backend ValueError exceptions, even
those with a code attribute, follow the unavailable/reconciliation boundary;
their text is never copied into the model receipt. An unknown stop error keeps
control_state=unknown. A directory error never becomes a successful empty list.

### Task-qualified history and native workflow views

`runtime.task_history.read_history` and `recall_send(job_id=...)` authorize the
actual persistent task lineage and node/session descriptors in a read snapshot.
The native workflow leaves/logs actions share this port; the platform delegates
these actions to the library. A task with several retry sessions requires an
explicit SID. A send task stays limited to its original send_index. Its projection
reads exclude multi-send compact rows and check the original source-seq scope.
Node sessions retained after driver cleanup remain readable, with every actual
generation binding reported as physical-session scope rather than one generation's
exclusive history. A logical task never grants the driver's or caller's history.

The exception is an explicit global message_id/message_ids matching saved
creation call/receipt references: the actor must separately own their source
send, and the read is limited to those exact rows. It cannot widen into the
caller's whole transcript or let a child read its ancestor's call body. Missing
or mismatched references do not grant authority.

Raw global IDs use message_id(s); messages ids remains a seq range, projection
id/ids uses projection IDs. Public record IDs, seq, and send coordinates are
decimal strings. Pages retain the captured coordinate upper bound and bind the
actor, task, selection and bindings; every page rechecks permission. Projection
content is mutable even when its ID is unchanged, and the envelope says so.
Deleted/unavailable history is explicit, never an empty successful export.

Inline reads have 1..100 records and a total 1024..32000-character JSON budget;
one record can be sliced by char_offset. Inline media is screened. Complete
JSONL records export only to a new workspace file, with 10000-record/16MiB hard
limits, one consistent read transaction, reauthorization, exclusive publication,
and exception/cancellation cleanup. A small receipt budget is rejected before
publishing the file. Reads never run models/tools, acknowledge results, close
supervision, or establish billing settlement. Legacy unqualified recall retains
its prior contract; legacy workflow history adapters remain required work.

## Trusted host resource preparation (foundation only)

`runtime.workflow_resources` defines private immutable resource bindings and an
explicit host-only registry. Match exact adapter key/version and original
tenant/principal/main, resource revision, config/contract hashes, workspace,
budget scope and notification mode. Class-only historical profiles cannot be
converted into this evidence. A changed binding cannot substitute a current
executor, workspace or fresh budget pool.

Adapters implement async side-effect-free prepare/resolve and explicit current
authorization of the actual resource objects. Registry revocation, including
same-version re-registration across awaits, invalidates in-flight resolution.
Resolution is bounded and cancellable; private factory errors become fixed
codes. Adapter code is trusted Python, not a sandbox; resource preparation must
not dispatch models, write business artifacts or reserve/charge money.

The registry grants no task admission, stop, billing or delivery authority. Its
binding protocol version is separate from workflow execution profile versions.
Profile-v2 creation and restoration additionally bind original create fingerprints
and actual plan/admission/dispatch/engine checks. A prepared resource is not proof
that a prior execution stopped, a callback was accepted by a business system,
or a shared budget is durably settled. Existing profile-v1 restore restrictions
and SDK defaults are unchanged; no model tool parameters expose this registry.

### Bound creation profiles

Trusted hosts may configure `TaskRuntime(workflow_resources=registry)` and select
an adapter with `WorkflowToolPorts.resource_adapter=(key, version)`. Direct SDK
creation accepts `resource_binding`, mutually exclusive with explicit executor,
budget and file ports. The model/YAML parameter schema never exposes this seam;
bound execution requires the managed detached path.

New admissions freeze the actual effective spec/tool contract before resolving
resources and save the immutable binding in profile-v2 and the original create
fingerprint. Original profile-v1 interpretation is retained. Before building a
driver/engine, re-read that profile, resolve and reauthorize the same identity,
check the actual lease, and reject changed capacity/floor or rewound pool spend.
Exact original-call replay uses saved evidence without resolving or replenishing
resources; no handle means reconciliation, not replacement dispatch.

This slice admits only concrete locally managed InProcessExecutor/SharedBudget
and inbox capabilities. External executors, file effects, custom budgets and
business notification require additional stop/effect/receipt adapters. Binding
evidence alone cannot authorize those effects. None of this establishes hard
budget settlement or cross-process recovery.

### Bound restoration (local managed resources)

Restore plans may read validated bound profiles without invoking host code.
Admission additionally requires resources prepared from the original binding;
host resolution/authorization never runs in the store transaction. Recompute
the original plan, retain profile/binding hashes and observed spend in the same
new-generation admission, then reauthorize and fence the actual dispatch and
engine objects. Exact-call replay returns the original receipt without resolving
resources, replenishing pools or granting another dispatch.

An in-memory SharedBudget has no durable balance authority. The registry retains
its original object across local creation/restoration, keyed by original owner
and budget scope. A newly constructed equal-looking pool cannot erase other
scopes' consumption; a fresh process/registry without that continuity requires
reconciliation. This temporary local safeguard must be supplemented by the
durable BudgetProvider protocol for cross-process shared-budget recovery.
Spec-owned pools retain prior canonical usage; original runtime pools are reused
without adding that historical usage again. Each restored node settles against
its actual execution generation, while unaffected canonical outputs replay
without new requests or charges. Failed authorization after admission retains
explicit cancellation/reconciliation responsibility and starts no child request.

## Shared completion window

`runtime.completion` provides the required-work collector used by both the
native node adapter and host executor adapters. A managed owning engine installs
one `CompletionWindow` per logical node execution; host generic completion steps
defer to that owner. Direct legacy host execution delegates to the same collector
and text finalizer. This does not enable unregistered background/timer adapters.

Repeated checks share their original deadline and remaining supplemental rounds.
Managed supplemental work is also limited by the frozen original AgentSpec,
ContinuationPolicy and accepted continuation rounds. Retry sessions may share
their owning window; parallel nodes and new generations do not share it.
An explicit deadline check precedes each model call, including after host
resolution. Zero reported rounds still consume conservative allowance without
altering reported token usage. A concurrent collector or a repeated final-answer
attempt cannot allocate another supplement. Known usage survives limit and
result-persistence failures.

`completion_unresolved` is intra-execution control evidence: result/ACK handling
and final-answer repair must not be interpreted as a business failure eligible
for retry, fallback or automatic continuation. Normal initial business failures
retain their declared policies. Persistent task results, required-result gates,
original generation evidence and remaining reconciliation obligations still
govern completion; this collector is not a final supervision-close protocol or
a provider-wide hard budget. Root wake handoff, English constraints and durable
background/timer integration remain required work.
