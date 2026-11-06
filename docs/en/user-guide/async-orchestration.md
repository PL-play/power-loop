# Asynchronous orchestration

[简体中文](../../zh/user-guide/async-orchestration.md) | [User guide](../index.md)

This page describes unreleased 10.0.0, design 203 P3. Hosts start `Runtime.worker`
and submit inputs through `RuntimeClient`. Workers rebuild execution environments
from persisted inputs. The store owns results; notifications only wake workers.
Shared execution requires a shared store, transport and session ownership coordinator.
Default process-local adapters do not provide cross-machine coordination.

## Children and background work

`spawn_agent` admits an independent session and returns `job_ref`, `input_ref` and
`session_id`. Acceptance is separate from completion. The returned `JobRef/0` can
be awaited directly: it selects generation 1 after the first claim, or the actual
generation 0 terminal result if the task never starts. The start input binds 0→1
at claim and never follows subsequent resumes.

Children retain frozen tool grants and configuration. Delegation is one level only.
Later sends in the original main session can inspect, nudge or stop the original
child without its parent loop. Nudges join a running generation; interrupted tasks
require explicit resume. Inline `run_agent_spec` admits the same durable job and
waits for it. Waiting keeps session ownership but releases execution capacity,
including when configured capacity is one.

`background_run(action="tool", tool=..., args=...)` and `action="run", command=...`
admit independent background jobs. `check/list/wait/cancel` address the original
`task_id`; a wait timeout does not cancel it. Background-eligible tools declared
with `interrupt="background"` are admitted independently before dispatch. Steering
ends only their foreground wait. The actual background owner settles the tool's
original receipt against its own job and original root; admission does not double-charge it.

The result and parent inbox reference commit together before notification. Active
parents consume results at round boundaries; an idle parent gets a subsequent
worker execution. Duplicate notification does not duplicate delivery. Explicitly
cancelled scopes are not revived by late results. Children collect their own
background results before their final answer, retaining the existing round budget
and bounded output repair. Queries retain full results; model notices may be summarized.

Parent exit does not cancel independent work. Shell control addresses the original
resource. A lost resource becomes interrupted/unknown; a replacement worker must
not rerun the command. `_RUNNING`, `BackgroundManager`, `BackgroundRuntimeProjector`
and completion callbacks no longer own task execution or delivery.

## Answers and explicit resume

`request_user_input` stores the interaction, original job, send and round, then
releases capacity. Inspect the input to observe `waiting_input`. A normal send is
not an answer. A host can answer from another node:

```python
receipt = await runtime.client.answer(
    job_ref, interaction_id=interaction_id, wait_round=wait_round,
    value="blue", message_id="answer-1", actor=actor,
)
result = await runtime.client.wait(
    original_input_ref, actor=actor, deadline_ms=deadline_ms,
)
```

The answer input and matching tool message commit atomically. An identical repeat
returns the original answer state. Conflicting, expired, wrong-round or newly
submitted post-cancellation answers are rejected. Node changes preserve the job,
generation, send, cumulative rounds and billing. `StatefulAgentLoop.submit_input`
wraps the same endpoint.

Explicit resume requires a positive interrupted/unknown, uncancelled child generation.
Generation 1 resumes as 2 with a new input bound to 2; the original `JobRef/0` and
start input still read 1. Withdrawing an unbound resume input only voids that input.
Unknown tools are never automatically replayed. Cancelling `JobRef/0` is valid only
before claim; after claim it reports stale_generation. Cancelling an `InputRef`
uses its atomically bound actual generation.

## Retention and scope

Waiters and capacity are released at execution end. Completed checkpoints,
allocations and budget attempts use existing cleanup. Results, original calls and
descriptors remain until the task tree is eligible for existing retention cleanup;
parent return never deletes a live child session.

Workflow driver/node integration remains explicitly unavailable until P4. Timers,
inspection and business-host integration belong to later stages. This work adds no
product entry point for users to control arbitrary children outside the original
management scope, and no runtime compatibility for historical call formats.
