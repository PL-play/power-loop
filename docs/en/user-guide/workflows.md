# Workflows

A workflow is a frozen `WorkflowSpec` interpreted by a durable driver job. The
driver and every leaf execution have independent sessions in one shared store.
The same client/worker protocol runs locally or across host-managed processes.
The parent Python object does not own the workflow's lifetime.

## Submit and wait

Create a `Runtime`, register a versioned environment builder in
`RegistryRuntimeFactory`, start its worker and open a root session with a
`RuntimeDescriptor`. [Example 27](../../../examples/27_dynamic_workflow.py)
contains executable setup. With that runtime and target:

```python
from power_loop.workflow import create_workflow

spec = {
    "name": "review",
    "budget": {"max_tokens": 50000},
    "timeout_s": 300,
    "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "draft", "input": "Write a short proposal.",
         "spec": {"name": "draft", "system_prompt": "Write clearly.", "tools": []}},
        {"type": "agent", "id": "review", "inputs_from": ["draft"],
         "spec": {"name": "review", "system_prompt": "Check the proposal.", "tools": []}},
    ]},
}
handle = await runtime.client.workflow(target, spec, message_id="review-1", actor=actor)
result = await handle.wait(deadline_ms=None)
print(result.status, result.usage)

# Equivalent convenience wrapper; the same request ID reuses the original admission.
workflow = create_workflow(spec, client=runtime.client, target=target,
                          actor=actor, message_id="review-1")
result = await workflow.run()
```

The handle contains `run_id`, `job_ref` and `input_ref`, never a local task or
completion callback. A repeated request ID with changed content is rejected.
`client.wait(handle.input_ref, actor=actor, deadline_ms=...)` also works after
rebuilding the client. Its result is the common Runtime `Result`, whose
`structured_output` contains the workflow projection. A wait deadline stops the
wait; explicit cancellation controls execution.

## Graph behavior

- `sequence` runs steps in order. `inputs_from` appends upstream results;
  `{{input}}`, `{{variable}}` and `{{node.key}}` render leaf input and system prompt.
- `parallel` runs branches with `max_concurrency`; `foreach` iterates literal
  `items` or `items_from`, with distinct nested iteration identities. Reference a
  foreach aggregate outside its body, not an individual body leaf.
- `branch` chooses a case from an earlier result. `when` conditionally skips a leaf.
  Structured fields require `output_schema` on their producing leaf.
- `loop` repeats its body until the declared condition or `max_iterations`.
  `continue_nodes` retains those node sessions across iterations. It starts a new
  node execution in that session and does not recreate a prior Python stack.
- `retry` creates another attempt for declared known failures. `fallback` and
  `continue_on_error` retain their explicit graph meanings. Container
  `on_error="halt"` stops its outstanding siblings; `"continue"` collects failures
  and lets other branches finish. Inspect both `status` and `ok`/`errors`.
- `continuation` can extend round-limit/context-checkpoint results in the same
  node job/session. The default gate reads remaining todos; `gate="always"` uses
  the declared continuation count. Interrupted or unknown external work is not
  automatically retried.

IDs are globally unique in a definition. References must point to earlier nodes
on the same dependency path, not parallel siblings. JSON and YAML compile to the
same spec. Invalid definitions report their problems before admission.

## Recovery, control and explicit continuation

The original start input and an already selected positive JobRef keep their original
generation result, including an interrupted old driver. Inspect the workflow job
to select the new generation for waiting or cancellation after takeover.

A node is keyed by run, node path, iteration path and attempt. Admission saves its
job, session, start input and journal reference in one transaction. Results and
parent inbox notifications also commit together. Only the driver updates its own
node journal entries while consuming those notices.

A new worker acquires the driver's session and advances the original workflow job
generation conditionally. It reads the frozen graph, reuses completed results and
waits for healthy original leaves. Leaf results and fees retain their original
owners. Missing information may produce `interrupted`/`unknown`; no unknown external
action is replayed to manufacture a result.

```python
await handle.cancel("user stopped the workflow")  # closes new subtree dispatch
# For one running node, use its exact JobRef:
await runtime.client.cancel(node_ref, actor=actor, scope="self")
await runtime.client.nudge(node_ref, "additional guidance", message_id="note-1", actor=actor)

# After the workflow has a terminal result, use its current generation.
continued = await handle.resume(generation=current_generation, node_id="draft",
    message="Revise the proposal", message_id="revision-1")
result = await continued.wait()
```

Explicit continuation resumes the selected node job in a new generation and
recomputes its dependent tail using new attempts. Unchanged branches reuse their
results. A node appearing in several foreach/loop iterations is ambiguous for
this graph-level API and is rejected; inspect its concrete jobs and send a new,
explicitly scoped task. Running leaves use nudge/steer or answer; graph-level
resume requires a terminal workflow. A cancelled generation is not restarted.

Token usage and the workflow deadline remain cumulative across automatic and
explicit continuations. Existing results can still be read after budget
exhaustion. Limits on calls and money use the Runtime's configured `BudgetConfig`
and original logical ancestry; the graph token budget is a soft dispatch limit,
so already running parallel calls may finish after it is reached. Unsettled fees
remain visible and prevent supervision closure.

## Workers, model tools and resources

For shared execution, configure a shared store, `SessionOwnershipCoordinator`
and `MessageTransport`, and set `shared_execution=True`. The default memory
adapters are single-process. All workers must resolve the frozen versioned
profiles. Hosts decide worker routing and process/container isolation; the SDK
has no per-leaf process launcher or private leaf database.

`register_workflow_tool(registry)` exposes create/validate/status/list/leaves/logs,
node nudge/steer and cancellation. `register_workflow_tools` exposes the same
admission path under create/validate/status tool names. Model creation returns a
receipt immediately, and the result reaches the original parent inbox.
`WorkflowToolPorts` supports spec loading, policy transforms, tool-grant limits
and bounded presentation observers. Observers do not decide completion.

Host resource profiles may provide `resources.workflow_file_io` with
`output_path(node_id, iteration=...)` and `render_ref(path, slice_expr)` for
`output_file` and `@@FILEREF:path@@` inputs. Workers need access to the same intended
resources; Python objects are not serialized in node requests. File adapter
errors fail explicitly. Required child tasks and user answers remain attached to
the leaf's own session/job.

The unreleased design 203 branch uses schema 15/protocol 42 and fresh stores.
Old parent-loop workflow APIs, local resume handles, completion callbacks,
`eager_wake`, detached/wait/sync flags and subprocess executors are removed.
