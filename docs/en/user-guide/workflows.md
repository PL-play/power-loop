# Dynamic Workflows

[中文](../../zh/user-guide/workflows.md) | [User Guide](../index.md)

`power_loop.workflow` is an **optional** submodule for *code-driven* deterministic multi-agent orchestration: a declarative `WorkflowSpec` JSON whose leaves are ordinary sub-agents, interpreted by a deterministic engine. This complements — it does not replace — the *model-driven* ad-hoc delegation of `spawn_agent` / `run_agent` ([Sub-agents](subagents.md)).

Use a workflow when you (the developer) know the control flow up front (a pipeline, a fan-out, a branch). Use ad-hoc delegation when the *model* should decide whom to call.

## The spec

A `WorkflowSpec` is a tree of nodes. Five node types:

| Node | Does |
|---|---|
| `agent` | run one sub-agent (a leaf); keyed by `id` |
| `sequence` | run `steps` in order |
| `parallel` | run `branches` concurrently (barrier fan-in) |
| `foreach` | map a `body` over a runtime list (`items` or `items_from`), optionally concurrent |
| `branch` | pick a `case` from a prior agent's output |

```python
SPEC = {
    "name": "research_and_summarize",
    "input": "the Japanese tea ceremony",
    "budget": {"max_tokens": 200_000, "stop_at_remaining_pct": 5},
    "root": {
        "type": "sequence",
        "steps": [
            {"type": "agent", "id": "plan",
             "spec": {"name": "planner",
                      "system_prompt": 'Reply ONLY JSON: {"subtopics": ["...","...","..."]}'},
             "input": "Topic: {{input}}",
             "output_schema": {"name": "Plan", "schema": {
                 "type": "object", "required": ["subtopics"],
                 "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "research",
             "items_from": "plan.subtopics", "as": "subtopic",
             "parallel": True, "max_concurrency": 3,
             "body": {"type": "agent", "id": "researcher",
                      "spec": {"name": "researcher",
                               "system_prompt": "Write 2-3 factual sentences."},
                      "input": "Subtopic: {{subtopic}}"}},
            {"type": "agent", "id": "synthesize",
             "spec": {"name": "writer", "system_prompt": "Synthesize into one paragraph."},
             "inputs_from": ["research"]},
        ],
    },
}
```

Data flow between nodes:

- `{{input}}` / `{{var}}` — template substitution in a node's `input`.
- `output_schema` — make a leaf emit structured JSON (validated); downstream nodes read it.
- `items_from: "plan.subtopics"` — a `foreach` reads a key out of a prior agent's parsed payload; `inputs_from: ["research"]` feeds prior results in as text.

## Running it

```python
from power_loop import StatefulAgentLoop, AgentLoopConfig
from power_loop.workflow import create_workflow

loop = StatefulAgentLoop(llm=make_llm(), db_path=":memory:",
                         config=AgentLoopConfig(system_prompt="orchestrator"))

wf = create_workflow(SPEC, parent_loop=loop)   # validates on creation
result = await wf.run()                         # deterministic, in-process

print(result.status)                            # completed | failed | budget_exceeded | cancelled
print(result.results["plan"].payload)           # parsed structured output
print(result.results["synthesize"].text)        # final sub-agent text
print(result.usage["total_tokens"])
```

`create_workflow` validates the spec immediately; an invalid spec raises `WorkflowSpecError` listing **every** problem at once (so an LLM that authored it can repair in one shot). `result.results[node_id]` is an `AgentResult` (`status`, `text`, `payload`, `usage`, `session_id`, `db_path`).

See [example 27](../../../examples/27_dynamic_workflow.py).

### Let the model author workflows

Register the meta-tools so the main agent can emit a spec and run it itself:

```python
from power_loop.workflow import register_workflow_tools
register_workflow_tools(registry)   # adds create_workflow + workflow_status
```

(Don't grant these to a workflow's own leaves — nested runs are refused in this tier.)

## Executors

The engine runs leaves through an `Executor`. Two ship in the box.

### In-process (default)

`InProcessExecutor` runs each leaf in the parent process via `run_agent_spec`. Fast, simple, shares the parent's store. This is what `wf.run()` uses by default.

### Out-of-process (`SubprocessExecutor`)

Run **each leaf in its own OS process against its own SQLite file**, so the one-writer-per-file rule holds trivially and a crashed/killed/timed-out leaf can't corrupt the orchestrator or its siblings.

```python
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

executor = SubprocessExecutor(
    bootstrap=WorkerBootstrap(llm_from_env=True),  # child rebuilds the provider from env
    runs_dir="/tmp/wf_runs",
    timeout_s=120,
    delete_on_success=False,   # keep each leaf db for inspection (GC via reap_runs)
)
result = await WorkflowEngine(loop, executor=executor, run_id="run1").run(
    WorkflowSpec.from_json(SPEC)
)
```

The worker rebuilds its dependencies from **config alone** (`WorkerBootstrap`) — nothing live crosses the process boundary. Cancel = SIGTERM→SIGKILL; timeout/crash → `failed` (re-runnable on resume). Each leaf's db is left on disk for post-hoc inspection; GC with `reap_runs(runs_dir, older_than_s=…)` or `cleanup_run(runs_dir, run_id)`. To confine each leaf in a sandbox, inject a [`WorkerLauncher`](sandboxing.md#workerlauncher--sandbox-a-workflow-leaf-process). See [example 30](../../../examples/30_subprocess_isolation.py).

## Detached execution + completion wake

Run a workflow in the background and let it wake the parent agent when done (the parent `pass_turn`s out and is re-woken via a durable timer):

```python
from power_loop import TimerRunner
from power_loop.workflow import create_workflow, register_wake_guard, get_workflow, list_workflows

register_wake_guard(loop)                # dedupe the at-least-once wake timer
await TimerRunner(loop).start()          # REQUIRED: delivers the wake (see Timers)

wf = create_workflow(SPEC, parent_loop=loop, parent_session_id=parent_sid)
handle = await wf.start(detached=True)   # returns immediately
# inspect any time:
get_workflow(loop, parent_sid, handle.run_id, detail=True)   # status + per-step usage
list_workflows(loop, parent_sid)
```

The host must run a `TimerRunner` **and** call `register_wake_guard(loop)`, or the parent never wakes. Progress is published as `SYSTEM_LOG` events with `source="workflow"`.

## Resume across a process restart

A detached run journals its spec and each step's output. If the process dies mid-run, point a fresh loop (same `db_path`) at the run id and resume — completed steps are **replayed** (their sub-agent is not called again), only the unfinished tail re-runs:

```python
from power_loop.workflow import resume_run, resume_detached

result = await resume_run(loop, parent_sid, run_id)        # sync
# or, to resume in the background and wake the parent:
handle = await resume_detached(loop, parent_sid, run_id)
```

Re-supply `executor=` / `budget=` if the original used non-default ones — only *data* is journaled; runtime objects come from the live process. Each leaf gets a stable `metadata["idempotency_key"] = f"{run_id}:{node_id}"` so side-effecting tools can dedupe a re-run. `foreach` is atomic: it replays via its journaled aggregate or re-runs the whole fan-out.

## See also

- [Sub-agents](subagents.md) — the model-driven alternative
- [Sandboxing](sandboxing.md) — confining out-of-process leaves
- [Timers](timers.md) — the wake mechanism behind detached runs
- [Blackboard](blackboard.md) — coordinating peer agents
