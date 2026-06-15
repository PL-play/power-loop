# power-loop

[![PyPI](https://img.shields.io/pypi/v/power-loop.svg)](https://pypi.org/project/power-loop/)
[![Python](https://img.shields.io/pypi/pyversions/power-loop.svg)](https://pypi.org/project/power-loop/)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)

[Documentation](docs/en/index.md) · [中文文档](docs/zh/index.md) · [Examples](examples/README.md) · [Changelog](CHANGELOG.md)

**The agent runtime that disappears into your app.** One class, one SQLite file, zero infrastructure — and you get durable multi-turn sessions, tool calling, sub-agents, deterministic multi-agent **workflows that resume across a crash**, durable timers, and process-level sandboxing. No service to run, no framework to adopt, no Redis/Postgres/queue to stand up.

```python
from power_loop import StatefulAgentLoop, create_llm_service_from_env

loop = StatefulAgentLoop(llm=create_llm_service_from_env(), db_path="app.db")
sid = loop.new_session()
await loop.send("Remember my favorite color is teal.", session_id=sid)
print((await loop.send("What's my favorite color?", session_id=sid)).final_text)
# → "Your favorite color is teal."   (persisted in app.db; survives restarts)
```

That's the whole setup. The conversation is already durable, resumable, and tool-capable.

---

## Why power-loop

Most "agent frameworks" ask you to build your app *inside* them. power-loop is the opposite: a **library** you embed. You keep your HTTP layer, your auth, your queues, your RAG, your UI, your deploy. It just runs the agent loop — well, and durably.

- 🪶 **Featherweight.** No `pydantic`, no LangChain, no graph DSL to learn. The runtime is a handful of files; the public surface is essentially one class. The SDK-free core has **zero runtime dependencies** (pure stdlib) — the OpenAI/Anthropic transport is pulled in only via the extra you install.
- 💾 **Zero infrastructure.** Sessions, timers, sub-agent trees, workflow journals, the shared blackboard — all in **one SQLite file**. Copy the file, you've copied the state. Scale by sharding files across processes.
- 🔌 **Provider-agnostic.** Any OpenAI-compatible endpoint or the native Anthropic Messages API, selected by env vars. Swap models per sub-agent or per workflow step.
- ⏱️ **Durable by default.** Crash mid-run and `resume()`. Agents schedule their own wake-ups with **durable timers** that survive restarts. Workflows **replay completed steps and re-run only the unfinished tail** after a process death.
- 🧩 **Composable from one loop to a fleet.** Start with `send()`. Add tools. Spawn sub-agents. Fan out a deterministic **workflow** (`sequence` / `parallel` / `foreach` / `branch`). Run each leaf in its **own process and DB** behind a sandbox. Same primitives all the way up.
- 🛡️ **Isolation seams where it counts.** Tool-level sandboxing via a `ShellBackend` (drop in gVisor/Docker for `bash`); process-level sandboxing via a `WorkerLauncher` (wrap a whole sub-agent worker per leaf). power-loop stays sandbox-agnostic; you choose the policy.
- 🔬 **Built to be observed.** Typed events for every stream chunk, tool call, round, and **individual LLM call** (`LLM_CALL_STARTED`/`LLM_CALL_COMPLETED` carry `call_id`/`attempt`/`duration_ms`/per-call usage), plus a real `AGENT_ERROR` channel on crash. Every event is `ts`+`seq` stamped and totally ordered. Per-run + per-session token accounting. Hard per-run token budgets. One-line JSON event logging (with secret-name redaction by default).
- ✅ **Real-LLM tested.** A dedicated `tests/real/` suite runs the library — workflows, resume, sandboxed subprocess agents, structured output, compaction — against a live model, not just mocks.

---

## Install

The core imports SDK-free; pick the transport extra you use (the vendor SDK is only
pulled in when you actually construct that provider):

```bash
pip install 'power-loop[openai]'      # any OpenAI-compatible endpoint
pip install 'power-loop[anthropic]'   # the native Anthropic Messages API
pip install 'power-loop[all]'         # both
```

Point it at any OpenAI-compatible endpoint (or `POWER_LOOP_PROVIDER=anthropic`):

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-...
POWER_LOOP_MODEL=gpt-4o-mini
```

Python 3.10+. See [Getting Started](docs/en/getting-started.md).

---

## What you get

| Capability | One-liner | Docs |
|---|---|---|
| **Stateful sessions** | Durable multi-turn memory + cross-process resume, backed by SQLite | [Sessions](docs/en/user-guide/sessions.md) |
| **Tool calling** | JSON-Schema-validated tools; built-in `bash`/file/search/skills presets | [Tools](docs/en/user-guide/tools.md) |
| **Sub-agents** | Delegate to a child loop via `AgentSpec` (own prompt/tools/model) | [Sub-agents](docs/en/user-guide/subagents.md) |
| **Dynamic workflows** | Declarative JSON DSL (`sequence`/`parallel`/`foreach`/`branch`) the LLM can author; deterministic engine | [feasibility](docs/dynamic-workflow-feasibility.md) |
| **Workflow resume** | Journals each step; after a crash, replays completed steps and re-runs only the tail | [Workflows](docs/en/user-guide/workflows.md) |
| **Subprocess executor** | Each workflow leaf in its own OS process + own DB; sandbox per leaf | [Sandboxing](docs/en/user-guide/sandboxing.md) |
| **Shared blackboard** | A scoped, durable coordination board multiple agents read/write | [Blackboard](docs/en/user-guide/blackboard.md) |
| **Durable timers** | Agents schedule their own wake-ups; survive restarts; one-shot or recurring | [examples/26](examples/26_timers.py) |
| **Hooks** | Veto/observe at every lifecycle point (round, tool, compaction, timer) | [Hooks](docs/en/user-guide/hooks.md) |
| **Typed events** | Streaming, audit, metrics — strongly typed payloads | [Events](docs/en/user-guide/events.md) |
| **Context compaction** | Auto-summarize old turns; never splits a tool-call pair | [Compaction](docs/en/user-guide/compaction.md) |
| **Retry / timeout / cancel** | Unified cancellation token; provider-aware retry | [Retry & Cancel](docs/en/user-guide/retry-cancel.md) |
| **Structured output** | `output_schema` → provider `response_format` → parsed & validated | [Structured](docs/en/user-guide/structured-output.md) |
| **Token budgets & usage** | Per-run cap (`budget_exceeded`), per-run + per-session accounting | [Budgets & usage ↓](#hard-token-budgets--usage-accounting-no-event-plumbing) |
| **Pluggable memory** | Cross-session recall via a `MemoryProvider` Protocol | [Memory](docs/en/user-guide/memory.md) |
| **Recall compacted detail** | `recall_compacted` tool pulls back exact turns compaction folded out | [Compaction](docs/en/user-guide/compaction.md) |
| **Stable error codes** | Every `PowerLoopError` carries a machine-readable `code` (`llm.timeout`, `session.pending`, `tool.not_found`, …); branch on `exc.code` | [API: error codes](docs/en/api/index.md#error-codes) |
| **Crash recovery** | `heal_pending` / `resume` / `abort_pending` for runs killed mid tool-call | [Pending recovery](docs/en/user-guide/sessions.md#pending-recovery) |

---

## Highlights

### Deterministic multi-agent workflows — that the model can author, and that survive a crash

Sub-agent delegation is *model-driven* ("go do this"). When you want **code-driven, deterministic** orchestration — fan out over a list, branch on a result, run a pipeline — describe it as a `WorkflowSpec` and let the engine interpret it. The only LLM calls are the leaves; `sequence`/`parallel`/`foreach`/`branch` are plain code.

```python
from power_loop.workflow import create_workflow

spec = {
    "name": "research", "input": "the Japanese tea ceremony",
    "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "plan",
         "spec": {"name": "planner", "system_prompt": "Break the topic into 3 subtopics."},
         "output_schema": {"name": "Plan", "schema": {"type": "object", "required": ["subtopics"],
            "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "foreach", "id": "research", "items_from": "plan.subtopics", "as": "t",
         "parallel": True, "max_concurrency": 3,
         "body": {"type": "agent", "id": "r",
                  "spec": {"name": "researcher", "system_prompt": "Write 2 sentences on {{t}}."},
                  "input": "Subtopic: {{t}}"}},
        {"type": "agent", "id": "write",
         "spec": {"name": "writer", "system_prompt": "Synthesize the notes."},
         "inputs_from": ["research"]},
    ]},
}
result = await create_workflow(spec, parent_loop=loop).run()
```

Validated on creation (every problem reported at once — perfect for an LLM to repair). Run it **detached**, and the parent agent is woken on completion via a durable timer. Crash halfway through the fan-out? `resume_run(loop, parent_sid, run_id)` replays the planner + finished researchers from the journal and re-runs only what's left. Register it as a tool and the agent builds and submits workflows itself.

### Run untrusted sub-agents in real sandboxes — without sandboxing the parent

The default executor runs leaves in-process. The **subprocess executor** runs each leaf in its own OS process against its own SQLite file (so the one-writer-per-file rule holds trivially), and a `WorkerLauncher` lets you wrap that process — per leaf, by inspecting its granted tools — in gVisor / Docker / firejail. A safe orchestrator can spawn a `bash`-wielding child that runs fully confined:

```python
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, create_workflow

ex = SubprocessExecutor(
    bootstrap=WorkerBootstrap(llm_from_env=True, tool_preset="core"),
    launcher=my_gvisor_launcher,   # wraps the worker command per leaf; fail-closed
    timeout_s=120,
)
await create_workflow(spec, parent_loop=loop, executor=ex).run()
```

A crashed or killed worker becomes a failed leaf — which `resume` re-runs. Each leaf's private DB is kept for inspection (or GC'd).

### Durable timers — the agent wakes itself up

A timer is **data, not a task**: a row in the store that survives restarts and cascade-deletes with its session. Give the agent the `schedule_wakeup` tool and it can say "check back in 10 minutes" — for real.

```python
from power_loop import TimerRunner
loop.schedule_timer(sid, delay_s=600, note="check the export job")
loop.schedule_timer(sid, delay_s=60, note="heartbeat", interval_s=3600)  # recurring
await TimerRunner(loop).start()   # fires due timers as normal follow-ups
```

### Structured output you can branch on

```python
from power_loop import AgentLoopConfig, StructuredOutputSpec, parse_structured

schema = {"type": "object", "required": ["label"], "properties": {"label": {"type": "string"}}}
fmt = StructuredOutputSpec(name="Triage", schema=schema).to_openai_response_format()
loop = StatefulAgentLoop(llm=llm, db_path="app.db",
                         config=AgentLoopConfig(response_format=fmt))   # also per sub-agent / workflow node
res = await loop.send("Classify: 'my card was charged twice'", session_id=sid)
triage = parse_structured(res.final_text, schema=schema)   # {"label": "billing"}
```

(In a workflow, a node's `output_schema` does this automatically so downstream `items_from` / `branch.on` can read the parsed payload — see the workflow example above.)

### Hard token budgets + usage accounting, no event plumbing

```python
res = await loop.send("…", session_id=sid)          # config: max_tokens_per_run=50_000
res.usage      # {"prompt_tokens":…, "completion_tokens":…, "total_tokens":…, "calls": 2}
res.status     # "budget_exceeded" if the per-run cap was hit (stops before the next call)
loop.get_session_stats(sid)   # cumulative, persisted per session
```

### Crash recovery

```python
# a run killed mid tool-call leaves unresolved tool_calls; opt into self-healing:
res = await loop.send("…", session_id=sid, heal_pending=True)
# or recover explicitly: loop.resume(sid) / loop.abort_pending(sid)
```

More in [`examples/`](examples/README.md) — 37 runnable programs from `00_hello_world.py` to the full chatbot, dynamic workflows, the memory/compaction/recall demos (`31`–`33`), and the durability/scaling/observability demos (`34`–`36`).

---

## Honest scope

power-loop **orchestrates**; it does not, by itself, **isolate**. The built-in `bash`/file tools run in-process and inherit the host environment — convenient for trusted, local use, **not a security boundary**. For untrusted/model-authored commands, inject a sandbox via the `ShellBackend` seam (tool-level) or run leaves through the `SubprocessExecutor` + `WorkerLauncher` (process-level). Keep secrets in your orchestrator.

**One store file = one writer process.** Per-session ordering is an in-process `asyncio.Lock`; two processes calling `send()` on the same session bypass it. Run one process per store file (shard sessions across files), or put your own distributed lock in front. A session only advances while a `send()`/`resume()` or a timer firing is in flight.

---

## Public API & stability

Stable imports are re-exported from `power_loop` (`StatefulAgentLoop`, `AgentLoopConfig`, `StatefulResult`, `ToolDefinition`, `ToolRegistry`, …). Tiers:

| Tier | Meaning |
|---|---|
| **Stable** | Backward compatible across minor releases. Listed in `power_loop.STABLE_API`. |
| **Provisional** | Re-exported from the top-level package during 0.x; may change. |
| **Internal** | `power_loop.core.*` etc.; no compatibility promise. |

See the [API reference](docs/en/api/index.md).

## Development

```bash
git clone https://github.com/PL-play/power-loop.git && cd power-loop
pip install -e ".[dev]"
ruff check .
pytest -q --no-real     # unit suite; drop --no-real to run the live-LLM suite
```

## Links

[Docs](docs/README.md) · [Architecture](docs/en/architecture.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)
