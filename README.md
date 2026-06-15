# power-loop

[![PyPI](https://img.shields.io/pypi/v/power-loop.svg)](https://pypi.org/project/power-loop/)
[![Python](https://img.shields.io/pypi/pyversions/power-loop.svg)](https://pypi.org/project/power-loop/)
[![CI](https://github.com/PL-play/power-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/PL-play/power-loop/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)

**English** · [中文](README.zh.md) · [Docs](docs/en/index.md) · [Examples](examples/README.md) · [Changelog](CHANGELOG.md)

> **The agent runtime that disappears into your app.** One class, one SQLite file, zero infrastructure — and you get durable multi-turn sessions, tool calling, sub-agents, deterministic multi-agent **workflows that resume across a crash**, durable timers, and process-level sandboxing. No service to run, no framework to adopt, no Redis/Postgres/queue to stand up.

```python
from power_loop import StatefulAgentLoop, create_llm_service_from_env

loop = StatefulAgentLoop(llm=create_llm_service_from_env(), db_path="app.db")
sid = loop.new_session()
await loop.send("Remember my favorite color is teal.", session_id=sid)
print((await loop.send("What's my favorite color?", session_id=sid)).final_text)
# → "Your favorite color is teal."   (persisted in app.db; survives restarts)
```

That's the whole setup. The conversation is already durable, resumable, and tool-capable.

```bash
pip install 'power-loop[openai]'      # or [anthropic], or [all]
```

> **1.0 — stable.** The public API is frozen under SemVer (a breaking change requires a major bump), machine-enforced by a baseline guard in CI. The **core has zero runtime dependencies** (pure stdlib; verified by a CI job that imports it with nothing else installed). See [Stability](#stability--semver) and the [honest caveats](#honest-scope) — a young, single-maintainer project says so plainly.

---

## Start here

| You are… | Go to |
|---|---|
| 🚀 **New** — show me the 5-minute version | [Getting Started](docs/en/getting-started.md) |
| 🛠️ **Learning by building** | [Tutorials](docs/en/tutorials/index.md) — chatbot · tools · human-in-the-loop · multi-agent |
| 🧩 **Browsing runnable code** | [39 examples](examples/README.md) — `00_hello_world.py` → full chatbot |
| 📚 **Looking something up** | [User Guide](docs/en/user-guide/index.md) · [API reference](docs/en/api/index.md) |
| 🤔 **Deciding if it fits** | [How it compares](#how-it-compares) · [Honest scope](#honest-scope) |

**Find your way by goal:** persist across restarts → [Sessions](docs/en/user-guide/sessions.md) · give it tools → [Tools](docs/en/user-guide/tools.md) / [Extending](docs/en/user-guide/extending-tools.md) · multi-agent → [Workflows](docs/en/user-guide/workflows.md) · sandbox untrusted code → [Sandboxing](docs/en/user-guide/sandboxing.md) · run long-term (backup/retention) → [Sessions](docs/en/user-guide/sessions.md) · monitor (logs/metrics/traces) → [Observability](docs/en/user-guide/observability.md) · measure & scale → [Scaling](docs/en/user-guide/scaling.md) · connect an MCP server → [Extending](docs/en/user-guide/extending-tools.md) · survive crashes → [Pending recovery](docs/en/user-guide/sessions.md#pending-recovery).

---

## Why power-loop

Most "agent frameworks" ask you to build your app *inside* them. power-loop is the opposite: a **library you embed**. You keep your HTTP layer, your auth, your queues, your RAG, your UI, your deploy. It just runs the agent loop — well, and durably.

- 🪶 **Featherweight & zero-dependency.** No `pydantic`, no LangChain, no graph DSL. A compact, pure-stdlib core (~17k lines) whose public surface is essentially one class — and **zero runtime dependencies**. The OpenAI/Anthropic transport is pulled in only by the extra you install.
- 💾 **Zero infrastructure.** Sessions, timers, sub-agent trees, workflow journals, the shared blackboard — all in **one SQLite file**. Copy the file, you've copied the state. Scale by sharding files across processes.
- ⏱️ **Durable by default.** Crash mid-run and `resume()`. Agents schedule their own **durable timers** that survive restarts. Workflows **replay finished steps and re-run only the unfinished tail** after a process death. The store **survives version upgrades** (a real schema-migration ladder) and can be **pruned, VACUUMed, and exported**.
- 🧩 **Composable from one loop to a fleet.** Start with `send()`. Add tools. Spawn sub-agents. Fan out a deterministic **workflow** (`sequence`/`parallel`/`foreach`/`branch`). Run each leaf in its **own process and DB** behind a sandbox. Same primitives all the way up.
- 🛡️ **Isolation seams where it counts.** Tool-level sandboxing via a `ShellBackend` (drop in gVisor/Docker for `bash`); process-level via a `WorkerLauncher` (wrap a whole sub-agent worker per leaf). power-loop stays sandbox-agnostic; you choose the policy.
- 🔬 **Built to be observed.** Typed events for every stream chunk, tool call, round, and **individual LLM call** — each `seq`-ordered + monotonic-clock stamped. Pluggable sinks behind extras: durable **JSONL** (with `replay`), **Prometheus/StatsD** metrics, an **OpenTelemetry** span tree. Per-run + per-session token accounting and hard per-run budgets.
- 🔌 **Open ecosystem.** Provider-agnostic (any OpenAI-compatible endpoint or native Anthropic, by env var). Bring any tool via the `ToolRegistry`, or connect a **Model Context Protocol** server with one adapter.
- ✅ **Real-LLM tested.** A dedicated `tests/real/` suite runs the library — workflows, resume, sandboxed subprocess agents, structured output, compaction, a live MCP server — against a real model, not just mocks.

---

## What you get

| Capability | One-liner | Docs |
|---|---|---|
| **Stateful sessions** | Durable multi-turn memory + cross-process resume, in one SQLite file | [Sessions](docs/en/user-guide/sessions.md) |
| **Tool calling** | JSON-Schema-validated tools; built-in `bash`/file/search/skills presets | [Tools](docs/en/user-guide/tools.md) · [Extending](docs/en/user-guide/extending-tools.md) |
| **Sub-agents** | Delegate to a child loop via `AgentSpec` (own prompt/tools/model) | [Sub-agents](docs/en/user-guide/subagents.md) |
| **Dynamic workflows** | JSON DSL (`sequence`/`parallel`/`foreach`/`branch`) the LLM can author; deterministic engine | [Workflows](docs/en/user-guide/workflows.md) |
| **Workflow resume** | Journals each step; after a crash, replays completed steps and re-runs only the tail | [Workflows](docs/en/user-guide/workflows.md) |
| **Process sandboxing** | Each workflow leaf in its own OS process + own DB; wrap each in gVisor/Docker per leaf | [Sandboxing](docs/en/user-guide/sandboxing.md) |
| **Durable timers** | Agents schedule their own wake-ups; survive restarts; one-shot or recurring | [Timers](docs/en/user-guide/timers.md) |
| **Context compaction** | Auto-summarize old turns (never splits a tool-call pair); `recall_compacted` to pull originals back | [Compaction](docs/en/user-guide/compaction.md) |
| **Durability ops** | Schema-migration ladder, retention/prune, VACUUM, `export_session`/`import_session`, graceful `aclose()` | [Sessions](docs/en/user-guide/sessions.md) |
| **Observability** | Typed `seq`-ordered events → durable JSONL + `replay`, Prometheus/StatsD metrics, OpenTelemetry spans; opt-in backpressure | [Observability](docs/en/user-guide/observability.md) |
| **Scale, measured** | A `bench/` harness with real numbers; opt-in read-only WAL connection pool; honest single-process ceiling | [Scaling](docs/en/user-guide/scaling.md) |
| **MCP tools** | Surface a Model Context Protocol server's tools as power-loop tools | [Extending](docs/en/user-guide/extending-tools.md) |
| **Hooks & events** | Veto/observe at every lifecycle point; strongly-typed event payloads | [Hooks](docs/en/user-guide/hooks.md) · [Events](docs/en/user-guide/events.md) |
| **Structured output** | `output_schema` → provider `response_format` → parsed & validated | [Structured](docs/en/user-guide/structured-output.md) |
| **Pluggable memory** | Cross-session recall via a `MemoryProvider` Protocol | [Memory](docs/en/user-guide/memory.md) |
| **Retry / cancel / budgets** | Provider-aware retry, a unified cancellation token, hard per-run token caps | [Retry & Cancel](docs/en/user-guide/retry-cancel.md) |
| **Stable error codes** | Every `PowerLoopError` carries a frozen machine-readable `code` — branch on `exc.code` | [API: error codes](docs/en/api/index.md#error-codes) |
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

Validated on creation (every problem reported at once — ideal for an LLM to repair). Run it **detached** and the parent agent is woken on completion via a durable timer. Crash halfway through the fan-out? `resume_run(loop, parent_sid, run_id)` replays the planner + finished researchers from the journal and re-runs only what's left. Register it as a tool and the agent builds and submits workflows itself.

### Run untrusted sub-agents in real sandboxes — without sandboxing the parent

The default executor runs leaves in-process. The **subprocess executor** runs each leaf in its own OS process against its own SQLite file (the one-writer-per-file rule holds trivially), and a `WorkerLauncher` wraps that process — per leaf, by inspecting its granted tools — in gVisor / Docker / firejail.

```python
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, create_workflow

ex = SubprocessExecutor(
    bootstrap=WorkerBootstrap(llm_from_env=True, tool_preset="core"),
    launcher=my_gvisor_launcher,   # wraps the worker command per leaf; fail-closed
    timeout_s=120,
)
await create_workflow(spec, parent_loop=loop, executor=ex).run()
```

### Durable, operable storage — the part most "agent libraries" skip

The on-disk store is the product, so it's built to run for the long haul:

```python
store.export_session(sid)                 # full session → a JSON archive (incl. compacted turns)
store.prune_compacted_messages(sid)       # opt-in retention of folded-out originals
store.vacuum(); store.checkpoint()        # reclaim disk
async with StatefulAgentLoop(...) as loop:  # graceful aclose(): drain in-flight sends, then close
    ...
```

It survives upgrades (a `PRAGMA user_version` migration ladder refuses a newer-than-code DB rather than corrupting it), and reads can run on an opt-in read-only WAL pool so they don't queue behind the writer.

### Observe everything, export anywhere

```python
from power_loop.contrib.jsonl_sink import attach_jsonl_sink, replay
from power_loop.contrib.metrics_sink import attach_metrics_sink, PrometheusBackend

attach_jsonl_sink(bus, "events.jsonl")        # durable; replay("events.jsonl") later
attach_metrics_sink(bus, PrometheusBackend()) # power-loop[prometheus] · or StatsD, or OpenTelemetry spans
```

Every event carries a process-wide `seq` and a monotonic clock, so streams totally-order and reconstruct. Sync subscribers run inline by default; opt into a bounded-queue background dispatcher when a sink might block.

### Connect a Model Context Protocol server

```python
from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools   # power-loop[mcp]

client = await StdioMCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]).connect()
await register_mcp_tools(registry, client, prefix="fs.")   # MCP tools → power-loop ToolDefinitions
```

The seam is a tiny `MCPToolSource` Protocol, so the `mcp` SDK is optional and any client works.

> More: hard token budgets, structured output, crash recovery, memory, the blackboard — see [`examples/`](examples/README.md) (39 runnable programs) and the [docs](docs/en/index.md).

---

## How it compares

power-loop is a **kernel**, not a platform — that's the whole trade-off.

- **vs. LangChain / LangGraph / LlamaIndex / CrewAI / AutoGen** — those are batteries-included frameworks with large ecosystems (connectors, vector stores, integrations) and heavy dependency trees. power-loop deliberately ships **none of that**: a compact (~17k-line) pure-stdlib core with zero dependencies, and you bring your own tools (or an MCP server). You get durable sessions, crash-resumable workflows, and real sandbox seams out of the box; you do **not** get a bundled RAG stack or 100 connectors.
- **Choose power-loop** when you want to *embed* an agent in an existing app, keep your dependency surface tiny, and care about durability + isolation + a stable contract.
- **Choose a framework** when you want batteries included, a big integration catalog, and don't mind the weight.

Honestly: power-loop is **behind on ecosystem breadth** (integrations, community, age) and **ahead on embeddability, durability, and a machine-guarded stable API**. Pick accordingly.

---

## Install & configure

```bash
pip install 'power-loop[openai]'      # any OpenAI-compatible endpoint
pip install 'power-loop[anthropic]'   # native Anthropic Messages API
pip install 'power-loop[all]'         # both transports + skills/pdf/observability/mcp extras
```

Point it at any OpenAI-compatible endpoint (or `POWER_LOOP_PROVIDER=anthropic`):

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-...
POWER_LOOP_MODEL=gpt-4o-mini
```

Python 3.10+. See [Getting Started](docs/en/getting-started.md). Optional extras: `skills`, `pdf`, `prometheus`, `statsd`, `otel`, `mcp`.

---

## Stability & SemVer

As of **1.0**, the **STABLE** API (listed in `power_loop.STABLE_API`) is under SemVer: a breaking change requires a major bump (`2.0.0`), enforced by a frozen-baseline test in CI — including the flagship `StatefulAgentLoop` *and the LLM contract needed to construct it* (so you can build, use, and implement a custom provider from STABLE-only symbols). Error `.code` strings are frozen too.

| Tier | Meaning |
|---|---|
| **Stable** | Backward-compatible within a major version; in `power_loop.STABLE_API`. |
| **Provisional** | Re-exported from the top level; may change in a future minor. |
| **Internal** | `power_loop.core.*` etc.; no compatibility promise. |

See the [API reference](docs/en/api/index.md).

---

## Honest scope

power-loop **orchestrates; it does not, by itself, isolate.** The built-in `bash`/file tools run in-process and inherit the host environment — convenient for trusted, local use, **not a security boundary**. For untrusted/model-authored commands, inject a sandbox via the `ShellBackend` seam (tool-level) or run leaves through `SubprocessExecutor` + `WorkerLauncher` (process-level). Keep secrets in your orchestrator. See [SECURITY.md](SECURITY.md).

**One store file = one writer process.** Per-session ordering is an in-process `asyncio.Lock`; two processes calling `send()` on the same session bypass it. Run one process per store file (shard sessions across files) — the [scaling guide](docs/en/user-guide/scaling.md) has measured numbers and the multi-process pattern. Multi-writer horizontal scale is out of scope.

**Maturity.** A 1.0 tag here is a confidence statement about the **API/durability contract** — not a claim of years of field-hardening. power-loop is young, primarily a single maintainer, with limited public production track record. The contract is machine-guarded and the project is MIT + forkable; weigh the bus factor for your use.

---

## Project & links

- **Used by:** DeepTalk — the agent runtime for a 1-on-1 relationship-IM product's in-conversation agents. *(Using it in production? PR a line here.)*
- **Develop:** `pip install -e ".[dev]"` · `ruff check .` · `pytest -q --no-real` (drop `--no-real` for the live-LLM suite).
- [Docs](docs/en/index.md) · [Architecture](docs/en/architecture.md) · [Roadmap to 1.0](ROADMAP_1.0.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)
