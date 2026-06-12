# power-loop

[Documentation](docs/en/index.md) | [中文文档](docs/zh/index.md) | [Examples](examples/README.md) | [Changelog](CHANGELOG.md)

Embeddable, stateful agent execution for Python.

power-loop gives application code one small interface, `StatefulAgentLoop`, and handles the repetitive agent runtime work around it: multi-turn LLM loops, tool calls, hooks, events, context compaction, sub-agents, retry/cancel, structured output, memory, and SQLite-backed session persistence.

It is a library, not a service or a full application framework. You keep ownership of product logic, HTTP APIs, auth, queues, RAG, UI, and deployment.

### Scope: orchestration, not isolation

power-loop **orchestrates** the agent loop; it does **not sandbox** tool execution. The
built-in `bash` / file tools run **in-process** (a `subprocess` shell inheriting the host
environment) — convenient for trusted, local use, but **not a security boundary**. If your
agent runs model-authored or otherwise untrusted commands, run them in your own sandbox
(container / gVisor / microVM) and inject it via the `ShellBackend` seam
(`runtime.exec_backend`); power-loop launches the persistent shell through your backend.
Keep secrets in your orchestrator — the loop does not scrub the tool environment for you.

**One store file = one process.** Per-session serialization is an in-process
`asyncio.Lock`; the SQLite store itself happily opens from multiple processes,
but two processes calling `send()` on the same session bypass all ordering
guarantees and can interleave histories. Run one process per store file (scale
by sharding sessions across processes/files), or put your own distributed lock
in front.

A session still only runs while a `send()` / `resume()` call is in flight — but since
0.11 there are **durable timers** ("wake this session at T with this note"): rows in the
session store, created by the agent (`schedule_wakeup` tool) or the host
(`loop.schedule_timer`), fired by a `TimerRunner` you explicitly start (or by your own
scheduler polling `store.due_timers()`). No runner running = nothing fires.

## Install

```bash
pip install power-loop
```

For local development:

```bash
git clone https://github.com/PL-play/power-loop.git
cd power-loop
pip install -e ".[dev]"
```

Python 3.10+ is required.

## Quick Example

```python
import asyncio

from power_loop import AgentLoopConfig, StatefulAgentLoop, create_llm_service_from_env


async def main() -> None:
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm,
        db_path="./power_loop_sessions.db",
        config=AgentLoopConfig(
            system_prompt="You are a concise assistant.",
            max_rounds=4,
        ),
    )

    sid = loop.new_session(metadata={"user_id": "demo"})
    first = await loop.send("My favorite color is teal.", session_id=sid)
    second = await loop.send("What is my favorite color?", session_id=sid)

    print(second.final_text)


asyncio.run(main())
```

Configure any OpenAI-compatible endpoint with environment variables:

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-...
POWER_LOOP_MODEL=gpt-4o-mini
```

See [Getting Started](docs/en/getting-started.md) for the complete first run.

## What It Provides

| Capability | Where to read more |
|---|---|
| Stateful sessions and cross-process resume | [Sessions](docs/en/user-guide/sessions.md) |
| Tool calling with JSON Schema validation | [Tools](docs/en/user-guide/tools.md) |
| Lifecycle hooks for control flow | [Hooks](docs/en/user-guide/hooks.md) |
| Typed events for streaming, audit, and metrics | [Events](docs/en/user-guide/events.md) |
| Context compaction | [Compaction](docs/en/user-guide/compaction.md) |
| Sub-agents with `AgentSpec` | [Sub-agents](docs/en/user-guide/subagents.md) |
| Retry, timeout, and cancellation | [Retry & Cancel](docs/en/user-guide/retry-cancel.md) |
| Structured JSON output | [Structured Output](docs/en/user-guide/structured-output.md) |
| Pluggable cross-session memory | [Memory](docs/en/user-guide/memory.md) |
| Provider configuration | [Providers](docs/en/user-guide/providers.md) |

### Per-call overrides

Build one loop and reuse it across callers; restrict tools or swap the system
prompt **per `send`** without rebuilding (the model only *sees* the allowed
tools). Ideal for multi-tenant hosts.

```python
# loop registered with all tools; this run exposes only "get_weather"
await loop.send("…", session_id=sid, tools=["get_weather"])

# per-run system prompt override (precedence: per-call > session > config)
await loop.send("…", session_id=sid, system_prompt="You are a terse bot.")
```

The same overrides are available on `send_sync()`. When `follow_up()` is idle
and falls back to a new send, it accepts them too. A follow-up queued into an
already running call keeps that call's active tool and prompt policy.

For a multi-tenant host that reuses one registry across workspaces, build an
**unbound** registry and supply the workspace at invocation time:

```python
from power_loop import RuntimeEnv, create_default_tool_registry, runtime_env_context

registry = create_default_tool_registry(preset="core", bind=False)
with runtime_env_context(RuntimeEnv(workspace_dir=tenant_workspace)):
    result = await registry.invoke_async("read_file", {"path": "README.md"})
```

See [`examples/23_per_send_overrides.py`](examples/23_per_send_overrides.py).

### Token usage accounting

Every `send()` returns the run's cumulative token usage — summed over all LLM
calls of that run (tool loops make several) — so cost accounting needs no event
plumbing:

```python
res = await loop.send("…", session_id=sid)
res.usage
# {"prompt_tokens": 1234, "completion_tokens": 56, "cache_read_tokens": 0,
#  "reasoning_tokens": 0, "total_tokens": 1290, "calls": 2}
```

For per-call, real-time metering subscribe to the `usage_updated` event (one
per LLM call, tagged with `session_id`). See
[`examples/25_token_usage.py`](examples/25_token_usage.py).

**Budget guardrail** — cap real provider tokens per run (rounds are cheap,
tokens are money):

```python
config = AgentLoopConfig(max_rounds=24, max_tokens_per_run=50_000)
res = await loop.send("…", session_id=sid)
res.status  # "budget_exceeded" when the cap is hit
```

Checked at round boundaries: the round that crosses the budget finishes
cleanly (no dangling tool_calls), then the loop stops without paying for the
next LLM call. A `status_changed` event with `kind="budget_exceeded"` fires.

**Session statistics** — cumulative accounting persisted in the store,
bumped once per finished send:

```python
stats = loop.get_session_stats(sid)
# SessionStatsRow(sends=12, rounds=29, llm_calls=31, tool_calls=18,
#                 prompt_tokens=…, completion_tokens=…, total_tokens=…,
#                 first_send_at=…, last_send_at=…)
loop.list_session_stats()  # every session, most recently active first
```

**Structured event logging** — one JSON line per event, stdlib-only:

```python
from power_loop.contrib.logging_sink import attach_logging_sink

bus = AgentEventBus(suppress_subscriber_errors=True)
attach_logging_sink(bus)   # or events={AgentEventType.USAGE_UPDATED}
loop = StatefulAgentLoop(llm=llm, event_bus=bus, ...)
```

### Crash recovery: `heal_pending`

A run killed mid tool-call leaves the session with unresolved `tool_calls`;
the next `send()` raises `SessionPendingError` (the message protocol forbids
continuing). Orchestrators whose runs can legitimately die (human interrupts,
process restarts) can opt into self-healing:

```python
res = await loop.send("…", session_id=sid, heal_pending=True)
# stale tool_calls are aborted with synthetic <aborted> results, then the
# send proceeds. Default remains raise — healing discards in-flight work.
```

Or recover explicitly with `resume(sid)` / `abort_pending(sid)`.

### Durable timers

A timer is **data, not a task**: it lives in the session store, survives
restarts, and cascade-deletes with its session. Firing is a normal message —
`follow_up` delivers it (idle session → a regular send; mid-run → injected at
the next round boundary), so there is exactly one path into a conversation.

One-shot vs recurring is **declared at creation**: `every_seconds` on the
tool / `interval_s` on the API. A recurring timer re-arms after each delivery
at fire-time + interval (fixed-delay — periods missed while the process was
down collapse into one), and `cancel` is the only way it ends.

```python
from power_loop import TimerRunner, HookPoint, TimerFireCtx

# agent-side: register the default tools schedule_wakeup / list_wakeups /
# cancel_wakeup / current_time — the model schedules its own wake-ups
# (schedule_wakeup(delay_seconds, note, every_seconds=None)).
# host-side:
loop.schedule_timer(sid, delay_s=600, note="check the export job")
loop.schedule_timer(sid, delay_s=60, note="heartbeat", interval_s=3600)  # recurring

runner = TimerRunner(loop)        # scans store.due_timers(), re-arms stale rows
await runner.start()

# optional orchestrator veto point before every delivery:
def gate(ctx: TimerFireCtx):
    if my_system_is_busy():
        ctx.postpone_s = 60       # or HookDirective.SKIP / BREAK (cancel)
loop.hooks.register(HookPoint.TIMER_FIRE, gate)
```

Semantics are **at-least-once**: a process dying mid-fire re-arms the row and
it may deliver twice — dedupe in the `TIMER_FIRE` hook if that matters. A
`timer_fired` event reports every outcome (`delivered` / `queued` / `skipped`
/ `cancelled` / `postponed` / `error`). See
[`examples/26_timers.py`](examples/26_timers.py).

## Public API

Stable imports are re-exported from `power_loop`:

```python
from power_loop import (
    AgentLoopConfig,
    StatefulAgentLoop,
    StatefulResult,
    ToolDefinition,
    ToolRegistry,
)
```

The stability tiers are:

| Tier | Meaning |
|---|---|
| Stable | Backward compatible across minor releases. Listed in `power_loop.STABLE_API`. |
| Provisional | Available from the top-level package during 0.x, but may change. |
| Internal | Submodule imports such as `power_loop.core.*`; no compatibility promise. |

See the [API reference](docs/en/api/index.md) for the current surface.

## Examples

The `examples/` directory is ordered from minimal usage to full chatbot composition:

```bash
python examples/00_hello_world.py
python examples/02_tool_calling.py
python examples/19_full_chatbot.py
```

The full list is in [examples/README.md](examples/README.md).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q --no-real
```

Real LLM examples/tests use `POWER_LOOP_*` or the legacy `OPENAI_COMPAT_*` variables.

## Project Links

- [Documentation index](docs/README.md)
- [Architecture](docs/en/architecture.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
