# Async orchestration & long-running work

[中文](../../zh/user-guide/async-orchestration.md) | [User Guide](../index.md)

power-loop has several ways to do work that doesn't finish inside one LLM round —
[background tasks](advanced-runtime-tools.md), [sub-agents](subagents.md),
[workflows](workflows.md), and [durable timers](timers.md). They all share **one model** for how
an async result gets back into the agent, how it's persisted, and how it interacts with
[projection](send-context-projection.md) / [compaction](compaction.md). This page is that shared
model — read it once, then the per-feature pages make sense. It also shows how to build **your own**
async tool with the same behavior, using only public, callable APIs (no magic), and ends with
troubleshooting.

## The one rule: power-loop is host-driven (no daemon)

`StatefulAgentLoop` is **stateless at the service layer** and **runs no background threads of its
own** to advance a conversation. The store (`pl_*` tables) is the source of truth; the loop reads it
fresh on every call (read-latest). **Your host process drives every wake.** Nothing inside
power-loop will, on its own, notice "a timer is due" or "a background job finished" and start a new
round — *you* (or a small runner power-loop ships) must call back in.

The whole API surface for driving a session:

| Call | Use it when | What it does |
|---|---|---|
| `loop.send(text, sid)` | a new user turn | Appends a user message, runs rounds until the loop yields (done / waiting / pending / budget). Allocates a new `send_index`. |
| `loop.follow_up(text, sid)` | an async result / out-of-band nudge | If a `send` is **in-flight** on this session, queues `text` and the running loop injects it as a `<follow_up>` user message at the **next round boundary**. If the session is **idle**, it's just a `send`. This is the universal "wake / steer" path. |
| `loop.resume(sid)` | continue after a crash, or after a `pending_tools` yield | Re-executes the persisted pending tool_calls, then continues the loop. |
| `loop.submit_input(sid, interaction_id, value)` | answer a paused `request_user_input` | Resolves the human-in-the-loop pause and continues. |

> The loop **yields** (returns to you) with a `status`: `completed`, `waiting_for_input`,
> `pending_tools`, `hit_round_limit`, `budget_exceeded`, `cancelled`, `degraded`. The status tells
> you which call resumes it (`waiting_for_input` → `submit_input`; `pending_tools` → `resume`;
> everything else → `send`/`follow_up`).

Everything below is a variation on "async work persists its state in the store, then something calls
one of those four methods to bring the result back in."

## How each async result re-enters the loop

| Mechanism | How it runs | How the result gets back to the agent |
|---|---|---|
| **Background task** ([`background_run`](advanced-runtime-tools.md)) | a daemon thread; the tool returns a `task_id` immediately | **Automatic, in-loop:** a `RuntimeProjector` (`BackgroundRuntimeProjector`, on by default) runs at the *start of every round* and injects any **unseen** finished/updated tasks as a `<background_updates>` system message, then marks them seen. So a running loop picks them up on its next round; an **idle** loop picks them up on its next `send`/`follow_up`/`resume`. |
| **Sub-agent** ([`spawn_agent` / `AgentSpec`](subagents.md)) | runs its **own** loop to completion, inside the parent's tool call (depth-capped by `MAX_SPAWN_DEPTH`) | **Inline:** the child's result is the tool's return value, so it lands in the parent's tool-result row in the same round — no separate wake needed. |
| **Workflow** ([`create_workflow`](workflows.md)) | an in-process (or subprocess) executor steps through the DSL | Synchronous steps return inline; a **detached** workflow re-enters via a completion wake (a `follow_up`) — same as a background task. |
| **Timer** ([`schedule_wakeup`](timers.md)) | a durable `pl_timers` row; nothing fires it by itself | A **`TimerRunner`** (or your own scheduler polling `store.due_timers()`) finds the due row and calls `loop.follow_up(note, sid)` — queued if the session is mid-run, a fresh `send` if idle. A `TIMER_FIRE` hook can veto/skip/postpone first. |
| **Pending tool_calls** (a crash, or a `pending_tools` yield) | the assistant emitted tool_calls; results weren't all written | The persisted `session_state.pending` is replayed by `loop.resume(sid)`. |
| **Human input** (`request_user_input`) | the loop paused with `waiting_for_input` | `loop.submit_input(sid, interaction_id, value)`. |

Two consequences worth internalizing:

- **A finished background job does not, by itself, run a round.** It only becomes visible the next
  time the loop takes a round — which means the next `send`/`follow_up`/`resume`. If your agent
  kicked off a long job and then went idle, *you* decide when to wake it (e.g. a timer, or your own
  "job done" callback calling `follow_up`).
- **Sub-agents block the parent round; background tasks don't.** Choose `spawn_agent` when you want
  the result *now* in this turn; choose `background_run` when you want to keep talking and pick the
  result up later.

## Persistence, session state & crash recovery

Every row the loop produces is appended through the **sink** to `pl_messages` in monotonic `seq`
order (never reset), each stamped with the current `send_index`. The durable per-session state lives
in `pl_session_state`: `next_seq`, `round_index`, `last_compact_seq`, and **`pending`** (a JSON
snapshot of in-flight tool_calls). Background tasks live in `pl_background_tasks`, timers in
`pl_timers`, agent notes in `pl_notes`.

Because the store is the source of truth, a session **resumes cross-process from `(dsn, session_id)`**
— there is no in-memory session state to lose. What you must handle:

- **A crash mid-tool-call leaves `pending` set.** The next `loop.send()` on that session raises
  **`SessionPendingError`** rather than silently dropping the half-finished turn. Recover with
  `loop.resume(sid)` (re-execute the pending tools and continue), `loop.abort_pending(sid)` (discard
  them), or `loop.send(..., heal_pending=True)` (abort-then-send). Pick one deliberately — this is a
  feature, not a bug: it stops a dangling tool pair from corrupting the next turn.
- **`resume()` re-executes tool handlers**, so a handler that has side effects must be **idempotent**
  (or guard against double-execution) — a resumed tool call runs again.
- **The per-session lock and the follow-up queue are in-memory** (`_locks[sid]`,
  `_follow_up_queues[sid]`). They serialize concurrent sends *within one process* and survive nothing
  on restart. If a follow-up was only queued (not yet drained) when the process died, it's lost — so
  durable wake-ups belong in **timers** (which are persisted), not in a bare `follow_up`. And two
  processes sharing one store are **not** mutually serialized by these locks (see Troubleshooting).

## Interaction with projection & compaction

`send_index` is allocated **once per `send()`** and inherited by every row that send produces —
**including** rows appended by `resume()`, drained `follow_up`s, and a timer's `follow_up`-delivered
note. So an async result that arrives mid-send is part of **that send**, not a new one. A timer that
wakes an idle session via `follow_up` becomes a fresh `send` (its own `send_index`).

- Under a **[projecting representation](send-context-projection.md)**
  (`representation=ProjectedRepresentation(...)`), a send is projected at end-of-send. The
  reason-gate **defers** projecting a send that yielded `waiting_for_input` / `pending_tools` (it
  re-finalizes under the same `send_index` when the resume completes), so an interrupted async turn
  isn't projected twice or half-projected. A missing or stale-version projection row falls back to
  rendering that send **verbatim** from `pl_messages` (never dropped); the representation never throws —
  it degrades. Folded sends remain recoverable via `recall_send`.
- Under the **default `VerbatimRepresentation`**, async rows are ordinary history and the
  **[fold_strategy](compaction.md)** folds them normally. `representation` and `fold_strategy` are
  two **orthogonal** axes — both apply at once (a projected representation is still folded by the
  configured `fold_strategy` once `trigger_ratio` is hit).
- `BackgroundRuntimeProjector`'s `<background_updates>` injection is a **transient per-round**
  message (it is not persisted as a normal turn) — it surfaces results without bloating the audit log.

## Building your own async-wake tool

Nothing above is private machinery. A custom tool that "kicks off async work and brings the result
back" uses the same public, callable seams ([Extending tools](extending-tools.md) covers the basics):

1. **Define + register** a tool (`ToolDefinition` + an **`async def` handler** — see the async
   gotcha below — via `ToolRegistry.register`).
2. **Reach the session** from inside the handler with `get_tool_runtime_context()` → `store`,
   `session_id`, `loop`, `config`. Persist whatever you need (your own table, or reuse
   `pl_background_tasks` via the background API).
3. **Bring the result back**, choosing the re-entry that fits:
   - *let the running loop pick it up* — write to `pl_background_tasks` (or expose your own
     `RuntimeProjector` that injects unseen state each round, like `BackgroundRuntimeProjector`);
   - *wake an idle session later* — schedule a durable timer (`loop.schedule_timer(sid,
     delay_s=..., note=...)`), or call `loop.follow_up(result, sid)` from your completion callback;
   - *block this turn for the result* — just `await` your work in the handler and return it (that's
     what sub-agents do).

```python
from power_loop import ToolDefinition, get_tool_runtime_context

async def _start_export(rows: int) -> str:
    ctx = get_tool_runtime_context(required=True)          # store / session_id / loop / config
    sid = ctx.session_id
    job_id = await my_jobs.start(rows)                      # your async system
    # Durable wake: when the job finishes, fire a timer that re-enters via follow_up.
    # (Or, from your own "job done" callback elsewhere: await ctx.loop.follow_up(summary, sid).)
    await ctx.loop.schedule_timer(sid, delay_s=5, note=f"export job {job_id} should be done — check it")
    return f"Started export job {job_id}; I'll check back shortly."

EXPORT_TOOL = ToolDefinition(
    name="start_export",
    description="Begin a long export; you'll be reminded to check the result.",
    input_schema={"type": "object", "properties": {"rows": {"type": "integer"}}, "required": ["rows"]},
)
# registry.register(EXPORT_TOOL, _start_export)
```

The wake APIs you'll reach for — `loop.follow_up`, `loop.schedule_timer` / `cancel_timer`,
`loop.resume`, `loop.submit_input` — plus `TimerRunner`, `runtime_env_context` / `RuntimeEnv`
(inject per-call backends into tools), `RuntimeProjector` (inject transient per-round state), and
`register_spawn_agent` / `run_agent_spec` are all in the package's `STABLE_API`. If you need a
backend (a DB pool, an HTTP client, a sandbox) inside a tool without globals, inject it per send via
`runtime_env_context(...)` and read it with `get_tool_runtime_context()`.

## Troubleshooting

**“My background job finished but the agent never reacted.”** A finished job is only *injected* on
the next round, and an idle loop takes no rounds. Wake it: schedule a timer, or `follow_up` from your
completion callback. Also confirm the `BackgroundRuntimeProjector` is in `runtime_projectors`
(it's in the default set) — without it, `pl_background_tasks` updates are never surfaced.

**“Background task is stuck at `running` and the result was never written.”** The daemon thread
writes its terminal status back via `run_coroutine_threadsafe` onto the loop's event loop. If that
loop is **closed during shutdown** before the write lands, the write is dropped. Drain/await
in-flight work before closing the loop (`await loop.aclose(...)`), and treat a long-`running` task as
possibly-orphaned on the next start.

**“`SessionPendingError` on `send()`.”** A prior run crashed after the assistant emitted tool_calls
but before all results were written. Call `resume(sid)` (finish them), `abort_pending(sid)`
(discard), or `send(..., heal_pending=True)`. Don't ignore it — it's protecting the next turn from a
dangling tool pair.

**“My async tool runs but its result never appears / it ran on the wrong thread.”** Register the
handler as **`async def`** (or a callable whose `__call__` is async). A *sync lambda that returns a
coroutine* is detected as sync, run via `asyncio.to_thread`, and its coroutine is never awaited.
Relatedly, async handlers must be invoked via `invoke_async` (the loop does this) — `invoke()` on an
async handler raises `AsyncToolInSyncContext`. And if your handler spawns an `asyncio.create_task`,
copy the context (`contextvars.copy_context()`) or it loses `session_id` / `store`.

**“A timer fired twice.”** Delivery is **at-least-once**: a crash between claiming a timer (`firing`)
and finishing it lets the stale-row recovery re-arm it. Make the effect idempotent, and/or use the
**`TIMER_FIRE` hook** (the one place to dedupe on `timer_id` / veto / postpone before delivery).

**“A queued `follow_up` (or timer note) vanished after a restart.”** The follow-up queue is
in-memory. Use a **durable timer** for anything that must survive a crash; `follow_up` is for
steering a live, in-process run.

**“Two processes on the same session stepped on each other.”** The per-session lock is in-process
only — it does **not** coordinate across processes. Run one writer per session, or add your own
distributed lock keyed by `session_id`; the store stays consistent (it serializes its own writes),
but the loops won't take turns on their own.

**“I switched a session to projection mode and old turns look folded/compressed.”** That's the
one-time mode-switch migration (see [Send-context projection](send-context-projection.md)); it's
best-effort and never throws. Pre-projection rows (`send_index = NULL`) render verbatim as a prefix;
a session with in-place `compact_note` history degrades to verbatim rendering rather than
mis-partitioning.

## See also

- [Advanced runtime tools](advanced-runtime-tools.md) — `background_run`, runtime context, runtime projectors, hooks
- [Sub-agents](subagents.md) · [Workflows](workflows.md) · [Durable timers](timers.md)
- [Extending tools](extending-tools.md) — the tool definition/handler/register recipe
- [Send-context projection](send-context-projection.md) · [Compaction](compaction.md) · [Sessions](sessions.md)
