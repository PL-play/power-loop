# Durable Timers

[中文](../../zh/user-guide/timers.md) | [User Guide](../index.md)

Timers let an agent (or your host) say *"wake this session at time T with this note."* They are **durable** — a row in the `timers` table, not an in-memory task — so they survive process restarts. Firing is just a normal turn: the note is delivered into the session via `follow_up` (a plain `send` when idle, a queued injection when mid-run).

## Creating timers

Two ways, both writing the same rows.

**The agent schedules its own wake-ups** via the default tools (`schedule_wakeup`, `cancel_wakeup`, `list_wakeups`):

```python
from power_loop import create_default_tool_registry, get_tool_definitions
from power_loop.tools import DEFAULT_TOOL_HANDLERS

registry = create_default_tool_registry(preset="core", workspace_dir=ws)
for d in get_tool_definitions(include=["schedule_wakeup", "cancel_wakeup", "list_wakeups"]):
    registry.register(d, DEFAULT_TOOL_HANDLERS[d.name], overwrite=True)
# Now the model can call schedule_wakeup(delay_seconds=3600, note="check the build").
```

**The host schedules externally** via the loop API:

```python
t = loop.schedule_timer(sid, delay_s=3600, note="check the report")   # or due_at_ms=…
loop.list_timers(sid)
loop.cancel_timer(sid, t.timer_id)
```

## Firing them: TimerRunner

Timers only fire while a `TimerRunner` is running — it scans the store for due rows and delivers them. (If you poll `loop.store.due_timers()` from your own scheduler, you don't need it.)

```python
from power_loop import TimerRunner

runner = TimerRunner(loop)
await runner.start()    # re-arms stale rows, then scans every scan_interval
# ...
await runner.stop()
```

See [example 26](../../../examples/26_timers.py).

## One-shot vs recurring

Recurrence is declared at creation. No `interval` → one-shot (`firing → fired`). Set `every_seconds` (tool) / `interval_s` (API) → the timer re-arms at fire-time + interval (fixed-delay) until cancelled. **Cancelling is the only way a recurring timer ends.**

```python
loop.schedule_timer(sid, delay_s=60, note="heartbeat", interval_s=300)   # every 5 min
```

## The TIMER_FIRE hook — the orchestrator's veto point

Before every delivery the runner runs `HookPoint.TIMER_FIRE` with a `TimerFireCtx`. The hook decides what happens:

```python
from power_loop import HookDirective, HookPoint

def gate(ctx):                 # ctx: TimerFireCtx(session_id, timer_id, note, due_at, message)
    if system_is_busy():
        ctx.postpone_s = 300   # re-arm 5 min later
    # ctx.directive = HookDirective.SKIP    # drop THIS firing (recurring still re-arms)
    # ctx.directive = HookDirective.BREAK   # cancel the timer entirely

hooks.register(HookPoint.TIMER_FIRE, gate)
```

No hook registered → deliver. This is also where you **dedupe after a re-fire** (see below).

## Delivery semantics

Timers are **at-least-once**. A claim is a compare-and-set (`armed → firing`); a row stuck in `firing` because a process died mid-fire is re-armed by the recovery sweep and may deliver twice. A *live* slow delivery is **not** reclaimed — the runner heartbeats the `firing` row while `follow_up` runs (cadence = a quarter of `stale_firing_s`, overridable via `TimerRunner(heartbeat_interval_s=…)`). If exactly-once matters for a side effect, dedupe in the `TIMER_FIRE` hook.

## See also

- [Hooks](hooks.md) — the full `TIMER_FIRE` contract
- [Sessions](sessions.md) — `follow_up` and the durable store
- [Workflows](workflows.md) — detached runs use a timer to wake the parent
