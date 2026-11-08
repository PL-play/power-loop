# Timers in runtime 11

A deployment uses one `schedule_wakeup` tool and one authoritative `TimerBackend`.
The default `Runtime` binds `StoreTimerBackend` and starts/stops `TimerRunner` with
its worker. No Redis is required. A standalone `RuntimeClient` starts no driver.
Set `timers_enabled=False` to disable timers. An explicit host backend requires
an explicit driver implementing `start`, `stop`, and `status`; startup refuses an
unready driver rather than falling back to local storage.

The model calls `schedule_wakeup(action="schedule", delay_seconds=60, note="check")`,
`action="list"`, or `action="cancel", timer_id=...`. Add `every_seconds` for a
recurring timer. Only main agents may create timers; query and cancel also check
identity. The SDK supplies the backend with a trusted `TimerContext`: store,
session, actor, original job/tool call, stable operation ID, and host context.
These fields never come from model arguments.

Hosts can use the same backend:

```python
from power_loop import TimerContext
source = TimerContext(store.store_id, target.session_id, authenticated_actor)
timer = await runtime.timer_backend.create(source, delay_s=60, note="check")
items = await runtime.timer_backend.list(source, statuses=("armed", "firing"))
await runtime.timer_backend.cancel(source, timer.timer_id)
```

The driver takes bounded indexed due batches, conditionally claims an occurrence,
and calls `RuntimeClient.append` with a stable message ID. It waits for acceptance
only. Busy sessions consume the note at a round boundary; idle sessions start a
turn. A slow model never keeps the timer scan open. Periodic timers schedule the
next occurrence from delivery time plus interval; missed intervals are not replayed.
Cancellation prevents future occurrences, including rearming a claimed periodic
timer. It does not cancel an input already accepted by the runtime.

Pass `AgentHooks` explicitly to the runtime/driver for `TIMER_FIRE`:
`CONTINUE` delivers, `SKIP` skips this occurrence, `BREAK` cancels, and positive
`postpone_s` delays it. The hook may replace `ctx.message`. A custom driver can
receive an `AgentEventBus` through `TimerRunner(event_bus=...)` for result events.
Driver `status()` reports readiness, last scan/error and the latest bounded due page.

On infrastructure failure an occurrence may be lost, duplicated or unknown.
Stale claimed occurrences are ended without replay; periodic future occurrences
can continue. Terminal rows retain the existing seven-day cleanup policy.

DeepTalk uses `ApiTimerBackend` and dispatcher readiness. The API retains business
authorization, limits, and a new run/budget for each wake; SDK timers are unused.
See [runnable example 26](../../../examples/26_timers.py) and
[the runtime contract](../../runtime-11.md).

Register finite child review explicitly with `register_wait_handoff_tool(registry, policy=WaitPolicy(deadline_at_ms=...))`.
The factory freezes the deadline. The original main can continue, cancel or defer the child, within its original deadline.
Review reuses the persisted interaction and current TimerBackend; an unknown timer creation is never replayed automatically.
