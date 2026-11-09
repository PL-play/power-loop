# power-loop 12.0.0

An embeddable agent runtime. Main agents, children, background work and workflows
use one ID-addressed client/worker execution path. The default composition is
single-process, with no Redis dependency. Shared execution requires explicit
shared ownership and messaging adapters.

Current task schema: **15**. Protocol: **42**. Descriptor schema: **1**.
Use a fresh SDK database or table prefix; historical tasks are not imported.
This branch is a source delivery, not a published or deployed release. See the
P7 delivery report for the full regression status.

- [Current API and startup guide](docs/runtime-11.md)
- [Runnable single-process example](examples/design203_single_process.py)
- [Runnable default timers](examples/26_timers.py)
- [Changelog](CHANGELOG.md)
- [Authoritative stable exports](power_loop/__init__.py)
- [中文](README.zh.md)

Install the checkout with `python -m pip install -e '.[dev]'`. From an isolated
test directory, run `python /absolute/path/to/examples/design203_single_process.py`.
The demo uses a fixed model; a real deployment resolves models, tools, hooks and
resources in its registered `RuntimeFactory`.

`StatefulAgentLoop` accepts `runtime=`. Direct LLM/DSN construction is retired.
An acceptance receipt is not completion: wait for its original `InputRef`.
Normal authorization, accounting, context and cleanup remain required. Failures
may leave interrupted/unknown results; unknown external actions are not replayed.

Finite child waits use `WaitPolicy` and `register_wait_handoff_tool`: the factory
sets a deadline, the child releases capacity, and its original main explicitly
continues, defers or cancels the same task. Configure the timer backend/driver in
the Runtime; see [the contract and assembly](docs/runtime-11.md#finite-child-review).

## Host Responsibility: Periodic Data Cleanup

⚠️ `purge_expired()` is a host-invoked interface; the SDK does not auto-clean.
If the host does not call it, the following tables grow unbounded:
- `job_events` (~10-15 rows per send)
- `jobs` (1 row per send)
- `follow_up_queue` (per drain)
- `hook_events`

Recommendation: call once daily with `limit=1000`, repeat until it returns all zeros.

```python
from power_loop.runtime.store.retention import purge_expired
from power_loop.agent.runtime_store import now_ms

# Example: scheduler service cleanup task
async def run_daily_cleanup(store):
    now = now_ms()
    total = 0
    while True:
        counts = await purge_expired(store, now, limit=1000, retention_days=None)
        batch = sum(counts.values())
        total += batch
        if batch == 0:
            break
    return total
```

Default retention periods (can be overridden via `retention_days` parameter):
- `task_trees` (jobs + job_events): 30 days
- `executors`: 1 day
- `follow_ups`: 7 days
- `hook_events`: 30 days
- `usage_observations`: 90 days
- `runtime_state`: 30 days

