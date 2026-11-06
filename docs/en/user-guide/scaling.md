# Scaling

[中文](../../zh/user-guide/scaling.md) | [User Guide](../index.md)

power-loop is an **embeddable** kernel over a **pluggable store**: SQLite by default (zero infra), or PostgreSQL/MySQL by DSN. This page states the concurrency model plainly, gives **measured** numbers from the bundled harness, lists the tuning knobs, and describes how to scale out. It is honest about the ceiling: a single SQLite writer has one — and a server backend is the way past it.

## The model

- **One writer per session.** The async store offloads blocking SQLite I/O to a worker thread under a single writer lock — this is what keeps `next_seq` collision-free and each multi-statement write atomic. (WAL mode, `journal_mode=WAL` / `synchronous=NORMAL`, is on, so reads never block the writer.) On PostgreSQL/MySQL, per-session sequence allocation is multi-writer-safe via a `SELECT … FOR UPDATE` row lock.
- **One asyncio event loop per process.** A single `StatefulAgentLoop` drives any number of concurrent sessions, each serialized by a per-**session** `asyncio.Lock`. Since 3.19.0 that lock lives in a process-wide registry keyed on `session_id`, so the guarantee holds even if the host builds several `StatefulAgentLoop` objects over one store (e.g. rebuilding a cached loop after a config edit). Before 3.19.0 the lock was per-instance, and two loops over one session took two different locks — no mutual exclusion at all. SQLite work runs in a worker thread so a slow write/read doesn't freeze the loop; PostgreSQL/MySQL drivers are natively async.
- **The loop holds no authoritative state.** All of it lives in the store, so loops are cheap to create and any session resumes from a `dsn` + `session_id` (ideal for web handlers / workers / cold starts). The per-session active-window cache (`session_cache_size`, default 256, `0` disables; inspect via `loop.cache_stats`) is a pure accelerator over the durable projection — it never changes what the model sees.

What a **single SQLite file** is **not**: a multi-writer store. Many processes writing one SQLite file is out of scope. To scale past one writer you have two paths: **shard SQLite files across processes** (below), or **point the DSN at PostgreSQL/MySQL** — a real multi-writer server, same code, same conformance suite (see [Storage backends](storage-backends.md)).

## Tuning knobs

| Knob | Where | Effect |
|---|---|---|
| Backend choice | `dsn=` on `StatefulAgentLoop` / `open_store(...)` | SQLite (default, single-writer file) vs PostgreSQL/MySQL (multi-writer server). The first lever once one writer isn't enough. See [Storage backends](storage-backends.md). |
| `session_cache_size` | `StatefulAgentLoop(session_cache_size=N)` | LRU of per-session active-window caches; skips re-reading the active history on hot multi-send paths. Default `256`; `0` disables. Pure accelerator. |
| Compaction | `AgentLoopConfig(compactor=DefaultCompactor(...))` | Keeps the **active history bounded** (≈ `max_tokens`), which bounds per-round read + token-estimate cost. The single biggest lever for a long-lived session. |
| Retention | `prune_compacted_messages` / `prune_usage_rounds` / `prune_timers` + `vacuum()` | Reclaim disk from folded-out originals / old usage rows; opt-in, caller-driven. (`vacuum`/`checkpoint` are SQLite-only; no-ops on PG/MySQL — use their native tooling.) |
| `max_tokens` | `AgentLoopConfig` | The context budget; also the rough cap on active-history size when a compactor is on. |

The store itself offloads its own blocking I/O — SQLite statements run in a worker thread under a single writer lock, and PostgreSQL/MySQL are natively async — so reads don't need any opt-in connection pool. You scale read/write throughput by choosing a server backend or sharding SQLite files across processes, not by tuning a pool.

## Measured numbers

Run the bundled harness yourself — a deterministic `FakeLLM` (no provider) over a real store, so the numbers reflect *store/loop* overhead:

```bash
python -m bench            # full sweep → JSON
python -m bench --smoke    # fast subset (also the CI smoke)
```

The figures below were recorded on a developer VM (Python 3.12, in-memory store, `latency_s=0`) — **illustrative, not a spec**. Record your own on representative hardware; CI only asserts the harness runs and reports monotone numbers, never absolute thresholds (runners are noisy).

**Fan-out** (N concurrent sessions, one send each):

| sessions | sessions/sec | send p50 | send p99 |
|---:|---:|---:|---:|
| 1 | ~198 | 5.0 ms | 5.0 ms |
| 8 | ~679 | 10.1 ms | 11.4 ms |
| 32 | ~1035 | 24.9 ms | 27.6 ms |
| 128 | ~1031 | 109 ms | 122 ms |
| 512 | ~986 | 450 ms | 503 ms |

Throughput plateaus around **~1000 sessions/sec** (the single writer) and per-send latency grows with concurrency past that — the expected single-writer ceiling.

**Big history** (per-send cost vs active-history size, no compaction):

| active messages | send p50 | send p99 |
|---:|---:|---:|
| 1,000 | 9.6 ms | 11.4 ms |
| 10,000 | 92 ms | 102 ms |
| 50,000 | 511 ms | 559 ms |

Per-send cost grows ~linearly with active-history size — because each send must load the full active window. **This is what compaction prevents:** with a compactor on, the active window stays ≈ `max_tokens` regardless of total turns, so this cost stays flat. A long single session *without* compaction degrades steadily (the harness's sequential-throughput scenario shows the same drift). Keep a compactor on for long-lived sessions.

## Scaling out

There are two ways past a single writer.

**A. Shard SQLite files across processes.** With SQLite the model is one file per writer. To use more cores / handle more load, run N processes, **each with its own DB file** and its own `StatefulAgentLoop`:

```
process A → loopA → dsn="shard-a.db"
process B → loopB → dsn="shard-b.db"
```

Route a given session to a fixed process (e.g. hash the session id). Do **not** point two processes at the same SQLite file for writing — the store does not coordinate cross-process writers (a stray second writer is caught only by the `(session_id, seq)` primary key raising `IntegrityError`, not prevented).

**B. Use a server backend.** Point the DSN at PostgreSQL/MySQL (`dsn="postgresql://…"` / `dsn="mysql://…"`). Now many processes can write the *same* logical store concurrently — per-session sequence allocation is multi-writer-safe via a `SELECT … FOR UPDATE` row lock, so different sessions append in parallel without colliding. A given session's pending-state machine still assumes one writer at a time. You can either serialize that session's sends in your own dispatcher/queue layer, or let power-loop do it — see [Running several processes over one store](#running-several-processes-over-one-store) below. See [Storage backends](storage-backends.md) for provisioning and preconditions.

## Running several processes over one store

The default `MemorySessionOwnershipCoordinator` serves one process. Shared deployments configure the unified Runtime with shared ownership and transport adapters:

```python
runtime = Runtime(
    store=store, factory=factory, node_id="worker-1",
    coordinator=shared_coordinator, transport=shared_transport,
    shared_execution=True, session_lease_ttl_s=90.0,
    host_context_fields={"app": {"trace_id"}},
)
```

Keep `node_id` stable; the default `boot_id` is generated once per interpreter startup. Shared mode rejects process-only adapters and never falls back to memory after failure. The SDK has no Redis dependency.

Ownership uses `(store_id, session_id)`. `SessionOwnershipCoordinator` exposes `acquire`, `renew`, and `release`; original tokens fence renewal and release. The SDK reads the expected store epoch **before** acquire, installs the grant conditionally, and releases/reacquires on conflict. A handle's monotonic deadline starts before the shared request, so a delayed response cannot extend permission.

Host adapters must pass `power_loop.testing.SessionOwnershipContractTests`. `power_loop.testing.db_session_ownership.DatabaseSessionOwnershipCoordinator` is a SQLite adapter for real same-host process tests, not the production cross-machine adapter.

Clients persist inputs without owning the target session. Workers acquire before constructing the execution environment; a busy worker leaves the input pending. Each independent child owns its own session. Loss cancels only this session's execution and prevents new SDK dispatch. Hooks cannot replace ownership loss or cancellation with tool success/retry. Already running external processes are outside that cancellation guarantee.

On takeover, only unfinished jobs in the acquired session are interrupted. The original main-send input retains an `interrupted` result at its original generation, and a new send repairs pending message format. Healthy child sessions are untouched. Rebuilding under an existing grant preserves its epoch and job generation. Per-job lease renewal and parent execution credentials have been removed; logical lineage, cumulative budgets and explicit cancellation still apply. Child resume and workflow continuation are subsequent implementation stages.

`get_execution_context()` returns an immutable snapshot of SDK identity, request/job references, resources and cancellation. Inside an execution, `with bind_host_context("app", {"trace_id": "trace-1"}):` temporarily binds allowlisted JSON values; exit resets them. Independent execution clears inherited ownership, send, tool invocation and cancellation context. Persist restart-surviving business state with the existing `runtime_state` API.

Runtime hooks provide `execution_acquired`, `execution_lost`, `execution_released`, `message_accepted`, and `execution_finished`. Notifications default to a one-second timeout, log failures, and cannot delay required cancellation indefinitely.

### Sizing the TTL

The TTL is a **failure-detection window**, not a per-round budget. Renewal runs on a background
task every TTL/3, independent of round boundaries — a round that takes ten minutes does not
threaten the lease. What does threaten it is the event loop being *starved* (a synchronous tool
blocking the loop, a long CPU-bound stretch) or a failed renewal, so size the TTL
above the longest stall you expect. The same number sets how long a genuinely crashed holder's
session stays locked before another process can take it: raise it and a live-but-slow holder is
less likely to be presumed dead; lower it and recovery after a real crash is faster.

### What the lease does not give you

Local deadline checks and loss notification cancel session tasks. Message-state writes retain a `next_seq` condition that rejects a writer behind the new owner. Failures may lose results or leave them unknown; takeover repairs message format and continues without replaying old external actions.

Note also that the lease protects the **session**, not whatever else your run touches: if your
tools write to a local filesystem workspace, two processes on different hosts have different
directories, and the lease will not save you. Shared storage or host affinity is a separate
problem.

## Honest caveats

- **The measured ceiling above is single-SQLite-writer and environment-sensitive.** Disk fsync latency and CPU dominate; record authoritative numbers on your reference hardware.
- **Multi-writer scale-out is a backend choice, not magic.** PostgreSQL/MySQL let many processes write one logical store, but a single session is still one writer at a time — either your dispatcher enforces that, or the application provides a shared session coordinator. The numbers below were recorded against SQLite.
- **Compaction is the main scaling lever** for long sessions; without it, per-send cost grows with history.
- **Whether the ceiling is "enough" is your call** against your expected concurrent-session load — and which backend you point the DSN at.

## Host inspection and reconciliation

Use `power_loop.runtime.task_inspection.list_open_roots(manager, after_job_id="", limit=100)`
to discover unfinished root tasks, and `power_loop.runtime.model_budgets.list_unsettled(manager,
after_attempt_id="", limit=100)` to discover unacknowledged fees and outstanding dispatches.
Responses contain `roots` or `attempts` and `next_cursor` (`None` at the end).
These trusted host queries cover the store; use the original task actor when calling
inspection or reconciliation. Each page is limited to 1,000 rows and skips finished history.
Executor heartbeats extend the current row's lease and do not accumulate event rows.
