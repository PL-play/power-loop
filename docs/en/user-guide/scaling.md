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

With a server backend, two processes can be handed the same session — a retry, a rebalance, a
blue/green overlap. The in-process lock cannot see across processes, so either your dispatcher
guarantees affinity, or you opt in:

```python
loop = StatefulAgentLoop(
    llm=llm, store=store,                      # PostgreSQL / MySQL — SQLite is single-host
    config=AgentLoopConfig(
        system_prompt="…",
        distributed_sessions=True,
        session_lease_ttl_s=90.0,
    ),
)
```

Each `send()` then takes a **lease row** for the session, renews it from a background task while
it works, and releases it at the end. A process that cannot take the lease raises `SessionBusy`:

```python
from power_loop import SessionBusy

try:
    result = await loop.send(text, session_id=sid)
except SessionBusy:
    ...  # reschedule; never spin — the holder may work for minutes
```

`follow_up()` handles this for you: when another process holds the session it writes the input to
a shared queue and returns `FollowUpQueued`. The holder drains that queue at its next round
boundary, so steering still folds into the live run — across processes.

Requires schema v7; `open_store` migrates automatically (it only adds tables).

### Sizing the TTL

The TTL is a **failure-detection window**, not a per-round budget. Renewal runs on a background
task every TTL/3, independent of round boundaries — a round that takes ten minutes does not
threaten the lease. What does threaten it is the event loop being *starved* (a synchronous tool
blocking the loop, a long CPU-bound stretch) or several renewals failing in a row, so size the TTL
above the longest stall you expect. The same number sets how long a genuinely crashed holder's
session stays locked before another process can take it: raise it and a live-but-slow holder is
less likely to be presumed dead; lower it and recovery after a real crash is faster.

### What the lease does not give you

A lease is not a hard mutual-exclusion guarantee. A holder that stalls past its TTL is presumed
dead and another process takes over while the first may still be running — the classic distributed
-lock hazard. The `fence` column exists for the fix (a monotonic token that lets writes from a
dispossessed holder be rejected) but is not yet enforced, so treat the lease as a strong reduction
in the odds, not an impossibility proof. `renew_session_lease()` returning `False` is the signal a
holder has been dispossessed.

Note also that the lease protects the **session**, not whatever else your run touches: if your
tools write to a local filesystem workspace, two processes on different hosts have different
directories, and the lease will not save you. Shared storage or host affinity is a separate
problem.

## Honest caveats

- **The measured ceiling above is single-SQLite-writer and environment-sensitive.** Disk fsync latency and CPU dominate; record authoritative numbers on your reference hardware.
- **Multi-writer scale-out is a backend choice, not magic.** PostgreSQL/MySQL let many processes write one logical store, but a single session is still one writer at a time — either your dispatcher enforces that, or you opt into `distributed_sessions`. The numbers below were recorded against SQLite.
- **Compaction is the main scaling lever** for long sessions; without it, per-send cost grows with history.
- **Whether the ceiling is "enough" is your call** against your expected concurrent-session load — and which backend you point the DSN at.
