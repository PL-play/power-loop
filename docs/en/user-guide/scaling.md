# Scaling

[中文](../../zh/user-guide/scaling.md) | [User Guide](../index.md)

power-loop is an **embeddable, single-process** kernel over one SQLite file. This page states the concurrency model plainly, gives **measured** numbers from the bundled harness, lists the tuning knobs, and describes the multi-process pattern. It is honest about the ceiling: there is one.

## The model

- **One SQLite file, one writer.** All writes go through a single connection guarded by an in-process `threading.RLock` — this is what keeps `next_seq` collision-free and each multi-statement write atomic. WAL mode (`journal_mode=WAL`, `synchronous=NORMAL`) is on.
- **One asyncio event loop per process.** A single `StatefulAgentLoop` drives any number of concurrent sessions, each serialized by its own `asyncio.Lock`. Blocking store I/O on the write path is offloaded with `asyncio.to_thread` so a slow write/read doesn't freeze the loop.
- **Reads can run concurrently** (opt-in, see [Read pool](#read-pool)) — WAL allows many readers alongside the one writer.

What this is **not**: a multi-writer / horizontally-scaled store. Many processes writing one logical store is **out of scope**. For more throughput, run more processes — one DB file per process (see [Multi-process](#multi-process)).

## Tuning knobs

| Knob | Where | Effect |
|---|---|---|
| `read_pool_size` | `SessionStore.open(read_pool_size=N)` | N extra read-only connections so reads don't serialize behind the writer (file DB only; `:memory:` declines it). Default `0`. |
| Compaction | `AgentLoopConfig(compactor=DefaultCompactor(...))` | Keeps the **active history bounded** (≈ `max_tokens`), which bounds per-round read + token-estimate cost. The single biggest lever for a long-lived session. |
| Retention | `prune_compacted_messages` / `prune_usage_rounds` / `prune_timers` + `vacuum()` | Reclaim disk from folded-out originals / old usage rows; opt-in, caller-driven. |
| `max_tokens` | `AgentLoopConfig` | The context budget; also the rough cap on active-history size when a compactor is on. |

### Read pool

```python
store = SessionStore.open("app.db", read_pool_size=4)
loop = StatefulAgentLoop(llm=llm, store=store)
```

Reads (`load_active_messages`, `load_all_messages`) check out one of N read-only (`query_only=ON`) connections and run concurrently with the writer instead of queuing behind its lock. A pooled reader in WAL mode sees every transaction committed before the read began — which is all a per-send history load needs. Default off (`0`); ignored for `:memory:` (each in-memory connection is a *separate* database). Worth enabling under read-heavy fan-out.

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

## Multi-process

The store's model is **one file, one process**. To use more cores / handle more load, run N processes, **each with its own DB file** and its own `StatefulAgentLoop`:

```
process A → loopA → SessionStore.open("shard-a.db")
process B → loopB → SessionStore.open("shard-b.db")
```

Route a given session to a fixed process (e.g. hash the session id). Do **not** point two processes at the same file for writing — the store does not coordinate cross-process writers (a stray second writer is caught only by the `(session_id, seq)` primary key raising `IntegrityError`, not prevented).

## Honest caveats

- **The measured ceiling is single-process and environment-sensitive.** Disk fsync latency and CPU dominate; record authoritative numbers on your reference hardware.
- **Multi-writer horizontal scale is out of scope for 1.0.** The deliverable is a *measured* single-process ceiling plus the one-db-per-process pattern above.
- **Compaction is the main scaling lever** for long sessions; without it, per-send cost grows with history.
- **Whether the ceiling is "enough" is your call** against your expected concurrent-session load.
