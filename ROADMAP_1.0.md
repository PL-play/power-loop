# Road to 1.0 — remaining production-hardening (post-0.14.1)

> Companion to `HARDENING_PLAN.md` (which tracked the correctness bugs C1–C8 + the C1
> double-compaction follow-up, all fixed in 0.14.x) and to the historical `ROADMAP.md`
> (the pre-integration M0–M3 plan). This file covers the **production-hardening** gaps that
> remain between **0.14.1** and a defensible **1.0** tag. Every item was verified against 0.14.1
> source. Last updated: 2026-06-15.

The 1.0 gap is **not features** — it is hardening the on-disk `SessionStore` (the value prop),
plus honest **measured** scale and pluggable observability seams. Work is overwhelmingly
**additive** (new methods, new `contrib` sinks behind extras), which is why a 0.x line can absorb it.
All storage/API breaks are concentrated into **one** release (0.15.0), after which the
`PRAGMA user_version` gate guarantees forward survival.

**Total effort:** ~32–38 person-days across four minor releases (~3–4 month calendar, single
maintainer). A defensible 1.0-ready *core* (critical-path subset) lands in ~4–4.5 weeks; the rest
trails as 1.0.x.

---

## Phase 0.15.0 — Durable store: survive upgrades, reclaim space, shut down cleanly ✅ DONE
*Foundational. The migration gate must land before any other schema change so existing ≤0.14.1 `.db`
files survive. The single storage-shape inflection. **All items shipped; validated end-to-end against a
real LLM in `tests/real/test_real_durability.py`.***

| ID | Item | Effort | API |
|----|------|--------|-----|
| **OPS-1** ✅ | `PRAGMA user_version` gate + ordered additive migration ladder (`CURRENT_SCHEMA_VERSION` + `MIGRATIONS` tuple; fold the timers-only `_micro_migrate` into step 1; refuse to open a newer-than-code DB). **Prerequisite for every later schema change.** *(`test_session_store_migrations.py`)* | M (1.5–2d) | additive |
| **OPS-2** ✅ | Opt-in retention: `prune_compacted_messages` / `prune_usage_rounds` / `prune_timers` (caller-driven, irreversible; preserves `compact_note` so `meta['ord']` ordering is intact). *(`test_session_store_retention.py`)* | M (1.5d) | additive |
| **OPS-3** ✅ | Reclamation: `auto_vacuum=INCREMENTAL` for fresh DBs + `vacuum(incremental=…)` + `checkpoint(mode=…)`. *(`test_session_store_retention.py`)* | S–M (1d) | additive |
| **OPS-4** ✅ | `export_session()/import_session()` — full durable state as a `schema_version`-stamped JSON + `backup()` via `sqlite3` online backup. *(`test_session_store_export.py`)* | M (1.5d) | additive |
| **OPS-5** ✅ | `async aclose()`/quiesce + `__aenter__/__aexit__` on `StatefulAgentLoop`: closing flag, drain in-flight sends (acquire all per-session locks), `AgentEventBus.drain()` pending async tasks, `checkpoint(TRUNCATE)` then close — fixes the `close()`-races-a-`to_thread`-write `ProgrammingError`. *(`test_stateful_loop_aclose.py`)* | M–L (2–2.5d) | additive |

## Phase 0.16.0 — Measure the single-process ceiling; de-bottleneck the read path ✅ DONE
*Convert "reasoned" → "measured". All items shipped: harness + read pool + offload + token-estimate bound + docs.*

| ID | Item | Effort | API |
|----|------|--------|-----|
| **SCALE-1** ✅ | `bench/` harness + deterministic `FakeLLM`, 3 scenarios (FANOUT / BIG-HISTORY / THROUGHPUT) → JSON report; non-blocking CI smoke (`bench.yml`). **The priority — done.** *(`python -m bench [--smoke]`, `tests/bench/test_bench_smoke.py`; big_history already shows the O(history) per-round cost SCALE-4 targets)* | L (3–4d) | additive |
| **SCALE-2** ✅ | Opt-in read-only WAL connection pool (`open(read_pool_size=N)`) so reads run concurrently with the single writer instead of serializing behind its lock; `query_only=ON` readers, file-DB only (`:memory:` falls back), default off. *(`test_session_store_read_pool.py` incl. a held-write-lock concurrency test)* | M (2–3d) | additive |
| **SCALE-3** ✅ | Offload the per-send `load_active_messages` read via `to_thread` (the O(history) hot-path read; the other named ops are cold sync helpers). *(`test_store_offload.py`)* | S (1d) | none |
| **SCALE-4** ✅ | Bound the default-on per-round O(history) `estimate_tokens` scan: pipeline keeps a self-invalidating incremental estimate, handed to the compactor via `CompactionContext.current_tokens` (measured 5ms@10k / 26ms@50k per round → O(1)). *(`test_token_estimate_cache.py`)* | M (1–2d) | additive |
| **SCALE-5** ✅ | `docs/{en,zh}/user-guide/scaling.md` grounded in measured harness numbers (fan-out plateau ~1000/s; big-history linear; sequential drift) + read-pool + retention knobs + one-db-per-process multi-process pattern + honest caveats. Examples `34` (durability) + `35` (scaling/read-pool). | M (1–2d) | none |

## Phase 0.17.0 — Observability: durable, replayable, ordered events + metrics/trace bridges ✅ DONE
*One canonical event serializer; bridges behind optional extras (core stays SDK-free). ~8–10.5 days.*

| ID | Item | Effort | API |
|----|------|--------|-----|
| **OBS-1** ✅ | `AgentEvent.to_dict()/from_dict()` carrying `ts`+`seq`+`mono`; `logging_sink` now emits the envelope (seq/ts). `from_dict` presence-checks timing (doesn't re-stamp or advance the global seq counter). **Foundation.** *(`test_event_serialization.py`)* | S (0.5–1d) | additive |
| **OBS-6** ✅ | Monotonic `mono` (`perf_counter`) field on `AgentEvent` for skew-free latency/span math (survives wall-clock rollback). *(`test_event_serialization.py`)* | S (0.5d) | additive |
| **OBS-2** ✅ | Durable rotating JSONL sink (`attach_jsonl_sink`) + `replay()`; redaction factored into shared `contrib/_redact.py`. *(`test_event_serialization.py`)* | M (1–2d) | additive |
| **OBS-3** ✅ | Backpressure: documented "sync subscribers must not block" contract + opt-in `sync_dispatch="thread"` (bounded queue + `on_overflow` drop policy + `shutdown()`); default inline → no regression. *(`test_event_bus_backpressure.py`)* | L (2–3d) | additive |
| **OBS-4** ✅ | Metrics sink: dep-free `MetricsBackend` Protocol + event→metric mapping; shipped `PrometheusBackend` (`[prometheus]`) / `StatsDBackend` (`[statsd]`), lazy-imported. *(`test_metrics_sink.py`)* | M (1.5–2d) | additive |
| **OBS-5** ✅ | OpenTelemetry span bridge: session→round→llm/tool span tree from the paired events, behind `[otel]` extra (lazy import). *(`test_otel_sink.py`)* | L (2–3d) | additive |

## Phase 0.18.0 — Ecosystem & supply-chain: MCP, provenance, zero-dep core, governance
*Least coupled to the core; benefits from a stable kernel to point adopters at. ~7.5–9 days.*

| ID | Item | Effort | API |
|----|------|--------|-----|
| **ECO-4** | Fix stale coverage target `--cov=llm_client` → `power_loop._vendor.llm_client` (or just `--cov=power_loop`). *(land first)* | S (0.25d) | none |
| **ECO-2** | `_vendor/llm_client/VENDOR.md` (upstream repo + pinned commit + version + license + sync date + local mods) + `scripts/sync_vendor.sh`. *(precedes ECO-3)* | S (1d) | none |
| **ECO-3** | Delete dead vendored `qwen_image.py` + `web_search.py` → drop `certifi` → **literally zero-dependency core**. *(dep ECO-2; verify real-LLM HTTPS smoke)* | S (0.5d) | none |
| **ECO-1** | MCP client adapter: `MCPToolSource` Protocol + `StdioMCPClient` behind `[mcp]` extra, mapping MCP `inputSchema` → `ToolDefinition`. **Biggest ecosystem reach.** | M (3–4d) | additive |
| **ECO-6** | Extension cookbook + `ToolRegistry` recipe + 1–2 examples (HTTP-API tool, in-memory vector retrieval) — **NOT** bundled connectors. *(dep ECO-1)* | M (2–3d) | none |
| **ECO-5** | `SECURITY.md` (reporting channel + "in-process tools are NOT a sandbox; use `ShellBackend`/`SubprocessExecutor`"; best-effort no-SLA). | S (0.5d) | none |
| **ECO-7** | Bus-factor surrogates: documented `RELEASING` process + README "Used by" stub + governance note. | S (0.5–1d) | none |

---

## Critical path (gates a 1.0 tag)

```
OPS-1 → OPS-3 → OPS-5              (DB survives upgrades · reclaimable · clean shutdown)
SCALE-1 → SCALE-2 → SCALE-5       (measured ceiling · de-bottlenecked reads · honest docs)
OBS-1 → OBS-2                     (durable, replayable event record)
ECO-2 → ECO-3 → ECO-5 → ECO-7     (provenance-clean · zero-dep · security policy · reproducible release)
```
Metrics/OTel/MCP/cookbook (OBS-4/5, ECO-1/6) are valuable but can trail as 1.0.x.

## Breaking changes (batched into 0.15.0, except ECO-3)
- **OPS-1**: `PRAGMA user_version` semantics; refuses to open a DB written by *newer* code. Load-bearing — must precede every later schema change so legacy `user_version=0` files are migrated, not silently skipped.
- **OPS-2**: `prune_compacted_messages` permanently deletes folded originals → breaks recall-from-raw-compacted for pruned rows. Opt-in, irreversible, loudly documented.
- **OPS-3**: `auto_vacuum=INCREMENTAL` for *fresh* DBs only (existing files untouched; no blocking VACUUM at open).
- **ECO-3** (0.18.0): drops `certifi` → empty base dependency set. Harmless for supported consumers; verify a live-HTTPS real-LLM smoke after removal.

## Honest caveats (NOT code-fixable)
- **Bus factor of 1** (single author, ~115 commits, ~3-month project). ECO-7 surrogates lower fork/onboarding risk and make the dependency legible — they don't manufacture a second maintainer.
- **No production-usage track record.** Maturity accrues with adoption + time; the "Used by" stub (DeepTalk's agent runtime) is a ledger, not a substitute. A 1.0 tag is a confidence statement about the API/durability contract, not years of field-hardening.
- **Security capacity** is bounded by one maintainer — SECURITY.md sets best-effort, no-SLA expectations honestly.
- **Operational policy is the integrator's**: retention cadence, VACUUM/checkpoint timing, metrics backend + PII policy, SIGTERM wiring. The kernel exposes hooks; it cannot choose a retention policy without risking data loss.
- **The measured ceiling is single-process** and environment-sensitive; CI asserts loose non-regression bounds, authoritative numbers on stated reference hardware. **Multi-writer horizontal scale is out of scope for 1.0** — deliverable is a measured single-process ceiling + the one-db-per-process pattern.
- **Legacy DB drift is best-effort**: ≤0.14.1 files at `user_version=0` with hand-modified schema can't be auto-reconstructed.
- **Vendored `llm_client` license**: ECO-2 surfaces upstream compatibility; if incompatible with MIT, that is a legal/relicensing problem to state, not hide.
