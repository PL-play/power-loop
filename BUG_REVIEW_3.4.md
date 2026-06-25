# power-loop systematic review — v3.4.0

_Date: 2026-06-25 · Method: 19-dimension multi-agent review (subsystem deep-reads + cross-cutting
sweeps) → adversarial refute-by-default verification (3 perspective-diverse skeptics per high/critical
finding) → synthesis. 108 agents, ~6.0M tokens, ~2.4h._

Baseline before review: **`tests/unit` 965 passed** (green). So every finding below is a gap the
existing suite does **not** cover.

## Headline

| Severity | Confirmed | Notes |
|---|---|---|
| Critical | **0** | none found |
| High | **7** | data-loss / loop-abort / DoS / portability |
| Medium | **15** | |
| Low | **32** | |
| **Total confirmed** | **54** | out of 70 raised (16 refuted) |

No critical / no remote-code-exec / no data-corruption-of-committed-history was found — the core
persistence and compaction paths (which carried the worst bugs in the 2.0/3.0 reviews) held up. The
risk has **migrated to the newest subsystems**: send-context **projection** (3.0→3.4) and the
**workflow DSL** durability/cleanup path. Seven themes below.

> Verification note: the run hit a session limit at the very end, which lost the verification of the
> two cross-cutting sweep dimensions (`xcut-concurrency`, `xcut-api-security`) and the `synthesize`
> step. I re-verified those by hand (see [§Manually re-verified](#manually-re-verified-cut-by-session-limit))
> and wrote this synthesis directly.

---

## Cross-cutting themes (synthesis)

1. **Projection mode is where the data-integrity risk now lives.** It is the newest path and the least
   battle-tested. `stateful-loop-1` (HIGH) silently drops the user's submitted answer / replayed tool
   result on `submit_input()`/`resume()`; `projection-1` (MED) permanently loses recall hints on
   migration-seeded folds; `projection-2/3` (LOW) mis-handle JSON-coerced config and corrupt rows.
   Verbatim/default mode is unaffected in every case — the defects are projection-specific.

2. **The "an open `assistant(tool_calls)` must be answered" invariant is fragile across *every*
   history transform.** It breaks in `budget.trim_history` (`async-control-1`, HIGH — emits a dangling
   tool_call → provider 400), in projection (`stateful-loop-1`), in the Anthropic transport
   (`llm-transport-3` thinking blocks, `llm-transport-4` tool_result images), and in
   `AgentMessage.to_openai_message` (`exec-skills-structured-7`). Each transform re-implements pairing
   protection independently and at least four get it wrong.

3. **"Observability must never break the loop" is asserted but inconsistently enforced.**
   `metrics_sink` (HIGH) lets backend exceptions propagate out of `publish()` and abort the run;
   `otel_sink` guards `.end()`/`.set_status()` but not span *creation* or `set_attribute` (MED);
   `JsonlSink` does inline disk I/O on the publish thread (LOW) and `SQLiteSink` can raise (LOW). The
   contract needs to be enforced at the bus, not per-sink.

4. **The MySQL backend passes parity tests but is not usable for real workloads.** `store-dialect-1`
   (HIGH): all free-text/JSON columns are `TEXT` (64 KiB cap) — a single >64 KiB tool result, system
   prompt, or tool_calls_json hard-fails (`DataError 1406`) under strict mode, or silently truncates
   otherwise. Parity tests never store >64 KiB. (`store-dialect-3`, LOW: prefix length can overflow
   the 64-char identifier limit.)

5. **The workflow DSL under-validates LLM-authored specs and is fragile on cancel/cleanup.**
   `create_workflow` is model-facing, yet fanout is unbounded (`workflow-engine-1`, HIGH — 100k tasks /
   sessions / LLM calls with no ceiling); a grandchild holding stdout open hangs a leaf and the whole
   run forever (`workflow-durability-1`, HIGH — no process group + an unbounded final drain); driver/
   leaf sessions leak (`workflow-engine-2`, MED); plus terminal-status, double-resume, reap-mid-run,
   and reference/schema validation gaps (`workflow-durability-2/3/4`, `workflow-engine-3/4/5`).

6. **Multimodal / vision is broken end-to-end.** `prompt-sink-provider-1` (HIGH): structured content
   is `json.dumps`'d on persist and never parsed back, so the model receives a literal JSON *string*
   instead of an image — on the **first** send, not just after restart. `llm-transport-4` drops images
   inside tool_results; `llm-transport-2` doesn't gate image support by capability.

7. **Fail-soft gaps: one bad input aborts a whole batch.** A single non-UTF-8 `SKILL.md` kills *all*
   skills (`exec-skills-structured-1`); a malformed `CONTEXT_COMPACT_THRESHOLD` crashes the whole send
   (`compaction-1`); odd-quote prose rejects valid JSON (`exec-skills-structured-2`); a `limit` arg
   bypasses the 5 MB read cap (`default-tools-1`).

A secondary theme: **dead / misleading config** — `POWER_LOOP_SUPPORTS_*` overrides are never wired
(`llm-transport-1`), and `StructuredOutputSpec.examples` / `wrong_type:` reason / `ToolArgsValidator`
are public-but-unused (`exec-skills-structured-4/5/6`). Memory previously recorded `runtime_config.retry`
as similarly dead.

---

> **Status (2026-06-25):** **7 HIGH + 14/15 MEDIUM + 30/32 LOW fixed-or-documented** with regression
> tests (each verified fail-pre-fix / pass-post-fix where applicable); H2 verified against the live
> MySQL server, projection/durability/tool-use/retry/subagent against the live LLM. Full unit suite
> **1016 green**. See CHANGELOG `[Unreleased]`. **Deferred (3, all Anthropic-transport + untestable
> against the configured OpenAI-compatible provider):** `llm-transport-3` (M10, extended-thinking
> signatures), `llm-transport-2` and `llm-transport-4` (LOW, ModelCapabilities + tool_result images)
> — they need a real Anthropic-thinking endpoint and a `LLMResponse`+persistence round-trip.

## HIGH (7) — full detail

### H1 · `stateful-loop-1` · Projection mode silently drops the submitted answer on `submit_input()`/`resume()`
**`agent/stateful_loop.py:762-770, 814-822, 1091-1129`; reader `1290-1296, 1341-1352`; `agent/sink.py:156`; `runtime/history_sanitize.py:86-103`** · data-integrity · 3/3 votes · *empirically reproduced*

`submit_input()`, `resume()` (via `_execute_pending`), and `abort_pending()` append their tool rows
out-of-band by calling `sink.on_message_appended()` with a dict that has **no `send_index` key** →
persisted as `send_index=NULL`. The in-flight send index is never bumped (by design), so in projection
mode the reader partitions the NULL row into `legacy_rows` (rendered as the temporally-*first* prefix)
while the assistant `tool_calls` that owns it sits in `current_rows` at the tail. `align_tool_calls()`
then drops the leading orphan tool-result and refuses to synthesize a placeholder for the trailing
call → **the LLM never sees the answer and replies blind.** With `repair_corrupt_history=True` the row
is physically deactivated — permanent audit-log erasure.
**Fix:** stamp the current `send_index` onto every out-of-band tool row (the sink already forwards it);
add a projection-mode regression test for `submit_input`/`resume`.

### H2 · `store-dialect-1` · MySQL `TEXT` (64 KiB) cap hard-fails large content that SQLite/PG accept
**`runtime/store/dialect.py:355-439`** · data-integrity / portability · 3/3 votes · *reproduced on live MySQL 3307*

Every free-text/JSON column in the MySQL DDL is `TEXT` (65,535-byte cap). Under default
`STRICT_TRANS_TABLES`, a >64 KiB `content` / `system_prompt` / `tool_calls_json` write raises
`DataError(1406, "Data too long")` and aborts the store transaction (or silently truncates under a
lax sql_mode). The same write round-trips on Postgres/SQLite. The 13-table column set is otherwise
byte-identical across dialects — this is a pure type-mapping divergence that makes MySQL silently
unusable for real agent payloads.
**Fix:** map LLM-payload columns to `LONGTEXT` (or ≥`MEDIUMTEXT`) in `MySQLDialect`; add a >64 KiB
parity test on all three backends; ship an `ALTER … MODIFY … LONGTEXT` migration (new schema version)
for existing MySQL stores.

### H3 · `workflow-engine-1` · `foreach` fanout is unbounded (cost / DoS explosion)
**`workflow/spec.py:587-646, 681-685`; `workflow/engine.py:412-451, 397-410, 370`** · design · 3/3 votes

`create_workflow` is an LLM-facing tool, so the spec is model-authored and may be hallucinated/
adversarial. The validator caps nothing: not `items` length, not `items_from` runtime length, not
`max_concurrency` (`_int_ge1` accepts 1_000_000). At runtime `_gather_branches` eagerly
`ensure_future`s **one task per item**; `max_concurrency` throttles execution but not task/session
creation. Each leaf persists a DB session and fires a real LLM call. The only ceiling is the
*optional* budget.
**Fix:** clamp `max_concurrency` (e.g. ≤64) and cap literal/runtime `items` length at validation;
enforce a hard per-run leaf ceiling independent of the optional budget.

### H4 · `workflow-durability-1` · Unbounded final drain + no process group → a grandchild hangs the whole run forever
**`workflow/subprocess_executor.py:308-311, 282-285`** · resource-leak / liveness · 3/3 votes · *primitive reproduced*

On timeout/cancel, `_terminate` does SIGTERM→SIGKILL each draining `comm` under `wait_for(..., grace)`,
then falls through to a bare `await asyncio.shield(comm)` with **no timeout**. The worker is spawned
without `start_new_session`, and kill signals only the direct PID — never the group. A grandchild that
inherited fds 1/2 keeps the stdout pipe open, so `communicate()` never sees EOF and the final drain
blocks forever, hanging the leaf, its branch, and the detached run. `_await_proc`'s 0.1 s poll loop
spins indefinitely in the same situation.
**Fix:** `start_new_session=True` + `os.killpg(...)` on terminate; bound the final drain with
`wait_for` and `comm.cancel()`→return `(None, None)`; give `_await_proc` a hard ceiling.

### H5 · `async-control-1` · `trim_history` body-front trim emits a dangling `assistant(tool_calls)` → provider 400
**`runtime/budget.py:177-186`** · correctness · 3/3 votes · *reproduced* · **public API** (`trim_history` is in `__all__`)

The body-rebuild loop keeps body messages from the front until the budget is hit, then `break`s — with
**no** tool-call/result atomicity guard (unlike the tail-cut and last-resort paths). If the budget runs
out right after an `assistant(tool_calls)` but before its `tool` result, the kept body ends on an
unanswered tool_call, immediately followed by the user-boundary tail → OpenAI/Anthropic both reject the
next call with `tool_calls must be followed by tool messages`.
**Fix:** after the loop, pop trailing `assistant(tool_calls)` whose results aren't all kept (mirror the
tail-boundary atomicity logic); add the matching unit test.

### H6 · `prompt-sink-provider-1` · Multimodal content is stringified on persist and never parsed back
**`agent/sink.py:362-370`; `agent/stateful_loop.py:1043, 1776-1792`** · data-integrity · 3/3 votes

`_as_text` `json.dumps`'s non-string content; `_row_to_loop_message` reloads it with **no
`json.loads`**. Because `_run_loop` rebuilds working history from the store even for the *current*
send, the model receives the image/structured content as a literal JSON string on the **first** send.
Vision requests silently produce wrong answers.
**Fix:** round-trip structured content losslessly (a `content_json` column or meta flag → `json.loads`
on reload), or reject non-string content at `send()` with a clear error.

### H7 · `contrib-observability-1` · `metrics_sink` backend exceptions propagate out of `publish()` and abort the loop
**`contrib/metrics_sink.py:59-86`** · correctness · 3/3 votes · *reproduced*

The metrics `_handler` calls `backend.incr/observe` with no try/except. Real backends raise (StatsD
socket `OSError`, prometheus_client `ValueError` on bad/duplicate labels). The default
`DEFAULT_EVENT_BUS` is created with `suppress_subscriber_errors=False`, and the pipeline emits with no
try/except at the call site → a metrics hiccup aborts the in-progress LLM round. The sibling
`otel_sink` documents this intent; `metrics_sink` has no guard.
**Fix:** wrap the backend dispatch in a log-and-swallow try/except (mirror `otel_sink`); don't rely on
the caller's bus suppressing.

---

## MEDIUM (15)

| ID | Area | Issue | Fix gist |
|---|---|---|---|
| `stateful-loop-2` | loop | `hit_round_limit` wrap-up summary returned to caller but **never persisted** to the transcript (success branch only) | append the assistant summary before finalizing, like the degraded branch |
| `pipeline-runner-1` | events | `STREAM_STARTED` per retry attempt but `STREAM_COMPLETED` once → unbalanced stream events on any LLM retry | emit COMPLETED per-attempt in `_do_call`'s finally |
| `projection-1` | projection | folded compact loses its `recall_send` hint whenever `compact_from_send==0` (migration-seeded folds) | gate on `to_send>=1`, not `lo>0` |
| `default-tools-1` | tools | `read_file` with a `limit` passes `max_bytes=None` → bypasses the 5 MB cap, loads whole file | enforce `TEXT_FILE_MAX_BYTES` unconditionally / read only the prefix |
| `workflow-engine-2` | workflow | driver session (+ linked leaves) never closed → unbounded session-row leak per run | `close_session` in a `finally` |
| `workflow-engine-3` | workflow | `foreach` `as` var not validated against `{{var}}` grammar → silent per-iteration input corruption | reject `as` not matching `^[A-Za-z_]\w*$` |
| `workflow-engine-4` | workflow | reference validation checks id existence, not ordering/reachability → forward/cross-branch refs pass then fail at runtime | validate against completed-before set per execution path |
| `workflow-durability-2` | workflow | `budget_exceeded` not in `_TERMINAL_STATUSES` → journal never frozen against late step writes | add it to `_TERMINAL_STATUSES` |
| `llm-transport-1` | transport | `POWER_LOOP_SUPPORTS_*` capability overrides are **dead config** (zero callers) | wire `capability_overrides_from_env` into `to_openai_compatible()` |
| `llm-transport-3` | transport | Anthropic extended-thinking signature blocks dropped → breaks multi-turn tool use with thinking on | preserve thinking blocks (+signature) and re-emit them first |
| `async-control-2` | timers | `TimerRunner.stop()` blocks for the full duration of an in-flight timer-fired run (no prompt shutdown) | plumb a `CancellationToken` into the timer-fired send |
| `exec-skills-structured-1` | skills | one non-UTF-8 `SKILL.md` aborts the whole `SkillLoader` (all skills lost) | per-file try/except, `errors="replace"`, quarantine |
| `exec-skills-structured-2` | structured | `parse_structured` rejects valid JSON when prose preamble has an odd `"` count | reset string-state at each top-level `{` |
| `contrib-observability-2` | otel | `_start`/`set_attribute` unguarded → breaks otel_sink's own "never break the loop" guarantee | wrap `handle()` body in try/except |
| `contrib-observability-6` | security | redaction is **key-name only** — secrets in *values* (bash strings, headers-as-text) persisted in cleartext | document + optional value-pattern redactor |

---

## LOW (32)

| ID | Area | Issue |
|---|---|---|
| `pipeline-runner-3` | events | no-tools follow-up drain skips `ROUND_COMPLETED`/`ROUND_END`/`on_round_ended` |
| `pipeline-runner-4` | data | `on_assistant_tool_calls` overwrites the correct DB `assistant_seq` with the in-memory history index |
| `pipeline-runner-5` | events | `@phase` decorator leaks unbalanced start/end events on error/retry-failure paths |
| `store-core-2` | concurrency | `upsert_background_task` monotonic `last_seen_at` bump is a non-atomic read-modify-write |
| `store-core-3` | concurrency | `create_timer`/`add_note` id-allocation race for sessions with no `session_state` row (PG/MySQL) |
| `store-dialect-3` | correctness | `table_prefix` length unvalidated → long prefix overflows MySQL's 64-char identifier limit |
| `session-store-legacy-1` | data | `checkpoint()` ignores a `BUSY` result → `-wal` not truncated when a pooled reader holds a lock |
| `session-store-legacy-2` | api | schema-version namespace divergence (legacy `PRAGMA user_version=1` vs new store) |
| `compaction-1` | correctness | malformed `CONTEXT_COMPACT_THRESHOLD` env crashes the whole send (trigger not fail-soft) |
| `compaction-2` | data | length-preserving `SESSION_START`/`ROUND_START` swap defeats the compaction alignment guard |
| `projection-2` | correctness | `ProjectionRenderConfig` bool/format knobs not coerced from JSON (`"false"` reads as `True`) |
| `projection-3` | correctness | `render_user_row` iterates a non-list `input`/`human` char-by-char (corrupt/custom rows) |
| `default-tools-2` | leak | `BackgroundManager.tasks` grows unbounded, pinning per-session store/event_loop forever |
| `default-tools-3` | correctness | bash home-scope guard false-positives on dirs whose path is a superstring of `POWER_LOOP_HOME` |
| `default-tools-4` | leak | persistent bash buffers all output in memory until sentinel/timeout (no streaming cap) |
| `subagent-coordination-2` | leak | EPHEMERAL sub-agent session leaked when `child_loop.send()` raises (no try/finally) |
| `subagent-coordination-3` | correctness | sub-agent blackboard author label is the raw session UUID, never the spec name |
| `workflow-engine-5` | correctness | `output_schema` shape unvalidated → malformed schema passes, breaks structured output at runtime |
| `workflow-durability-3` | concurrency | `resume_run`/`resume_detached` don't guard a still-running run → second concurrent engine |
| `workflow-durability-4` | data | `reap_runs` can delete a still-live but quiet leaf's DB mid-run (mtime-only liveness) |
| `llm-transport-2` | design | Anthropic transport ignores `ModelCapabilities`; image data-url support not gated |
| `llm-transport-4` | correctness | Anthropic image content inside `tool_result` flattened to text and dropped |
| `llm-transport-5` | correctness | OpenAI streaming tool-call accumulator collides concurrent calls when provider omits id/index |
| `memory-hooks-2` | concurrency | shared single `MemoryRecallHook` memoizes recall across concurrent sessions |
| `async-control-4` | perf | timer `heartbeat_interval_s=0` → busy re-stamp loop on the firing row |
| `prompt-sink-provider-2` | design | `SQLiteSink` violates the `MessageSink` "MUST NOT raise on normal paths" contract |
| `exec-skills-structured-4` | api | `StructuredOutputError` documents a `wrong_type:` reason the code never produces |
| `exec-skills-structured-5` | api | `StructuredOutputSpec.examples` is public but silently dropped from `response_format` |
| `exec-skills-structured-6` | design | `ToolArgsValidator` protocol exported but never used as an injection seam |
| `exec-skills-structured-7` | correctness | `AgentMessage.to_openai_message` can emit a tool-role message with no `tool_call_id` |
| `contrib-observability-4` | perf | `JsonlSink` writes+`flush()` inline on the publish thread, stalling the loop |
| `contrib-observability-5` | data | `JsonlSink` with `max_bytes>0` + `backup_count=0` silently discards all events on rotation |

---

## Manually re-verified (cut by session limit)

The two cross-cutting sweep dimensions and `memory-hooks-3` had **0/0 votes** — verification was lost
to the session limit, *not* refuted. I checked them by reading the code:

| Finding | My verdict |
|---|---|
| `xcut-api-security-1` — `_safe()` allows `.`/`..` so `run_id` escapes `runs_dir` | **Real, LOW.** `_safe` (`subprocess_executor.py:95`) replaces `/`→`_` but keeps `.`, so `run_id=".."` → `runs_dir/..` escapes exactly **one** level. Can't chain (slashes sanitized) and `run_id` is framework-generated. Harden: reject `.`/`..`. |
| `xcut-api-security-2` — bash home guard bypassable via substring | **Real, LOW** — same root as `default-tools-3`. The guard (`default_tools.py:283-320`) is substring-based and self-documents (lines 313-316) as non-exhaustive; the real boundary is the exec backend. |
| `xcut-api-security-3` — redaction key-name-only | **Duplicate** of `contrib-observability-6` (already counted). |
| `xcut-concurrency-1` — shared recall memoization across sessions | **Duplicate** of `memory-hooks-2` (already counted). |
| `xcut-concurrency-2` — async SQLite serializes reads behind the write lock | **Not a bug** — `store/backends/sqlite.py:10` documents this as the intentional single-connection design; the read pool lives in legacy `session_store.py`. |
| `memory-hooks-3` — `add_note` reuses `MAX(note_id)+1` | **Real, LOW.** `store.py:1041` is under a state-row lock (no PK collision), but deleting the newest note and re-adding reuses its id → stale reference staleness. FIFO eviction removes the *lowest*, so it isn't triggered there. |

Net new from this batch: **+2 genuine LOW** (run_id hardening, note_id reuse). No change to the
headline.

---

## Refuted (16, for transparency)

These were raised and rejected by adversarial verification (so the report stays honest about false
positives): `pipeline-runner-2` (compaction `phase` only "started" — observation true, not a defect),
`store-core-1`, `store-dialect-2`, `session-store-legacy-3`, `projection-4`, `subagent-coordination-1`
(claimed cross-session race doesn't exist — child gets parent's hooks but not the memoization race),
`memory-hooks-1` (1/3 — real logic but contested impact), `memory-hooks-3` (re-classified Real-LOW
above), `async-control-3`, `exec-skills-structured-3` (misreads `re.MULTILINE` vs JSON quoting),
`contrib-observability-3`, plus the 5 `xcut-*` 0/0 entries re-adjudicated above.

---

## Suggested fix order

1. **H1 + H6 + H2** — silent data loss / corruption first: projection drops the submitted answer,
   multimodal stringification, MySQL 64 KiB. (H1/H6 are one-day fixes; H2 needs a migration.)
2. **H5 + H7** — runtime aborts that surface as opaque provider-400s / crashed runs.
3. **H3 + H4** — workflow safety: clamp fanout, fix the subprocess hang (process group + bounded drain).
4. **Theme 2 cleanup** — extract one shared tool-call/result atomicity guard and route every history
   transform (trim/projection/transport) through it; it's currently re-implemented 4+ times.
5. **Theme 3 cleanup** — enforce "observability never breaks/stalls the loop" at the bus
   (`suppress_subscriber_errors` default + async sink dispatch) rather than per-sink.
6. Medium/low fail-soft + dead-config items as cleanup.

## Coverage gaps / follow-up for a next pass

- **No real-LLM projection test exercises `submit_input`/`resume`** (would have caught H1) — add one.
- **No >64 KiB payload parity test** across the 3 backends (would have caught H2).
- The **legacy `session_store.py` vs new `store/` package** divergence (schema-version namespace,
  read-pool semantics, checkpoint BUSY handling) deserves a dedicated reconciliation pass.
- A focused **concurrency stress harness** (interleaved `submit_input`/timer-fire/follow_up + parallel
  sub-agents on one session) — the cross-cutting concurrency sweep was the least-covered dimension this
  run (its verification was the part lost to the session limit).
- One finder dimension lost its output to a `StructuredOutput` retry-cap failure mid-run; a re-run of
  just that dimension is cheap insurance.
