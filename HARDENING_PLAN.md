# power-loop: Hardening Plan to 1.0 (continuation past 0.13.1)

> **Status — shipped in 0.14.0 (2026-06-15):** all confirmed correctness bugs **C1–C8** fixed
> (each red-before/green-after); **H1** correctness lockdown complete (incl. C12); **H2** test/CI
> rigor (coverage gate, strict markers, property tests, examples-smoke, import-without-extras leg,
> security-branch tests — which found & fixed a real `rm -rf /<sysdir>` false-negative); **H3**
> packaging (lazy SDK imports + extras, vendored `llm_client`, featherweight core, top-level LLM
> re-exports, STABLE_API single-source + SemVer guard, py.typed, Beta classifier); **H4** the
> high-value observability (per-call LLM events, error codes, logging hygiene + redaction; ts/seq
> envelope); **H5.1** the bind-injection footgun; plus the **H7** compaction-coordination track
> (recall_compacted tool + widened Compactor). Deferred as acceptable: H4.3 subagent provenance,
> H5.2 bounded-growth, C10/C11/C13, H2.8/H2.10.
>
> Continues `ROADMAP.md` (M0–M3, most of M3 shipped) toward a 1.0-grade "very high standard" bar. Tracks below are named **H1–H6** to sit after M3 without renumbering it. Same conventions as ROADMAP: every track ends with README + CHANGELOG updates or it doesn't count; LLM-behavior items require ≥1 real-LLM test.
>
> Method: this plan is the output of a 6-dimension code audit (core correctness · workflow/subprocess/blackboard · test rigor · packaging/API/typing · observability · architecture/DX), followed by an **adversarial verification pass** that re-read the cited code for every concrete bug-claim and tried to refute it. Only claims that survived are in §1; the refuted/downgraded ones are listed for credibility.
>
> Last updated: 2026-06-15.

---

## 0. Honest verdict

0.13.1 is **architecturally 1.0 and engineering-wise mid-0.x**. The seam design (Protocol + default impl + `runtime_env_context` injection), the single SQLite persistence base, and the durable-timer / CAS / heartbeat machinery are genuinely strong, and the 0.13.1 fixes were real. But the verification pass surfaced **a cluster of latent correctness bugs that all share one root cause** — state mutated outside the canonical write path, with no realignment — plus a packaging story that contradicts the library's own headline positioning, and an observability surface that cannot support the OTel bridge the ROADMAP already promises. None of this breaks the happy path; **all of it breaks under failure, cancellation, restart, memory recall, or a long-lived host** — exactly the regimes a 1.0 "eyes-closed production-ready" claim must cover. The honest gap is: the kernel's hardest paths (timers, subprocess cancel/resume) are adversarially tested, but the *combinations* (recall + compaction, halt + fan-out, eager-wake + GC) and the *non-default provider's own unit tests* are not. This is a finishable gap — concentrated, well-localized, and mostly addressed by making three invariants explicit (index alignment, fan-out cancellation, wake durability) and making the OpenAI transport + import-without-extras testable.

---

## 1. Confirmed correctness issues (survived adversarial verification)

Ordered by blast radius. Each is grounded in code re-read during the audit.

### C1 — Memory recall desyncs `pipeline.history` from `sink._history_seqs` → compaction marks the WRONG DB rows `compacted_out` `[high · conf:high]`
- **Where:** `power_loop/core/pipeline.py:374` (`self.history[insert_at:insert_at] = tagged`) inserts recalled `memory_*` messages directly, never through `_append_message`/the sink. `_history_seqs` (`power_loop/agent/sink.py:96-99,120,160,172`) is only grown by `init_history_seqs` + `on_message_appended`, so after recall `len(history) == len(_history_seqs) + len(tagged)` and every index `≥ insert_at` is shifted. `on_compaction` (`sink.py:209-228`) maps `fold_start_idx/fold_end_idx` → `_history_seqs[...]`, now off by `len(tagged)`; `record_compaction` (`session_store.py:695-698`) runs `UPDATE messages SET state='compacted_out' WHERE seq BETWEEN from_seq AND to_seq` on the wrong seqs and then rewrites `_history_seqs` incorrectly, **cascading into all later compactions**.
- **Why it matters:** silent, persisted, self-propagating state corruption gated only on the optional-but-documented `MemoryProvider` running with the default-on compactor — i.e. a first-class supported config. Corrupts resume/`load_active_messages`.
- **Fix:** Make `_history_seqs` the single index-alignment source. Preferred: route recalled messages through `_append_message` with a transient/synthetic seq the sink records, keeping `history` and `_history_seqs` 1:1; alternative: pass fold *message objects* (or their stored seqs) to `on_compaction` instead of positional indices. Regression test: `StatefulAgentLoop` + `MemoryProvider` recalling ≥1 msg + a firing compactor, assert the `compacted_out` seq set equals exactly the folded range.
- **Follow-up (fixed 0.14.1):** the 0.14.0 fix kept the index map aligned across recall, but a *different* same-root manifestation slipped the acceptance net — a **second compaction within one run**. The `compact_note` got a fresh HIGH identity `seq` placed at a LOW logical index, so `_history_seqs` went **non-monotonic**; the next fold's translated `BETWEEN from_seq AND to_seq` inverted (`from_seq > to_seq`) → DB marked nothing while the in-memory fold proceeded → divergence + inverted audit rows; and even a single note reloaded below the kept tail. Fixed by decoupling identity (`seq`) from logical position: mark by **explicit seq-set** (`seq IN (…)`, monotonicity-independent), store the note's logical `ord` in `meta`, order `load_active_messages` by it, and track a parallel `_history_ord` in the sink. See `tests/unit/test_compaction_double_fold.py` (11 cases incl. a 200-example property test) and CHANGELOG 0.14.1. **Lesson:** the original C1 regression test folded only ONCE per run — the multi-fold-in-one-run regime was untested.

### C2 — `parallel`/`foreach` `on_error="halt"` orphans sibling sub-agents AND clobbers the finalized journal `[high · conf:high]`
- **Where:** `power_loop/workflow/engine.py:357-360` and `:391-395` call `asyncio.gather(..., return_exceptions=(on_error=="continue"))`; default `on_error="halt"` (`spec.py:117,137`) ⇒ `return_exceptions=False`. `gather` re-raises on first failure but **does not cancel in-flight siblings** (engine passes bare coroutines, no `Task` handles; `self._cancel` is never flipped on halt — confirmed no `.cancel()` in `engine.py`). Orphans keep calling the real LLM, then `_emit_step → make_on_step → journal.record_step` (`runner.py:53-59`) fires **after** `_bg` already caught `WorkflowRunError` and ran `journal.finalize/fail` (`runner.py:151-162`). `record_step` is an uncas'd read-modify-write (`journal.py:147-164`; the lock is released between read and write) → the late step reverts `status` to `'running'` and nulls the result.
- **Why it matters:** corrupts a finalized run's status/result, breaks resume/introspection, burns real tokens on work whose result is discarded.
- **Fix:** Gather into explicit `asyncio.Task`s; on first failure under halt, flip `self._cancel` and `task.cancel()` the pending siblings, await them `return_exceptions=True`, then raise `WorkflowRunError`. Independently, make `record_step`/`update` no-ops once the journal status is terminal (idempotent late-write guard).

### C3 — `eager_wake` fires an untracked `follow_up` task that can be GC'd → parent wake permanently lost `[high · conf:high]`
- **Where:** `power_loop/workflow/runner.py:163-172`: claims the wake (`journal.update(..., woke=True)`, line 170) **then** `asyncio.create_task(loop.follow_up(...))` with no retained reference (line 172) — unlike the `_bg` task added to `task_set` at 174-176, and unlike `resume.py:59-60`'s explicit `_RESUME_TASKS` GC guard. CPython holds only a weak ref to a bare `create_task`, so it may be collected mid-flight; `_suppress` (`runner.py:180-185`) only swallows the synchronous `create_task()` call, never the coroutine's later exception. Because `woke=True` is already committed, `make_wake_guard` (`runner.py:199-201`) SKIPs the durable timer ⇒ **the durable safety net is deliberately disabled**.
- **Why it matters:** deterministic liveness loss of a `pass_turn`'d parent agent when `follow_up` raises (session closed / lock edge) or the task is GC'd. Opt-in (`eager_wake` defaults False) but public API.
- **Fix:** Deliver-then-claim: retain the follow_up task in a task set with an `add_done_callback`, and only set `woke=True` from its success path — OR keep claim-first but have the done-callback re-open `woke=False` on failure so the durable timer still fires. At minimum, retain the reference.

### C4 — Unexpected exception in `pipeline.run()` escapes with no `SESSION_ENDED` and no error event; `AGENT_ERROR` is documented but never emitted `[high · conf:high]`
- **Where:** `power_loop/agent/stateful_loop.py:678-699` wraps `pipeline.run()` in `try/finally` with **no `except`** (finally only resets the loop token); `send()` (`:180-195`) has no guard either. In `pipeline.run()` the only broad `except Exception` (`pipeline.py:946`) covers *only* `execute_tool`, not the ~11 unguarded `hooks.run_typed_async(...)` calls (`core/hooks.py:112-118` has no try/except). `SESSION_ENDED` fires only via `_finalize` (`pipeline.py:315`), `SESSION_STARTED` fires at `:646` before the loop. `AGENT_ERROR`/`AgentErrorPayload` are defined (`contracts/events.py:51`, `event_payloads.py:269-272`), exported, and **documented as emitted** (`docs/en/user-guide/events.md:118`, `docs/events.md:338`) — yet grep finds zero publishers.
- **Why it matters:** a crashing run yields a raw traceback and strands every subscriber that saw `SESSION_STARTED` with no terminal; the advertised error channel is dead code, a documentation lie.
- **Fix:** Wrap the loop in `except Exception`: emit `AGENT_ERROR(AgentErrorPayload(...))`, call `_finalize("error")` so `SESSION_ENDED` fires, then re-raise (or return `status="error"`). Add a test asserting both events on a raising hook.

### C5 — `reap_runs` aborts the whole GC sweep on a concurrent unlink (unguarded `f.stat()`) `[medium · conf:high]`
- **Where:** `power_loop/workflow/subprocess_executor.py:133-134`: `files = list(run_dir.iterdir())` then `max((f.stat().st_mtime for f in files), ...)`. WAL sidecars (`session_store.py:438` sets `journal_mode=WAL`) are created/removed by the separate worker process; `delete_on_success` (`:238`) removes dbs mid-sweep. A `FileNotFoundError` from `f.stat()` propagates out, aborting the `for run_dir in base.iterdir()` loop and leaving all later dirs un-GC'd. The sibling `_remove_db` (`:99-107`) already swallows `FileNotFoundError/OSError` — the asymmetry is the bug.
- **Fix:** Guard per-file `try/except FileNotFoundError/OSError` (skip missing) or wrap each `run_dir` body in `try/except OSError; continue` — mirror `_remove_db`.

### C6 — Sync `publish()` silently disconnects async-subscriber exceptions even when `suppress_subscriber_errors=False` `[medium · conf:high]`
- **Where:** `power_loop/core/events.py:103-107`: with a running loop, async handlers are wrapped in `loop.create_task(_run_coro(...))` fire-and-forget; `_await_handler_result` (`:75-84`) re-raises only when the flag is False, but that re-raise now runs **inside the detached task**, surfacing only as a GC "Task exception was never retrieved" warning. The running-loop branch is the normal runtime path (`core/phase.py:149,188`, `pipeline.py:263 _emit`). `publish_async` honors the flag correctly.
- **Fix:** Attach a done-callback to the created task that logs/re-raises per the flag; document that async subscribers must use `publish_async`. (Pairs with C3 — both are untracked-task footguns.)

### C7 — Store read-modify-write atomicity comes from the `RLock`, not the DB transaction; the comment overstates it `[medium · conf:high]`
- **Where:** `session_store.py:436` opens with `isolation_level=""` (deferred). In legacy `sqlite3` deferred mode the transaction auto-BEGINs only before the first DML, **not** before a leading `SELECT`. In `append_message` (`:603-633`), `record_compaction` (`:685-744`), `upsert_background_task` (`:890-934`), `create_timer` (`:1165-1179`) the `SELECT next_seq/MAX(...)` runs in autocommit, outside the write txn. Protection is the `self._lock` (RLock), not isolation — but the comment at `:428-435` claims the deferred mode "restored the atomicity those methods document." The atomicity test (`tests/unit/test_session_store_atomicity.py:11-29`) opens its `with` with an INSERT first, never exercising the leading-SELECT pattern.
- **Fix:** Either issue `BEGIN IMMEDIATE` at the top of each RMW method so the SELECT runs inside the reserved write txn, **or** correct the comment (in-process atomicity = lock; cross-process safety = `(session_id, seq)` PK raising `IntegrityError`). Add a test whose `with` begins with a SELECT then DML and asserts rollback.

### C8 — All SQLite store calls run synchronously inside async methods → one session's write/compaction freezes the event loop `[medium · conf:high]`
- **Where:** `session_store.py:19-21` tells callers to wrap in `asyncio.to_thread`, but `pipeline.py:426,432,706,806,847` and `stateful_loop.py` call the store directly (grep: zero `to_thread` in pipeline/stateful_loop/sink). With `busy_timeout=5000` (`:441`), a contended write blocks the awaiting coroutine — and thus the whole loop — up to 5s. The `StatefulAgentLoop` docstring (`stateful_loop.py:85-88`) advertises "any number of sessions concurrently."
- **Fix (1.0 bar):** Wrap all store/sink calls from async pipeline code in `asyncio.to_thread` (the RLock already makes the store thread-safe). Minimum acceptable: downgrade the concurrency claim in the docstring.

### Lower-severity confirmed (fix opportunistically)
- **C9** `[medium]` SESSION_START/ROUND_START hook replacing `self.history` wholesale desyncs `_history_seqs` — same root cause as C1 (`pipeline.py:644-645`). Fix with C1; at minimum forbid message replacement when a non-Null sink is attached.
- **C10** `[medium]` Subprocess worker seeds each leaf as a **root** session `spawn_depth=0` (`worker.py:170`), resetting the cross-process recursion ceiling. Mitigated today (`tool_preset=None` ⇒ no spawn tools); fix by threading parent `spawn_depth` into `WorkerJob`, or refuse spawn tools on subprocess leaves unless opted in.
- **C11** `[low]` `pending.assistant_seq` stores `len(self.history)` not the DB seq (`pipeline.py:805`); informational-only (resume keys off `round_index`+`tool_calls`). Pass `self._assistant_seq` (already the real seq, `sink.py:162`).
- **C12** `[low]` Per-session `_locks`/`_follow_up_queue_locks` never popped on `close_session` (`stateful_loop.py:119-121,130-132`) — slow monotonic leak in long-lived hosts. Pop the three keys in `close_session`.
- **C13** `[low]` `_db_path_for` uuid suffix orphans a failed leaf's db on every resume re-run (`subprocess_executor.py:192-199`); only mtime GC reclaims. Make path deterministic per `(run_id, node_id)` (keep per-iteration index for foreach).
- **C14** `[low]` `_dangerous_command_reason` regex/blocklist branches and `_validate_bash_command_scope` (`tools/default_tools.py:256,279-305`) are security-critical and untested — covered under H2.

### Checked-and-dismissed (credibility signal)
- **REFUTED — "Detached run dies silently when parent closed mid-flight (journal frozen at 'running')":** `journal.update` does `get_runtime_state` first and `if j is None: return None` (`journal.py:115-123`); `_delete_session_tree` co-deletes `session_runtime_state` + `sessions` in one txn (`session_store.py:548-574`). `finalize/fail` no-op safely when the parent is gone — the guard the claim wanted already exists. **Not a bug.**
- **REFUTED / DOWNGRADED — "`import power_loop` is a CRITICAL bug":** the eager dual-SDK import chain is real (`__init__.py:145 → provider.py:27,29 → anthropic_factory.py:10 / llm_factory.py:20`) and `import power_loop` does `ModuleNotFoundError` without `anthropic` in a stripped env — but `pyproject.toml:25-33` declares BOTH SDKs as hard deps, so a supported `pip install` never fails. This is a **packaging/positioning defect (H3.1), not a correctness blocker.** Severity: medium.
- **DOWNGRADED to low:** the STABLE-tier 3-way disagreement (docstring vs `STABLE_API` vs `__all__`) is docs/contract metadata only — confirmed `__init__.py:9-16` vs `:207-228`, every entry resolves, nothing mis-exports.
- **DOWNGRADED to medium / scoped:** `bind=True` default shadowing outer `runtime_env_context(shell_backend=...)` is real (`tools/__init__.py:33,97-103`) but the **blackboard half is unreachable** via the default registry (blackboard tools aren't default tools), and **no shipped example/README path actually triggers a non-sandboxed bash**. Latent footgun, not a live break.
- **CONFIRMED-but-accepted:** the `follow_up` terminal-drain TOCTOU (`stateful_loop.py:218-221`, `pipeline.py:814-833`) is documented best-effort for the DeepTalk dispatch model — no code change required; keep the explanatory comment.

---

## 2. Phased hardening roadmap (H1–H6)

Ordered by leverage. **H1 and the import split (H3.1) are the only things that gate everything else** — both are short. H2 must land the regression tests *for H1's bugs* before H1's fixes are declared done.

---

### H1 · Correctness lockdown — the index-alignment & fan-out invariants
**Goal:** eliminate every confirmed silent-state-corruption path. The theme is a single invariant per subsystem, made explicit and enforced.

| # | Deliverable | Files | Fixes |
|---|---|---|---|
| H1.1 | `_history_seqs` is the *sole* index-alignment source. Route recall through `_append_message` (transient seq) **or** pass fold message-objects to `on_compaction`. Forbid hook history-replacement under a non-Null sink (or realign). | `core/pipeline.py:374,644-645`, `agent/sink.py:209-228` | C1, C9 |
| H1.2 | Fan-out halt cancels siblings: gather explicit `Task`s, flip `self._cancel`, `task.cancel()` pending, await `return_exceptions=True`, then raise. | `workflow/engine.py:357-360,391-395` | C2 |
| H1.3 | Late-write idempotency: `record_step`/`update` are no-ops once journal status is terminal. | `workflow/journal.py:115-164` | C2 |
| H1.4 | Eager-wake deliver-then-claim + retain task; durable timer survives follow_up failure. | `workflow/runner.py:163-172` | C3 |
| H1.5 | `AGENT_ERROR` wired: `except Exception` around the loop → emit error event + `_finalize("error")` → re-raise. | `agent/stateful_loop.py:678-699`, `core/pipeline.py:315` | C4 |
| H1.6 | `reap_runs` per-file stat guard. | `workflow/subprocess_executor.py:133-134` | C5 |
| H1.7 | Sync `publish()` done-callback honors `suppress_subscriber_errors`; doc async→`publish_async`. | `core/events.py:103-109` | C6 |
| H1.8 | Store-atomicity: `BEGIN IMMEDIATE` in each RMW method **or** corrected comment + leading-SELECT atomicity test. | `session_store.py:428-435,603-633,685-744,890-934,1165-1179` | C7 |
| H1.9 | Offload store/sink I/O via `asyncio.to_thread` from async pipeline paths. | `core/pipeline.py:426,432,706,806,847`, `agent/stateful_loop.py` | C8 |
| H1.10 | Opportunistic: pop session lock dicts on close (C12); pass real seq to `on_assistant_tool_calls` (C11); deterministic leaf db path (C13); thread `spawn_depth` into `WorkerJob` (C10). | as cited | C10–C13 |

**Acceptance:** new regression tests (H2.1) reproduce C1/C2/C3/C4/C5 *before* the fix (red) and pass after (green); `pytest --no-real` green; no `compacted_out` mis-map under recall+compaction; a halted fan-out leaves zero orphan tasks (`asyncio.all_tasks()` assertion); a GC'd/raising eager follow_up still wakes via the durable timer.
**Effort:** ~L (the cluster is localized but each needs its own adversarial test). **Depends on:** nothing. **Blocks:** H4 (AGENT_ERROR feeds the event envelope work), H5.

---

### H2 · Test & CI rigor — close the breadth gaps
**Goal:** make the regression set adversarial for *combinations* and the default provider; make CI able to *see* the import-without-extras and security-branch blind spots.

- **H2.1 — Regression tests for every H1 bug** (the gate for declaring H1 done): recall+compaction seq-set equality; halt fan-out orphan-cancellation; eager-wake GC/raise durability; AGENT_ERROR + SESSION_ENDED on raising hook; `reap_runs` concurrent-unlink survival. *Files:* `tests/unit/test_compact.py`, new `tests/unit/test_workflow_fanout.py`, `tests/unit/test_workflow_detached.py:167-189`, new `tests/unit/test_pipeline_error_terminal.py`.
- **H2.2 — `tests/unit/test_openai_transport.py`** mirroring `test_anthropic_transport.py`: fake `AsyncOpenAI` returning a scripted chunk async-iterator (text deltas, a tool_call delta split across chunks, a usage chunk, a mid-tool-call truncation to hit the stream-resume path at `llm_factory.py:496`). Asserts text/tool-call/usage accumulation and `_request_kwargs` mapping. Moves the **default** provider from nightly-only to per-PR. `[high]`
- **H2.3 — Coverage gate:** add `pytest-cov` to dev extras; CI runs `--cov=power_loop --cov=llm_client --cov-report=term-missing --cov-fail-under=70` (ratchet up over time). *Files:* `pyproject.toml:36-41`, `.github/workflows/ci.yml:38`.
- **H2.4 — Security-branch tests** (parametrized) for `_dangerous_command_reason`: each blocklisted binary (incl. `/usr/bin/sudo`), `rm -rf /`, `rm -fr ~`, `rm -rf $HOME`, `cat foo > /dev/sda`, **plus negative** cases (`rm -rf ./build`, `echo dd`); + `_validate_bash_command_scope` in/out of `home_rw_allowlist`. *File:* `tests/unit/test_default_tools.py` covering `default_tools.py:246,256,279-305`. `[high]` (C14)
- **H2.5 — Import-without-extras CI leg:** after H3.1, a matrix job installing the OpenAI-only subset runs `python -c "import power_loop"` + an OpenAI smoke. Until then, an `xfail` test documenting the constraint. `[medium]`
- **H2.6 — Property tests (`hypothesis`)** for the three hand-rolled state machines: (a) `parse_structured` either raises `StructuredOutputError` or returns `json.loads` of the embedded object; (b) `DefaultCompactor.maybe_compact` over random histories always passes `_assert_no_orphan_tools` and never drops the first system message (generalizes the orphan-tool regression); (c) `WorkflowSpec.from_json(spec.to_dict())` round-trips. *Files:* `structured.py:88-206`, `compact.py:156-240`, `spec.py:183-235`.
- **H2.7 — Marker taxonomy honesty:** add `addopts = "--strict-markers --strict-config"`; a conftest hook failing collection if any `tests/unit|integration` file lacks exactly one tier marker; **either** populate `tests/integration/` with real multi-component scenarios (workflow+subprocess+resume) **or** drop the tier from CI/docs. *Files:* `pyproject.toml:90-94`, `conftest.py`.
- **H2.8 — Strengthen weak assertions:** `test_compact.py:143-159` → deterministic trigger (set `CONTEXT_COMPACT_THRESHOLD`) + positive assertion that the atomic `assistant(tool_calls)+tool` pair survives; extend the `run_grep` rg-spy (`test_default_tools.py:142-156`) to return rc=2/rc=1/rc=0-with-truncation. `[low]`
- **H2.9 — PR-time examples smoke:** import each `examples/NN_*.py` under the echo/stub provider (`stub_provider.EchoLLMService`, `POWER_LOOP_PROVIDER=echo`) and assert it constructs without raising — catches public-API rename breakage per-PR; leave semantic validation to nightly. *File:* new `tests/unit/test_examples_smoke.py`.
- **H2.10 — mypy covers transports:** CI runs `mypy power_loop llm_client` (start lenient); ratchet `check_untyped_defs=true` first on `llm_client/interface.py` and `llm_factory.py`. *Files:* `pyproject.toml:84`, `ci.yml:35`.

**Acceptance:** coverage ≥70% on both packages and rising; default transport + all security branches covered per-PR; `pytest -m unit` runs the full unit set (markers strict); examples smoke green. **Effort:** L. **Depends on:** H1 fixes exist (H2.1 tests them); H3.1 (H2.5).

---

### H3 · Packaging & public API for 1.0
**Goal:** make the "featherweight" positioning true, stop shipping a squat-prone top-level package, and make the type story real.

- **H3.1 — Lazy transport imports + optional extras** `[critical-for-positioning]`: move `from llm_client.anthropic_factory import ...` and `from llm_client.llm_factory import ...` *inside* `create_llm_service_from_config` (mirror the existing lazy `EchoLLMService` at `provider.py:182-183`); keep `from llm_client.interface import ...` at module level (interface is SDK-free). Move `anthropic`/`openai` out of `dependencies` into `[project.optional-dependencies]` (`anthropic=[...]`, `openai=[...]`, `all=[...]`); raise a clear `ImportError` with install hint at construction if the chosen SDK is missing. Fix `README.md:29`. *Files:* `provider.py:27-29`, `pyproject.toml:24-33`, `README.md:29`.
- **H3.2 — Vendor `llm_client`** `[high]`: move `llm_client/` → `power_loop/_vendor/llm_client/` (or `power_loop/llm/`), rewrite the ~10 absolute imports (`provider.py`, `compact.py`, `structured.py`, `stub_provider.py`, `core/state.py`, `core/pipeline.py`, `agent/stateful_loop.py`, `contracts/hook_contexts.py`, `workflow/worker.py`), drop `llm_client*` from `pyproject.toml:55-56` packages.find. The wheel's `top_level.txt` currently ships a bare un-prefixed `llm_client` — a real namespace-collision hazard with a 0-byte `__init__.py`.
- **H3.3 — `py.typed`** `[high]`: add empty `power_loop/py.typed` + register as package-data; without it, per PEP 561 the entire annotated API is invisible to downstream mypy/pyright. (Only add `_vendor`'s after it is type-checked under H2.10.)
- **H3.4 — Re-export the public LLM contract** `[high]`: add `LLMService, LLMRequest, LLMResponse, LLMStreamChunk, LLMTokenUsage, OpenAICompatibleChatConfig, AnthropicChatConfig` to `__init__.py`/`__all__` (sourced from the SDK-free `interface.py`), so users writing `llm.after` hooks (`LlmAfterCtx.output: LLMResponse | None`, `hook_contexts.py:134,146`) don't import across into a private package.
- **H3.5 — Right-size dependencies** `[medium]`: drop `socksio` (never imported; httpx pulls it transitively), move `python-dotenv` to dev/examples extra (only in `examples/` + guarded `conftest.py:21`), move `pyyaml` to a `[skills]` extra (`skills.py:29,96` already degrades gracefully). Sync `requirements.txt` or delete it for pyproject-as-single-source. *File:* `pyproject.toml:28-31`.
- **H3.6 — STABLE-tier single source of truth** `[low]`: make `STABLE_API` (`__init__.py:207-228`) authoritative; add a test asserting docstring STABLE list == `STABLE_API`; decide `FollowUpQueued`'s tier; add a CI guard test that fails if a `STABLE_API` symbol is removed/renamed without a major bump (enforce the line-6 SemVer promise).
- **H3.7 — Bump classifier** `[low]`: `Development Status :: 3 - Alpha` → `4 - Beta` now, `5 - Production/Stable` at 1.0 (`pyproject.toml:14`).

**Acceptance:** `pip install power-loop` (no extras) → `python -c "import power_loop"` succeeds with zero SDKs; wheel `top_level.txt` lists only `power_loop`; downstream mypy sees power-loop types; `STABLE_API`-guard test green. **Effort:** M. **Depends on:** nothing (H3.1 unblocks H2.5). **Blocks:** 1.0 tag.

---

### H4 · Observability for the OTel bridge
**Goal:** make the ROADMAP-M3 "AgentEvent → OTel span" a thin subscriber, not a rewrite; surface per-call latency/cost/retries.

- **H4.1 — Per-call LLM events** `[high]`: emit `LLM_CALL_STARTED`/`LLM_CALL_COMPLETED` inside `call_llm` (`pipeline.py:526-607`) with `call_id`, `round_index`, `attempt`, `model`, per-call `token_usage` (the `LLMResponse.token_usage`, **not** the cumulative `ctx` total), `duration_ms` (`time.perf_counter()` around `llm.complete()`), `finish_reason`, success/error. Today there's zero latency instrumentation and `USAGE_UPDATED` fires once/round with cumulative totals, so retries are invisible. Optional pluggable `cost_callback(model, usage) → usd`.
- **H4.2 — AgentEvent envelope: `ts` + monotonic `seq` + optional `trace_id/span_id/parent_id`** `[high]`: `contracts/events.py:61-88` currently has none; populate in `_emit` (`pipeline.py:261`). This is the single change that turns the M3 OTel bridge from fabricated-timing-and-out-of-band-parentage into a clean span emitter.
- **H4.3 — Subagent event provenance** `[medium]`: thread `parent_session_id` + `depth` (and `source="subagent"`) onto every event a child pipeline emits (child shares the parent bus, `runtime/spec.py:274-281`), so a subscriber reconstructs the call tree from the stream alone without querying `SessionStore.parent_session_id`.
- **H4.4 — Machine-readable error detail** `[medium]`: add `reason/error_type/error_message/rounds` to `AgentLoopResult`/`StatefulResult` (`types.py:78-91`) populated on degraded/timeout paths (today that detail lives only on the transient event, `pipeline.py:754-763`); add a class-level `code: str` to each `PowerLoopError` (`'llm.timeout'`, `'session.pending'`, `'tool.not_found'`) so callers key on a stable code, not class identity.
- **H4.5 — Logging hygiene** `[low/medium]`: add `logging.getLogger("power_loop").addHandler(NullHandler())` in `__init__.py`; standardize all module loggers on `getLogger(__name__)` (today `timers.py:47`/`stateful_loop.py:61` hard-code names, breaking subtree routing). Add a key-name redaction denylist (`api_key/authorization/token/password/secret`) to `contrib/logging_sink.py:81-88` + a `redact_keys` param — tool inputs and request messages currently land verbatim at INFO.

**Acceptance:** a test subscriber reconstructs the parent→child call tree and orders events by `seq` from the bus alone; per-call latency + retry attribution visible without DB; an OTel example bridge (~50 LOC subscriber) emits correctly-parented spans. **Effort:** M–L. **Depends on:** H1.5 (AGENT_ERROR feeds the taxonomy).

---

### H5 · Architecture / seam hardening
**Goal:** close the injection footgun and the unbounded-growth leaks so long-lived hosts stay flat.

- **H5.1 — `bind=True` must not shadow outer `runtime_env_context`** `[medium]`: merge/honor outer `shell_backend`/`blackboard` instead of resetting the contextvar to a `from_env` snapshot (`tools/__init__.py:97-103`); add `shell_backend`/`blackboard` params to `create_default_tool_registry`; regression test the bound-default-registry + outer-context combination (untested today, `test_per_send_overrides.py:204-207`); document `bind=False` in `docs/.../blackboard.md`.
- **H5.2 — Bound growth in long-lived hosts** `[low]`: `SqliteBlackboard._locks` (`blackboard.py:135-142`) and `SubprocessExecutor` lock/dict growth — key by `board_id` in a bounded LRU/`WeakValueDictionary` or accept+document. Add an optional FIFO/auto-prune board policy so a full board degrades instead of hard-erroring (`blackboard.py:180-183`).
- **H5.3 — Subprocess robustness** `[low]`: bounded timeout on the final `await asyncio.shield(comm)` (`subprocess_executor.py:274-277`) returning "failed: child unreaped" instead of blocking forever on a D-state child; cap/truncate buffered worker stdout (`proc.communicate`, `:230`).
- **H5.4 — Workflow retention** `[low]`: `workflow:index` grows unbounded (`journal.py:104-108`); add an optional retention cap (N most-recent runs per parent, or prune-on-finalize older than T) + a public `prune` helper symmetric with `reap_runs`. On resume, cancel the prior attempt's durable wake timer or carry the attempt number in the wake note (`resume.py:139-148`) so a stale re-fire can't deliver an old attempt's note.

**Acceptance:** a 10k create/close-session loop leaves the in-memory lock dicts flat; the default registry under an outer `shell_backend` context routes to the injected backend (regression test green). **Effort:** M. **Depends on:** none (independent of H1).

---

### H6 · DX & docs
**Goal:** every public capability has doc + example + a real-LLM test; nothing in the docs is a lie.

- **H6.1** Reconcile `docs/events.md` / `events.md` with the actual emitted set (AGENT_ERROR now real per H1.5; add the H4.1 per-call events).
- **H6.2** Document the injection footgun fix (`bind=False` guidance), the optional-extras install matrix (`pip install power-loop[openai]`), and the redaction story.
- **H6.3** Per ROADMAP guardrail #3: every track ends with README + CHANGELOG; new public capabilities (per-call events, error codes, extras) get an `examples/NN_*.py` + a `real_llm` test.
- **H6.4** A short "production checklist" doc: wrap store I/O in `to_thread`, schedule `reap_runs`/`prune`, attach a redacting sink, set concurrency expectations.

**Acceptance:** docs lint (no reference to unemitted events); every `STABLE_API` symbol has a doc anchor + example. **Effort:** S–M. **Depends on:** H1, H3, H4 (documents their outcomes).

---

## 3. Quick wins this week (≤8, S/M, high payoff)

1. **H1.6** `reap_runs` per-file `stat` guard (S) — one `try/except`, stops the GC sweep from aborting. `subprocess_executor.py:133-134`.
2. **H1.4** Retain the eager `follow_up` task (S) — prevents permanent parent-wake loss. `runner.py:172`.
3. **H1.5** Wire `AGENT_ERROR` + `_finalize("error")` (S) — kills the documented-but-dead error channel and the terminal-less crash. `stateful_loop.py:678-699`.
4. **H3.1** Lazy transport imports + `[openai]`/`[anthropic]` extras + fix `README.md:29` (M) — makes "featherweight" true; unblocks the import CI leg.
5. **H3.3** Add `power_loop/py.typed` (S) — every downstream type-checker suddenly sees the API.
6. **H2.2** `test_openai_transport.py` with a fake `AsyncOpenAI` (M) — the **default** provider gets per-PR coverage for the first time.
7. **H1.3** Make `journal.record_step`/`update` no-ops once terminal (S) — neutralizes the C2 journal-clobber even before the cancel work lands.
8. **H4.2** Add `ts` + monotonic `seq` to `AgentEvent` (S) — cheap, and it's the keystone for the OTel bridge.

---

## 4. Definition of Done for 1.0

- **Correctness:** zero known hot-path **and** known-under-failure bugs from §1 (C1–C10 fixed; C11–C14 fixed or explicitly documented-as-accepted with a tracking issue). Every fixed bug has a red-before/green-after regression test.
- **Import:** `pip install power-loop` (no extras) → `python -c "import power_loop"` succeeds with **zero** SDKs; provider construction raises a clear `ImportError` with an install hint when the chosen SDK is absent. CI proves it on a dedicated leg.
- **Packaging:** wheel `top_level.txt` lists only `power_loop`; `py.typed` shipped; `STABLE_API` is the single source of truth with a CI guard enforcing the SemVer promise; classifier `5 - Production/Stable`; dependency surface contains only what shipped code imports.
- **Tests/CI:** coverage ≥85% on `power_loop` and ≥75% on the vendored transports, gated (`--cov-fail-under`); the default OpenAI transport, all `_dangerous_command_reason` branches, and `_validate_bash_command_scope` covered per-PR; `hypothesis` property tests on the three parsers/state-machines; `--strict-markers` with an honest tier taxonomy; examples import-smoke per-PR + real-LLM nightly.
- **Typing:** CI runs `mypy power_loop llm_client`; `check_untyped_defs=true` on `interface.py` + `llm_factory.py` at minimum; no `# type: ignore` without a reason code.
- **Observability:** every `AgentEvent` carries `ts` + monotonic `seq`; per-LLM-call latency/usage/retry events exist; a documented ~50-LOC OTel bridge subscriber works; `AgentLoopResult` carries machine-readable `reason/error_type/code`.
- **Docs/DX:** every `STABLE_API` capability has doc + `examples/NN_*.py` + a real-LLM test; no doc references an unemitted event; production checklist published; CHANGELOG current.
- **Perf baseline (new for 1.0):** a recorded micro-benchmark of `import power_loop` time (post-lazy-import) and a single-round send latency floor, checked in `tests/` or `bench/` so regressions are visible.

---

## 5. Explicitly NOT doing (anti-scope-creep, aligned with ROADMAP 显式不做)

- ❌ **No async DB driver / aiosqlite rewrite.** C8 is fixed with `to_thread` offload, not a storage-engine swap. (SQLite-single-base is a deliberate constraint.)
- ❌ **No router LLM, planner, or DAG workflow engine.** The dynamic JSON workflow + `run_agent(spec)` stays as-is; H1.2 hardens fan-out *failure*, it does not add scheduling semantics.
- ❌ **No long-term memory / vector store / RAG.** `MemoryProvider` remains an injected seam; H1 fixes its *alignment* bug, not its scope.
- ❌ **No multi-provider routing / cost-optimization router.** H4's cost hook is a pluggable callback, not a router.
- ❌ **No business semantics (IM / sessions / accounts).** Stays in the DeepTalk `agent` shell.
- ❌ **No new transports beyond OpenAI-compatible + Anthropic** until a concrete consumer needs one — but the extras split (H3.1) makes adding one cheap later.
- ❌ **No E2EE, no distributed/multi-node SessionStore.** Single-process embeddable kernel; cross-process safety is the `(session_id, seq)` PK + subprocess isolation, not a clustered store.
