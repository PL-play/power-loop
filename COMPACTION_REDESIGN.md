# power-loop 3.0: Orthogonal Context Architecture — Representation × FoldStrategy (FINAL)

## Revision 2 — LOCKED user decisions (override anything below that conflicts)

1. **Drop `deterministic` fold entirely** (concat/truncate isn't compaction). Shipped FoldStrategy impls = **`LLMSummaryFold` (default)** + **`AgenticFold`** only. Delete `DeterministicFold` AND `MinimalFold`. Every fold is an LLM call.
   - Consequence: projection's *per-send terse rendering* (representation, `max_chars`) STAYS (cheap, no LLM); but **folding old sends now always costs an LLM call** under projection too. `ProjectedRepresentation` loses its old `compact()`.
   - Drop the `is_pure` flag — all folds are async/awaited under the lock (bounded by `fold_timeout`). No sync-in-tx fast path.
   - "No never-fold" floor = the default LLM fold at the normal trigger (no MinimalFold). Legacy `compactor=None` → `LLMSummaryFold`.
2. **Implement ALL combos, fully config-driven.** {verbatim, projection} × {default(LLMSummaryFold), agentic(AgenticFold)} = **4 combos**, all selectable purely by config. Every strategy/representation knob is a constructor/config param (keep_last_sends, trigger_ratio, summary_max_tokens, max_rounds, max_chars, …) surfaced through `AgentLoopConfig`, DeepTalk `runtime_config`, and the admin UI.
3. **Clean break on the API** (power-loop is internal/monorepo): replace `Compactor`/`HistoryProjector`/the mutual-exclusion outright; update DeepTalk in lockstep. **No** deprecation-kwarg shims, **no** legacy `Compactor`/`HistoryProjector` adapters, **no** submodule import shims, **no** 4.0-migration ceremony. (§5.2, §7.1, §7.2, §7.3, the adapter paths in §5.2, and the `_Legacy*` adapters are DROPPED.)
   - KEEP, though: the **stored-session data migration** — schema v2→v3 add (`representation_kind`,`fold_version`), mode-switch re-fold, and `recall_send`/`recall_compacted` recovery — so existing live agent sessions don't break on deploy. This is data safety, not API back-compat.
4. Default fold = `LLMSummaryFold` (matches today's `DefaultCompactor`). Verbatim still routed through `pl_project_messages` (substrate-agnostic fold). Side effects via transactional `note_ops`. Mode-switch best-effort/never-throws.

5. **Custom impls are first-class.** `Representation` and `FoldStrategy` are public, documented, `@runtime_checkable` Protocols = the official extension points. Anyone can ship a custom representation or fold strategy by implementing the standard interface and passing it in config — no inheritance required. (The clean break only drops the *legacy* `Compactor`/`HistoryProjector` adapters; the new Protocols are the open contract.) Export both Protocols + `FoldContext`/`FoldResult`/`NoteOp`/`ProjectedSend`/`ProjectMessageRow` in `__init__`/`STABLE_API`.

Everything below stands EXCEPT where it describes deterministic fold, MinimalFold, `is_pure`, or API-level shims/adapters/4.0-migration — those are removed per the above.

---


**Status:** spec, implementation-ready (revised; all critical/high critique issues resolved)
**Author:** lead engineer (context subsystem)
**Supersedes:** the `history_projector ⊻ compactor` mutual-exclusion model (`platform/docs/design/14`, `power_loop/agent/types.py:118-127`)
**SemVer:** **3.0.0** (breaking — `AgentLoopConfig` shape changes + a one-row pl_project_messages schema add; see §7)

> This revision resolves every critical/high finding from the adversarial review and the actionable mediums. The biggest changes vs the draft: (a) the back-compat shim is a `__post_init__` dataclass-safe mapping (NOT a hand-written `__init__`); (b) old submodule import paths survive as shim modules; (c) `VerbatimRepresentation.render` is explicitly rewritten to render `compact` rows; (d) the fold runs **inside** the loop's contextvar+session scope so agentic notes resolve; (e) the two-phase write is replaced by a **single-tx fold under lock** with a durable "needs-fold" recovery marker; (f) representation rows are **stamped with a representation KIND** + the migration **deletes prior-representation rows**; (g) a real `fold_version` column is added (the "no schema change" claim is retracted); (h) intra-send budget control is **kept**; (i) deterministic fold's output is **token-bounded**, not char-bounded; (j) COMPACT_BEFORE/AFTER hooks and AutoCompactStatusPayload are **preserved + re-emitted**; (k) legacy `Compactor` keeps working via a real adapter; (l) default fold = `LLMSummaryFold` (matches today).

---

## 0. Problem statement and the prior-discussion finding

Today power-loop models context handling as a **one-of switch**: a session is EITHER `{verbatim history + in-place Compactor}` OR `{projection + no compactor}`. The two are forced apart by `AgentLoopConfig._validate_projector_compactor_exclusion` (`types.py:118-127`) and by `_default_compactor()` (`types.py:18-20,52`) making a `DefaultCompactor` present by default (so merely setting a projector raises).

This conflates **two genuinely independent concerns**:

- **Representation** — *how a finished send is recorded and rendered to the model*: raw verbatim `pl_messages` rows, vs a per-send projected summary.
- **Fold/compaction** — *how N records become 1 compact record once a budget is hit*. Its essence is "turn large text into small text". It MAY be pure (deterministic concat), LLM-backed (single summary call), or agentic (LLM + tools writing notes/memory).

The user requires these to be **orthogonal axes**, both usable in any combination, with compaction available under EITHER representation.

### Prior discussion located (requirement 4 — honored)

The prior discussion is **`platform/docs/design/14-send-context-projection.md`** (verified present; §1 动机, §A7 投影层压缩, §D4 实现修订, §D7 模式切换永不抛, §D8 切换迁移压缩 + agentic 记忆压缩器). Settled positions and how 3.0 treats each:

- **§1 (动机) — decoupling thesis:** `pl_messages` was both audit log AND 1:1 LLM context; the only slimmer (`DefaultCompactor`) rewrote `pl_messages` in place (`compacted_out` + `compact_note` + `meta.ord`) — "the root of a whole class of corruption bugs (C1 / double-fold / C9 / G8)". The fix was **separation**: `pl_messages` stays append-only audit; `pl_project_messages` holds LLM context; fold moves into the derived layer. → **Kept and pushed further: fold is ALWAYS a derived-layer operation, never an in-place rewrite of `pl_messages`.**
- **§D4 — "reuse policy, not code":** projection fold deliberately did NOT reuse `DefaultCompactor`'s code (it couples in-memory history + LLM summary + sink seq-index machinery). → **Conclusion reversed, lesson kept:** 3.0 introduces ONE `FoldStrategy` that operates on a derived record abstraction (never on `pl_messages` seq-index machinery), so LLM/agentic strategies become reusable across representations WITHOUT the sink/`ord` coupling.
- **§A7 — mutual-exclusion rationale:** the compactor's `compact_note` insertion + `meta.ord` reordering breaks the projection reader's `send_index` partitioning. → **THE concrete blocker, removed** by removing in-place `pl_messages` folding entirely (no more `compact_note` row written, so nothing reorders `ord`).
- **§D8 — mode-switch compaction:** switching default→projection folds prior history once via `projector.compact()`, seeding from any `compact_note`; best-effort, idempotent (`projection_migrated`), never throws (§D7). → **Generalized into one migration that runs on ANY representation/fold-strategy change, driven by the configured `FoldStrategy`.**
- **§D8 — two fold interfaces kept separate** (`Compactor.maybe_compact` in-place vs `HistoryProjector.compact` projection). → **Merged into one `FoldStrategy`.** This is the single deliberate reversal, sound now because the in-place substrate it conflicted with is gone.

**Net honoring statement:** the two-table separation, audit-immutability invariant, recall-on-demand philosophy ("loss = not-sent-next-turn, not data-gone"), and best-effort/never-throw mode switch are all preserved. The one deliberate reversal is merging the two fold interfaces.

---

## 1. Target model: two orthogonal axes

```
context handling = Representation  ×  FoldStrategy
                   {verbatim, projection}   {deterministic, llm_summary, agentic}
```

- **Representation** answers "how is each finished send turned into LLM-visible records, and how is it rendered back?" Per-send build + render. NEVER decides what to drop.
- **FoldStrategy** answers "given already-built records and a budget, which span collapses into one compact record, and what does that record say?" NEVER builds or renders per-send records.

The loop composes them: representation produces records; the loop measures the rendered prefix; when over budget it hands the foldable span to the fold strategy; the strategy returns one compact record covering that span. Both axes feed the SAME derived store (`pl_project_messages`); `pl_messages` is append-only audit under every combination.

### 1.1 The common record abstraction (makes fold representation-agnostic)

The single thing that lets one `FoldStrategy` work on BOTH raw and projected history is a common record type: `ProjectMessageRow` (`store/types.py:186-207`) with `kind ∈ {user, project, compact}`. We elevate it to **the** context record and give the verbatim representation a faithful encoding into it.

- **Both representations emit `ProjectMessageRow`s.** Verbatim emits a `project` row carrying the verbatim message list (`content={"messages":[...]}`); projection emits `user`/`project` summary rows. Both persist to `pl_project_messages`.
- **The fold strategy consumes `ProjectMessageRow`s and emits one `compact` `ProjectMessageRow`.** It sees `kind`, `content`, `rendered_text`, `send_index`, `source_seq_lo/_hi` — not whether the rows underneath were verbatim or projected.
- **`pl_messages` is the audit/fallback for both.** `recall_send(N)` re-expands send N from `pl_messages.send_index == N` regardless of representation. `recall_compacted` stays fully functional for legacy + NULL-`send_index` recovery (§6.3).

Because verbatim is ALSO routed through `pl_project_messages` (today's `IdentityProjector` already proves byte-identical for the user/project rows), **there is no in-place `pl_messages` folding path at all** — exactly what made the two axes conflict.

> **CRITICAL invariant (review C1):** every representation's `render()` MUST handle `kind='compact'`. In particular `VerbatimRepresentation.render` is **rewritten** (it is NOT byte-moved from `IdentityProjector`): for `kind in {user, project}` it emits `content['messages']` verbatim; for `kind == 'compact'` it emits a synthesized `{"role":"user","content": rendered_text or content['summary']}`. Without this, the newly-unlocked `(verbatim, *fold)` combos would silently drop the folded prefix on the next send. A PR-1 regression test asserts the compact summary appears in assembled history after a `(verbatim, DeterministicFold)` fold fires.

### 1.2 Axis interfaces (Protocols)

Two Protocols, in two files, neither importing the other.

**Representation** — `power_loop/runtime/representation.py` (new file; the old `history_projector.py` becomes a deprecation shim that re-exports, §7.2):

```python
@runtime_checkable
class Representation(Protocol):
    """How a finished send is recorded into pl_project_messages and rendered back.
    Build + render only. NEVER decides what to drop (that is FoldStrategy)."""

    kind: str               # 'verbatim' | 'projection' | a subclass id. Stamped on every row
                            # (representation_kind column). The reader NEVER renders a row whose
                            # representation_kind != this rep's kind (falls back to verbatim from
                            # pl_messages). Distinguishes the two now-independent axes per-row.
    version: int            # secondary compat key within a kind. >= 1.

    def project_send(
        self, send_rows: list[MessageRow], *, send_index: int,
        tool_registry: ToolRegistry | None,
    ) -> ProjectedSend: ...

    def render(self, rows: list[ProjectMessageRow]) -> list[LoopMessage]:
        """Render stored rows (user/project/compact) into LLM messages.
        MUST handle kind='compact' (emit content['rendered_text'] or content['summary'] as a
        user-role text block), because a fold can be applied under any representation."""
        ...
```

What moved OUT of the old `HistoryProjector`: `keep_last_sends`, `trigger_ratio`, `compact()` — fold concerns, now on `FoldStrategy`. `Representation` is pure build+render.

Shipped representations:
- `VerbatimRepresentation` (`kind="verbatim"`, `version=1`): emits one `project` row `{"messages":[...verbatim...]}`; render branches on kind (see §1.1 invariant).
- `ProjectedRepresentation` (`kind="projection"`, `version=1`): today's `DefaultDeterministicProjector` MINUS its fold knobs/`compact()`. The `human`/`tools`/`final_text` build + terse render (already handles `kind='compact'` at `history_projector.py:324-325`).
- DeepTalk subclass `DeepTalkRepresentation` (`kind="projection"`, may override `version` if its rendering diverges): today's `DeepTalkProjector` — drops `final_text`, strips `_NOISE_TOOLS`. **Because it shares `kind="projection"` with the base `ProjectedRepresentation` but renders differently, it MUST bump `version` (e.g. `version=2`) so a row written by one is not rendered by the other** — the reader gates on `(kind, version)` (review §6/low-2 on stamping).

**FoldStrategy** — `power_loop/runtime/fold.py` (new file; old `compact.py` becomes a deprecation shim, §7.2):

```python
@dataclass(frozen=True)
class FoldContext:
    """Everything a strategy needs at fold time, for BOTH representations and ALL side-effect
    classes (deterministic/llm/agentic). The loop builds it INSIDE the active session +
    contextvar scope (set_current_loop), so note/memory tools resolve (review §C2)."""
    session_id: str
    round_index: int
    representation: Representation        # re-render rows to text to summarize (substrate-agnostic)
    llm: Any | None = None               # main loop LLM; None only in a no-LLM unit harness
    summary_llm: Any | None = None       # optional cheaper dedicated summary model
    tool_registry: ToolRegistry | None = None   # agentic side effects (note action add/update/...)
    memory: MemoryProvider | None = None        # agentic memory side effects
    max_tokens: int | None = None        # the budget (drives self-bounding of summary length)
    current_tokens: int | None = None    # loop's incremental rendered-prefix estimate (SCALE-4)


@dataclass(frozen=True)
class FoldResult:
    """One compact record + the SEND span it covers. The loop is the authority on the persisted
    compact's FULL from_send (it rolls a prior compact forward), so FoldResult exposes ONLY the
    newly-folded hi cursor + the strategy's own summary. (review §low on from_send footgun.)"""
    content: dict[str, Any]              # {"summary": "..."} — the compact row's content
    rendered_text: str | None            # optional precomputed render (else representation.render)
    folded_to_send: int                  # last folded send_index == the new read cursor
    note_ops: tuple[NoteOp, ...] = ()    # OPTIONAL: deferred note writes the loop applies in the
                                         # SAME tx as the compact UPSERT, stamped with the span, so
                                         # a discarded fold leaves no orphan notes (review §med-atomicity).


@runtime_checkable
class FoldStrategy(Protocol):
    """Turn a span of context records (an optional leading prior 'compact' + the oldest live
    user/project sends) into ONE compact record. May have side effects per concrete strategy."""

    keep_last_sends: int     # most-recent finished sends kept individually before older fold. >= 1.
    trigger_ratio: float     # fold when rendered prefix tokens >= max_tokens * trigger_ratio. (0,1].
    fold_id: str             # stable id (e.g. 'deterministic'/'llm_summary'/'agentic') stamped on
                             # compact rows (fold_version column) so a fold-strategy switch can be
                             # detected per-row and re-folded (review §6/high stamping).
    is_pure: bool            # True ⇒ no I/O ⇒ may run inside the locked tx synchronously
                             # (DeterministicFold). False ⇒ LLM/tools ⇒ runs inside the tx but the
                             # loop knows it awaits network (see §3.2 single-tx-under-lock note).

    async def fold(
        self, rows: list[ProjectMessageRow], *, context: FoldContext,
    ) -> FoldResult | None:
        """Fold ``rows`` (loop-filtered to the foldable span) into one compact. Return None to
        decline (nothing foldable / soft-fail). MUST NOT touch pl_messages or the store directly —
        the loop persists the result (and any note_ops) atomically. MAY read text via
        context.representation.render(rows). MAY collect note/memory ops into FoldResult.note_ops
        (preferred) OR write them directly via context.tool_registry/memory (legacy path, only
        safe because the fold now runs inside the session scope AND inside the lock — §3.2)."""
        ...
```

Differences from the old `Compactor`:
1. `fold` is async and receives `llm` + `tool_registry` + `memory` + `representation` via `FoldContext` — agentic side effects work for ANY strategy under ANY representation. (The old `HistoryProjector.compact()` had no `llm` param — `history_projector.py:110` — which is why projection could only fold deterministically.)
2. `fold` operates on `ProjectMessageRow`s (the common record), not on in-memory `list[dict]` with seq-index machinery. A strategy calls `context.representation.render(rows)` to get plain text, then summarizes — substrate-independent. This is what lets LLM/agentic strategies be reused without the sink/`ord` coupling that motivated §D4.
3. `fold` returns `folded_to_send` (the cursor) only; no `fold_start_idx`/`fold_end_idx` into a live list, no `CompactionPlan` applied to `self.history`.

### 1.3 The three shipped fold strategies

All in `fold.py`; all usable under verbatim OR projection (requirement 3).

- **`DeterministicFold`** (`keep_last_sends=4`, `trigger_ratio=0.75`, `is_pure=True`, `fold_id="deterministic"`). No LLM, no side effects. Lifts `DefaultDeterministicProjector.compact()`: roll a leading prior `compact` forward, render each user/project row via `context.representation.render` to `#N user:`/`#N agent:` lines, **token-bound** the tail (see §1.4), append the `recall_send(#N)` marker on truncation. Returns `FoldResult(content={"summary": ...}, folded_to_send=max(folded))`.

- **`LLMSummaryFold`** (`keep_last_sends=4`, `trigger_ratio=0.75`, `summary_max_tokens=5000`, `is_pure=False`, `fold_id="llm_summary"`). Single LLM call, no tools. `DefaultCompactor._summarize_async` re-homed: render the span via `context.representation.render(rows)` + `_stringify`, send ONE `LLMRequest` (no tools, no system_prompt) to `context.summary_llm or context.llm`, `_strip_summary` the `<summary>…</summary>` body. Soft-fail (`return None`) on any exception → loop keeps the span unfolded this send. Rolls a prior compact's summary into the prompt.

- **`AgenticFold`** (`keep_last_sends=4`, `trigger_ratio=0.75`, `max_rounds=4`, `is_pure=False`, `fold_id="agentic"`). LLM + tools + notes/memory side effects. `AgenticMemoryCompactor._agentic_summarize` re-homed onto `FoldContext`: a FLAT bounded tool-use loop (NOT a nested `StatefulAgentLoop`) that issues `note(action=add|update)`. **Side-effect channel:** the strategy collects note operations into `FoldResult.note_ops` (preferred) which the loop applies transactionally with the compact UPSERT; if a strategy instead calls the tools directly, that now resolves because the fold runs inside the session/contextvar scope (§3.2). On ANY failure it falls back to `LLMSummaryFold` semantics, so it never blocks a fold. The side effects are an explicit, documented property of THIS strategy (requirement 3).

`keep_last_sends >= 1` on every strategy (a strategy must keep the in-flight context coherent). There is no public `keep_last_sends=0` "never fold" path — see §5.5.

### 1.4 Deterministic fold output is TOKEN-bounded (review §high — leaky char-cap)

The old `DefaultDeterministicProjector` bounded its compact only by `max_compact_chars` (default 4000), decoupled from the token trigger — so it could never guarantee the prefix drops below `max_tokens × trigger_ratio`, re-triggering forever (the C-series class of bug). **`DeterministicFold` couples its output bound to the loop budget:**

- After concatenating, it iteratively drops the OLDEST folded lines (preserving the `…[older folded sends omitted — use recall_send(#N)]` marker) until `estimate_tokens(context.representation.render([the_compact_row])) <= max_tokens * trigger_ratio * COMPACT_BUDGET_FRACTION` (default `COMPACT_BUDGET_FRACTION = 0.5`, i.e. the rolled compact must fit in half the trigger budget, leaving room for the kept tail).
- `max_compact_chars` remains a constructor knob but is now a *secondary* cap (min of the two). `max_compact_chars=0` no longer means "unbounded" — the token bound always applies. (Documented behavior change.)
- **Property test (required):** after `DeterministicFold` on a span far over budget, `estimate_tokens(rendered prefix incl. the new compact) < max_tokens * trigger_ratio`. This makes §5.5's "deterministic is the floor that always bounds growth" actually true.

---

## 2. The unified Fold/Compaction interface (detail)

This is the heart of requirement 2.

| concern | answer |
|---|---|
| **signature** | `async def fold(self, rows, *, context: FoldContext) -> FoldResult \| None` |
| **receives llm** | yes — `context.llm` (+ optional `context.summary_llm`) |
| **receives tools/notes** | yes — `context.tool_registry`, `context.memory` (agentic side-effect channel); plus `FoldResult.note_ops` for transactional note application |
| **input substrate** | `list[ProjectMessageRow]` — works on verbatim (`kind=project`, `content.messages`) AND projection (`kind=user/project`) rows identically; may include one leading prior `compact` row to roll forward |
| **how it reads text** | `context.representation.render(rows)` → `list[LoopMessage]` → stringify. Representation-agnostic. |
| **output** | one `FoldResult` = compact `content` + optional `rendered_text` + `folded_to_send` (the new cursor) + optional `note_ops` |
| **side effects** | per-strategy. `DeterministicFold`=none; `LLMSummaryFold`=LLM only; `AgenticFold`=LLM+notes/memory. The interface ALLOWS them; the strategy decides. |
| **must NOT do** | touch `pl_messages`, mutate `self.history`, write the store directly, recurse into the loop. The loop persists `FoldResult` + `note_ops` atomically. |

### 2.1 Why side effects live on the strategy, not the interface contract

Requirement 3: compaction's essence is "large text → small text"; side effects are strategy-dependent. So `FoldStrategy.fold` is conceptually a function from `(rows, context) → result`, and persistence of facts is a strategy's own privilege exercised through capabilities `FoldContext` hands it. `DeterministicFold` is provably side-effect-free (ignores tools); `AgenticFold` writes notes. **Atomicity (review §med):** note writes are collected into `FoldResult.note_ops` and applied **in the same transaction** as the compact UPSERT (§3.2), each note stamped with `(from_send, to_send)`; note add/update operations no-op if a note for that span already exists. So a discarded/retried fold never leaves orphan or duplicate notes. The audit-trail invariant is unaffected (notes are additive; folded detail stays recoverable in `pl_messages`).

### 2.2 Rolling a prior compact forward (nothing lost)

The loop, not the strategy, owns the authoritative compact span. When it folds, it prepends the existing latest `compact` row to `rows`; the strategy rolls its `summary` forward. The loop then persists the new compact with `compact_from_send = prior.compact_from_send if prior else min(span the loop handed in)` and `compact_to_send = result.folded_to_send`. **The loop asserts `result.folded_to_send == max(send_index of user/project rows it handed in)`** and rejects (logs + skips fold) otherwise — making the cursor contract type-and-assertion-enforced, not docstring-enforced (review §low on `from_send` footgun). Because `FoldResult` no longer carries `from_send`, a third-party strategy cannot double-count the prior compact's range.

### 2.3 Superseded compact rows are deleted (review §med — leak + invisible re-finalize)

Under strict serialization `compact_to_send` only grows, so each fold writes a NEW compact row at a NEW `send_index`, leaving the prior (lower-`send_index`) compact behind. **The compact UPSERT now also DELETEs prior compact rows with `compact_to_send < new_to_send` in the same tx** (keep only the latest). This prevents the slow leak and a confusing audit, and removes the orphan-compact failure mode where `load_project_messages(after_send_index=cutoff)` skips a lower compact whose folded sends sit below the surviving higher cutoff.

---

## 3. How the loop wires it per send

Orchestration lives in `StatefulAgentLoop._run_loop` (already hosts the reader/writer). **The in-round `AgentPipeline.prepare_round` compactor path is REPLACED, not deleted** — see §3.4 (intra-send budget control is kept).

### 3.1 Per-send order of operations (unified for both representations)

```
_run_loop(sid):
  rep  = config.representation        # always set (default VerbatimRepresentation)
  fold = config.fold_strategy         # always set (default LLMSummaryFold — matches today)
  current_send_index = read send_index runtime_state (>=1 else None → degrade verbatim)

  migrate_axes_metadata_if_needed()   # translate pre-3.0 metadata → 3-axis WITHOUT firing fold (§4.4)
  maybe_migrate_on_switch(rep, fold)  # fold-driven migration ONLY on a genuine axis change (§4)

  # READER — assemble history from pl_project_messages (+ verbatim fallback from pl_messages)
  history = assemble_context(sid, rep, fold, current_send_index)

  align_tool_calls(history)            # always-on sanitizer backstop (unchanged)

  async with runner.session_async(sid):       # the contextvar + session scope ...
    set_current_loop(self)
    result = pipeline.run(history)             # in-round budget control kept (§3.4)
    # WRITER + FOLD — STILL INSIDE the session scope, so note/memory tools resolve (review §C2)
    if current_send_index is not None and status not in {waiting_for_input, pending_tools}:
        await write_send_projection_and_maybe_fold(sid, current_send_index, rep, fold)
    # crash-recovery: a "needs_fold" marker (§3.2) left by a prior interrupted run is retried here
```

> **Move (review §C2):** in today's code `_write_send_projection` runs at `stateful_loop.py:1423` — AFTER `reset_current_loop(loop_token)` (line 1398) and AFTER the `async with self._runner.session_async` block exits. `AgenticFold`'s `note(action=add)` resolves store+session via `get_current_loop()/get_session_id()` contextvars (`runtime_state.py:35-41`, `default_tools.py:1374`), which are None there → it would always hit its except-fallback and silently degrade `(agentic) → (llm_summary)`. **3.0 moves the projection-write + fold INSIDE the `async with session_async(...)` / `set_current_loop` block** so the contextvars are live. Required integration test: `AgenticFold` actually writes a note (`count_notes` increases), not just returns a summary.

### 3.2 The end-of-send writer + fold (where the token trigger lives)

`_write_send_projection` becomes representation-agnostic and gains the unified fold call. **The fold runs inside ONE locked transaction** (the draft's Phase-A/Phase-B split is dropped — it broke the cross-loop serialization it claimed to keep, and orphaned rows on crash; review §critical + §high):

1. Load this send's `pl_messages` rows; `projected = rep.project_send(rows, send_index, tool_registry)`.
2. Open `store.write_send_projection_locked` (one tx, `session_state` row lock — `store.py:642-662`, locking unchanged):
   - UPSERT this send's `user`/`project` rows, **stamped with `representation_kind = rep.kind` and `representation_version = rep.version`**.
   - Snapshot `prior = latest compact`, `cutoff = prior.compact_to_send`, `live = project rows after cutoff for kinds user/project`.
   - **Trigger check uses the incremental estimate first** (review §med-perf): `cur = context.current_tokens if available else estimate_tokens(rep.render(prefix))`. Only when `cur >= max_tokens * fold.trigger_ratio` AND `len(live_sends) > fold.keep_last_sends` do we fold. Full `rep.render(...)` is reserved for AFTER the trigger fires (to pick/summarize the span), never for the trigger check on the hot path. This avoids the per-send O(history) re-render for verbatim sessions.
   - `to_compact = ([prior] if prior else []) + oldest live[:-keep]`.
   - `result = await fold.fold(to_compact, context=FoldContext(...))`.
     - **`is_pure` strategies** (DeterministicFold) execute synchronously — no await yields to other store ops on the SQLite single-writer connection (review §med — async-in-tx on SQLite). They are `async def` but never await; the store calls them directly inside the tx.
     - **non-pure strategies** (LLMSummaryFold/AgenticFold) DO await network I/O **inside the lock**. This is the honest, documented tradeoff: a fold holds the `session_state` lock for the duration of one LLM call. It is bounded by an explicit `fold_timeout` (default 60s; on timeout → soft-fail, "rows written, no fold", retried next send). Single-process DeepTalk (one loop per conversation) never contends; shared-store deployments serialize correctly (the whole point of the lock). **No data lock is held across I/O on SQLite because SQLite shared-store is single-process by construction; for PG/MySQL the row lock is exactly the serialization we want.**
   - if `result` is not None and the cursor assertion (§2.2) passes:
     - DELETE prior compact rows with `compact_to_send < result.folded_to_send` (§2.3).
     - UPSERT a `kind='compact'` row at `send_index=result.folded_to_send` with `compact_from_send = prior.compact_from_send if prior else min(folded span)`, `compact_to_send = result.folded_to_send`, `representation_kind = rep.kind`, `representation_version = rep.version`, **`fold_version = fold.fold_id`**.
     - apply `result.note_ops` in this SAME tx, each idempotent on `(span, content-hash)`.
   - a raising/None/timeout fold → "rows written, no fold" (degrade, never throw). The store wraps the fold call so a misbehaving strategy can't abort the per-send-row commit.
3. **Crash recovery (review §high — never-sends-again over-budget):** before opening the fold tx, if `cur >= threshold` the writer first stamps a durable `needs_fold = current_send_index` runtime_state key (cheap, separate tx); it clears it on a successful fold commit (or on a deliberate no-fold decision). At the START of the next `_run_loop` for this session, if `needs_fold` is set and still over budget, the loop re-attempts the fold there. Additionally, `aclose()`/finalization attempts a best-effort fold in a `finally` so a session that completes and never sends again is still bounded on its last run.

This keeps the single-tx atomicity and cross-loop serialization that `write_send_projection_locked` was built to provide (`store.py:626-633`), while making the LLM-fold at-least-once + idempotent (the `(session_id, to_send, 'compact')` UPSERT key makes a retry a no-op once committed).

### 3.3 Threading llm/tools into the fold for BOTH representations

The only thing the fold needs that differs by representation is `render`. By passing `context.representation`, a single `LLMSummaryFold`/`AgenticFold` renders verbatim rows (`kind=project`, `content.messages`) or projected rows (`kind=user/project`) identically into text, then summarizes. So `(verbatim, llm_summary)` and `(projection, llm_summary)` are the same code path, differing only in what `rep.render` produces — satisfying requirement 3's "usable under EITHER representation".

### 3.4 In-round (intra-send) budget control is KEPT (review §med — single send > budget)

A single send can run up to `max_rounds` (default 24) rounds; a tool-heavy send can blow the window MID-send, before any end-of-send fold runs. The draft deleted in-round compaction entirely — a regression for exactly DeepTalk's long-tool-using power-loop agents. **3.0 keeps an in-round budget control, re-expressed in the derived layer:**

- `AgentPipeline.prepare_round` no longer writes a `compact_note` to `pl_messages` (that is the in-place path we remove). Instead, when the in-flight send's rendered rounds exceed `max_tokens * trigger_ratio`, the pipeline calls the configured `fold_strategy` on the **in-flight send's older rounds only**, producing a transient in-memory compact that is spliced into `self.history` for the next LLM call AND written append-only as an `intra_send` projection row keyed `(session_id, send_index, 'compact')` so a resume re-reads it. `pl_messages` is never mutated. This is intra-send folding; cross-send folding stays end-of-send (§3.2).
- **Verbatim + single oversized send (review §high — math):** if even the last kept round alone exceeds budget, the intra-send fold summarizes within that round's tool transcript (the fold renders rows to text and summarizes a partial send). The documented precondition: `(verbatim, fold)` guarantees a bounded prefix as long as a single LLM *round*'s prompt fits in `max_tokens`; a single round that alone exceeds `max_tokens` is unrecoverable for any strategy and surfaces as `loop.degraded` (same as today). §5.5's "no unbounded growth" claim is scoped to this precondition explicitly.

### 3.5 Public hooks + events preserved (review §critical — COMPACT hooks; §med — AutoCompactStatusPayload)

- `HookPoint.COMPACT_BEFORE` / `COMPACT_AFTER` and `CompactBeforeCtx` / `CompactAfterCtx` are public (`__init__.py:391-392,419`, `contracts/hooks.py`). `HookPoint`/`HookDirective` are in the frozen STABLE_API (`test_packaging.py:43`). **They are KEPT and RE-EMITTED from the new fold paths** (both intra-send §3.4 and end-of-send §3.2): `COMPACT_BEFORE` fires before a fold with `CompactBeforeCtx.messages = rep.render(foldable span)` and honors `HookDirective.SKIP` (skips the fold this round/send); `COMPACT_AFTER` fires after with before/after message counts. No silent removal; no STABLE-surface violation.
- `AutoCompactStatusPayload` (exported in `__all__`; emitted at `pipeline.py:618-628`) is KEPT and **emitted from the new fold path** with `trigger="fold_applied"`, `before_tokens`/`after_tokens` from the rendered prefix. Consumers (admin/observability) keep receiving events.

---

## 4. Mode-switch compaction semantics

Requirement 4: switching representation OR fold-strategy must still compact appropriately, honoring §D8.

### 4.1 What a "switch" is now (3-axis, per-row + per-session)

Per-session metadata records `{representation_kind, representation_version, fold_id}` on first run (replacing the old `history_mode`/`projector_version` keys). A switch fires when the current `(rep.kind, rep.version, fold.fold_id)` differs from the recorded tuple. **Crucially, fold-strategy change ALSO triggers migration** (review §high — fold-only switch with existing compact): rolling a foreign-strategy compact forward is unsafe (e.g. a char-capped deterministic concat fed into an agentic extractor compounds imprecision while the reader trusts `compact_to_send` as an un-revisitable cutoff). So a `fold_id` change re-folds the entire prior span from the `pl_messages` audit under the new strategy. The per-row `fold_version` column lets the migration detect which compacts were produced by the now-old strategy.

### 4.2 Migration policy (generalized from §D8)

On the FIRST send after a genuine axis change (representation kind/version OR fold_id), the migration runs in ONE tx:

1. **CLEAR (review §critical — stale prior-representation compact):** DELETE all `pl_project_messages` rows for the session whose `(representation_kind, representation_version)` differs from the new representation, AND all `compact` rows whose `fold_version` differs from the new `fold_id`. This guarantees `latest_project_compact` and the `after_send_index` cursor can never observe a stale prior-representation/strategy compact at a higher `send_index` than the freshly written rows. (`clear_projection(session_id, keep_kind, keep_version, keep_fold)` at the head of `write_projection_migration`.)
2. **Re-project** every prior send under the NEW representation (`rep.project_send` on each send's `pl_messages` rows), stamping the new `(kind, version)`.
3. **Fold** all but the most-recent `keep_last_sends` into ONE `compact` via the CURRENT `fold.fold(...)` — so a switch into `llm_summary`/`agentic` gets an LLM-quality migration compact (requirement 4 "compact appropriately"). A legacy `compact_note`'s text seeds a synthetic leading `compact` row so its content rolls forward, never lost (§4.3).
4. Persist atomically (`write_projection_migration`, `store.py:665-706`, extended to clear + call `fold.fold` inside the session scope so agentic notes resolve); mark `projection_migrated`; idempotent; best-effort — on failure fall back to verbatim rendering this send and retry next send. **Never throws** (honoring §D7).

`(projection → verbatim)` is safe to render (verbatim from `pl_messages` always works); we still re-home the prefix into `VerbatimRepresentation` rows on first send so the session becomes representation-native.

### 4.3 Legacy `compact_note` handling (review §med — send-index alignment)

Pre-3.0 sessions may have a `compact_note` row in `pl_messages` (Mechanism 1) covering a **seq** range (folded by EXCHANGE, not by send). The migration:
- Derives the note's covered send boundary from the `compactions` table (`CompactionRow.from_seq/to_seq`) → `max_covered_send`. The synthetic seed compact's `compact_to_send = max_covered_send` aligns to a real send boundary.
- Re-projects ONLY sends **entirely above** the legacy fold's `to_seq` (treats the note as authoritative for its span). Sends straddling the boundary are NOT double-represented.
- Marks `recall_compacted` as the recovery path for the legacy `COMPACTED_OUT` rows in the note's span; `recall_send` covers sends above it. This preserves audit recoverability (requirement 7) for straddling sends (which `recall_send` alone could not).
- 3.0 NEVER writes a new `compact_note` to `pl_messages`. The reader EXCLUDES `compact_note` from the verbatim partition (as today, `stateful_loop.py:1247-1253`).

### 4.4 Existing-session metadata translation WITHOUT spurious migration (review §high — re-migration)

Existing sessions recorded `history_mode ∈ {projection, default}` + `projector_version`/etc., NOT the new 3-tuple. On first 3.0 open, **translate before comparing**: if `representation_kind` is absent but `history_mode` is present, derive `representation_kind = ('projection' if history_mode=='projection' else 'verbatim')`, `representation_version = projector_version or 1`, `fold_id = <the fold the legacy config maps to>` (§5.3/§8.2), and write the 3-tuple WITHOUT firing a fold-migration (translated == current ⇒ no switch). Only fire §4.2 when the translated tuple genuinely differs from the configured one. **Required regression test:** opening a pre-3.0 projection session and a pre-3.0 verbatim session in 3.0 with matching config does NOT run migration and does NOT rewrite history.

---

## 5. Config changes

### 5.1 New `AgentLoopConfig` shape

```python
@dataclass
class AgentLoopConfig:
    ...
    representation: Representation = field(default_factory=_default_representation)   # VerbatimRepresentation()
    fold_strategy: FoldStrategy = field(default_factory=_default_fold_strategy)       # LLMSummaryFold()  (matches today)
    migrate_history_on_switch: bool = True
    max_tokens: int | None = 8000            # the SHARED fold budget (rep-agnostic)
    # ── deprecated, accepted for one minor cycle, mapped in __post_init__ ──
    compactor: Any = _UNSET
    history_projector: Any = _UNSET
    migrate_history_on_projection_switch: Any = _UNSET   # alias → migrate_history_on_switch
    ...

    def __post_init__(self) -> None:
        self._migrate_legacy_axes()      # maps deprecated fields → representation/fold_strategy
        self._validate_config()          # max_tokens > 0 (always required), axes non-None
        object.__setattr__(self, "_initialized", True)
```

DELETED: `_validate_projector_compactor_exclusion` and its `__setattr__` cross-field guard. `__setattr__` re-validation stays but only guards `max_tokens > 0` and non-None axes. `_initialized` guard pattern kept.

- `_default_representation()` → `VerbatimRepresentation()` (library default = byte-identical to today's verbatim user/project rows, byte-identical history below the fold threshold).
- `_default_fold_strategy()` → **`LLMSummaryFold()`** (review §high — default-change). This MATCHES today's library default (`DefaultCompactor`, an LLM summarizer); it does NOT silently swap to a terse deterministic concat. Callers wanting deterministic set `fold_strategy=DeterministicFold()`.

### 5.2 `__post_init__` legacy mapping (review §critical — dataclass-safe shim)

**Do NOT hand-write `__init__` on a dataclass** (it replaces the generated one and silently unsets the ~20 real fields — the draft's biggest self-inflicted break). Instead `_migrate_legacy_axes()` runs in `__post_init__`:

```python
def _migrate_legacy_axes(self) -> None:
    if self.compactor is not _UNSET or self.history_projector is not _UNSET:
        warnings.warn("history_projector/compactor are deprecated in 3.0; use representation/"
                      "fold_strategy", DeprecationWarning, stacklevel=3)
        rep, fold = _resolve_legacy_axes(self.history_projector, self.compactor)
        # only fill when the caller didn't ALSO pass the new fields (new wins, review §med precedence)
        if self.representation is _DEFAULT_SENTINEL: object.__setattr__(self, "representation", rep)
        if self.fold_strategy   is _DEFAULT_SENTINEL: object.__setattr__(self, "fold_strategy", fold)
    if self.migrate_history_on_projection_switch is not _UNSET:
        warnings.warn("migrate_history_on_projection_switch → migrate_history_on_switch",
                      DeprecationWarning, stacklevel=3)
        object.__setattr__(self, "migrate_history_on_switch",
                           bool(self.migrate_history_on_projection_switch))
    object.__setattr__(self, "compactor", _UNSET)
    object.__setattr__(self, "history_projector", _UNSET)
    object.__setattr__(self, "migrate_history_on_projection_switch", _UNSET)
```

`migrate_history_on_projection_switch` keeps a deprecated read alias property forwarding to `migrate_history_on_switch` (review §high — renamed-field no-shim). Both are removed in 4.0.

`_resolve_legacy_axes` mapping:
- `history_projector` is a shipped projection projector → `representation = ProjectedRepresentation(...)` (or `DeepTalkRepresentation`), `fold_strategy = DeterministicFold(keep_last_sends, trigger_ratio, max_compact_chars)` from the projector.
- `history_projector` is an arbitrary custom `HistoryProjector` → wrap in a `_LegacyProjectorRepresentation` adapter (delegates `project_send`/`render`) + `DeterministicFold` from its knobs.
- `history_projector=None`/`_UNSET`, `compactor=DefaultCompactor(...)` → `VerbatimRepresentation()`, `LLMSummaryFold(trigger_ratio, keep_last_sends=keep_last_n, summary_max_tokens, summary_llm)`.
- `compactor=AgenticMemoryCompactor(...)` → `VerbatimRepresentation()`, `AgenticFold(...)`.
- `compactor` is an arbitrary custom `Compactor` → wrap in `_LegacyCompactorFold` adapter (review §med — example 16): translates the new `fold(rows, context)` into a call to the old `maybe_compact(messages, llm=, max_tokens=, round_index=, context=CompactionContext(...))`, mapping the returned `CompactionPlan(fold_start_idx, fold_end_idx, summary_text, ...)` index-span back to a send-span `FoldResult`. So `example 16`'s `TailOnlyCompactor` and any external custom `Compactor` keep running. A test instantiates `TailOnlyCompactor` and runs a loop end-to-end.
- `compactor=None` → `VerbatimRepresentation()`, **`MinimalFold()`** — NOT the aggressive default (review §high/§low — silent on→off behavior change). `MinimalFold` is a `DeterministicFold` subclass with `trigger_ratio=0.98` and a large keep, i.e. "effectively off — fold only at the brink to avoid a provider 400". This honors §5.5 (no truly-unbounded path) while not silently injecting aggressive mid-session folding into sessions that explicitly disabled compaction. The remap is documented as a behavior change in the CHANGELOG and logged once per build (§8.2).

### 5.3 The combos

| representation | fold_strategy | reachable today? |
|---|---|---|
| verbatim | deterministic | NO → now yes |
| verbatim | llm_summary | yes (today's default) |
| verbatim | agentic | yes |
| projection | deterministic | yes (today's projection) |
| projection | llm_summary | NO → now yes |
| projection | agentic | NO → now yes |

All 6 now expressible (requirement 5).

### 5.4 Session migration — covered by §4 (no new path).

### 5.5 Decision: keep a `none` fold? — **No public "never fold".** (requirement 6)

Compaction is fundamental. Justification:
- "No fold ever" → unbounded context → eventual provider 400 / `loop.degraded`. A footgun, not a feature.
- Today's "no fold" affordances (`compactor=None`; `IdentityProjector` with `keep_last_sends=0`) are de-facto bugs-waiting-to-happen.
- The floor is **`MinimalFold`** (`trigger_ratio≈0.98`, large keep) — folds only at the brink, cheaply, deterministically, token-bounded (§1.4), with full `recall_send` recovery. Legacy `compactor=None` maps here (§5.2), so "effectively off" callers get the closest safe equivalent, and the upgrade is documented + logged — not silent.
- The `(verbatim, *)` parity tests assert byte-identical history **below the fold threshold** (the only honest steady-state contract, since folding always eventually fires). A unit-only `_NeverFold` lives in the test tree for "prove the seam folds nothing" seam tests; it is NOT public.

This resolves the draft's internal inconsistency (review §high — "no none" vs byte-identical-verbatim vs the `compactor=None` mapping): verbatim-truly-never-fold is intentionally gone, documented as a breaking behavior change with a migration note, and the parity test is reworded.

---

## 6. Storage & recall

### 6.1 Storage: one derived table for both representations — **one schema add**

- `pl_messages` — append-only audit, never folded in place under any combo. `send_index` is the universal bridge. 3.0 NEVER calls `record_compaction` for new folds.
- `pl_project_messages` — the SINGLE derived context store for BOTH representations. Verbatim sessions write `project` rows here too. `compact` rows (fold output) keyed `(session_id, compact_to_send, 'compact')`.

**Schema change (review §critical/§high — retract "no schema change"):** add two nullable columns to `pl_project_messages`:
- `representation_kind TEXT` — the producing representation's `kind`. The reader renders a row ONLY when `(representation_kind, representation_version) == (rep.kind, rep.version)`, else falls back to verbatim from `pl_messages`. Without this, two representations both defaulting to `version=1` (e.g. `VerbatimRepresentation` and `ProjectedRepresentation`, or base vs `DeepTalkRepresentation`) would mis-render each other's rows with NO fallback (silent corruption on switch). This plus the migration CLEAR (§4.2 step 1) means the reader NEVER renders a row written under a different representation.
- `fold_version TEXT` — on `compact` rows, the producing `FoldStrategy.fold_id`, so a fold-strategy switch is detectable per-row (§4.1) and the migration re-folds stale compacts.

This bumps `CURRENT_SCHEMA_VERSION` to **3** with a 3-dialect `v2→v3` `ALTER TABLE ADD COLUMN` (guarded by the same `_column_exists`-on-the-open-tx pattern the v1→v2 `send_index` add used — MEMORY notes the SQLite single-conn deadlock trap: probe via the migration tx, not a separate `db.fetch`). Legacy rows have NULL `representation_kind` → reader treats them as "unknown representation" → verbatim fallback (safe). `record_compaction` / `compactions` table / `COMPACTED_OUT` state / `meta['ord']` reordering become **legacy-read-only** (no new writes). `prune_compacted_messages` likewise.

### 6.2 `recall_send` — works under every combo (scoped honestly)

`recall_send(N)` reads `pl_messages WHERE send_index == N`. Since `pl_messages` is never folded in place in 3.0, it is correct for any send carrying a `send_index`, under verbatim or projection, deterministic or LLM or agentic fold. Two fixes (review §high):
- **Filter states** to `{ACTIVE, COMPACTED_OUT}` (exclude `DROPPED`, the sanitizer's protocol-invalid repair rows), so a recall never surfaces repaired corruption as real history. (Today it uses `load_all_messages` → all states.)
- **Scope the claim:** `recall_send` covers any send that carries a `send_index`. Legacy NULL-`send_index` rows (pre-projection / export→import) are recovered by `recall_compacted` (§6.3), NOT `recall_send`. §6.4's recoverability guarantee is stated for both tools together, not `recall_send` alone.

`recall_send` is **unconditionally registered** by the loop's default tool set (folding is always on; replacing the `projection_on`-gated registration at `loop_cache.py:289-293`).

### 6.3 `recall_compacted` — kept FUNCTIONAL (not just shimmed) (review §med/§low + §med-NULL)

`recall_compacted` reads `pl_messages` rows in `COMPACTED_OUT` state (`default_tools.py:1508`). It stays fully registered + functional because it is the recovery path for: (a) legacy pre-3.0 in-place-compacted sessions, and (b) **legacy NULL-`send_index` rows the migration folds** (which `recall_send` cannot address). Changes:
- On a 3.0-native session with zero `COMPACTED_OUT` rows but existing projection `compact` rows, return a **redirect** message: "This session uses send-context projection; folded detail is recoverable via `recall_send(#N)` (available sends N..M)" — instead of a bare misleading "nothing folded" (review §low — silent empty).
- Tool description updated NOW to "legacy + NULL-send_index recovery; prefer `recall_send` for projected sends".
- It is NOT removed in 4.0 until the §7 4.0 data migration (below) is proven complete.

### 6.4 Audit trail intact (requirement 7)

Under all six combos: `pl_messages` keeps every original row, never flipped/deleted by a new fold. `compact` rows carry `source_seq_lo/_hi`. Full detail of any folded send is recoverable via `recall_send(N)` (sends with `send_index`) OR `recall_compacted` (legacy/NULL-send_index). Agentic-fold side effects additionally persist span-stamped facts to `notes` — additive, never destructive, transactional with the compact (§2.1). The §1 "loss = not-sent-next-turn, not data-gone" invariant holds uniformly.

---

## 7. SemVer impact & deprecation/shim path

**Breaking → 3.0.0.**

Breaking surface:
- `AgentLoopConfig.compactor` / `.history_projector` → deprecated, accepted one cycle, mapped (§5.2).
- `Compactor` / `HistoryProjector` Protocols → superseded by `FoldStrategy` / `Representation`; old Protocols kept importable (shim) + old custom impls run via adapters (§5.2).
- In-place `pl_messages` folding removed (`record_compaction` no longer called for new folds); the `compact_note`/`COMPACTED_OUT`/`meta.ord` path becomes legacy-read-only. COMPACT_BEFORE/AFTER hooks + AutoCompactStatusPayload KEPT and re-emitted (§3.5).
- **Default fold UNCHANGED in behavior** (`LLMSummaryFold` == today's `DefaultCompactor`); `compactor=None` → `MinimalFold` (documented behavior change).
- Schema v2 → v3 (one ADD COLUMN ×2; legacy rows safe).

### 7.1 Deprecation timeline
1. **3.0.0:** new axes authoritative. Deprecated kwargs accepted via `__post_init__` mapping with `DeprecationWarning`. Old class names aliased + old custom impls run via adapters. `recall_compacted` fully functional. Schema v3.
2. **3.1.0:** shim + aliases + adapters present; warnings escalate; CHANGELOG flags removal.
3. **4.0.0:** remove deprecated kwargs/aliases/adapters AND `record_compaction`/`compactions`/`COMPACTED_OUT`/`recall_compacted`/`prune_compacted_messages` — **only after** the 4.0 data migration below.

### 7.2 Submodule import shims (review §critical — DeepTalk imports + §low — PR ordering)
The new files are `power_loop/runtime/representation.py` and `power_loop/runtime/fold.py`. **The old module paths SURVIVE as thin shim modules** for one minor cycle:
- `power_loop/runtime/history_projector.py` — module-level `DeprecationWarning`, re-exports `HistoryProjector`, `IdentityProjector`, `DefaultDeterministicProjector`, `ProjectedRow`, `ProjectedSend`, `ProjectedCompact` (aliases onto the new symbols/adapters; `ProjectedSend` is shared).
- `power_loop/runtime/compact.py` — re-exports `Compactor`, `CompactionContext`, `CompactionPlan`, `DefaultCompactor`, `AgenticMemoryCompactor`, `DEFAULT_COMPACTION_AGENT_PROMPT`. `CompactionContext`/`CompactionPlan` stay REAL (used by the legacy `Compactor` adapter, §5.2), not behaviorally-incompatible aliases.
- Tests assert `from power_loop.runtime.compact import DefaultCompactor` and `from power_loop.runtime.history_projector import DefaultDeterministicProjector, ProjectedSend` still resolve in 3.0. This keeps DeepTalk's direct submodule imports (`agent/app/loop_cache.py:41`, `agent/app/projector.py:22`) working until PR-7 switches them — so `main` (and the agent service) stays importable across the whole PR sequence.

### 7.3 The 4.0 data migration (review §med — concretely specified now)
Before 4.0 drops `recall_compacted`/`COMPACTED_OUT`: a one-time migration **re-stamps** residual `COMPACTED_OUT` rows with a synthetic `send_index` (derived from `compactions.from_seq/to_seq` → send boundary) and flips them to `ACTIVE`-but-excluded-from-the-active-window, so `recall_send` can address them after `recall_compacted` is gone. Un-reopened pre-3.0 sessions are handled by **lazy on-open conversion** (the §4 migration already runs on first 3.0 open) PLUS an optional batch migration script; `recall_compacted` stays shimmed until the batch is proven complete (telemetry: zero sessions with `COMPACTED_OUT` rows and no `send_index`). 4.0 removal is gated on that proof — not committed blindly.

CHANGELOG: a `[3.0.0] BREAKING` section documenting the axis split, the removed mutual exclusion, the schema add, the `compactor=None → MinimalFold` behavior change, the kept-but-re-homed hooks/events, and the migration table.

---

## 8. DeepTalk downstream changes

### 8.1 `agent/app/loop_cache.py` `_build`

Replace the binary `projection_on` branch (`loop_cache.py:318-351`) with two independent reads + precedence (review §med — both-keys precedence):

```python
# NEW keys win over legacy when both present (read-time shim only fills new keys when absent).
rep_kind  = (rc.get("representation") or _legacy_representation(rc) or "verbatim").lower()
fold_kind = (rc.get("fold_strategy") or _legacy_fold(rc) or "llm_summary").lower()
rep_cfg   = rc.get("representation_config") or _legacy_rep_cfg(rc) or {}
fold_cfg  = rc.get("fold_config") or _legacy_fold_cfg(rc) or {}

representation = (
    DeepTalkRepresentation(max_chars=int(rep_cfg.get("max_chars", 300)))   # version=2 (≠ base)
    if rep_kind == "projection" else VerbatimRepresentation()
)
fold_strategy = {
    "deterministic": lambda: DeterministicFold(keep_last_sends=..., trigger_ratio=...,
                                               max_compact_chars=int(fold_cfg.get("max_compact_chars", 4000))),
    "llm_summary":   lambda: LLMSummaryFold(keep_last_sends=..., trigger_ratio=...,
                                            summary_max_tokens=int(fold_cfg.get("summary_max_tokens", 5000)),
                                            summary_llm=_maybe_summary_llm(fold_cfg.get("summary_model"))),
    "agentic":       lambda: AgenticFold(keep_last_sends=..., trigger_ratio=..., max_rounds=4),
    "minimal":       lambda: MinimalFold(),
}[fold_kind]()

cfg = AgentLoopConfig(..., max_tokens=max_tokens,
                      representation=representation, fold_strategy=fold_strategy)
```

- `recall_send` registration (`loop_cache.py:289-293`) + `RECALL_SEND_NOTE` append (line 329): unconditional now (folding always on).
- Build-log (line 360): report both axes, e.g. `context=projection/llm_summary`.
- `_build_fingerprint` (lines 79-88) hashes the whole `runtime_config` → new keys auto-invalidate cached loops.
- `DeepTalkProjector` → `DeepTalkRepresentation` (`agent/app/projector.py`), subclassing `ProjectedRepresentation`, **`version=2`** (so its rows are never rendered by the base projection rep, §1.2). Loses fold knobs (→ `fold_config`); keeps only representation overrides. `send_message`'s `project()` hook (`speak.py:31-58`) unchanged.

### 8.2 Read-time config shim + lazy normalization (review §med — both-keys + visibility)

Existing definitions store `runtime_config.context_mode` + `runtime_config.compactor`. The shim (`_legacy_*` helpers above) maps old→new at read time, filling new keys ONLY when absent:
- `context_mode == "projection"` → `representation="projection"`, `fold_strategy="deterministic"`, carry `projection.{max_chars}`→`representation_config`, `projection.{keep_last_sends,trigger_ratio,max_compact_chars}`→`fold_config`.
- `context_mode == "verbatim"` + `compactor == "default"` → `representation="verbatim"`, `fold_strategy="llm_summary"`.
- `compactor == "agentic"` → `fold_strategy="agentic"`.
- `compactor == "none"` → `fold_strategy="minimal"` (NOT aggressive deterministic; review §high/§low). **Logged once per build** at INFO ("upgraded legacy compactor=none → minimal fold; folds only at the brink") so operators see the behavior change.

**Precedence:** new keys win; legacy fills only gaps. On the first admin save through the new UI, the row is **normalized to the new keys** (legacy keys dropped) — a lazy one-row migration so no definition carries conflicting old+new keys. Agent unit tests: a definition carrying ONLY legacy keys, ONLY new keys, and BOTH.

### 8.3 Admin UI (`admin/web/src/agents/`)

- `defaults.ts`: replace `context_mode` + `compactor` + bundled `projection.*` + dead `compaction.*` with `representation: "verbatim"|"projection"`, `fold_strategy: "deterministic"|"llm_summary"|"agentic"|"minimal"`, `representation_config: {max_chars}`, `fold_config: {keep_last_sends, trigger_ratio, max_compact_chars, summary_model, summary_max_tokens}`. Add `REPRESENTATION_OPTIONS` + `FOLD_STRATEGY_OPTIONS`. Default `representation="projection"`, `fold_strategy="deterministic"` (preserves current DeepTalk behavior).
- `AgentDefinitionForm.tsx` "上下文策略" section (760-815): TWO orthogonal selects; knob sub-forms conditional per axis (representation→`max_chars`; fold split by strategy: deterministic→`max_compact_chars`, llm_summary→`summary_model`+`summary_max_tokens`, all→`keep_last_sends`+`trigger_ratio`). Delete the legacy "上下文压缩 (DefaultCompactor · 旧版逐字参数)" block (817-862) — fold its useful knobs into the llm_summary group. Remove the "两者互斥" text. **One-time banner** when a definition was auto-upgraded from `compactor=none` → minimal.
- `formFieldTips.tsx`: add tips for `representation`, `fold_strategy`, split knobs; prune dead `compaction.*` tips (264-329).
- Admin backend: `inspector.py:294-298` expose `representation` + `fold_strategy` (replacing `definition_context_mode`); `agent_sessions.py:313-367` `/projections` endpoint reports `representation` + `fold_strategy` (metadata records the 3 axes per §4.1). The 投影 tab reads `send_index` + the new `representation_kind`/`fold_version` columns (B3 send_index real-column note in MEMORY).

---

## 9. Phased implementation plan (small, independently-testable PRs)

Each PR leaves `main` green **including the in-repo agent import** (the submodule shims, §7.2, land in PR-1 and stay until PR-7).

**PR-1 — Common record + interfaces + submodule shims (no behavior change).**
- New `representation.py`/`fold.py` with `Representation`/`FoldStrategy` Protocols, `FoldContext`/`FoldResult`/`NoteOp`. Old `history_projector.py`/`compact.py` become re-export shim modules (DeprecationWarning).
- `VerbatimRepresentation` (with the rewritten kind-branching `render`, §1.1), `ProjectedRepresentation` (= `DefaultDeterministicProjector` minus fold), `DeterministicFold` (token-bounded, §1.4) + `MinimalFold`.
- Tests: re-point `test_history_projector.py`; assert `VerbatimRepresentation` round-trips byte-identical below threshold AND renders a `compact` row's summary; `DeterministicFold.fold` matches the old `compact()` for user/project, and the token-bound property holds; submodule-import shim tests; STABLE_API/`HookPoint` members intact.

**PR-2 — `LLMSummaryFold` + `AgenticFold` + legacy adapters.**
- Port `DefaultCompactor._summarize_async`→`LLMSummaryFold.fold`; `AgenticMemoryCompactor._agentic_summarize`→`AgenticFold.fold` (note_ops collection). `_LegacyCompactorFold` + `_LegacyProjectorRepresentation` adapters.
- Tests: `fold(rows, context)` for both, parametrized over representation (renders verbatim AND projected rows); agentic emits `note_ops`; fallback on failure; `TailOnlyCompactor` (example 16) runs via the adapter end-to-end.

**PR-3 — Config axes + `__post_init__` mapping; remove exclusion; preserve hooks.**
- Add `representation`/`fold_strategy`/`migrate_history_on_switch` + deprecated mapped fields (§5.1/§5.2). Delete `_validate_projector_compactor_exclusion`. Keep COMPACT_BEFORE/AFTER + AutoCompactStatusPayload, re-emitted from fold paths.
- Tests: legacy-kwarg construction maps + warns (incl. `migrate_history_on_projection_switch`); new+old kwargs → new wins; all other dataclass fields still settable; hook-fires test under the new fold path.

**PR-4 — Loop wiring: unified single-tx end-of-send fold; intra-send fold; move into session scope.**
- Schema v2→v3 (`representation_kind`, `fold_version`); store `write_send_projection_locked` stamps them + deletes superseded compacts + applies note_ops; single-tx fold under lock (is_pure sync / non-pure awaits-under-lock + `fold_timeout`); `needs_fold` recovery marker. Move `_write_send_projection`+fold INSIDE the `session_async`/`set_current_loop` scope. Re-express `prepare_round` as derived-layer intra-send fold (§3.4).
- Tests: `(projection, deterministic)` identical to today below threshold; `(projection, llm_summary)` folds via one scripted LLM call; `AgenticFold` actually writes a note (count increases); incremental token estimate drives the trigger (no O(history) re-render on the hot path); intra-send fold bounds a long single send; concurrency: two loops sharing the store never double-fold/orphan (serialized under lock); crash between rows-commit and fold → `needs_fold` retried next run; schema v3 ALTER on SQLite/PG/MySQL (deadlock-safe probe).

**PR-5 — Verbatim routed through `pl_project_messages` + unified reader.**
- Reader assembles verbatim history from `VerbatimRepresentation` rows (fallback to `pl_messages`); reader gates render on `(representation_kind, version)` and `fold_version`. Verbatim sessions write project rows end-of-send and fold in the derived layer.
- Tests: `(verbatim, deterministic/llm_summary/agentic)` fold end-of-send; `pl_messages` never gets a new `compact_note`; byte-identical prompt to pre-3.0 verbatim below threshold; a cross-representation row is NEVER rendered by the wrong rep (falls back).

**PR-6 — Mode-switch migration generalized + metadata translation.**
- §4: fire on representation kind/version OR fold_id change; CLEAR prior-rep/strategy rows first (§4.2 step 1); run `fold.fold` for the migration compact inside the session scope; seed legacy `compact_note` with send-boundary alignment (§4.3); 3-axis metadata; translate pre-3.0 metadata WITHOUT spurious migration (§4.4); never throw.
- Tests: verbatim↔projection switches; fold_id-only change re-folds; legacy `compact_note`→aligned projection compact; pre-3.0 projection AND verbatim sessions open in 3.0 with matching config → NO migration, NO rewrite; `test_real_projection_migration.py` extended for an LLM-fold migration.

**PR-7 — Legacy read-only + recall + DeepTalk downstream.**
- `record_compaction`/`COMPACTED_OUT`/`compactions`/`prune_compacted_messages` → legacy-read-only (assert no new 3.0 writes). `recall_send` excludes DROPPED, unconditionally registered. `recall_compacted` kept functional + redirect message + updated description. DeepTalk: switch `agent/app/{loop_cache,projector}.py` to new symbols (the submodule shims can now be removed in a later minor); two-axis `_build` + read-time shim + lazy normalization; admin `defaults.ts`/`AgentDefinitionForm.tsx`/`formFieldTips.tsx`/inspector/`/projections`.
- Tests: `test_recall_compacted.py` on a synthetic legacy session + NULL-send_index recovery; `test_recall_send.py` parametrized across all six combos + DROPPED-excluded; 3.0-native session → zero new `COMPACTED_OUT`; agent `_build` across six combos + legacy/new/both config keys; admin web build/lint; real-LLM E2E for `(projection, llm_summary)` and `(verbatim, agentic)` (asserts a note is written).

**PR-8 — Docs + 3.0 release.**
- Rewrite `compaction.md`/`send-context-projection.md` as one "Context: representation × fold" guide; update `platform/docs/design/14` (D1 schema add, new D-section for the axis split + single-tx-under-lock + intra-send fold + kept hooks); CHANGELOG `[3.0.0] BREAKING`; new example `42_orthogonal_context.py` showing `(verbatim, agentic)` and `(projection, llm_summary)`; the 4.0 data-migration script + telemetry gate (§7.3). PyPI 3.0.0.

### 9.1 Regression surface to hold green
- `test_compaction_double_fold.py` — port to "projection compact never inverts span / cursor monotone / superseded compacts deleted" property tests over `pl_project_messages` (invariant still matters for rolled-forward compacts).
- `test_history_sanitize.py` — `align_tool_calls` unchanged.
- `test_project_messages_store.py` — store CRUD + add v2→v3 migration coverage on SQLite/PG/MySQL.
- `test_history_projection_loop.py` — richest surface; rewritten per PR-4/5/6.
- `test_packaging.py` — STABLE_API + COMPACT hook members + AutoCompactStatusPayload intact.
- `examples/16_custom_compactor.py`, `33_coordinating_compactor.py`, `31_memory_with_compaction.py`, `04_compaction.py`, `32_recall_compacted.py` — keep running via adapters/shims (PR-2/PR-7).
- Real suite (`test_real_projection*.py`, `test_real_agentic_compaction.py`, `test_real_compaction.py`, `test_real_sanitize_compaction.py`) — re-pointed; add `(verbatim, agentic)` + `(projection, llm_summary)`.
- Conformance: full unit suite green across SQLite + PG5433 + MySQL3307; `ruff` clean. (Per MEMORY: FULLY wipe stale PG/MySQL test tables across ALL prefixes — `pl_`/`plrc_`/`plg2_`, the `pl\_%` LIKE misses the latter — before runs; `rm` stale `./power_loop_sessions.db` before the real suite; run via `power-loop/.venv`.)

---

## 10. Summary of decisions (decisive)

1. **Two Protocols:** `Representation {VerbatimRepresentation, ProjectedRepresentation(+DeepTalkRepresentation, version=2)}` (build+render, stamps `representation_kind`+`version`) × `FoldStrategy {DeterministicFold, LLMSummaryFold, AgenticFold, MinimalFold}` (fold, stamps `fold_version`).
2. **One unified fold interface:** `async fold(rows: list[ProjectMessageRow], *, context: FoldContext) -> FoldResult | None`, receiving `llm`/`summary_llm`/`tool_registry`/`memory`/`representation`; side effects are a per-strategy privilege returned as transactional `note_ops`. `VerbatimRepresentation.render` is rewritten to render `compact` rows.
3. **Common record = `ProjectMessageRow`**; verbatim encodes as a `project` row carrying the verbatim message list — fold is substrate-independent.
4. **In-place `pl_messages` folding removed**; in-round budget control KEPT, re-homed to the derived layer (intra-send fold). COMPACT_BEFORE/AFTER hooks + AutoCompactStatusPayload preserved + re-emitted.
5. **Fold runs inside the session/contextvar scope, inside ONE locked tx** (is_pure sync; non-pure awaits under lock, bounded by `fold_timeout`), with a durable `needs_fold` recovery marker — preserving atomicity + cross-loop serialization the draft's Phase A/B split broke, and making agentic notes resolve.
6. **Fold is always on; no public `none`** (`keep_last_sends >= 1`); the floor is `MinimalFold` (token-bounded, brink-only). Legacy `compactor=None` → `MinimalFold` (documented + logged behavior change, not silent aggressive folding).
7. **Schema v2 → v3** (`representation_kind`, `fold_version`) — the "no schema change" claim is retracted; the columns are required to keep the two axes per-row distinguishable and the reader fallback safe. Audit trail intact under every combo; `recall_send` (send_index sends, DROPPED-excluded) + `recall_compacted` (legacy/NULL-send_index) together preserve recoverability.
8. **Mode switch always compacts** via the current `fold.fold` (LLM-quality when LLM/agentic), with a CLEAR-prior-rows step and send-boundary-aligned legacy seeding; fold-strategy change ALSO migrates; pre-3.0 metadata is translated without spurious migration; best-effort, idempotent, never throws.
9. **3.0.0** with a dataclass-safe `__post_init__` legacy mapping (NOT a hand-written `__init__`), submodule import shims for the renamed files, real adapters so old custom `Compactor`/`HistoryProjector` impls keep running, and a concretely-specified, telemetry-gated 4.0 data migration before any legacy-recall removal. Default fold = `LLMSummaryFold` (matches today's `DefaultCompactor`).

**Key files this spec edits:** `power_loop/runtime/compact.py` → shim re-exporting `power_loop/runtime/fold.py`; `power_loop/runtime/history_projector.py` → shim re-exporting `power_loop/runtime/representation.py`; `power_loop/agent/types.py` (config + `__post_init__` mapping); `power_loop/agent/stateful_loop.py:1106-1616` (reader/writer/fold wiring moved inside session scope + generalized migration + metadata translation); `power_loop/core/pipeline.py:581-662` (re-home in-round compaction to derived-layer intra-send fold; keep + re-emit COMPACT hooks/AutoCompactStatusPayload); `power_loop/agent/sink.py:244-349` (retire `on_compaction` in-place apply; intra-send projection write); `power_loop/runtime/store/{schema.py,store.py}` (v3 ADD COLUMN; legacy-read-only `record_compaction`; single-tx `write_send_projection_locked` stamping + superseded-compact delete + note_ops + clear-on-migration); `power_loop/tools/default_tools.py:1488-1595` (recall state filter + redirect; unconditional `recall_send`). DeepTalk: `agent/app/loop_cache.py:182-360`, `agent/app/projector.py`, `admin/web/src/agents/{defaults.ts,AgentDefinitionForm.tsx,formFieldTips.tsx}`, `admin/backend/app/services/inspector.py:294`, `admin/backend/app/routers/agent_sessions.py:313`.
