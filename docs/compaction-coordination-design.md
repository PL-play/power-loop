# Design draft — Compaction coordination (widen `Compactor` + retrieve-on-demand)

> Status: **Phase 1 + Phase 2 SHIPPED** (recall_compacted tool; CompactionContext +
> widened Compactor). Phase 3 (compaction-hook span enrichment) remains optional/deferred.
> Original draft below. Proposes a way for context
> compaction to (1) coordinate with the injected `MemoryProvider`/tools so important
> detail is captured *before* it leaves the active window, and (2) let the agent pull
> previously-compacted detail back *on demand*. Stays inside the library's seam
> philosophy: memory/RAG remain injected, never built in-core.
>
> Maps to a new hardening track **H7 · Compaction coordination** (post-quick-wins).
> Last updated: 2026-06-15.

---

## 1. Problem

Today's `DefaultCompactor` is a solid but **"blind" summary-buffer**:

- `Compactor.maybe_compact(messages, *, llm, max_tokens, round_index)` only sees the
  message list + an LLM. It has **no access** to the configured `MemoryProvider`,
  notes, tools, store, or workspace (`runtime/compact.py`).
- It folds the oldest cuttable span into **one** `compact_note` (lossy prose) and
  marks the folded rows `compacted_out` in the store (`agent/sink.py` →
  `session_store.record_compaction`). The next `send` loads only **active** rows
  (`load_active_messages`), so folded detail is no longer sent to the model.
- `MemoryProvider.remember` runs **only at session end** (`pipeline._finalize`), never
  at fold time. So detail folded mid-session that *should* be remembered long-term is
  **not captured** — it only survives as lossy summary text.

Two gaps, two fixes:

| Gap | Fix | Effort |
|---|---|---|
| Compactor can't **capture** must-keep detail before folding | **B** — widen the `Compactor` seam with an optional `CompactionContext` (memory + session + read accessor), back-compat | M |
| Folded detail is gone from the model's view even when it's still in the DB | **Retrieve tool** — `recall_compacted` surfaces `compacted_out` rows on demand | S |

> **Key reframing.** Folded messages are **not deleted** — they are `state=compacted_out`
> rows still in the DB (`MessageState.COMPACTED_OUT`). "Loss" = "not sent next turn," not
> "data gone." So the cheapest, highest-leverage move is *retrieval on demand*, not a
> lossless summary. The two fixes are complementary: **capture** handles cross-session /
> long-term durability; **retrieve** handles within-session "I need that detail back."

---

## 2. Part B — widen the `Compactor` protocol

### 2.1 New optional context

```python
# power_loop/runtime/compact.py
@dataclass(frozen=True)
class CompactionContext:
    """Read/coordinate handle handed to a compactor at fold time. Optional and
    additive — DefaultCompactor ignores it; a custom compactor uses it to capture
    must-keep detail into the *injected* MemoryProvider/notes before folding."""
    session_id: str | None
    memory: MemoryProvider | None          # the configured provider, or None
    fetch_messages: Callable[[int, int], list[dict]]  # (from_seq, to_seq) -> rows, read-only
    round_index: int
```

`maybe_compact` gains an optional keyword (the Protocol default makes it back-compat):

```python
async def maybe_compact(
    self, messages, *, llm, max_tokens, round_index,
    context: CompactionContext | None = None,   # NEW, optional
) -> CompactionPlan | None: ...
```

### 2.2 Back-compatibility (critical — `Compactor` is on the public-ish surface)

A pre-existing user compactor implements the **old** signature (no `context`). The
pipeline must NOT pass `context` to those, or it raises `TypeError`. Mechanism:

- The pipeline caches, per compactor instance, whether its `maybe_compact` accepts
  `context` (via `inspect.signature(...).parameters` — looking for `context` or a
  `VAR_KEYWORD`). Passes `context=` only when accepted. This is the same
  signature-aware-dispatch idiom already used for async tool handlers
  (`tools/registry.py` caches `is_async`).
- `DefaultCompactor.maybe_compact` adds `context=None` and **ignores it** → zero
  behavior change to the default path.

Rejected alternatives (documented so the choice is auditable):
- *Always pass `context`, require `**kwargs`* — breaks every existing compactor.
- *Contextvar `get_compaction_context()`* — zero signature change and idiomatic to
  this codebase (RuntimeEnv/current_loop are contextvars), but less discoverable/typed
  than an explicit param for a typed Protocol. **Fallback if introspection proves ugly.**

### 2.3 What a coordinating compactor looks like (lives in `examples/`, not core)

```python
class CoordinatingCompactor(DefaultCompactor):
    async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
        plan = await super().maybe_compact(
            messages, llm=llm, max_tokens=max_tokens, round_index=round_index)
        if plan and context and context.memory:
            slice_ = messages[plan.fold_start_idx : plan.fold_end_idx + 1]
            # persist the about-to-be-folded slice for long-term recall; the provider
            # extracts facts and distinguishes this from a session-end snapshot by status.
            await context.memory.remember(
                snapshot=MemorySnapshot(session_id=context.session_id or "",
                                        messages=slice_, status="compaction"),
                session_id=context.session_id)
        return plan
```

No new `MemoryProvider` method: capture reuses `remember` with `status="compaction"`,
so providers that don't care simply ignore that status. (Documented as the convention.)

### 2.4 Pipeline wiring

`prepare_round` (`core/pipeline.py`) builds a `CompactionContext` from
`self.config.memory`, `self.session_id`, and a `fetch_messages` closure backed by
`self.store` (read-only). Passes it to `maybe_compact` only if accepted. No change to
how the plan is applied or persisted (H1.1 alignment is untouched).

---

## 3. Retrieve-on-demand tool — `recall_compacted`

A default tool (opt-in via tool preset) that surfaces this session's
`compacted_out` rows + compaction audit rows to the model.

```python
def recall_compacted(
    query: str | None = None,      # case-insensitive substring filter over content
    from_seq: int | None = None,   # or an explicit seq window
    to_seq: int | None = None,
    limit: int = 20,
) -> str: ...
```

- **Data source:** `get_current_loop().store` + `get_session_id()` (the contextvar
  pattern every session-aware tool already uses). Reads `load_all_messages(sid)`,
  filters `state is COMPACTED_OUT`, applies the seq window / keyword filter, truncates.
  `list_compactions(sid)` gives the audit ranges for a "what was compacted when" view.
- **Scope/safety:** strictly the **current session's own** rows — no cross-session,
  no external I/O, read-only. Lower risk than `bash`/file tools. Off unless the host
  enables it in the preset.
- **In-core = keyword/seq only.** No embeddings/vector search in the library (that
  stays the injected `MemoryProvider`'s job — consistent with HARDENING_PLAN "显式不做").
  A vector-backed variant is an `examples/` recipe.
- **Why high-leverage:** zero extra LLM cost at fold time; leverages data already
  persisted; turns "compaction lost it" into "the model asks for it back."

---

## 4. Delivery plan — incremental, test-first, real-LLM at each phase

Per the "边做边测试 / 真实 LLM 测试" guidance: each phase ships **red-before/green-after
unit tests + ≥1 real-LLM test + an example + a doc update** before the next starts.

### Phase 1 — `recall_compacted` tool  *(smallest, highest leverage, no protocol change)*
- Implement the tool + register it in a preset.
- **Unit:** seed `compacted_out` rows → tool returns them; `query`/seq-window filters;
  empty result; **session isolation** (never returns another session's rows); ACTIVE
  rows excluded.
- **Real-LLM:** a session compacts, then the agent is asked something answerable
  **only** from folded detail → it calls `recall_compacted` and recovers it. New
  `examples/NN_recall_compacted.py` + entry in `tests/real/test_examples.py`.
- **Docs:** `tools.md` + a "retrieve compacted detail" note in `compaction.md`.

### Phase 2 — `CompactionContext` + widened `maybe_compact`  *(the custom-compactor interface)*
- Add `CompactionContext`, the optional param, the introspection-based pass-through.
- **Unit:** custom compactor receives a populated context (memory/session/fetch);
  **old-signature compactor still works** (back-compat guard — red-before if we passed
  `context` unconditionally); `DefaultCompactor` behavior byte-identical to today.
- **Real-LLM:** a `CoordinatingCompactor` example that on fold `remember`s the slice;
  assert the captured fact **survives into a new session via recall** (end-to-end:
  fold → remember → new session → recall → answer). `examples/NN_coordinating_compactor.py`.
- **Docs:** `compaction.md` "Coordinating with memory" section; `memory.md` cross-link;
  document the `status="compaction"` convention.

### Phase 3 *(optional, only if needed)* — enrich the compaction hooks
- Give `COMPACT_BEFORE`/a new `COMPACT_CAPTURE` hook the **planned fold span** + the
  context handle, so capture can also be done from a hook without a custom compactor.
- Defer unless a concrete consumer wants the hook path over the compactor path.

---

## 5. Risks & non-goals

- **No memory/RAG in core.** Capture only *coordinates* with the injected
  `MemoryProvider`; retrieval is keyword/seq only. Vector search stays an example.
- **Back-compat is a hard requirement.** No existing `Compactor` may break; covered by
  an explicit old-signature regression test in Phase 2.
- **Cost.** Capture-at-fold adds one provider `remember` per fold (provider-side, opt-in
  via a coordinating compactor). Retrieve adds a tool call only when the model needs it.
  `DefaultCompactor` + no tool = today's cost exactly.
- **Not** building agentic/sub-agent compaction (Option C) here — powerful but costly;
  revisit only on concrete need.

---

## 6. Public-API surface added (for the 1.0 stability table)

| Symbol | Tier | Note |
|---|---|---|
| `CompactionContext` | PROVISIONAL | new dataclass in `runtime/compact` |
| `Compactor.maybe_compact(..., context=None)` | STABLE (additive) | optional kwarg; back-compat |
| `recall_compacted` tool | PROVISIONAL | opt-in preset tool |
| `MemorySnapshot.status == "compaction"` | convention | documented, not enforced |
