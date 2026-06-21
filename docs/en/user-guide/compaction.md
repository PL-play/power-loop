# Compaction

[中文](../../zh/user-guide/compaction.md) | [User Guide](../index.md)

Context compaction prevents long sessions from hitting the LLM's context window limit. It summarizes old messages and replaces them with a compact system note — **default-on**.

## How It Works

```mermaid
flowchart TD
    A[Round start] --> B{tokens > threshold?}
    B -->|No| E[Run round normally]
    B -->|Yes| C[LLM summarizes old messages]
    C --> D[Replace with compact_note]
    D --> E
```

1. Before each round, `estimate_tokens(messages)` is compared against `max_tokens × trigger_ratio` (default 0.75).
2. If over the threshold, the compactor identifies the **oldest messages** that can be safely folded.
3. A summary LLM call produces a compact note (`role=system, name=compact_note`).
4. The old messages are marked `compacted_out` in the store; the note is inserted.

## Configuration

```python
from power_loop.runtime.compact import DefaultCompactor
from power_loop import AgentLoopConfig

compactor = DefaultCompactor(
    trigger_ratio=0.75,        # compact when > 75% of max_tokens
    keep_last_n=4,             # always keep last 4 exchanges
    summary_max_tokens=512,    # max tokens for the summary
)

config = AgentLoopConfig(compactor=compactor)

# Disable compaction
config_no_compaction = AgentLoopConfig(compactor=None)
```

### Absolute Threshold

Set `CONTEXT_COMPACT_THRESHOLD=6000` in the environment to use an absolute token count instead of `trigger_ratio`. Useful when your model has a known context window (e.g., 8192 for gpt-4o-mini).

## Invariants

The compactor enforces strict rules to keep the message protocol valid:

| Rule | Why |
|---|---|
| **System messages preserved** | `role=system` messages (including prior `compact_note`s) are never folded |
| **Last N exchanges preserved** | The most recent `keep_last_n` user-bounded exchanges are always kept |
| **Tool-call pairs atomic** | `assistant(tool_calls) ↔ tool(tool_call_id=...)` pairs are never split — the compactor walks back to keep them together |
| **At most once per round** | `round_compacted=True` flag prevents double-compaction |
| **Soft-fail** | If the summary LLM call fails, the loop continues with the original (uncompacted) history |

## Persistence & memory recall

When a `SQLiteSink` is attached (the default for a `StatefulAgentLoop` with a store), a fold also persists: the folded rows are marked `compacted_out`, a `compact_note` row is appended, and the `compactions` table gets an audit row. The sink translates the compactor's **in-memory history indices** into **store row seqs** through a parallel `index → seq` map it keeps aligned with `pipeline.history`.

This matters when [memory recall](memory.md) is also on. Recalled `memory_*` messages are spliced into the **front** of the history (the system region) but are **never persisted** — the `MemoryProvider` owns them. The sink records a placeholder for each so its `index → seq` map stays aligned; otherwise a later fold would map indices to the **wrong** rows and mark the wrong messages `compacted_out`. The two features compose cleanly: recalled facts survive folds (system region is preserved) and never leak into the store.

> **Safety net.** If that map ever drifts out of alignment — e.g. a `SESSION_START`/`ROUND_START` hook that *replaces* `ctx.messages` wholesale without the sink's knowledge — the sink **skips persisting that compaction** rather than risk marking the wrong rows. The in-memory fold still applies (the LLM call is unaffected); the un-persisted compaction simply re-triggers next round, and a resume stays correct because the active rows were left untouched. If you mutate history in a hook, prefer appending over wholesale replacement.

See [`examples/31_memory_with_compaction.py`](../../../examples/31_memory_with_compaction.py) for recall + compaction in one session.

### Retrieving folded detail on demand

Folded messages are not deleted — they remain `compacted_out` rows in the store. The optional **`recall_compacted`** tool lets the agent pull them back when the `compact_note` lacks a specific detail (an exact value, path, decision). It reads only the current session's folded rows (read-only), filtered by keyword or seq range. Add it to the agent's tools (`include=["recall_compacted", ...]` or the `full` preset). See [`examples/32_recall_compacted.py`](../../../examples/32_recall_compacted.py) and the [Tools guide](tools.md).

## Custom Compactor

Implement the `Compactor` protocol to plug in your own strategy:

```python
from power_loop.runtime.compact import Compactor, CompactionPlan

class MyCompactor:
    async def maybe_compact(
        self, messages, *, llm, max_tokens, round_index
    ) -> CompactionPlan | None:
        # Your logic here
        # Return CompactionPlan(fold_start_idx, fold_end_idx, summary_text, ...)
        # Return None to skip
        return None

config = AgentLoopConfig(compactor=MyCompactor())
```

### Coordinating with memory (capture before folding)

`maybe_compact` may take an **optional** `context: CompactionContext` — the configured `MemoryProvider`, the `session_id`, and a read-only `fetch_messages(from_seq, to_seq)` accessor. A custom compactor can use it to persist must-keep detail into [memory](memory.md) *before* the fold drops it from the active window:

```python
from power_loop import MemorySnapshot
from power_loop.runtime.compact import DefaultCompactor

class CoordinatingCompactor(DefaultCompactor):
    async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
        plan = await super().maybe_compact(
            messages, llm=llm, max_tokens=max_tokens, round_index=round_index)
        if plan and context and context.memory:
            slice_ = messages[plan.fold_start_idx : plan.fold_end_idx + 1]
            await context.memory.remember(
                snapshot=MemorySnapshot(session_id=context.session_id or "",
                                        messages=slice_, status="compaction"),
                session_id=context.session_id)
        return plan
```

The `context` parameter is **opt-in and back-compatible**: the pipeline only passes it to compactors whose `maybe_compact` accepts it (signature-checked), so existing compactors with the old signature keep working unchanged, and `DefaultCompactor` ignores it. Memory stays an injected seam — the library never persists it for you. The `status="compaction"` value is a convention so providers can distinguish a fold-time capture from a session-end snapshot. See [`examples/33_coordinating_compactor.py`](../../../examples/33_coordinating_compactor.py).

## Agentic memory-aware compaction (opt-in)

`DefaultCompactor` summarizes a slice in **one** LLM call. `AgenticMemoryCompactor` instead runs a **bounded, memory-aware agent loop** at the fold: the model first uses memory tools (by default the existing `note_add` / `note_update`) to **persist durable facts into the session's notes**, then writes the `compact_note` summary. This separates *long-term memory* (kept as notes, surfaced on later turns) from the *working-context summary* (compressed), so the agent forgets less across many folds.

```python
from power_loop.runtime.compact import AgenticMemoryCompactor

config = AgentLoopConfig(
    compactor=AgenticMemoryCompactor(
        trigger_ratio=0.75, keep_last_n=4,   # inherited from DefaultCompactor (trigger + span)
        max_rounds=4,                        # cap on the compaction agent's tool rounds
        # memory_tools=my_registry,          # default: a registry of note_add / note_update
        # system_prompt=...,                 # default: DEFAULT_COMPACTION_AGENT_PROMPT
    ),
)
```

- **Default behavior is unchanged** — this is opt-in; the default stays `DefaultCompactor` (single call).
- **Safe**: the loop is a flat, bounded tool-use loop (not a nested `StatefulAgentLoop`), so it can never recurse into another compaction. The note tools resolve the live session from the loop's contextvars (set for the whole run), so notes land on the right session. On **any** failure (no tool support, malformed output, exception) it **falls back to the single-call summary** — it never blocks a fold.
- **Cost**: it makes several LLM calls per fold (extract + summarize) instead of one — the trade for richer memory. Reuse the `summary_llm` arg to point the compaction agent at a cheaper model.

## Events

Subscribe to compaction events for observability:

```python
bus.subscribe(AgentEventType.STATUS_CHANGED, lambda e: print(
    f"Compacted: {e.data.before_tokens} → {e.data.after_tokens} tokens"
) if getattr(e.data, "kind", None) == "auto_compact" else None)
```

## Token Estimation

The compactor uses a heuristic token estimator (~4 chars/token) defined in `power_loop/runtime/budget.py`. It's not billing-accurate but monotonic with content size — good enough for triggering decisions.

See also `trim_history()` in [budget.py](../../../power_loop/runtime/budget.py) for a pure-trim (no LLM) alternative.

## Next

- [Send-context projection](send-context-projection.md) — the opt-in alternative: project finished sends to plain text in a derived table instead of rewriting history in place (mutually exclusive with this compactor)
- [Memory](memory.md) — cross-session recall via `MemoryProvider`
- [Sessions](sessions.md) — understand the session lifecycle
