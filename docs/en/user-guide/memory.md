# Memory

[中文](../../zh/user-guide/memory.md) | [User Guide](../index.md)

`MemoryProvider` is a pluggable protocol for **cross-session recall**. The library does not implement a memory backend — you bring your own (SQLite facts, HTTP API diary, vector DB). The protocol tells the pipeline **when** to recall and **when** to persist.

## How It Works

```mermaid
flowchart TD
    A[send user_input] --> B[session.start]
    B --> C{memory configured?}
    C -->|No| G[Round loop]
    C -->|Yes| D[memory.recall]
    D --> E[tag_as_memory then system region]
    E --> F[MEMORY_RECALLED hook]
    F --> G
    G --> H[session.end]
    H --> I[memory.remember]
```

1. **Recall** — at `session.start`, before the first round. Recalled messages are injected after the leading `role=system` block (same region as `compact_note`, so the compactor preserves them).
2. **Remember** — at `session.end`. Receives a `MemorySnapshot` with the full final history, final text, status, and round count.

## Quick Start

```python
from power_loop import MemorySnapshot, AgentLoopConfig

class MyMemory:
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        # Return a list of dicts to inject as system messages
        return [{"content": "User prefers Python. Favorite color: blue."}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        # snapshot.messages, snapshot.final_text, snapshot.status, snapshot.rounds
        pass  # persist what you need

config = AgentLoopConfig(memory=MyMemory())
```

## MemorySnapshot

Passed to `remember()` at session end:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The session ID |
| `messages` | `list[dict]` | Full final history (after any compaction) |
| `final_text` | `str` | The last assistant reply |
| `rounds` | `int` | Total rounds completed |
| `status` | `str` | `"completed"` / `"cancelled"` / `"degraded"` / `"hit_round_limit"` |
| `metadata` | `dict` | Caller-provided metadata from `new_session(metadata=...)` |

## Injection Position

Recalled messages are tagged `role=system, name=memory_*` and inserted **after** the leading system block (which includes the `system_prompt` and any `compact_note`):

```
[system_prompt]          ← from AgentLoopConfig
[compact_note]           ← from compactor (if any)
[memory_0]               ← from recall
[memory_1]               ← from recall
[user msg 1]             ← conversation begins here
[assistant msg 1]
...
```

This position means memory messages share the compactor's system-region protection — they are never folded.

## Failure Model

Memory is **best-effort**. Failures never block the user from getting a reply:

| Failure | Behavior |
|---|---|
| `recall()` raises | Returns `[]` (no injection). Emits `MEMORY_FAILED(phase="recall")`. Loop continues. |
| `remember()` raises | Emits `MEMORY_FAILED(phase="remember")`. `StatefulResult` is returned unchanged. |

## MEMORY_RECALLED Hook

Filter or drop recalled messages before injection:

```python
hooks = AgentHooks()

async def gate_memory(ctx: MemoryRecalledCtx) -> None:
    if not user_has_consented(ctx.session_id):
        ctx.directive = HookDirective.SKIP  # drop everything
    # Or redact:
    for m in ctx.recalled:
        m["content"] = redact_pii(m.get("content", ""))

hooks.register(HookPoint.MEMORY_RECALLED, gate_memory)
```

## Example: SQLite Fact Memory

```python
class SqliteFactMemory:
    def __init__(self, db_path):
        self.db_path = db_path

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        rows = db.execute("SELECT key, value FROM facts").fetchall()
        return [{"content": "Known facts:\n" + "\n".join(
            f"- {r['key']}: {r['value']}" for r in rows
        )}] if rows else []

    async def remember(self, *, snapshot, session_id):
        # Extract FACT: key=value lines from final_text
        for key, value in parse_facts(snapshot.final_text):
            db.execute("INSERT OR REPLACE INTO facts VALUES (?, ?)", (key, value))
```

Full runnable version at [`examples/13_memory_sqlite.py`](../../../examples/13_memory_sqlite.py).

## Next

- [Retry & Cancel](retry-cancel.md) — handle LLM failures gracefully
- [Events](events.md) — observe memory events (`memory_recalled`, `memory_failed`)
