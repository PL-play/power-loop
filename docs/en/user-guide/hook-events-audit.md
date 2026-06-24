# Hook-injected context audit (`pl_hook_events`)

`LLM_BEFORE` hooks can inject **ephemeral** context into a single LLM call — most notably the
built-in [memory recall](memory.md) hook, which appends recalled notes at the request tail. That
injected context is sent to the model for that call and then discarded: it is never written to
`self.history` or the store (by design — it keeps the prompt prefix byte-stable for provider
prefix-caching). The cost is that it is **recorded nowhere**, so you cannot later audit *what extra
context the model actually saw* on a given round.

The optional **hook-events audit log** fixes that without changing any of the above.

## Enabling it

```python
from power_loop import AgentLoopConfig

config = AgentLoopConfig(
    memory=my_memory_provider,
    record_hook_events="full",   # "off" (default) | "metadata" | "full"
)
```

| mode | what is recorded |
|------|------------------|
| `"off"` (default) | nothing — zero overhead |
| `"metadata"` | per injected item: `role`, `name`, `source`, `chars`, and the `position` (tail/front) — **not** the text |
| `"full"` | the above **plus** the injected `content` text |

`"full"` stores the injected text verbatim with no per-item cap, so the table grows with large
RAG/memory blocks — use `"metadata"` if volume is a concern.

## What it does NOT do (the guarantee)

The audit is **observability only**. It is written exactly like `send_index` — onto the *sink copy*
of the round's assistant message, never onto the message that lives in `self.history`. So it can
**never** re-enter the conversation, reach the LLM request, or perturb prefix-caching. Turning it on
does not change the model's behavior in any way.

## Reading it

```python
events = await store.list_hook_events(session_id)            # all, chronological
events = await store.list_hook_events(session_id, message_seq=seq)  # for one message
```

Each [`HookEventRow`](../api/index.md) links to the assistant message it fed into
(`message_seq`) and the `round_index` / `send_index` that locate it. The `payload` is
`{v, items: [{role, name, source, chars, content?}], item_count, total_chars}`.

One row is written **per round** (the `LLM_BEFORE` hook runs each round; the memory block is
memoized once per send but re-injected every round), so a multi-round send yields one audit row per
round.

## Storage & lifecycle

- A dedicated table `{prefix}hook_events` (schema **v3**; the migration is an idempotent
  `CREATE TABLE` — it never alters the hot `messages` table). It shows up automatically in any
  generic `pl_*`-table inspector.
- Rows are deleted with the session (`close_session_tree`).
- The table is **audit-only** and is intentionally **not** part of `export_session` / `import_session`
  (the audit lives in the live store).

## Notes / edge cases

- Capture is an **identity diff** of the post-`LLM_BEFORE` message list against a pre-hook snapshot,
  so it records both tail- and front-positioned injection.
- It assumes `LLM_BEFORE` handlers **mutate `ctx.messages` in place** (the built-in contract). A
  handler that *replaces* `ctx.messages` with fresh copies makes the per-injection diff
  unresolvable; the row is then a small `kind="inject_unresolved"` marker (it still never affects
  context or caching).
