# Shared Blackboard

[中文](../../zh/user-guide/blackboard.md) | [User Guide](../index.md)

A **scoped shared blackboard** is a small, structured, mutable space that several agents read and write to coordinate — distinct from each agent's private message history. A planner leaves tasks; a worker claims and completes them; each sees the other's updates — *without* dumping one agent's whole transcript into the other's context.

It's the same abstraction whether the scope is a chat conversation or a workflow run; only the `blackboard_id` and lifetime differ. DeepTalk uses it (scope = `conversation_id`) for its in-room agent coordination board.

## The pieces

| Piece | Role |
|---|---|
| `Blackboard` (Protocol) | async `read` / `post` / `update` / `remove` — per-entry merge, never whole-doc overwrite |
| `SqliteBlackboard` | the default impl, persisted in the `SessionStore` sqlite (a board is *not* tied to a session) |
| `register_blackboard_tools(registry)` | adds the agent-facing `board_read` / `board_post` / `board_update` / `board_remove` tools |
| `RuntimeEnv(blackboard=, blackboard_id=)` | injects the live board + this agent's board id per run |
| `render_entries(entries, header=)` | format a board snapshot for prompt injection |

Agents with the **same** `blackboard_id` share a board; different ids are isolated. A board-less agent simply doesn't register the tools (default = isolated).

## Setup

```python
from power_loop import (
    SqliteBlackboard, ToolRegistry, register_blackboard_tools,
    RuntimeEnv, runtime_env_context, render_entries,
)

registry = ToolRegistry()
# The kind/status vocabularies are YOUR policy (they shape the tool schemas).
register_blackboard_tools(registry, kinds=("note", "task"), statuses=("open", "doing", "done"))

board = SqliteBlackboard(loop.store)
BOARD_ID = "project-x"
```

## Running an agent against the board

Inject the board per send (the same seam as `shell_backend`). The **author is stamped from session metadata** (`spec_name`), not supplied by the model — so attribution can't be spoofed. The host typically *projects* the current board into each turn's prompt (the "pull" side):

```python
sid = loop.new_session(metadata={"spec_name": "planner"})
snapshot = render_entries(await board.read(BOARD_ID), header="Shared board:", empty="(empty)")

with runtime_env_context(RuntimeEnv(blackboard=board, blackboard_id=BOARD_ID)):
    await loop.send(f"{snapshot}\n\nPost two tasks for your teammate.", session_id=sid)
```

Now a second agent with the *same* `BOARD_ID` sees those entries and can act on them:

```python
sid2 = loop.new_session(metadata={"spec_name": "worker"})
with runtime_env_context(RuntimeEnv(blackboard=board, blackboard_id=BOARD_ID)):
    await loop.send("Mark the first task done and leave a note.", session_id=sid2)

for e in await board.read(BOARD_ID):
    print(f"#{e.id} [{e.kind}·{e.status}] ({e.author}) {e.text}")
```

See [example 29](../../../examples/29_shared_blackboard.py) for the full planner/worker run.

## Tools the agent sees

| Tool | Action |
|---|---|
| `board_read` | snapshot the board (usually auto-shown at turn start; call to re-check) |
| `board_post` | add an entry (`text`, optional `kind`, `status`) |
| `board_update` | edit an entry's `text` / `status` by `entry_id` |
| `board_remove` | delete an entry by `entry_id` |

Entries are append-only with monotonic integer ids. Writes are per-entry (post / update one row), not whole-document last-write-wins — so concurrent authors don't clobber each other. `SqliteBlackboard` enforces `max_entries` and `max_text_len` caps and raises `BlackboardError` on violations.

## Direct API (no agent)

The board is a plain async object you can drive yourself — useful for seeding, tests, or a host-side audit view:

```python
e = await board.post(BOARD_ID, text="claim the plan", kind="task", status="open", author="alice")
await board.update(BOARD_ID, e.id, status="done")
await board.remove(BOARD_ID, e.id)
```

## Custom implementations

`Blackboard` is a `Protocol`, so a host can back it with anything (an HTTP API, a shared DB) as long as it keeps the per-entry merge semantics. DeepTalk's `DeeptalkBlackboard` routes every write through its REST API (honoring its single-writer invariant) while reusing power-loop's tools and projector unchanged.

## See also

- [Tools](tools.md) — registering and presetting tools
- [Configuration](configuration.md) — `RuntimeEnv` and per-send injection
- [Workflows](workflows.md) — orchestrating the agents that share a board
