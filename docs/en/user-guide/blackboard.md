# Shared blackboard

`StoreBlackboard(store)` is the default database-backed implementation of the `Blackboard` protocol. It supports SQLite, PostgreSQL and MySQL. Hosts choose a `board_id` and its lifetime; a board is independent of message sessions.

Each `post`, `update` and `remove` runs as an atomic read/modify/write in one database transaction, including creation of the first row. Independent instances and processes can update the same board without losing entries or allocating duplicate IDs. SQLite uses its write transaction; PostgreSQL and MySQL create the unique owner/key row without overwriting it, then lock it for the update. The entry and text limits still apply. Missing entries and invalid operations raise `BlackboardError`.

```python
from power_loop import StoreBlackboard

board = StoreBlackboard(store)
entry = await board.post("project-37", text="Review the plan", kind="task", status="open")
await board.update("project-37", entry.id, status="done")
entries = await board.read("project-37")
await board.remove("project-37", entry.id)
await board.delete("project-37")  # The host ends this board's scope.
```

For model tools, call `register_blackboard_tools(registry)` and have the current worker factory return an `ExecutionEnvironment` with `runtime_env=RuntimeEnv(blackboard=board, blackboard_id="project-37")`. This exposes the single `board` tool with `read/post/update/remove` actions. The model cannot select another board ID. Share a board only among authorized agents; the host supplies the scope and performs its final cleanup.

The factory must reconstruct the same board backend and scope when execution moves. A submitter's context variable is not a cross-process resource. DeepTalk uses `DeeptalkBlackboard`, whose writes go through its API and business tables; it does not write the SDK board table.

Run the credential-free example:

```bash
python examples/29_shared_blackboard.py
```

`SqliteBlackboard` is removed; no compatibility alias is provided. See [Runtime 11](../../runtime-11.md) for composition and lifecycle.
