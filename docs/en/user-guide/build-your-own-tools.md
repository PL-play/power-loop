# Build your own: recreating the built-ins from primitives

[中文](../../zh/user-guide/build-your-own-tools.md) | [User Guide](../index.md)

Every "special" built-in tool in power-loop — background tasks, sub-agents, timers, human-input,
the blackboard, memory — is built on **public primitives you can use too**. This page rebuilds each
one as a plain custom tool, so you can see there's no magic and adapt the pattern to your own
backends. All the code here is the runnable [`examples/42_build_your_own_tools.py`](https://github.com/) and is
exercised by `tests/unit/test_byo_tools.py`, so it stays honest.

> Prerequisite: the [Async orchestration](async-orchestration.md) model (host-driven; the
> `send`/`resume`/`submit_input`/`follow_up` wake API) and the [tool recipe](extending-tools.md).

## The primitives palette

Everything below uses only these public seams (all exported from `power_loop`):

| Primitive | What it gives a custom tool |
|---|---|
| `ToolDefinition` + `ToolRegistry.register(defn, handler)` | declare a tool + an **`async def`** handler |
| `get_tool_runtime_context()` → `ToolRuntimeContext(session_id, store, loop, config)` | reach the live session from inside a handler |
| `SessionStore` methods (`add_note`/`list_notes`, `get_runtime_state`/`set_runtime_state`, `upsert_background_task`/`list_unseen_background_updates`/`mark_background_seen`, …) | durable per-session state |
| `loop.schedule_timer` / `follow_up` / `resume` / `submit_input` | wake / continue the loop |
| `run_agent_spec(spec, input, *, parent_loop)` + `AgentSpec` | run a child agent |
| `RuntimeProjector` (+ `config.runtime_projectors`) | inject transient state into the prompt each round |
| `HumanInputRequired` | a control signal that pauses the loop for input |
| `MemoryProvider` (+ `config.memory`) | recall durable memory into context each send |

The rule for "from primitives": **never import the built-in's private module** — only the public
seams above. Where a built-in uses a private helper, you replicate that small bit of logic inline
(the per-feature *Gaps* note where this happens).

---

## 1. Background task — `background_run` → `bg_run`

**Built-in:** `background_run` runs a shell command on a daemon thread, returns a `task_id`
immediately, and the result re-enters the loop automatically.

**Primitives:** `get_tool_runtime_context()` (→ `store`, `session_id`), `asyncio.get_running_loop()`
to capture the owning loop, `store.upsert_background_task(...)` to record status, and a
`RuntimeProjector` that each round injects `list_unseen_background_updates(...)` then
`mark_background_seen(...)`.

```python
class CustomBackgroundManager:
    async def run(self, command: str) -> str:
        ctx = get_tool_runtime_context(required=True)
        store, sid = ctx.store, ctx.session_id
        owning_loop = asyncio.get_running_loop()          # daemon writes back onto this loop
        task_id = uuid.uuid4().hex[:8]
        await store.upsert_background_task(sid, task_id=task_id, command=command, status="running")
        def _work():
            p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
            status = "completed" if p.returncode == 0 else f"failed({p.returncode})"
            asyncio.run_coroutine_threadsafe(
                store.upsert_background_task(sid, task_id=task_id, command=command,
                                            status=status, output_tail=(p.stdout + p.stderr)[:4000]),
                owning_loop)          # if the loop closed at shutdown this write is dropped (gotcha)
        threading.Thread(target=_work, daemon=True).start()
        return f"Background task {task_id} started."

class CustomBackgroundProjector(RuntimeProjector):       # the re-entry: install via config.runtime_projectors
    def __init__(self, *, mark_seen: bool = True): self.mark_seen = mark_seen
    async def project(self, *, store, session_id, round_index, context):
        updates = await store.list_unseen_background_updates(session_id)
        if not updates: return []
        body = "<background_updates>" + "".join(
            f'<task id="{t.task_id}" status="{t.status}">{(t.output_tail or "").strip()}</task>'
            for t in updates) + "</background_updates>"
        if self.mark_seen: await store.mark_background_seen(session_id, [t.task_id for t in updates])
        return [{"role": "user", "name": "background_updates", "content": body}]
```

**Parity:** starts concurrent work, returns immediately, and the result re-enters on a later round.
**Gotcha:** a custom projector must **replace** the default `runtime_projectors` (which already
includes `BackgroundRuntimeProjector`) via `AgentLoopConfig(runtime_projectors=(CustomBackgroundProjector(),))`,
or *both* inject. **Gaps:** the built-in adds dangerous-command validation, persistent bash sessions,
and an LRU session cache — none needed for functional parity.

---

## 2. Sub-agent — `spawn_agent` → `delegate`

**Built-in:** `spawn_agent` runs a child agent to completion inside the parent's tool call.

**Primitive:** `run_agent_spec(spec, task, *, parent_loop)` does exactly this — it creates a child
session under the parent, runs its loop, and returns `{"status", "final_text", ...}`.

```python
async def _delegate(task: str, max_rounds: int = 6) -> str:
    ctx = get_tool_runtime_context(required=True)
    spec = AgentSpec(name="delegate", system_prompt="You are a focused worker…",
                     tools=[], max_rounds=max_rounds)          # tools=[]: none; None: inherit parent's
    res = await run_agent_spec(spec, task, parent_loop=ctx.loop)
    out = res.get("final_text") or "(no output)"
    return out if res.get("status") == "completed" else f"[child {res.get('status')}] {out}"
```

**Parity:** full — runs the child inline and returns its answer; depth is capped by
`MAX_SPAWN_DEPTH`. **Gaps:** the built-in also exposes a per-child `model` override, the full
`SUBAGENT_*` event lifecycle, and `spawn_tool_call_id` audit linkage (observability, not behavior).

---

## 3. Mini-workflow — `WorkflowSpec` → `run_pipeline`

**Built-in:** the `WorkflowSpec` DSL/engine runs a declarative graph of steps with parallelism,
journaled resume, and structured outputs.

**Primitive:** the orchestration *core* is just calling `run_agent_spec` in sequence, threading each
step's output into the next:

```python
async def _run_pipeline(goal: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    carry = goal
    for step in ("research", "draft"):
        spec = AgentSpec(name=step, system_prompt=f"You are the '{step}' step.", tools=[], max_rounds=4)
        res = await run_agent_spec(spec, f"...\n\n{carry}", parent_loop=ctx.loop)
        carry = res.get("final_text") or carry
    return carry
```

**Parity:** a fixed sequential pipeline. **Gaps (what the real engine adds):** parallel / foreach /
branch nodes, journaled cross-restart resume, per-step structured-output schemas, detached
execution, and whole-spec validation. Reach for the built-in `WorkflowSpec` when you need those;
this shows the primitive underneath.

---

## 4. Durable timer — `schedule_wakeup` → `remind_me`

**Built-in:** `schedule_wakeup` arms a durable self-wake; the note fires back into the session later.

**Primitive:** `loop.schedule_timer(sid, delay_s=…, note=…)` writes the same `pl_timers` row. The
**host** must run a `TimerRunner` (or poll `store.due_timers()`) to actually fire it — power-loop
has no daemon (see [Async orchestration](async-orchestration.md) and [Timers](timers.md)).

```python
async def _remind_me(delay_seconds: int, note: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    timer = await ctx.loop.schedule_timer(ctx.session_id, delay_s=float(delay_seconds), note=note)
    return f"Reminder armed (#{timer.timer_id}) in {delay_seconds}s."
```

**Parity:** the durable wake-up itself. **Gaps:** the firing side — the `TIMER_FIRE` hook
(veto/skip/postpone), the stale-row heartbeat, and at-least-once dedupe — lives in `TimerRunner` /
your scheduler, not the tool. If you write your own scheduler, you own those.

---

## 5. Human input / approval — `request_user_input` → `ask_human`

**Built-in:** `request_user_input` pauses the loop with `status="waiting_for_input"` and
`pending_interactions`; the host resumes with `loop.submit_input(...)`.

**Primitive:** the pause is a **control-signal exception**, `HumanInputRequired` (public). A custom
tool gets *full parity* by raising it:

```python
async def _ask_human(prompt: str, options: list[str] | None = None) -> str:
    raise HumanInputRequired(kind="choice" if options else "text", prompt=prompt,
                             options=[{"value": o, "label": o} for o in (options or [])])
```

Host flow: `send()` → `result.status == "waiting_for_input"` → read
`result.pending_interactions[0]["interaction_id"]` → `loop.submit_input(sid, interaction_id, answer)`
→ the loop continues with the answer as the tool result.

**Parity:** full. **Gap:** this relies on the pipeline's explicit `except HumanInputRequired` catch.
`HumanInputRequired` is in `__all__` (public) but not in `STABLE_API`; the catch is a deliberate,
stable part of the framework, but it is an architectural dependency to be aware of.

---

## 6. Blackboard — `board_*` → `board_write` / `board_read`

**Built-in:** `board_*` is a shared scratchpad agents read/write to coordinate.

**Primitive:** the store's per-session key/value, `get_runtime_state` / `set_runtime_state` (values
are JSON-serializable):

```python
_BOARD_KEY = "byo_board"
async def _board_write(text: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    board = list(await ctx.store.get_runtime_state(ctx.session_id, _BOARD_KEY, default=[]) or [])
    board.append(text)
    await ctx.store.set_runtime_state(ctx.session_id, _BOARD_KEY, board)
    return f"Posted (entry #{len(board)})."
```

**Parity:** a session-scoped shared scratchpad (multiple agents *in the same session* see it).
**Gap:** the built-in `SqliteBlackboard` shares state across sessions by `board_id` and adds
versioned-JSON optimistic concurrency; this session-local version is simpler. For cross-session or
cross-process sharing, inject a board backend via `runtime_env_context` instead.

---

## 7. Memory / notes — `note_add` + recall → `remember` + `CustomNotesMemory`

**Built-in:** `note_add` writes durable notes; `SQLiteNoteMemory` (a `MemoryProvider`) recalls them
into context at the start of every send.

**Primitives:** `store.add_note` / `list_notes` for the tool, and the `MemoryProvider` protocol
(`config.memory`) + `render_notes(...)` for recall:

```python
async def _remember(content: str, pinned: bool = False) -> str:
    ctx = get_tool_runtime_context(required=True)
    note = await ctx.store.add_note(ctx.session_id, content.strip(), pinned=bool(pinned))
    return f"Remembered as note #{note.note_id}."

class CustomNotesMemory:                                   # set config.memory = CustomNotesMemory(store)
    def __init__(self, store, *, policy=DEFAULT_NOTES_POLICY): self._store, self._policy = store, policy
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        if session_id is None: return []
        text = render_notes(await self._store.list_notes(session_id), policy=self._policy)
        return [{"role": "system", "name": "memory_notes", "content": text}] if text else []
    async def remember(self, *, snapshot, session_id=None): return None   # notes are written live
```

**Parity:** durable notes written by the agent and recalled each send. **Gaps:** the built-in's
policy-checked add/update helpers (`NotesPolicy` enforcement of caps/eviction) are private — replicate
that small check inline if you need it; and `NoteRow` isn't exported, so treat returned notes by
attribute (`note.note_id`, `note.content`, `note.pinned`).

---

## Crash recovery (bonus)

None of these change the recovery story: a crash mid-tool-call leaves `pending` set; the next
`send()` raises `SessionPendingError`; recover with `resume` / `abort_pending` / `heal_pending`. See
[`examples/05_pending_recovery.py`](https://github.com/) and the recovery section of
[Async orchestration](async-orchestration.md).

## See also

- [`examples/42_build_your_own_tools.py`](https://github.com/) — all of the above, runnable, with a real-LLM demo
- [Async orchestration](async-orchestration.md) · [Extending tools](extending-tools.md)
- Deep dives: [Advanced runtime tools](advanced-runtime-tools.md) · [Sub-agents](subagents.md) · [Timers](timers.md) · [Workflows](workflows.md) · [Blackboard](blackboard.md) · [Memory](memory.md)
