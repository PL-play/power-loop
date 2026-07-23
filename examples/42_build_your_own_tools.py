"""42 · 用原语自造内置工具 / Build-your-own: recreate the built-ins from primitives

For each "special" built-in, a from-scratch custom tool built on ONLY public power-loop primitives
— no importing the built-in's private module, no magic. This file IS the canonical code referenced
by docs/user-guide/build-your-own-tools.md, and is exercised by tests/unit/test_byo_tools.py.

Features recreated (built-in → custom):
  1. spawn_agent        → delegate          (run_agent_spec + AgentSpec)
  2. schedule_wakeup    → remind_me         (loop.schedule_timer + a host TimerRunner)
  3. request_user_input → ask_human         (raise HumanInputRequired → waiting_for_input → submit_input)
  4. board_*            → board_write/read   (store.get_runtime_state / set_runtime_state)
  5. note add + recall  → remember + CustomNotesMemory (store.add_note/list_notes + MemoryProvider)
  6. WorkflowSpec       → run_pipeline       (a fixed run_agent_spec sequence — the orchestration PRIMITIVE)
  7. background_run     → bg_run + CustomBackgroundProjector (daemon + pl_background_tasks + RuntimeProjector re-entry)

Each section notes what it FULLY recreates and what only the built-in adds (see the guide for detail).

Run / 运行
----------
    python examples/42_build_your_own_tools.py        # needs .env with POWER_LOOP_*
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
import uuid
from tempfile import TemporaryDirectory
from typing import Any

from _helpers import make_llm

from power_loop import (
    DEFAULT_NOTES_POLICY,
    AgentLoopConfig,
    AgentSpec,
    HumanInputRequired,
    RuntimeProjector,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    get_tool_runtime_context,
    render_notes,
    run_agent_spec,
)

# ── 1. SUB-AGENT — recreate spawn_agent via run_agent_spec ──────────────────────────────────
# Parity: runs a child agent to completion inside this tool call, returns its final text inline.
# Only the built-in adds: per-child model override, the full SUBAGENT_* event lifecycle, and
# spawn_tool_call_id audit linkage (none needed for functional parity).


async def _delegate(task: str, max_rounds: int = 6) -> str:
    ctx = get_tool_runtime_context(required=True)
    if ctx.loop is None:
        return "Error: delegate needs an active loop."
    spec = AgentSpec(
        name="delegate",
        system_prompt="You are a focused worker. Do the task and give a concise final answer.",
        tools=[],  # [] = no tools (pure reasoning); None = inherit the parent's tools
        max_rounds=max(1, min(50, max_rounds)),
    )
    res = await run_agent_spec(spec, task, parent_loop=ctx.loop)
    out = res.get("final_text") or "(no output)"
    return out if res.get("status") == "completed" else f"[child {res.get('status')}] {out}"


DELEGATE_TOOL = ToolDefinition(
    name="delegate",
    description="Delegate a self-contained sub-task to a child agent; returns its answer.",
    input_schema={"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
    required_params=("task",),
)


# ── 2. TIMER — recreate schedule_wakeup via loop.schedule_timer ─────────────────────────────
# Parity: a durable self-wake-up; the note fires back as a follow_up later. The HOST must run a
# TimerRunner (or poll store.due_timers) — see main(). Only the built-in adds the TIMER_FIRE hook
# wiring and at-least-once dedupe helpers (orchestrator-side, not tool-side).


async def _remind_me(delay_seconds: int, note: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    timer = await ctx.loop.schedule_timer(ctx.session_id, delay_s=float(delay_seconds), note=note)
    return f"Reminder armed (#{getattr(timer, 'timer_id', timer)}) in {delay_seconds}s."


REMIND_TOOL = ToolDefinition(
    name="remind_me",
    description="Schedule a durable wake-up; you'll receive the note after the delay.",
    input_schema={
        "type": "object",
        "properties": {"delay_seconds": {"type": "integer"}, "note": {"type": "string"}},
        "required": ["delay_seconds", "note"],
    },
    required_params=("delay_seconds", "note"),
)


# ── 3. HUMAN INPUT — recreate request_user_input via HumanInputRequired ─────────────────────
# Parity: FULL. Raising the (public) HumanInputRequired control signal makes the loop yield
# status="waiting_for_input" with pending_interactions; the host resolves it with
# loop.submit_input(sid, interaction_id, value) and the loop continues. (HumanInputRequired is in
# __all__ but not STABLE_API — the pipeline explicitly catches it; that catch is the dependency.)


async def _ask_human(prompt: str, options: list[str] | None = None) -> str:
    raise HumanInputRequired(
        kind="choice" if options else "text",
        prompt=prompt,
        options=[{"value": o, "label": o} for o in (options or [])],
    )


ASK_HUMAN_TOOL = ToolDefinition(
    name="ask_human",
    description="Pause and ask the human a question (optionally with choices) before continuing.",
    input_schema={
        "type": "object",
        "properties": {"prompt": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}},
        "required": ["prompt"],
    },
    required_params=("prompt",),
)


# ── 4. BLACKBOARD — recreate board_* via the store's per-session runtime_state ──────────────
# Parity: a shared per-session scratchpad agents read/write. The built-in SqliteBlackboard adds
# cross-board sharing (by board_id) + versioned-JSON optimistic concurrency; this session-local
# version uses the public get_runtime_state/set_runtime_state (JSON-serializable values).

_BOARD_KEY = "byo_board"


async def _board_write(text: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    board = list(await ctx.store.get_runtime_state(ctx.session_id, _BOARD_KEY, default=[]) or [])
    board.append(text)
    await ctx.store.set_runtime_state(ctx.session_id, _BOARD_KEY, board)
    return f"Posted to board (entry #{len(board)})."


async def _board_read() -> str:
    ctx = get_tool_runtime_context(required=True)
    board = await ctx.store.get_runtime_state(ctx.session_id, _BOARD_KEY, default=[]) or []
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(board)) or "(board empty)"


BOARD_WRITE_TOOL = ToolDefinition(
    name="board_write",
    description="Append a line to the shared session scratchpad (visible to all agents here).",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    required_params=("text",),
)
BOARD_READ_TOOL = ToolDefinition(
    name="board_read",
    description="Read the shared session scratchpad.",
    input_schema={"type": "object", "properties": {}},
)


# ── 5. MEMORY — recreate note add + recall via store notes + a MemoryProvider ──────────────
# Parity: durable per-session notes the agent writes, recalled into context each send. The built-in
# adds policy-checked add/update helpers (NotesPolicy enforcement); replicate inline if you need it.


async def _remember(content: str, pinned: bool = False) -> str:
    ctx = get_tool_runtime_context(required=True)
    text = content.strip()
    if not text:
        return "Error: empty note."
    note = await ctx.store.add_note(ctx.session_id, text, pinned=bool(pinned))
    return f"Remembered as note #{note.note_id}."


REMEMBER_TOOL = ToolDefinition(
    name="remember",
    description="Save a short durable note; it is shown back to you at the start of each turn.",
    input_schema={
        "type": "object",
        "properties": {"content": {"type": "string"}, "pinned": {"type": "boolean"}},
        "required": ["content"],
    },
    required_params=("content",),
)


class CustomNotesMemory:
    """A MemoryProvider that recalls the session's notes as a system message each send (the bridge
    that makes written notes visible). recall() is the only required half; remember() is a no-op
    because the notes are written live by the tool."""

    def __init__(self, store: Any, *, policy: Any = DEFAULT_NOTES_POLICY) -> None:
        self._store = store
        self._policy = policy

    async def recall(
        self, *, messages: list[dict[str, Any]], session_id: str | None, budget_tokens: int = 1500
    ) -> list[dict[str, Any]]:
        if session_id is None:
            return []
        text = render_notes(await self._store.list_notes(session_id), policy=self._policy)
        return [{"role": "system", "name": "memory_notes", "content": text}] if text else []

    async def remember(self, *, snapshot: Any, session_id: str | None = None) -> None:
        return None


# ── 6. MINI-WORKFLOW — recreate the orchestration PRIMITIVE via a run_agent_spec sequence ────
# Parity: a fixed pipeline of sub-agent steps, each fed the previous output, from one tool call.
# The real WorkflowSpec engine adds: parallel/foreach/branch, journaled resume, structured-output
# schemas per step, detached execution, and validation — this shows only the sequential core.


async def _run_pipeline(goal: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    carry = goal
    for step in ("research", "draft"):
        spec = AgentSpec(name=step, system_prompt=f"You are the '{step}' step of a pipeline.", tools=[], max_rounds=4)
        instr = (
            f"List the key facts needed to achieve the goal.\n\nGoal: {carry}"
            if step == "research"
            else f"Using these facts, write the final answer.\n\nFacts:\n{carry}"
        )
        res = await run_agent_spec(spec, instr, parent_loop=ctx.loop)
        carry = res.get("final_text") or carry
    return carry


RUN_PIPELINE_TOOL = ToolDefinition(
    name="run_pipeline",
    description="Run a fixed research→draft pipeline of sub-agents and return the final result.",
    input_schema={"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]},
    required_params=("goal",),
)


# ── 7. BACKGROUND — recreate background_run via daemon + pl_background_tasks + a RuntimeProjector ─
# Parity: starts concurrent work, returns a task_id immediately, and the result re-enters the loop
# automatically (the projector injects unseen finished tasks each round). The built-in adds dangerous
# -command validation, persistent bash sessions, and an LRU session cache (none needed for parity).


class CustomBackgroundManager:
    """Runs a shell command on a daemon thread and writes its terminal status back to the store."""

    async def run(self, command: str) -> str:
        ctx = get_tool_runtime_context(required=True)
        store, sid = ctx.store, ctx.session_id
        owning_loop = asyncio.get_running_loop()  # the daemon thread writes back onto this loop
        task_id = uuid.uuid4().hex[:8]
        await store.upsert_background_task(sid, task_id=task_id, command=command, status="running")

        def _work() -> None:
            try:
                p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
                status = "completed" if p.returncode == 0 else f"failed({p.returncode})"
                out = (p.stdout + p.stderr).strip()[:4000]
            except Exception as exc:  # noqa: BLE001 — surface any failure as task output
                status, out = "failed", str(exc)[:4000]
            fut = asyncio.run_coroutine_threadsafe(
                store.upsert_background_task(sid, task_id=task_id, command=command, status=status, output_tail=out),
                owning_loop,
            )
            try:
                fut.result(timeout=10)
            except Exception:  # noqa: BLE001 — owning loop closed at shutdown → orphaned write (documented)
                pass

        threading.Thread(target=_work, daemon=True).start()
        return f"Background task {task_id} started; I'll see the result on a later turn."


class CustomBackgroundProjector(RuntimeProjector):
    """Re-entry: each round, inject finished/updated background tasks not yet seen, then mark seen.
    This is what makes a background result visible without the agent polling — mirrors the built-in
    BackgroundRuntimeProjector using only public store methods."""

    def __init__(self, *, mark_seen: bool = True) -> None:
        self.mark_seen = mark_seen

    async def project(
        self, *, store: Any, session_id: str, round_index: int, context: Any
    ) -> list[dict[str, Any]]:
        updates = await store.list_unseen_background_updates(session_id)
        if not updates:
            return []
        lines = ["<background_updates>"]
        for t in updates:
            lines.append(f'<task id="{t.task_id}" status="{t.status}">{(t.output_tail or "(running)").strip()}</task>')
        lines.append("</background_updates>")
        if self.mark_seen:
            await store.mark_background_seen(session_id, [t.task_id for t in updates])
        return [{"role": "user", "name": "background_updates", "content": "\n".join(lines)}]


BG_RUN_TOOL = ToolDefinition(
    name="bg_run",
    description="Run a shell command in the background; the result comes back to you on a later turn.",
    input_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    required_params=("command",),
)


# ── Registration ────────────────────────────────────────────────────────────────────────────

#: All custom (definition, handler-name) tools. The handler for bg_run is bound per-manager below.
_SIMPLE_TOOLS: tuple[tuple[ToolDefinition, Any], ...] = (
    (DELEGATE_TOOL, _delegate),
    (REMIND_TOOL, _remind_me),
    (ASK_HUMAN_TOOL, _ask_human),
    (BOARD_WRITE_TOOL, _board_write),
    (BOARD_READ_TOOL, _board_read),
    (REMEMBER_TOOL, _remember),
    (RUN_PIPELINE_TOOL, _run_pipeline),
)


def register_byo_tools(
    registry: ToolRegistry, *, include: set[str] | None = None, background_manager: CustomBackgroundManager | None = None
) -> None:
    """Register the build-your-own tools onto a registry. ``include`` selects a subset by name."""
    for defn, handler in _SIMPLE_TOOLS:
        if include is None or defn.name in include:
            registry.register(defn, handler, overwrite=True)
    if (include is None or "bg_run" in include) and background_manager is not None:
        registry.register(BG_RUN_TOOL, background_manager.run, overwrite=True)


async def main() -> dict[str, Any]:
    """Real-LLM smoke: the agent uses the custom remember + board + delegate tools as drop-ins."""
    from power_loop import SessionStore

    with TemporaryDirectory() as tmp:
        store = await SessionStore.open(f"{tmp}/sessions.db")
        registry = create_default_tool_registry(preset="core", bind=False)
        register_byo_tools(registry, include={"remember", "board_write", "delegate"})
        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You have custom tools: remember(content), board_write(text), delegate(task). "
                    "Use them when asked."
                ),
                memory=CustomNotesMemory(store),  # the bridge that recalls written notes each send
                max_rounds=6,
            ),
            tool_registry=registry,
        )
        sid = await loop.new_session()

        await loop.send(
            "Remember that my favorite color is teal, then post 'kickoff' to the shared board.",
            session_id=sid,
        )
        d = await loop.send("Delegate this to a sub-agent and report its answer: what is 2+2?", session_id=sid)

        notes = await store.list_notes(sid)
        board = await store.get_runtime_state(sid, _BOARD_KEY, default=[]) or []
        print("notes:", [n.content for n in notes])
        print("board:", board)
        print("delegate reply:", (d.final_text or "")[:120])
        await loop.aclose()
        return {
            "notes_count": len(notes),
            "board_entries": len(board),
            "delegate_replied": bool((d.final_text or "").strip()),
        }


if __name__ == "__main__":
    asyncio.run(main())
