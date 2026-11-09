"""42 · Custom tools on the current Runtime ports.

Run without a provider: python examples/42_build_your_own_tools.py
Timer/board/notes handlers use public SDK primitives. Delegation and
background aliases retain the SDK implementation's permissions and lifecycle.
The sequential research/draft command is a trusted host command: a worker slot
is released while it waits for each child, including at capacity one.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

from _runtime import demo_runtime

from power_loop import (
    DEFAULT_NOTES_POLICY,
    AgentSpec,
    LLMResponse,
    LLMService,
    StoreBlackboard,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    get_tool_runtime_context,
    register_spawn_agent,
    render_notes,
    run_agent_spec,
    timer_context,
)
from power_loop.runtime.pricing import Free


# Timers use the runtime's configured backend and its trusted invocation origin.
async def _remind_me(delay_seconds: int, note: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    timer = await ctx.loop.timer_backend.create(await timer_context(), delay_s=float(delay_seconds),
        due_at_ms=None, note=note, interval_s=None)
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
pricing=Free("local"))


# This example scopes each board to its authorized current session.
# Cross-session sharing must use a host-authorized stable board reference.
async def _board_write(text: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    entry = await StoreBlackboard(ctx.store).post(ctx.session_id, text=text)
    return f"Posted to board (entry #{entry.id})."


async def _board_read() -> str:
    ctx = get_tool_runtime_context(required=True)
    board = await StoreBlackboard(ctx.store).read(ctx.session_id)
    return "\n".join(f"{entry.id}. {entry.text}" for entry in board) or "(board empty)"


BOARD_WRITE_TOOL = ToolDefinition(
    name="board_write",
    description="Append a line to the shared session scratchpad (visible to all agents here).",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    required_params=("text",),
pricing=Free("local"))
BOARD_READ_TOOL = ToolDefinition(
    name="board_read",
    description="Read the shared session scratchpad.",
    input_schema={"type": "object", "properties": {}},
pricing=Free("local"))


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
pricing=Free("local"))


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
pricing=Free("composite"))


# Ordinary handlers and aliases are registered through the public registry.
_SIMPLE_TOOLS: tuple[tuple[ToolDefinition, Any], ...] = (
    (REMIND_TOOL, _remind_me),
    (BOARD_WRITE_TOOL, _board_write),
    (BOARD_READ_TOOL, _board_read),
    (REMEMBER_TOOL, _remember),
    (RUN_PIPELINE_TOOL, _run_pipeline),
)


def register_byo_tools(registry: ToolRegistry, *, include: set[str] | None = None) -> None:
    """Select ordinary custom handlers and aliases of lifecycle-aware tools."""
    for defn, handler in _SIMPLE_TOOLS:
        if include is None or defn.name in include:
            registry.register(defn, handler, overwrite=True)
    native = create_default_tool_registry(include=["background_run", "bash"], bind=False)
    register_spawn_agent(native)
    if include is None or "bg_run" in include:
        shell = native.get("bash")
        registry.register(shell.definition, shell.handler, overwrite=True)
    for alias, original in (("delegate", "spawn_agent"), ("bg_run", "background_run")):
        if include is None or alias in include:
            registered = native.get(original)
            registry.register(replace(registered.definition, name=alias), registered.handler, overwrite=True)


class DemoModel(LLMService):
    def __init__(self):
        self.round = 0

    async def complete(self, request, **kwargs):
        self.round += 1
        if self.round == 1:
            return LLMResponse(raw_text="", tool_calls=[
                {"id": "note", "type": "function", "function": {"name": "remember", "arguments": '{"content":"favorite color is teal"}'}},
                {"id": "board", "type": "function", "function": {"name": "board_write", "arguments": '{"text":"kickoff"}'}}])
        return LLMResponse(raw_text="Saved note and board entry.")

    async def close(self):
        pass


async def main() -> dict[str, Any]:
    registry = ToolRegistry()
    register_byo_tools(registry, include={"remember", "board_write"})
    async with demo_runtime(DemoModel(), registry) as (runtime, facade, target, actor):
        result = await facade.send(target, "Remember the color and post kickoff", message_id="demo-1",
            actor=actor, deadline_ms=int(time.time()*1000)+10000)
        notes = await runtime.store.list_notes(target.session_id)
        board = await StoreBlackboard(runtime.store).read(target.session_id)
        assert result.status == "completed" and len(notes) == len(board) == 1, (result, [(r.role, r.content) for r in await runtime.store.load_all_messages(target.session_id)])
        print(result.final_text)
        return {"notes_count": len(notes), "board_entries": len(board)}


if __name__ == "__main__":
    asyncio.run(main())
