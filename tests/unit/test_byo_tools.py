"""Build-your-own tools (examples/42) — deterministic parity tests for each reimplementation.

Each test drives the custom tool through a scripted-LLM loop (so the loop binds the tool runtime
context) and asserts it reproduces the built-in's observable effect, using only public primitives.
Loads the canonical code straight from the shipped example so the docs/example/test never drift.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop, ToolRegistry
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.runtime.store.store import SessionStore
from tests.unit.test_stateful_loop import _Scripted, _tool_resp

_EXAMPLE = Path(__file__).resolve().parents[1].parent / "examples" / "42_build_your_own_tools.py"


def _load_byo():
    sys.path.insert(0, str(_EXAMPLE.parent))  # so the example's `from _helpers import …` resolves
    spec = importlib.util.spec_from_file_location("byo42", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


byo = _load_byo()


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


def _registry(*names: str, manager=None) -> ToolRegistry:
    r = ToolRegistry()
    byo.register_byo_tools(r, include=set(names), background_manager=manager)
    return r


def _loop(store: SessionStore, llm, registry, *, memory=None) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(system_prompt="S", max_rounds=6, compactor=None, memory=memory),
    )


@pytest.mark.asyncio
async def test_memory_remember_persists_and_recalls(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "remember", '{"content": "favorite color is teal"}'),
        LLMResponse(raw_text="saved"),
        LLMResponse(raw_text="recalled"),
    ])
    loop = _loop(store, llm, _registry("remember"), memory=byo.CustomNotesMemory(store))
    sid = await loop.new_session()

    await loop.send("remember my color", session_id=sid)
    notes = await store.list_notes(sid)
    assert [n.content for n in notes] == ["favorite color is teal"]

    n = len(llm.calls)
    await loop.send("what is my color?", session_id=sid)
    # The MemoryProvider injected the note as a system message into the 2nd send's request.
    joined = "\n".join(str(m.get("content", "")) for m in llm.calls[n])
    assert "favorite color is teal" in joined


@pytest.mark.asyncio
async def test_blackboard_write_and_read(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "board_write", '{"text": "kickoff"}'),
        LLMResponse(raw_text="posted"),
        _tool_resp("c2", "board_read", "{}"),
        LLMResponse(raw_text="read"),
    ])
    loop = _loop(store, llm, _registry("board_write", "board_read"))
    sid = await loop.new_session()

    await loop.send("post kickoff", session_id=sid)
    assert (await store.get_runtime_state(sid, byo._BOARD_KEY, default=[])) == ["kickoff"]

    await loop.send("read the board", session_id=sid)
    tool_rows = [m for m in await loop.get_messages(sid) if m["role"] == "tool" and m["tool_call_id"] == "c2"]
    assert tool_rows and "kickoff" in str(tool_rows[0]["content"])


@pytest.mark.asyncio
async def test_timer_remind_me_arms_a_durable_timer(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "remind_me", '{"delay_seconds": 60, "note": "check the build"}'),
        LLMResponse(raw_text="armed"),
    ])
    loop = _loop(store, llm, _registry("remind_me"))
    sid = await loop.new_session()

    await loop.send("remind me later", session_id=sid)
    timers = await loop.list_timers(sid)
    assert [t.note for t in timers] == ["check the build"]


@pytest.mark.asyncio
async def test_human_input_ask_human_pauses_then_submit_resumes(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "ask_human", '{"prompt": "Proceed?", "options": ["yes", "no"]}'),
        LLMResponse(raw_text="continued after approval"),
    ])
    loop = _loop(store, llm, _registry("ask_human"))
    sid = await loop.new_session()

    paused = await loop.send("do the risky thing", session_id=sid)
    assert paused.status == "waiting_for_input"
    interaction = paused.pending_interactions[0]
    assert interaction["prompt"] == "Proceed?"
    assert interaction["options"][0]["value"] == "yes"

    resumed = await loop.submit_input(sid, interaction["interaction_id"], "yes")
    assert resumed.status == "completed"
    assert resumed.final_text == "continued after approval"


@pytest.mark.asyncio
async def test_subagent_delegate_runs_child_inline(store: SessionStore) -> None:
    # Order: parent calls delegate → the child (same scripted llm) answers → parent finishes.
    llm = _Scripted(responses=[
        _tool_resp("c1", "delegate", '{"task": "greet"}'),
        LLMResponse(raw_text="child says hello"),  # the child agent's only round
        LLMResponse(raw_text="parent done"),
    ])
    loop = _loop(store, llm, _registry("delegate"))
    sid = await loop.new_session()

    result = await loop.send("delegate a greeting", session_id=sid)
    assert result.status == "completed"
    tool_rows = [m for m in await loop.get_messages(sid) if m["role"] == "tool" and m["tool_call_id"] == "c1"]
    assert tool_rows and "child says hello" in str(tool_rows[0]["content"])


@pytest.mark.asyncio
async def test_mini_workflow_chains_two_subagent_steps(store: SessionStore) -> None:
    # parent run_pipeline → research child → draft child → parent finishes.
    llm = _Scripted(responses=[
        _tool_resp("c1", "run_pipeline", '{"goal": "plan a trip"}'),
        LLMResponse(raw_text="facts: pack light"),    # research step
        LLMResponse(raw_text="FINAL PLAN: go"),        # draft step
        LLMResponse(raw_text="pipeline done"),
    ])
    loop = _loop(store, llm, _registry("run_pipeline"))
    sid = await loop.new_session()

    result = await loop.send("run the pipeline", session_id=sid)
    assert result.status == "completed"
    tool_rows = [m for m in await loop.get_messages(sid) if m["role"] == "tool" and m["tool_call_id"] == "c1"]
    assert tool_rows and "FINAL PLAN: go" in str(tool_rows[0]["content"])  # last step's output carried out


@pytest.mark.asyncio
async def test_background_bg_run_executes_and_writes_back(store: SessionStore) -> None:
    """bg_run starts the command on a daemon thread, returns immediately, and the terminal status
    is written back to pl_background_tasks (asserted via list_background_tasks, which is immune to
    the seen-marker the loop's default projector sets)."""
    manager = byo.CustomBackgroundManager()
    llm = _Scripted(responses=[
        _tool_resp("c1", "bg_run", '{"command": "echo hi"}'),
        LLMResponse(raw_text="started"),
    ])
    loop = _loop(store, llm, _registry("bg_run", manager=manager))
    sid = await loop.new_session()

    await loop.send("run it in the background", session_id=sid)

    done = None
    for _ in range(60):  # the daemon writes back onto this running loop; poll for the terminal row
        tasks = await store.list_background_tasks(sid)
        if tasks and tasks[0].status.startswith("completed"):
            done = tasks[0]
            break
        await asyncio.sleep(0.1)
    assert done is not None and "hi" in (done.output_tail or "")


@pytest.mark.asyncio
async def test_background_custom_projector_reenters_once_then_marks_seen(store: SessionStore) -> None:
    """Re-entry parity, isolated from the loop: CustomBackgroundProjector injects a finished task
    once as <background_updates>, then marks it seen so it does not re-inject (mirrors the built-in
    BackgroundRuntimeProjector). A custom projector must REPLACE the default one (config
    runtime_projectors=...) or both would inject — that's the gotcha this isolation makes explicit."""
    from power_loop.runtime.store.types import SessionKind

    sid = await store.create_session(kind=SessionKind.ROOT)
    await store.upsert_background_task(sid, task_id="bg-1", command="echo hi", status="running")
    await store.upsert_background_task(sid, task_id="bg-1", command="echo hi", status="completed", output_tail="hi")

    proj = byo.CustomBackgroundProjector()
    first = await proj.project(store=store, session_id=sid, round_index=0, context=None)
    assert first and "hi" in str(first[0]["content"]) and "<background_updates>" in str(first[0]["content"])
    assert await proj.project(store=store, session_id=sid, round_index=1, context=None) == []
