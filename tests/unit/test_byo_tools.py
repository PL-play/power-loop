"""Build-your-own tools (examples/42) — deterministic parity tests for each reimplementation.

Each test drives the custom tool through a scripted-LLM loop (so the loop binds the tool runtime
context) and asserts it reproduces the built-in's observable effect, using only public primitives.
Loads the canonical code straight from the shipped example so the docs/example/test never drift.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, JobRef, RuntimeEnv, StoreBlackboard, ToolRegistry
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


def _registry(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    byo.register_byo_tools(registry, include=set(names))
    return registry


async def _loop(runtime_agent, store, llm, registry, *, memory=None, **kwargs):
    return await runtime_agent(model=llm, store=store, registry=registry,
        config=AgentLoopConfig(system_prompt="S", max_rounds=6, compactor=None, memory=memory), **kwargs)


@pytest.mark.asyncio
async def test_memory_remember_persists_and_recalls(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "remember", '{"content": "favorite color is teal"}'),
        LLMResponse(raw_text="saved"),
        LLMResponse(raw_text="recalled"),
    ])
    loop = await _loop(runtime_agent, store, llm, _registry("remember"), memory=byo.CustomNotesMemory(store))
    sid = loop.target.session_id

    await loop.send("remember my color")
    notes = await store.list_notes(sid)
    assert [n.content for n in notes] == ["favorite color is teal"]

    n = len(llm.calls)
    await loop.send("what is my color?")
    # The MemoryProvider injected the note as a system message into the 2nd send's request.
    joined = "\n".join(str(m.get("content", "")) for m in llm.calls[n])
    assert "favorite color is teal" in joined


@pytest.mark.asyncio
async def test_blackboard_write_and_read(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "board_write", '{"text": "kickoff"}'),
        LLMResponse(raw_text="posted"),
        _tool_resp("c2", "board_read", "{}"),
        LLMResponse(raw_text="read"),
    ])
    loop = await _loop(runtime_agent, store, llm, _registry("board_write", "board_read"))
    sid = loop.target.session_id

    await loop.send("post kickoff")
    assert [entry.text for entry in await StoreBlackboard(store).read(sid)] == ["kickoff"]

    await loop.send("read the board")
    tool_rows = [m for m in await store.load_active_messages(sid) if m.role == "tool" and m.tool_call_id == "c2"]
    assert tool_rows and "kickoff" in str(tool_rows[0].content)


@pytest.mark.asyncio
async def test_timer_remind_me_arms_a_durable_timer(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "remind_me", '{"delay_seconds": 60, "note": "check the build"}'),
        LLMResponse(raw_text="armed"),
    ])
    loop = await _loop(runtime_agent, store, llm, _registry("remind_me"))
    sid = loop.target.session_id

    await loop.send("remind me later")
    timers = await store.list_timers(sid)
    assert [t.note for t in timers] == ["check the build"]


@pytest.mark.asyncio
async def test_human_input_ask_human_pauses_then_submit_resumes(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[
        _tool_resp("c1", "ask_human", '{"prompt": "Proceed?", "options": ["yes", "no"]}'),
        LLMResponse(raw_text="continued after approval"),
    ])
    loop = await _loop(runtime_agent, store, llm, _registry("ask_human"))

    from power_loop.agent.runtime_store import now_ms
    from tests.unit.test_history_projection_loop import question
    receipt, pending = await question(loop, "ask before continuing")
    interaction = pending["pending_interactions"][0]
    assert interaction["prompt"] == "Proceed?"
    assert interaction["options"][0]["value"] == "yes"
    status = await loop.runtime.client.inspect(receipt.input_ref, actor=loop.actor)
    answer = await loop.runtime.client.answer(status.job_ref, interaction_id=interaction["interaction_id"],
        wait_round=pending["round_index"], value="yes", message_id="answer", actor=loop.actor)
    resumed = await loop.runtime.client.wait(answer.input_ref, actor=loop.actor, deadline_ms=now_ms()+10000)
    assert resumed.status == "completed" and resumed.final_text == "continued after approval"
    await loop.drain()


@pytest.mark.asyncio
async def test_subagent_delegate_returns_receipt_and_child_result(store, runtime_agent):
    from power_loop.agent.runtime_store import now_ms
    llm = _Scripted(responses=[_tool_resp("c1", "delegate", '{"task":"greet"}'),
        LLMResponse(raw_text="parent done"), LLMResponse(raw_text="child says hello")])
    loop = await _loop(runtime_agent, store, llm, _registry("delegate"))
    assert (await loop.send("delegate a greeting")).status == "completed"
    rows = await store.load_all_messages(loop.target.session_id)
    receipt = json.loads(next(row.content for row in rows if row.tool_call_id == "c1"))
    result = await loop.runtime.client.wait(JobRef(**receipt["job_ref"]), actor=loop.actor, deadline_ms=now_ms()+10000)
    assert result.status == "completed" and result.final_text == "child says hello"
    await loop.drain()


@pytest.mark.asyncio
async def test_mini_workflow_chains_two_subagent_steps(store, runtime_agent):
    llm = _Scripted(responses=[LLMResponse(raw_text="facts: pack light"), LLMResponse(raw_text="FINAL PLAN: go")])
    loop = await _loop(runtime_agent, store, llm, _registry("run_pipeline"),
        command=("run_pipeline", {"goal":"plan a trip"}))
    result = await loop.send("run pipeline")
    assert result.status == "completed" and result.final_text == "FINAL PLAN: go"
    assert any("facts: pack light" in str(row.get("content")) for row in llm.calls[1])


@pytest.mark.asyncio
async def test_background_bg_run_executes_after_parent_and_remains_queryable(store, runtime_agent, tmp_path):
    from power_loop.agent.runtime_store import now_ms
    llm = _Scripted(responses=[_tool_resp("c1", "bg_run", '{"action":"run","command":"echo hi"}'),
        LLMResponse(raw_text="started"), LLMResponse(raw_text="saw result"), LLMResponse(raw_text="next turn")])
    loop = await _loop(runtime_agent, store, llm, _registry("bg_run"), runtime_env=RuntimeEnv(workspace_dir=tmp_path))
    assert (await loop.send("run in background")).status == "completed"
    rows = await store.load_all_messages(loop.target.session_id)
    receipt = json.loads(next(row.content for row in rows if row.tool_call_id == "c1"))
    ref = JobRef(**receipt["job_ref"])
    result = await loop.runtime.client.wait(ref, actor=loop.actor, deadline_ms=now_ms()+10000)
    assert result.status == "completed" and "hi" in result.final_text
    await loop.drain()
    await loop.send("next turn")
    notices = [row for row in await store.load_all_messages(loop.target.session_id)
               if "[background_tool_results]" in str(row.content)]
    assert len(notices) == 1 and "hi" in notices[0].content
    assert await loop.runtime.client.wait(ref, actor=loop.actor, deadline_ms=now_ms()+10000) == result
