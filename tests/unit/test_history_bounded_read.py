"""7.1: history is read in bounded slices; send(history=False) and send metadata."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.store import SessionStore
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_stateful_loop import _Scripted, _tool_resp


def _script(n):
    """Each send: one tool call, then a final answer."""
    out = []
    for i in range(n):
        out += [_tool_resp(f"c{i}", "echo", f'{{"i": {i}}}'), LLMResponse(raw_text=f"answer {i}")]
    return out


async def _run(store, sends, *, cap=6, history=True, metadata=None):
    reg = ToolRegistry()
    reg.register(ToolDefinition("echo", ""), lambda **kw: f"echo {kw}")
    llm = _Scripted(_script(sends))
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg, config=AgentLoopConfig(
        system_prompt="S", max_tokens=100, max_rounds=5, representation=ProjectedRepresentation(),
        max_context_rows=cap))
    sid = await loop.new_session()
    for i in range(sends):
        await loop.send(f"question {i}", session_id=sid, history=history,
                        metadata=(metadata or {}) | {"i": i} if metadata is not None else None)
    return llm, sid


@pytest.mark.parametrize("cap", [6, None])
async def test_bounded_read_gives_the_same_prompt_as_a_full_read(monkeypatch, cap):
    full_store = await SessionStore.open(":memory:")
    bounded_store = await SessionStore.open(":memory:")
    try:
        bounded_llm, sid = await _run(bounded_store, 12, cap=cap)
        with monkeypatch.context() as m:
            m.delattr(SessionStore, "load_unsent_messages")   # no bounded reader → full read
            full_llm, _ = await _run(full_store, 12, cap=cap)
        assert bounded_llm.calls == full_llm.calls
        main = [c for c in bounded_llm.calls if "question 11" in str(c) and "summarizer" not in str(c)][-1]
        if cap:
            assert "[#1] question 0" not in str(main)   # outside the window: never read
    finally:
        await full_store.close()
        await bounded_store.close()


async def test_bounded_read_never_scans_the_whole_session(monkeypatch):
    store = await SessionStore.open(":memory:")
    try:
        _, sid = await _run(store, 10)
        calls = []
        original = SessionStore.load_active_messages

        async def spy(self, session_id, **kw):
            calls.append(kw)
            return await original(self, session_id, **kw)

        monkeypatch.setattr(SessionStore, "load_active_messages", spy)
        reg = ToolRegistry()
        reg.register(ToolDefinition("echo", ""), lambda **kw: "x")
        loop = StatefulAgentLoop(llm=_Scripted([LLMResponse(raw_text="ok")]), store=store, tool_registry=reg,
                                 config=AgentLoopConfig(system_prompt="S", max_tokens=100, max_rounds=5,
                                                        representation=ProjectedRepresentation(),
                                                        max_context_rows=6))
        await loop.send("next", session_id=sid)
        assert calls and all(kw.get("send_index") is not None for kw in calls)
    finally:
        await store.close()


async def test_history_false_sees_only_its_own_send_and_writes_no_projection():
    store = await SessionStore.open(":memory:")
    try:
        llm, sid = await _run(store, 3, history=False, metadata={"agent": "classify"})
        last = llm.calls[-1]   # final answer request of send 3
        assert "question 2" in str(last)
        assert "question 0" not in str(last) and "answer 0" not in str(last)
        assert not await store.load_project_messages(sid)
        rows = await store.load_all_messages(sid)
        users = [r for r in rows if r.role == "user"]
        assert [r.meta.get("send") for r in users] == [{"agent": "classify", "i": i} for i in range(3)]
        assert len([r for r in rows if r.role == "tool"]) == 3   # raw rows are still stored
    finally:
        await store.close()


async def test_history_false_then_normal_send_heals_only_the_window():
    store = await SessionStore.open(":memory:")
    try:
        llm, sid = await _run(store, 3, history=False)
        reg = ToolRegistry()
        reg.register(ToolDefinition("echo", ""), lambda **kw: "x")
        loop = StatefulAgentLoop(llm=_Scripted([LLMResponse(raw_text="ok")]), store=store, tool_registry=reg,
                                 config=AgentLoopConfig(system_prompt="S", max_tokens=100, max_rounds=5,
                                                        representation=ProjectedRepresentation()))
        result = await loop.send("with history", session_id=sid)
        assert result.status == "completed"
    finally:
        await store.close()
