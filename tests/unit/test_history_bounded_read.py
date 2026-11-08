"""7.1: history is read in bounded slices; send(history=False) and send metadata."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, RequestOptions
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_stateful_loop import _Scripted, _tool_resp


def _script(n):
    """Each send: one tool call, then a final answer."""
    out = []
    for i in range(n):
        out += [_tool_resp(f"c{i}", "echo", f'{{"i": {i}}}'), LLMResponse(raw_text=f"answer {i}")]
    return out


async def _run(runtime_agent, sends, *, cap=6, history=True, metadata=None):
    reg = ToolRegistry()
    reg.register(ToolDefinition("echo", "", pricing=Free("test_fixture")), lambda **kw: f"echo {kw}")
    llm = _Scripted(_script(sends))
    runner = await runtime_agent(model=llm, registry=reg, config=AgentLoopConfig(
        system_prompt="S", max_tokens=100, max_rounds=5, representation=ProjectedRepresentation(),
        max_context_rows=cap))
    for i in range(sends):
        await runner.send(f"question {i}", options=RequestOptions(history=history,
            metadata=(metadata or {}) | {"i": i} if metadata is not None else {}))
    return runner, llm


@pytest.mark.parametrize("cap", [6, None])
async def test_bounded_read_gives_the_same_prompt_as_a_full_read(monkeypatch, cap, runtime_agent):
    from power_loop import SessionStore
    _, bounded_llm = await _run(runtime_agent, 12, cap=cap)
    with monkeypatch.context() as patch:
        load = SessionStore.load_project_messages
        async def full_projection_read(self, *args, **kwargs):
            kwargs.pop("last_rows", None)
            return await load(self, *args, **kwargs)
        patch.setattr(SessionStore, "load_project_messages", full_projection_read)
        _, full_llm = await _run(runtime_agent, 12, cap=cap)
    assert bounded_llm.calls == full_llm.calls
    main = [c for c in bounded_llm.calls if "question 11" in str(c) and "summarizer" not in str(c)][-1]
    if cap:
        assert "[#1] question 0" not in str(main)


async def test_bounded_read_never_scans_the_whole_session(monkeypatch, runtime_agent):
    from power_loop import SessionStore
    runner, _ = await _run(runtime_agent, 10)
    calls = []
    original = SessionStore.load_active_messages
    async def spy(self, session_id, **kw):
        calls.append(kw)
        return await original(self, session_id, **kw)
    monkeypatch.setattr(SessionStore, "load_active_messages", spy)
    await runner.send("next")
    assert calls and all(kw.get("send_index") is not None for kw in calls)


async def test_history_false_sees_only_its_own_send_and_writes_no_projection(runtime_agent):
    runner, llm = await _run(runtime_agent, 3, history=False, metadata={"agent": "classify"})
    store, sid = runner.runtime.store, runner.target.session_id
    last = llm.calls[-1]
    assert "question 2" in str(last)
    assert "question 0" not in str(last) and "answer 0" not in str(last)
    assert not await store.load_project_messages(sid)
    rows = await store.load_all_messages(sid)
    users = [r for r in rows if r.role == "user"]
    assert [r.meta.get("send") for r in users] == [{"agent": "classify", "i": i} for i in range(3)]
    assert len([r for r in rows if r.role == "tool"]) == 3


async def test_history_false_then_normal_send_heals_only_the_window(runtime_agent):
    runner, llm = await _run(runtime_agent, 3, history=False)
    result = await runner.send("with history")
    assert result.status == "completed"
    assert "question 2" in str(llm.calls[-1])
