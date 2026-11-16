"""7.1: history is read in bounded slices; send(history=False) and send metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import pytest

from power_loop import AgentLoopConfig, RequestOptions
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent
from tests.unit.session_native_fixture import ACTOR, DESCRIPTOR
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
        max_context_rows=cap), options=RequestOptions(history=history))
    for i in range(sends):
        await runner.send({"text": f"question {i}", "metadata": metadata | {"i": i}}
                          if metadata is not None else f"question {i}")
    return runner, llm


@pytest.mark.parametrize("cap", [6, None])
async def test_bounded_read_gives_the_same_prompt_as_a_full_read(monkeypatch, cap, runtime_agent):
    from power_loop import SessionStore
    _, bounded_llm = await _run(runtime_agent, 12, cap=cap)
    with monkeypatch.context() as patch:
        load = SessionStore.load_projection_window
        async def full_projection_read(self, session_id, *, current_send_index, send_limit=None):
            return await load(self, session_id, current_send_index=current_send_index,
                              send_limit=None)
        patch.setattr(SessionStore, "load_projection_window", full_projection_read)
        _, full_llm = await _run(runtime_agent, 12, cap=cap)
    assert bounded_llm.calls == full_llm.calls
    main = [c for c in bounded_llm.calls if "question 11" in str(c) and "summarizer" not in str(c)][-1]
    if cap:
        assert "[#1] question 0" not in str(main)


async def test_bounded_read_never_scans_the_whole_session(monkeypatch, runtime_agent):
    from power_loop import SessionStore
    runner, _ = await _run(runtime_agent, 10)
    calls = []
    original = SessionStore.load_projection_window
    async def spy(self, session_id, *, current_send_index, send_limit=None):
        calls.append(send_limit)
        return await original(self, session_id, current_send_index=current_send_index,
                              send_limit=send_limit)
    monkeypatch.setattr(SessionStore, "load_projection_window", spy)
    await runner.send("next")
    assert calls and all(limit is not None for limit in calls)  # bounded by whole sends


@dataclass
class _RecordingProjector(ProjectedRepresentation):
    projected_sends: list[int] = field(default_factory=list)

    def project_send(self, send_rows, *, send_index, tool_registry):
        self.projected_sends.append(send_index)
        return super().project_send(send_rows, send_index=send_index,
                                    tool_registry=tool_registry)


async def test_cap_bounds_healing_to_the_recent_send_window(registry_store):
    """cap=6 on 20 legal raw-history sends must repair only the recent window, not all 20."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    target = await port.create_session(
        replace(DESCRIPTOR, definition_ref="fixture.agent@215"), actor=ACTOR)
    for index in range(20):
        await port.accept(target, f"question {index}", message_id=f"history-{index}",
                          actor=ACTOR)
        batch = await port.claim(target, node_id="history", boot_id="history")
        await port.final(batch, consumed=batch.input_refs, final_text=f"answer {index}")
    projector = _RecordingProjector()
    model = _Scripted([LLMResponse(raw_text="latest answer")])
    agent = await RunningAgent(
        store, model,
        config=AgentLoopConfig(representation=projector, max_context_rows=6,
                               context_budget_tokens=10**9),
    ).start(target=target)
    agent.actor = ACTOR
    try:
        result = await agent.send("latest", message_id="latest")
        assert result.status == "completed"
        healed = projector.projected_sends[:-1]
        # current send is 21; send_limit = cap + 1 = 7 -> absolute lower bound 14.
        assert healed == list(range(14, 21)), projector.projected_sends
        prompt = str(model.calls[0])
        assert "question 19" in prompt
        assert "question 0" not in prompt
    finally:
        await agent.aclose()


async def test_history_false_sees_only_its_own_send_and_writes_no_projection(runtime_agent):
    runner, llm = await _run(runtime_agent, 3, history=False, metadata={"agent": "classify"})
    store, sid = runner.runtime.store, runner.target.session_id
    last = llm.calls[-1]
    assert "question 2" in str(last)
    assert "question 0" not in str(last) and "answer 0" not in str(last)
    assert not await store.load_project_messages(sid)
    rows = await store.load_all_messages(sid)
    users = [r for r in rows if r.role == "user"]
    assert [json.loads(r.content)["items"][0]["content"]["metadata"] for r in users] == [
        {"agent": "classify", "i": i} for i in range(3)]
    assert len([r for r in rows if r.role == "tool"]) == 3


async def test_history_false_retains_originals_for_inspection(runtime_agent):
    runner, llm = await _run(runtime_agent, 3, history=False)
    rows = await runner.runtime.store.load_all_messages(runner.target.session_id)
    assert all(f"question {i}" in str(rows) for i in range(3))


async def test_cap_recent_window_with_sparse_projection_heals_only_missing(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    target = await port.create_session(
        replace(DESCRIPTOR, definition_ref="fixture.agent@215"), actor=ACTOR)
    for index in range(20):
        await port.accept(target, f"question {index}", message_id=f"history-{index}",
                          actor=ACTOR)
        batch = await port.claim(target, node_id="history", boot_id="history")
        await port.final(batch, consumed=batch.input_refs, final_text=f"answer {index}")
    sid = target.session_id
    # One send in the recent window already has a current projection; its neighbours do not.
    raw18 = await store.load_active_messages(sid, send_index=18)
    projected = ProjectedRepresentation().project_send(raw18, send_index=18,
                                                       tool_registry=None)
    for row in projected.rows:
        await store.upsert_project_message(
            sid, send_index=18, kind=row.kind, content=row.content,
            rendered_text=row.rendered_text, projector_version=1)
    projector = _RecordingProjector()
    model = _Scripted([LLMResponse(raw_text="latest answer")])
    agent = await RunningAgent(
        store, model,
        config=AgentLoopConfig(representation=projector, max_context_rows=6,
                               context_budget_tokens=10**9),
    ).start(target=target)
    agent.actor = ACTOR
    try:
        result = await agent.send("latest", message_id="latest")
        assert result.status == "completed"
        healed = projector.projected_sends[:-1]
        # 18 has a current projection and is skipped; the rest of the absolute recent
        # window [14, 20] is still repaired.
        assert healed == [14, 15, 16, 17, 19, 20], projector.projected_sends
        prompt = str(model.calls[0])
        assert "question 19" in prompt and "question 0" not in prompt
    finally:
        await agent.aclose()
