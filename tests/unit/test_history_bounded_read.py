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


async def _run(runtime_agent, sends, *, cap=6, history=True, metadata=None, store=None):
    reg = ToolRegistry()
    reg.register(ToolDefinition("echo", "", pricing=Free("test_fixture")), lambda **kw: f"echo {kw}")
    llm = _Scripted(_script(sends))
    runner = await runtime_agent(model=llm, registry=reg, store=store, config=AgentLoopConfig(
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


async def test_history_false_sees_only_its_own_send_and_writes_no_projection(
    runtime_agent, registry_store, monkeypatch,
):
    from power_loop.runtime.store.store import SessionStore
    from tests.unit.test_send_sql_profile import Profiled

    store, _ = registry_store
    window_reads = []
    original = SessionStore.load_projection_window

    async def load(*args, **kwargs):
        window_reads.append(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(SessionStore, "load_projection_window", load)
    profiled = Profiled(store._db)
    monkeypatch.setattr(store, "_db", profiled)
    runner, llm = await _run(runtime_agent, 3, store=store, history=False,
                             metadata={"agent": "classify"})
    store, sid = runner.runtime.store, runner.target.session_id
    assert window_reads == []
    writes = [r["sql"].lower() for r in profiled.records
              if r["sql"].lower().startswith(("insert", "update", "delete", "replace"))]
    assert not any(store.t.project_messages in sql or store.t.session_runtime_state in sql
                   for sql in writes), writes
    last = llm.calls[-1]
    assert "question 2" in str(last)
    assert "question 0" not in str(last) and "answer 0" not in str(last)
    assert not await store.load_project_messages(sid)
    rows = await store.load_all_messages(sid)
    users = [r for r in rows if r.role == "user"]
    assert [json.loads(r.content)["items"][0]["content"]["metadata"] for r in users] == [
        {"agent": "classify", "i": i} for i in range(3)]
    assert len([r for r in rows if r.role == "tool"]) == 3


@pytest.mark.parametrize("history,compact_to", [(0, None), (20, None), (20, 18), (20, 5), (20, 20)])
@pytest.mark.parametrize("send_limit", [None, 5])
async def test_window_has_one_raw_stats_header(registry_store, monkeypatch, history, compact_to, send_limit):
    from power_loop.runtime.store.projection_fold import STATE_KEY
    from power_loop.runtime.store.store import ProjectionWindowStats

    store, _ = registry_store
    sid = await store.create_session(DESCRIPTOR, actor=ACTOR)
    for si in range(1, history + 1):
        await store.upsert_project_message(sid, send_index=si, kind="user",
                                           content={"input": [str(si)]}, projector_version=1)
    if compact_to is not None:
        await store.upsert_project_message(sid, send_index=compact_to, kind="compact",
                                           content={"summary": "kept"}, compact_to_send=compact_to)
    stored = [dict(r) for r in await store._db.fetchall(
        f"SELECT * FROM {store.t.project_messages} WHERE session_id=?", (sid,))]
    columns = {"id", "session_id", "send_index", "kind", "content_json", "rendered_text",
               "source_seq_lo", "source_seq_hi", "completed_through_send_index", "compact_from_send",
               "compact_to_send", "projector_version", "token_estimate", "created_at"}
    # Include invalid JSON and significant whitespace/multibyte text; never reserialize it.
    for exists, raw in [(False, None), (True, None), (True, "{invalid"),
                        (True, '  {"revision":"多字节","v":999} \t ')]:
        await store._db.execute(
            f"DELETE FROM {store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
            (sid, STATE_KEY))
        if exists:
            await store._db.execute(
                f"INSERT INTO {store.t.session_runtime_state} (session_id,state_key,value_json,updated_at) "
                "VALUES (?,?,?,?)", (sid, STATE_KEY, raw, 0))
        returned = []
        fetchall = store._db.fetchall

        async def capture(sql, params=()):
            rows = await fetchall(sql, params)
            returned.append([dict(r) for r in rows])
            return rows

        with monkeypatch.context() as patch:
            patch.setattr(store._db, "fetchall", capture)
            current = max(2, history + 1)
            window = await store.load_projection_window(
                sid, current_send_index=current, send_limit=send_limit)
        assert len(returned) == 1
        headers = [r for r in returned[0] if r["id"] is None and r["kind"] == "__pl_stats__"]
        assert len(headers) == 1
        assert window.stats == ProjectionWindowStats(exists, raw, compact_to)
        assert headers[0]["content_json"] == raw
        assert headers[0]["rendered_text"] == (STATE_KEY if exists else None)
        assert all(set(r) == columns for r in returned[0])
        data = [r for r in returned[0] if r["id"] is not None or r["kind"] != "__pl_stats__"]
        lo = max((compact_to or 0) + 1, max(1, current - send_limit) if send_limit else 1)
        assert {r["id"]: r for r in data} == {
            r["id"]: r for r in stored if r["kind"] == "compact" or r["send_index"] >= lo}
        assert window.window_lo == lo
        assert [r.send_index for r in window.rows] == list(range(lo, history + 1))
        assert len(data) == len(window.rows) + int(compact_to is not None)


async def test_first_send_expects_no_stats_without_query(registry_store, monkeypatch):
    from power_loop.runtime.store.store import ProjectionWindowStats

    store, _ = registry_store

    async def unexpected(*args, **kwargs):
        pytest.fail("first send must not query the history window")

    monkeypatch.setattr(store._db, "fetchall", unexpected)
    for current in (0, 1):
        window = await store.load_projection_window("first", current_send_index=current)
        assert window.stats == ProjectionWindowStats(False, None, None)
        assert window.compact is None and window.rows == [] and window.window_lo == 1


async def test_history_false_retains_originals_for_inspection(runtime_agent):
    runner, llm = await _run(runtime_agent, 3, history=False)
    rows = await runner.runtime.store.load_all_messages(runner.target.session_id)
    assert all(f"question {i}" in str(rows) for i in range(3))


async def test_cap_recent_window_with_sparse_projection_heals_only_missing(registry_store):
    from dataclasses import asdict

    from power_loop.runtime.fold import LLMSummaryFold
    from power_loop.runtime.representation import ProjectionFoldPolicy

    from .test_projection_fold_statistics import assert_exact, cas_effects, cas_trace
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
    class IncrementalRecordingProjector(_RecordingProjector):
        def fold_statistics_policy(self):
            return ProjectionFoldPolicy.for_renderer(self, config={
                "max_chars": self.max_chars, "render": asdict(self.render_config)})

    projector = IncrementalRecordingProjector()
    # Establish genuinely healthy facts BEFORE the run. The first heal must invalidate
    # these facts; passing the old window at final save would cause an observable CAS.
    await store.prepare_send_projection_fold(sid, send_index=18,
        rows=[(r.kind, r.content, r.rendered_text) for r in projected.rows],
        source_seq_lo=None, source_seq_hi=None, projector_version=1, projector=projector,
        keep_last_sends=100, threshold=10**9)
    await assert_exact(store, sid, projector)
    model = _Scripted([LLMResponse(raw_text="latest answer")])
    agent = await RunningAgent(
        store, model,
        config=AgentLoopConfig(representation=projector, max_context_rows=6,
                               context_budget_tokens=10**9,
                               fold_strategy=LLMSummaryFold(keep_last_sends=100)),
    ).start(target=target)
    agent.actor = ACTOR
    try:
        async with cas_trace(store, sid, "run-lazy-heal") as probe:
            result = await agent.send("latest", message_id="latest")
        assert result.status == "completed"
        healed = projector.projected_sends[:-1]
        # 18 has a current projection and is skipped; the rest of the absolute recent
        # window [14, 20] is still repaired.
        assert healed == [14, 15, 16, 17, 19, 20], projector.projected_sends
        assert cas_effects(probe) == []
        assert sum("AS stats_json" in r["sql"] for r in probe.records) == len(healed) + 1
        assert len(model.calls) == 1
        await assert_exact(store, sid, projector)
        prompt = str(model.calls[0])
        assert "question 19" in prompt and "question 0" not in prompt
    finally:
        await agent.aclose()
