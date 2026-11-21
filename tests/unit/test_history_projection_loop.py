"""Step 3 integration: the projection layer wired into StatefulAgentLoop._run_loop.

Covers the reader (history = rendered projections of finished sends + the in-flight send
verbatim), the writer (project at end-of-send into pl_project_messages), and the must-fix
behaviors: child-session exclusion (by parent_session_id), reason-gate (waiting_for_input
defers), memory-recall composition, and that the default path (no projector) is unchanged.
Reuses the scripted-LLM helpers from test_stateful_loop.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import InputRef
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.fold import FoldResult
from power_loop.runtime.history_projector import DefaultDeterministicProjector
from power_loop.runtime.store.store import SessionStore
from tests.unit.test_stateful_loop import _echo_registry, _Scripted, _tool_resp


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_outcome", ["conflict", "db_error"])
async def test_c02_projection_outcome_keeps_business_result_and_does_not_replay(
    registry_store, runtime_agent, monkeypatch, projection_outcome,
):
    from power_loop import ProjectedRepresentation, ToolDefinition
    from power_loop.runtime.pricing import Free
    from power_loop.runtime.store.projection_fold import read_state
    from power_loop.tools.registry import ToolRegistry

    from .test_projection_fold_statistics import assert_exact

    store, _ = registry_store
    effects = []
    registry = ToolRegistry()
    registry.register(ToolDefinition("echo", "", pricing=Free("fixture")),
                      lambda **kw: effects.append(kw) or "tool fact")
    model = _Scripted([LLMResponse(raw_text="seed"), _tool_resp("c2", "echo", '{"text":"once"}'),
                       LLMResponse(raw_text="business success")])
    projector = ProjectedRepresentation()
    runner = await runtime_agent(store=store, model=model, registry=registry,
        config=AgentLoopConfig(representation=projector, context_budget_tokens=10**9))
    sid = runner.target.session_id
    assert (await runner.send("seed")).status == "completed"
    complete = model.complete
    changed = []
    during_invalid: list[bool | None] = []

    async def competing_complete(*args, **kwargs):
        if not changed:
            changed.append(True)
            await store.upsert_project_message(sid, send_index=1, kind="user", content={"input": ["corrected"]})
            st = await read_state(store, store._db, sid)
            during_invalid.append(st.get("invalid") if st else None)
        return await complete(*args, **kwargs)

    error_hits = []

    def fail_project(*args, **kwargs):
        error_hits.append(True)
        raise ConnectionError("derived projection failed")

    with monkeypatch.context() as patch:
        if projection_outcome == "conflict":
            patch.setattr(model, "complete", competing_complete)
        else:
            patch.setattr(projector, "project_send", fail_project)
        result = await runner.send("work")
    assert result.status == "completed" and result.final_text == "business success"
    assert len(model.calls) == 3 and len(effects) == 1
    if projection_outcome == "conflict":
        assert changed == [True] and during_invalid == [True]
        assert {r.send_index for r in await store.load_project_messages(sid)} == {1, 2}
        # B1 regression: Final MUST NOT overwrite concurrent invalidation with stale Claim snapshot.
        await assert_exact(store, sid, projector)
        kv_rows = await store._db.fetchall(
            f"SELECT 1 FROM {store.t.session_runtime_state} WHERE session_id=?", (sid,),
        )
        assert kv_rows == []
    else:
        assert error_hits == [True]
        assert {r.send_index for r in await store.load_project_messages(sid)} == {1}


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_state", ["null", "non_null_invalid", "non_null_other_healthy", "rebuild_deferred_to_next_send"])
async def test_b1_final_fold_stats_cas_protects_concurrent_mutations(
    registry_store, runtime_agent, monkeypatch, initial_state,
):
    """Design 217 M1b B1: Final conditional fold_stats_json write covers NULL & non-NULL snapshots,
    never misreports SessionLeaseLost, never replays tools, and never writes stale healthy stats."""
    from power_loop import ProjectedRepresentation, ToolDefinition
    from power_loop.runtime.pricing import Free
    from power_loop.runtime.store import projection_fold as pf
    from power_loop.tools.registry import ToolRegistry

    from .test_projection_fold_statistics import assert_exact

    store, _ = registry_store
    effects = []
    registry = ToolRegistry()
    registry.register(ToolDefinition("echo", "", pricing=Free("fixture")),
                      lambda **kw: effects.append(kw) or "tool fact")
    responses = (
        [_tool_resp("c1", "echo", '{"text":"first"}'), LLMResponse(raw_text="turn1 done")]
        if initial_state == "null"
        else [
            LLMResponse(raw_text="seed"),
            _tool_resp("c2", "echo", '{"text":"once"}'),
            LLMResponse(raw_text="turn2 done"),
            LLMResponse(raw_text="turn3 done"),
        ]
    )
    model = _Scripted(responses)
    projector = ProjectedRepresentation()
    runner = await runtime_agent(
        store=store, model=model, registry=registry,
        config=AgentLoopConfig(representation=projector, context_budget_tokens=10**9),
    )
    sid = runner.target.session_id
    if initial_state != "null":
        assert (await runner.send("seed")).status == "completed"
        before_st = await pf.read_state(store, store._db, sid)
        assert before_st is not None and not before_st.get("invalid")
    else:
        srow = await store._db.fetchone(
            f"SELECT fold_stats_json FROM {store.t.sessions} WHERE session_id=?", (sid,),
        )
        assert srow["fold_stats_json"] is None

    complete = model.complete
    mutated = []

    async def competing_complete(*args, **kwargs):
        if not mutated:
            mutated.append(True)
            if initial_state == "null":
                # Concurrent projection insert during Turn 1 turns fold_stats_json NULL -> invalid=True
                await store.upsert_project_message(
                    sid, send_index=5, kind="user", content={"input": ["preseeded_mid_turn1"]},
                )
            elif initial_state in ("non_null_invalid", "rebuild_deferred_to_next_send"):
                await store.upsert_project_message(
                    sid, send_index=1, kind="user", content={"input": ["corrected_long " * 30]},
                )
            else:
                # Concurrent non-invalid healthy rewrite (different revision) while Turn 2 is running
                await store.upsert_project_message(
                    sid, send_index=1, kind="user", content={"input": ["corrected_alt " * 25]},
                )
                snap = await pf.read_snapshot(store, sid)
                assert snap is not None
                rebuilt = await pf._rebuild(store, sid, projector, projector.fold_statistics_policy(), snap)
                assert rebuilt is not None and not rebuilt.get("invalid")
        return await complete(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(model, "complete", competing_complete)
        if initial_state == "rebuild_deferred_to_next_send":
            async def fail_rebuild(*args, **kwargs):
                raise RuntimeError("transient post-final rebuild error")

            patch.setattr(pf, "_rebuild", fail_rebuild)
        res = await runner.send("work")

    assert mutated == [True]
    assert res.status == "completed"
    assert len(effects) == 1  # tool executed strictly once, no replay or SessionLeaseLost
    if initial_state == "rebuild_deferred_to_next_send":
        # Even if post-final rebuild fails, Final MUST leave invalid=True (never stale healthy!)
        st_after = await pf.read_state(store, store._db, sid)
        assert st_after is not None and st_after.get("invalid") is True
        # Next send rebuilds to exact automatically!
        res3 = await runner.send("follow up")
        assert res3.status == "completed"
        await assert_exact(store, sid, projector)
    else:
        await assert_exact(store, sid, projector)
    kv_rows = await store._db.fetchall(
        f"SELECT 1 FROM {store.t.session_runtime_state} WHERE session_id=?", (sid,),
    )
    assert kv_rows == []


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@dataclass
class _DeterministicFold:
    """Test-only FoldStrategy (no LLM): delegates to the deterministic projector's ``compact()``
    so the loop's fold WIRING (span selection, keep_last_sends, roll-forward, migration) is
    assertable deterministically. The real LLMSummaryFold/AgenticFold are covered in tests/real."""

    keep_last_sends: int = 4
    trigger_ratio: float = 0.75
    fold_id: str = "test_deterministic"

    async def fold(self, rows, *, context):
        folded = [r.send_index for r in rows if r.kind in ("user", "project")]
        if not folded:
            return None
        rep = context.representation
        pc = rep.compact(rows) if hasattr(rep, "compact") else None  # raises propagate → degrade
        if pc is None:
            return None
        return FoldResult(content=pc.content, rendered_text=pc.rendered_text, folded_to_send=max(folded))


def _two_send_script() -> _Scripted:
    # send 1: echo tool then "done1"; send 2: "done2"
    return _Scripted(responses=[
        _tool_resp("c1", "echo", '{"text": "a"}'),
        LLMResponse(raw_text="done1"),
        LLMResponse(raw_text="done2"),
    ])


async def _projector_loop(
    runtime_agent, store: SessionStore, llm: _Scripted, projector, *, max_tokens: int = 8000,
    fold_strategy=None, target=None,
):
    # 3.0 API: representation = the (projection) projector; fold = a deterministic test fold seeded
    # with the projector's keep/trigger so fold mechanics match the old deterministic behavior.
    fs = fold_strategy or _DeterministicFold(
        keep_last_sends=int(getattr(projector, "keep_last_sends", 4)),
        trigger_ratio=float(getattr(projector, "trigger_ratio", 0.75) or 0.75),
    )
    return await runtime_agent(target=target,
        model=llm, store=store, registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=4, representation=projector, fold_strategy=fs,
            max_tokens=max_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_projection_written_and_fed_to_next_send(store: SessionStore, runtime_agent) -> None:
    llm = _two_send_script()
    loop = await _projector_loop(runtime_agent, store, llm, DefaultDeterministicProjector())
    sid = loop.target.session_id

    await loop.send("first")
    # End-of-send 1 wrote user+project projection rows…
    proj = await store.load_project_messages(sid)
    assert {(r.send_index, r.kind) for r in proj} == {(1, "user"), (1, "project")}
    # …and pl_messages is untouched (the immutable audit still holds send 1's structured rows).
    assert any(m.tool_calls for m in await store.load_active_messages(sid))

    n = len(llm.calls)
    await loop.send("second")
    req = llm.calls[n]  # send 2's first LLM request (history + runtime)
    # The historical prefix is PLAIN TEXT — no tool-call protocol leaks into send 2's context.
    assert all("tool_calls" not in m and "tool_call_id" not in m for m in req)
    joined = "\n".join(str(m.get("content", "")) for m in req)
    assert "first" in joined and "done1" in joined   # send 1 rendered (human + final_text)
    assert "second" in joined                          # send 2's own user (in-flight, verbatim)


# ── H1 (BUG_REVIEW_3.4): out-of-band tool rows must carry the in-flight send_index ──
# Recovery and abort_pending append tool rows out-of-band of the
# pipeline. In projection mode the reader partitions active rows by send_index; a NULL-send_index
# tool row falls into the legacy prefix and renders BEFORE its own assistant tool_call, so
# align_tool_calls drops it as an orphan — silently losing the answer. These three tests assert the
# rows are stamped with the active send's index (so they stay paired in current_rows) AND that the
# answer actually reaches the resumed LLM request.


async def _projection_loop_with_tools(runtime_agent, store, llm, tool_registry, *, max_rounds=3):
    return await runtime_agent(
        model=llm, store=store, registry=tool_registry,
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=max_rounds,
            representation=DefaultDeterministicProjector(), max_tokens=8000,
        ),
    )


@pytest.mark.asyncio
async def test_unknown_pending_tool_is_not_replayed_and_keeps_its_send_index(store, runtime_agent):
    calls = []
    registry = _echo_registry()
    registered = registry.get('echo')
    registry.register(registered.definition, lambda **kw: calls.append(kw) or 'unexpected', overwrite=True)
    llm = _Scripted([LLMResponse(raw_text='continued')])
    loop = await _projection_loop_with_tools(runtime_agent, store, llm, registry)
    sid = loop.target.session_id
    await store._db.execute(f'UPDATE {store.t.sessions} SET send_index=1 WHERE session_id=?', (sid,))
    await store.append_message(sid, role='user', content='old input', send_index=1)
    call = {'id':'unknown-tool','type':'function','function':{'name':'echo','arguments':'{"text":"original"}'}}
    seq = await store.append_message(sid, role='assistant', send_index=1, round_index=0, tool_calls=[call])
    await store.set_pending(sid, {'assistant_seq':seq})
    result = await loop.send('continue with new work')
    assert result.status == 'completed' and not calls
    rows = [r for r in await store.load_all_messages(sid) if r.tool_call_id == 'unknown-tool']
    assert len(rows) == 1 and rows[0].send_index == 1 and '结果未知' in rows[0].content
    assert 'continue with new work' in str(llm.calls[-1])


@pytest.mark.asyncio
async def test_default_mode_writes_projections(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="hi")])
    loop = await runtime_agent(
        model=llm, store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )
    sid = loop.target.session_id
    await loop.send("hi")
    assert {(r.send_index, r.kind) for r in await store.load_project_messages(sid)} == {
        (1, "user"), (1, "project")}


@pytest.mark.asyncio
async def test_child_session_projected_without_tool_context(store: SessionStore, runtime_agent) -> None:
    from power_loop.tools.spawn_agent import register_spawn_agent
    registry = _echo_registry()
    register_spawn_agent(registry)
    llm = _Scripted([_tool_resp('spawn','spawn_agent','{"task":"go","system_prompt":"CHILD"}'),
                     LLMResponse(raw_text='parent done'), LLMResponse(raw_text='child done')])
    loop = await _projection_loop_with_tools(runtime_agent, store, llm, registry)
    await loop.send('delegate')
    rows = await store.load_all_messages(loop.target.session_id)
    created = json.loads(next(row.content for row in rows if row.tool_call_id == 'spawn'))
    result = await loop.runtime.client.wait(InputRef(**created['input_ref']), actor=loop.actor, deadline_ms=now_ms()+10000)
    assert result.status == 'completed'
    child = created['session_id']
    assert {(r.send_index, r.kind) for r in await store.load_project_messages(child)} == {(1, 'user'), (1, 'project')}
    await loop.drain()


@pytest.mark.asyncio
async def test_projection_compaction_keeps_last_n(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 6)])
    # tiny max_tokens so the token-driven fold trigger fires on these short sends
    loop = await _projector_loop(runtime_agent, store, llm, DefaultDeterministicProjector(keep_last_sends=2), max_tokens=10)
    sid = loop.target.session_id
    for i in range(1, 4):  # 3 sends → send 1 folded (keep last 2)
        await loop.send(f"m{i}")
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 1
    assert "d1" in compact.content["summary"] and "m1" in compact.content["summary"]
    # folded user/project rows REMAIN (append-only; recoverable)
    rows = await store.load_project_messages(sid)
    assert any(r.send_index == 1 and r.kind == "project" for r in rows)
    # the next send's reader window = compact + sends 2,3
    after = await store.load_project_messages(sid, after_send_index=compact.compact_to_send)
    assert sorted({r.send_index for r in after if r.kind == "project"}) == [2, 3]


@pytest.mark.asyncio
async def test_projection_compaction_rolls_prior_forward(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 7)])
    loop = await _projector_loop(runtime_agent, store, llm, DefaultDeterministicProjector(keep_last_sends=2), max_tokens=10)
    sid = loop.target.session_id
    for i in range(1, 5):  # 4 sends → compact now spans 1-2, nothing lost
        await loop.send(f"m{i}")
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 2
    assert "d1" in compact.content["summary"] and "d2" in compact.content["summary"]


class _CountingMem:
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [{"content": "a persistent note"}]

    async def remember(self, *, snapshot, session_id):
        return None


async def _mem_projector_loop(runtime_agent, store, *, memory, memory_budget_tokens, max_tokens):
    return await runtime_agent(
        model=_Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 6)]),
        store=store, registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=4,
            representation=DefaultDeterministicProjector(keep_last_sends=2),
            fold_strategy=_DeterministicFold(keep_last_sends=2, trigger_ratio=0.75),
            max_tokens=max_tokens, memory=memory, memory_budget_tokens=memory_budget_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_memory_headroom_lowers_projection_fold_trigger(store: SessionStore, runtime_agent) -> None:
    """P0: the projection fold trigger uses config.effective_context_budget(), so
    configuring memory (which reserves memory_budget_tokens) LOWERS the threshold
    and makes folding fire earlier on identical sends. Guards stateful_loop.py
    against a regression that reverts the trigger to raw max_tokens."""
    # No memory: threshold = 200 * 0.75 = 150 → 3 tiny sends stay under → NO fold.
    loop_a = await _mem_projector_loop(runtime_agent, store, memory=None, memory_budget_tokens=0, max_tokens=200)
    sid_a = loop_a.target.session_id
    for i in range(1, 4):
        await loop_a.send(f"m{i}")
    assert await store.latest_project_compact(sid_a) is None  # no headroom reserved → no fold

    # Memory + large budget: effective = max(1, 200-190) = 10 → threshold ≈ 7 →
    # the SAME 3 sends now exceed it → fold fires. (memory is injected at the LLM
    # tail, not the projection snapshot, so only the THRESHOLD changes.)
    store_b = await SessionStore.open(":memory:")
    try:
        loop_b = await _mem_projector_loop(runtime_agent, store_b, memory=_CountingMem(), memory_budget_tokens=190, max_tokens=200)
        sid_b = loop_b.target.session_id
        for i in range(1, 4):
            await loop_b.send(f"m{i}")
        assert await store_b.latest_project_compact(sid_b) is not None  # headroom → folds earlier
    finally:
        await loop_b.aclose()
        await store_b.close()


@pytest.mark.asyncio
async def test_rebuilt_worker_projects_same_session_consistently(store: SessionStore, runtime_agent) -> None:
    # The locked write path keeps ONE coherent projection when two loops share a store on the same
    # session: every send projected exactly once, no duplicate (send_index,kind) rows, one compact
    # (regression for #5/#8 — sequential here so it stays deterministic).
    p = DefaultDeterministicProjector(keep_last_sends=2)
    llm_a = _Scripted(responses=[LLMResponse(raw_text=f"a{i}") for i in range(1, 4)])
    llm_b = _Scripted(responses=[LLMResponse(raw_text=f"b{i}") for i in range(1, 4)])
    loop_a = await _projector_loop(runtime_agent, store, llm_a, p, max_tokens=10)
    sid = loop_a.target.session_id
    await loop_a.send('m1')
    await loop_a.aclose()
    loop_b = await _projector_loop(runtime_agent, store, llm_b, p, max_tokens=10, target=loop_a.target)
    for i in range(2, 5):
        await loop_b.send(f'm{i}', message_id=f'replacement-{i}')

    rows = await store.load_project_messages(sid)
    assert sorted(r.send_index for r in rows if r.kind == "project") == [1, 2, 3, 4]  # once each
    keys = [(r.send_index, r.kind) for r in rows]
    assert len(keys) == len(set(keys))  # no duplicate/clobbered (send_index, kind) rows
    assert await store.latest_project_compact(sid) is not None  # folded into one coherent compact


def test_config_explicit_representation_and_fold_axes():
    cfg = AgentLoopConfig(representation=DefaultDeterministicProjector(), context_budget_tokens=10**9)
    assert cfg.representation is not None
    assert cfg.fold_strategy is not None


def test_config_max_tokens_revalidates_on_reassignment() -> None:
    # max_tokens drives the fold trigger, so a non-positive reassignment is rejected + rolled back.
    cfg = AgentLoopConfig()
    assert cfg.representation is not None and cfg.fold_strategy is not None
    with pytest.raises(ValueError):
        cfg.max_tokens = 0
    assert cfg.max_tokens != 0  # rolled back to the prior valid value
    cfg.max_tokens = 4242
    assert cfg.max_tokens == 4242




@pytest.mark.asyncio
async def test_projection_compact_larger_fold_span(store: SessionStore, runtime_agent) -> None:
    # Over many sends the compact rolls prior folds forward and never loses content: every send is
    # either inside the compact range or still has a live project row.
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 11)])
    loop = await _projector_loop(runtime_agent, store, llm, DefaultDeterministicProjector(keep_last_sends=2), max_tokens=10)
    sid = loop.target.session_id
    for i in range(1, 8):  # 7 sends
        await loop.send(f"m{i}")
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.compact_from_send == 1
    assert "d1" in compact.content["summary"] and "d2" in compact.content["summary"]  # prior preserved
    live = {r.send_index for r in await store.load_project_messages(sid)
            if r.kind == "project" and r.send_index > compact.compact_to_send}
    covered = set(range(1, compact.compact_to_send + 1)) | live
    assert covered.issuperset(set(range(1, 8)))  # nothing lost


@dataclass
class _StubMemory:
    text: str

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [{"content": self.text}]

    async def remember(self, *, snapshot, session_id):
        return None


@pytest.mark.asyncio
async def test_projection_composes_with_memory_recall(store: SessionStore, runtime_agent) -> None:
    llm = _two_send_script()
    loop = await runtime_agent(
        model=llm, store=store, registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=4, context_budget_tokens=10**9,
            representation=DefaultDeterministicProjector(),
            memory=_StubMemory(text="MEMORY: user prefers terse replies"),
        ),
    )
    await loop.send("first")
    n = len(llm.calls)
    r = await loop.send("second")
    assert r.status == "completed"  # no crash: recall front-insert + projected history coexist
    joined = "\n".join(str(m.get("content", "")) for m in llm.calls[n])
    assert "MEMORY: user prefers terse replies" in joined   # recalled memory present
    assert "done1" in joined and "second" in joined          # projected past + current send


# ── reliability: lazy projection repair + fold-failure degradation ──


@pytest.mark.asyncio
async def test_missing_projection_row_is_repaired_lazily(store: SessionStore, runtime_agent) -> None:
    # A best-effort end-of-send projection write can fail/crash, leaving a past send with rows in
    # pl_messages but none in pl_project_messages. The reader rebuilds that send's projection
    # from the audit log and feeds the rebuilt projection, never the raw protocol rows.
    llm = _two_send_script()  # send 1 has a tool call → the audit log carries tool_calls
    # migrate=False so we exercise the per-send VERBATIM FALLBACK path specifically (with migration
    # on, an empty projection table would instead trigger the one-time migration).
    loop = await _projector_loop(runtime_agent, store, llm, DefaultDeterministicProjector())
    sid = loop.target.session_id
    await loop.send("first")
    assert await store.load_project_messages(sid)  # send 1 was projected…
    # …simulate the projection write having failed for send 1: drop its derived rows.
    await store._db.execute(
        f"DELETE FROM {store.t.project_messages} WHERE session_id=? AND send_index=?", (sid, 1)
    )
    assert not await store.load_project_messages(sid)

    n = len(llm.calls)
    await loop.send("second")
    req = llm.calls[n]
    joined = "\n".join(str(m.get("content", "")) for m in req)
    assert "first" in joined and "done1" in joined  # send 1 recovered (NOT dropped)
    # 6.9.0：不再逐字回退——缺投影行的 send 在装配时现场投影并落库（确定性、无 LLM），
    # 请求里是蒸馏文本而不是原始 tool 协议行；投影行也已重新出现在表里。
    assert not any("tool_calls" in m for m in req)
    assert [r for r in await store.load_project_messages(sid) if r.send_index == 1]


@pytest.mark.asyncio
async def test_stale_projector_version_is_rebuilt_with_the_current_projection(store: SessionStore, runtime_agent) -> None:
    # Bumping the projector version between runs must not silently mis-render rows written by the
    # old version: a past send whose projection rows carry a different version is rebuilt.
    loop1 = await _projector_loop(runtime_agent, store, _two_send_script(), DefaultDeterministicProjector(version=1))
    sid = loop1.target.session_id
    await loop1.send("first")
    assert {r.projector_version for r in await store.load_project_messages(sid)} == {1}

    # A new run with a version-2 projector on the SAME session.
    llm2 = _Scripted(responses=[LLMResponse(raw_text="done2")])
    await loop1.aclose()
    loop2 = await _projector_loop(runtime_agent, store, llm2, DefaultDeterministicProjector(version=2), target=loop1.target)
    await loop2.send("second", message_id="version2")
    req = llm2.calls[0]
    joined = "\n".join(str(m.get("content", "")) for m in req)
    assert "first" in joined and "done1" in joined  # send 1 still present…
    # 6.9.0：版本不匹配 → 用当前版本投影器现场重投影（不再逐字回退），表里出现 v2 行。
    assert not any("tool_calls" in m for m in req)
    assert 2 in {r.projector_version for r in await store.load_project_messages(sid) if r.send_index == 1}


class _BoomCompactProjector(DefaultDeterministicProjector):
    def compact(self, rows):  # type: ignore[override]
        raise RuntimeError("boom in compact")


@pytest.mark.asyncio
async def test_compact_failure_degrades_to_no_fold(store: SessionStore, runtime_agent) -> None:
    # A projector whose compact() raises must NOT abort the whole locked write: the per-send rows
    # still commit (so nothing is lost), only the fold is skipped.
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 4)])
    # keep_last_sends=1 + max_tokens=1 (threshold 0) → the fold triggers every send → compact() raises.
    loop = await _projector_loop(runtime_agent, store, llm, _BoomCompactProjector(keep_last_sends=1), max_tokens=1)
    sid = loop.target.session_id
    statuses = [(await loop.send(f"m{i}")).status for i in range(1, 4)]

    assert statuses == ["completed", "completed", "completed"]  # no crash despite compact() raising
    assert await store.latest_project_compact(sid) is None  # fold skipped (degraded)
    proj_sends = {r.send_index for r in await store.load_project_messages(sid) if r.kind == "project"}
    assert proj_sends == {1, 2, 3}  # every send's rows committed (not lost with the failed fold)














@pytest.mark.asyncio
async def test_default_config_uses_the_projection_layer(store: SessionStore, runtime_agent) -> None:
    # The library default is the projection representation: a second send on a default-configured
    # loop must read send 1's stored projection, not raw pl_messages rows.
    first = await runtime_agent(
        model=_Scripted([LLMResponse(raw_text="done1")]), store=store, registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
    )
    sid = first.target.session_id
    await first.send("first")
    assert await store.load_project_messages(sid)
    await first.aclose()

    llm2 = _Scripted([LLMResponse(raw_text="done2")])
    second = await runtime_agent(target=first.target, model=llm2, store=store,
                                 registry=_echo_registry(),
                                 config=AgentLoopConfig(system_prompt="S", max_rounds=4))
    r = await second.send("second", message_id="second")
    assert r.status == "completed"
    joined = "\n".join(str(m.get("content", "")) for m in llm2.calls[0])
    assert "first" in joined and "done1" in joined
    assert not any("tool_calls" in m for m in llm2.calls[0])
    await second.aclose()


@pytest.mark.asyncio
async def test_orphan_tool_row_sanitized_from_prompt(store: SessionStore, runtime_agent) -> None:
    # A corrupt orphan tool-result row in pl_messages must NOT reach the provider (it would 400 and
    # brick the session forever). The always-on backstop drops it from the assembled prompt; with
    # repair off (default) the audit row itself is left intact.
    llm = _Scripted(responses=[LLMResponse(raw_text="ok")])
    loop = await runtime_agent(
        model=llm, store=store, registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=2),  # default mode
    )
    sid = loop.target.session_id
    await store.append_message(sid, role="tool", tool_call_id="ghost", content="orphan result")
    r = await loop.send("hi")
    assert r.status == "completed"  # not bricked
    assert not any(m.get("tool_call_id") == "ghost" for m in llm.calls[0])  # orphan never sent
    assert any(m.tool_call_id == "ghost" for m in await store.load_all_messages(sid))  # audit intact


@pytest.mark.asyncio
async def test_interrupted_send_is_lazily_projected_on_next_send(store: SessionStore, runtime_agent) -> None:
    """6.9.0 self-heal（真实事故）：一个被进程重启杀在半途的 send 永远走不到 end-of-send 投影，
    它的原始行（含 tool 协议行）以前会逐字进入之后每个 send 的上下文（一个 80 轮 send = 182K 字符
    ≈ 115K tokens，每轮 2–3 分钟）。现在下一个 send 装配上下文时现场投影并落库。"""
    big = "BIGRAWTOOLRESULT-" * 300
    llm = _Scripted(responses=[LLMResponse(raw_text="after heal")])
    loop = await _projection_loop_with_tools(runtime_agent, store, llm, _echo_registry())
    sid = loop.target.session_id
    # 模拟中断的 send 1：只有原始行，没有投影行
    await store._db.execute(f'UPDATE {store.t.sessions} SET send_index=1 WHERE session_id=?', (sid,))
    await store.append_message(sid, role="user", content="kick off", send_index=1)
    await store.append_message(
        sid, role="assistant", send_index=1, round_index=0,
        tool_calls=[{"id": "tc-h", "function": {"name": "echo", "arguments": '{"text":"x"}'}}],
    )
    await store.append_message(sid, role="tool", content=big, send_index=1,
                               tool_call_id="tc-h", round_index=0)
    assert not [r for r in await store.load_project_messages(sid) if r.send_index == 1]

    await loop.send("next")  # send 2

    healed = [r for r in await store.load_project_messages(sid) if r.send_index == 1]
    assert {r.kind for r in healed} >= {"user", "project"}, "中断的 send 1 应被现场投影并落库"
    req = llm.calls[0]
    roles = [m.get("role") for m in req]
    assert "tool" not in roles, "send 1 的原始 tool 协议行不该再逐字进入 send 2 的上下文"
    joined = "\n".join(str(m.get("content") or "") for m in req)
    assert big not in joined, "182K 式原文不该原样带入——应是蒸馏后的投影"
