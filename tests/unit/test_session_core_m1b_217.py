"""Design 217 M1b focused verification suite across SQLite, PostgreSQL 16, and MySQL 8.4.

Covers every M1b engineering item and adversarial audit probe:
  - Double-row ('user' + 'project') single multi-value INSERT & Turn 2 first non-system role='user' (Probe 10)
  - Inline sessions.fold_stats_json v3: 12-hex SHA256 dedup, 512 cap, 513 oldest eviction, <10 KB,
    compact clears dedup, 0 session_runtime_state SQL (M0 Decision 2)
  - send_index interval pagination: load_projections never splits ('user','project'),
    load_projections_paged every SQL has LIMIT and reads full uncapped history (DISCOVERY 2026-11-19),
    'repaired' id-based arbitration (mid-turn checkpoint vs post-turn repair)
  - seed_context: empty session send_index=0 visible on Turn 1, unseeded cold Turn 1 pays 0 extra reads,
    double seed on empty session & seed on non-empty session increment send_index without colliding or
    overwriting old turns, and all 3 overflow modes ('truncate_oldest', 'compact', 'reject') (Probe 5)
  - Three-line format self-healing:
      Line 1: S09-R5 (_assistant_seq), S09-R6 (MAX(round_index)+1, reclaimed_pending_tool_seq, MAX(seq)+1),
              and M1c HookPoint.TURN_REPAIR seam (on_repair_interrupted_turn)
      Line 2: 0-SQL in-memory pre-LLM sanitizer
      Line 3 (Defense 3): HTTP 400 format error folds historical tool rounds by original send_index
              and persists per-send_index 'repaired' projections before single retry (Probe 10)
  - Secondary takeover on same send_index: same-tx DELETE + INSERT 'repaired' never collides (Probe 7)
  - Fence fixes:
      Probe 4A: stop() on child notifies root with sys_stop_{sid}_c{new_cancel_seq} (wake=0)
      Probe 4B: release_claim never resurrects closing/closed, preserves suspended inputs, propagates cancel_by
      Probe 4C: consume_pending_inbox permits suspended, voids cancel_by!=NULL rows & decrements counts
  - Public primitives: repair(mode='heal_broken'|'fold_tools_to_text'|'compact', transform=...)
    and context_budget(descriptor | target)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    ContextBudget,
    LLMResponse,
    RepairResult,
    ToolDefinition,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import InputRef, RuntimeRequestError
from power_loop.contracts.errors import SessionBusy
from power_loop.runtime.history_sanitize import (
    find_dangling_tool_calls,
    fold_tool_rounds_by_send_index,
    sanitize_request_messages,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import resolve_send_projections
from power_loop.runtime.store.projection_fold import (
    FOLD_DEDUP_MAX_KEYS,
    fold_dedup_hash,
    parse_state,
    update_fold_stats,
)
from power_loop.runtime.store.types import ProjectMessageRow
from power_loop.tools.registry import ToolRegistry
from tests.unit.session_native_fixture import ACTOR, DESCRIPTOR
from tests.unit.test_send_sql_profile import Profiled, RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _RecordingModel:
    """Minimal scripted LLM that records every message list sent to it."""

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.calls = 0

    async def complete(self, request, **kwargs):
        idx = self.calls
        self.calls += 1
        msgs = request.messages if hasattr(request, "messages") else request
        self.seen_messages.append([dict(m) for m in msgs])
        resp = self.responses[min(idx, len(self.responses) - 1)]
        if isinstance(resp, Exception):
            raise resp
        if callable(resp):
            return resp(idx, msgs)
        return resp

    async def close(self):
        return None


def _tool_registry():
    reg = ToolRegistry()

    async def echo(text: str = "") -> str:
        return f"echo:{text}"

    reg.register(
        ToolDefinition(
            "echo",
            "Echo text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            pricing=Free("fixture"),
        ),
        echo,
    )
    return reg


# ── 1. Double-Row Projection & Turn 2 First Non-System Role = 'user' (Probe 10) ──


async def test_m1b_double_row_single_insert_and_turn2_first_role_user(registry_store):
    store, _ = registry_store
    model = _RecordingModel([
        LLMResponse(raw_text="answer_1"),
        LLMResponse(raw_text="answer_2"),
    ])
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    agent = await RunningAgent(
        store, model, ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        profiled.records.clear()
        r1 = await agent.send("hello turn 1")
        assert r1.status == "completed" and r1.final_text == "answer_1"

        # Verify 1 single multi-value INSERT INTO project_messages with ('user', 'project')
        proj_inserts = [
            rec for rec in profiled.records
            if f"INSERT INTO {store.t.project_messages}" in rec["sql"]
        ]
        assert len(proj_inserts) == 1, proj_inserts
        # Two value tuples in a single INSERT
        assert proj_inserts[0]["sql"].count("NULL,NULL") == 2, proj_inserts[0]["sql"]
        assert all(store.t.session_runtime_state not in rec["sql"] for rec in profiled.records)

        prows = await store.load_project_messages(agent.target.session_id)
        assert [(r.send_index, r.kind) for r in prows] == [(1, "user"), (1, "project")]

        # Turn 2: first non-system message sent to LLM MUST be role='user'
        r2 = await agent.send("hello turn 2")
        assert r2.status == "completed" and r2.final_text == "answer_2"
        turn2_non_sys = [m for m in model.seen_messages[1] if m.get("role") != "system"]
        assert turn2_non_sys[0]["role"] == "user", turn2_non_sys
        assert any("hello turn 1" in str(m.get("content") or "") for m in turn2_non_sys[:1])
    finally:
        await agent.aclose()
        store._db = original


# ── 2. Inline fold_stats_json: 12-hex, 512 Cap, 513 Eviction, <10 KB, Compact Clear ──


async def test_m1b_fold_stats_json_512_cap_eviction_and_compact_clear(registry_store):
    store, _ = registry_store
    # Pure cap/eviction check (512 cap, 513th evicts oldest send_index, 12-hex keys, <10 KB)
    state: dict[str, Any] | None = None
    for si in range(1, FOLD_DEDUP_MAX_KEYS + 2):  # 1..513
        state = update_fold_stats(
            state,
            send_index=si,
            dedup_keys=[f"file:/repo/mod_{si}.py"],
            token_delta=10,
        )
    assert state is not None and state["v"] == 3
    assert len(state["dedup"]) == FOLD_DEDUP_MAX_KEYS == 512
    assert all(len(k) == 12 for k in state["dedup"])
    assert fold_dedup_hash("file:/repo/mod_1.py") not in state["dedup"]  # oldest evicted!
    assert fold_dedup_hash("file:/repo/mod_2.py") in state["dedup"]
    assert fold_dedup_hash("file:/repo/mod_513.py") in state["dedup"]
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    assert len(encoded.encode("utf-8")) < 10 * 1024  # < 10 KB

    # Compact clears dedup
    compacted = update_fold_stats(state, send_index=513, token_delta=50, compact_reset=True)
    assert compacted["dedup"] == {}
    assert compacted["compact_to_send"] == 513

    # End-to-end DB check: 0 SQL on session_runtime_state after repair(mode='compact')
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    sid = target.session_id
    await store.upsert_project_message(sid, send_index=1, kind="user", content={"input": ["u1"]}, rendered_text="u1")
    await store.upsert_project_message(sid, send_index=1, kind="project", content={"final_text": "a1"}, rendered_text="a1")
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET send_index=1,fold_stats_json=? WHERE session_id=?",
        (encoded, sid),
    )
    res = await port.repair(target, mode="compact", actor=ACTOR)
    assert isinstance(res, RepairResult) and res.send_index == 1 and res.repaired_projections == 1
    srow = await store._db.fetchone(
        f"SELECT fold_stats_json FROM {store.t.sessions} WHERE session_id=?", (sid,),
    )
    parsed = parse_state(srow["fold_stats_json"])
    assert parsed is not None and parsed["dedup"] == {} and parsed["compact_to_send"] == 1
    kv = await store._db.fetchall(
        f"SELECT 1 FROM {store.t.session_runtime_state} WHERE session_id=?", (sid,),
    )
    assert kv == []


# ── 3. send_index Interval Pagination + Uncapped Paged Read + 'repaired' Arbitration ──


async def test_m1b_interval_pagination_paged_limit_and_repaired_arbitration(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    sid = target.session_id

    # Seed 15 sends (30 rows: user + project each)
    for si in range(1, 16):
        await store.upsert_project_message(
            sid, send_index=si, kind="user", content={"input": [f"q{si}"]}, rendered_text=f"q{si}",
        )
        await store.upsert_project_message(
            sid, send_index=si, kind="project", content={"final_text": f"a{si}"}, rendered_text=f"a{si}",
        )

    # 3A: load_projections with send_limit=5 returns exact complete 5 sends (10 rows), never splitting user/project
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        profiled.records.clear()
        page_rows = await store.load_projections(sid, before_send_index=16, send_limit=5)
        assert {r.send_index for r in page_rows} == {11, 12, 13, 14, 15}
        assert len(page_rows) == 10
        assert all("LIMIT" in rec["sql"].upper() for rec in profiled.records)

        # 3B: load_projections_paged reads ALL 15 sends with page_size=4, every SQL has LIMIT (DISCOVERY 2026-11-19)
        profiled.records.clear()
        all_rows = await store.load_projections_paged(sid, before_send_index=16, page_size=4)
        assert {r.send_index for r in all_rows} == set(range(1, 16))
        assert len(all_rows) == 30
        assert len(profiled.records) >= 4
        assert all("LIMIT" in rec["sql"].upper() for rec in profiled.records)
    finally:
        store._db = original

    # 3C: 'repaired' id-based arbitration
    # Case 1: mid-turn checkpoint (repaired.id < user.id/project.id) MUST NOT shadow final user+project
    mid_rep = ProjectMessageRow(sid, 1, "repaired", {"messages": [{"role": "assistant", "content": "mid"}]}, "mid", None, None, None, None, 0, None, 0, id=10)
    fin_u = ProjectMessageRow(sid, 1, "user", {"input": ["q1"]}, "q1", None, None, None, None, 1, None, 0, id=11)
    fin_p = ProjectMessageRow(sid, 1, "project", {"final_text": "final"}, "final", None, None, None, None, 1, None, 0, id=12)
    _, resolved1 = resolve_send_projections([mid_rep, fin_u, fin_p])
    assert [r.kind for r in resolved1[1]] == ["user", "project"]

    # Case 2: post-turn repair (repaired.id > max(user.id, project.id)) supersedes project/turn
    post_rep = ProjectMessageRow(
        sid, 1, "repaired",
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "fixed"}]},
        "fixed", None, None, None, None, 0, None, 0, id=20,
    )
    _, resolved2 = resolve_send_projections([fin_u, fin_p, post_rep])
    assert [r.kind for r in resolved2[1]] == ["repaired"]


# ── 4. Probe 5: seed_context (Turn 1 Visible, Cold 0-Read, Double/Non-Empty, 3 Overflow) ──


async def test_m1b_probe5_seed_context_visibility_increment_and_overflow(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)

    # 4A: Empty session first seed -> send_index=0, kind='seed', visible on Turn 1
    model = _RecordingModel([LLMResponse(raw_text="turn1_ack"), LLMResponse(raw_text="turn2_ack")])
    agent = await RunningAgent(
        store, model, ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        r_seed = await agent.runtime.client.seed_context(
            agent.target,
            [
                {"role": "user", "content": "seeded_q"},
                {"role": "assistant", "content": "seeded_a"},
            ],
            actor=agent.actor,
        )
        assert r_seed["send_index"] == 0 and r_seed["seeded_count"] == 2
        res1 = await agent.send("live_turn_1")
        assert res1.status == "completed"
        t1_contents = " ".join(str(m.get("content") or "") for m in model.seen_messages[0])
        assert "seeded_q" in t1_contents and "seeded_a" in t1_contents and "live_turn_1" in t1_contents

        # 4B: Non-empty session seed increments send_index (no collision, preserves Turn 1)
        r_seed2 = await agent.runtime.client.seed_context(
            agent.target,
            [{"role": "user", "content": "mid_session_seed"}],
            actor=agent.actor,
        )
        assert r_seed2["send_index"] == 2  # Turn 1 was send_index=1; seed takes send_index=2
        prows = await store.load_project_messages(agent.target.session_id)
        kinds_by_si = {(r.send_index, r.kind) for r in prows}
        assert (0, "seed") in kinds_by_si
        assert (1, "user") in kinds_by_si and (1, "project") in kinds_by_si
        assert (2, "seed") in kinds_by_si
    finally:
        await agent.aclose()

    # 4C: Two consecutive seeds on an empty session -> send_index=0 then send_index=1 (no unique key error!)
    t_empty = await port.create_session(DESCRIPTOR, actor=ACTOR)
    s0 = await port.seed_context(t_empty, [{"role": "user", "content": "first seed"}], actor=ACTOR)
    s1 = await port.seed_context(t_empty, [{"role": "user", "content": "second seed"}], actor=ACTOR)
    assert s0["send_index"] == 0 and s1["send_index"] == 1

    # 4D: Unseeded cold Turn 1 pays 0 project_messages reads!
    t_cold = await RunningAgent(
        store, _RecordingModel([LLMResponse(raw_text="ok")]), ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=2, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        profiled.records.clear()
        await t_cold.send("cold hello")
        proj_reads = [
            rec for rec in profiled.records
            if store.t.project_messages in rec["sql"] and rec["sql"].lstrip().upper().startswith("SELECT")
        ]
        assert proj_reads == [], proj_reads
    finally:
        store._db = original
        await t_cold.aclose()

    # 4E: Three overflow modes ('reject', 'truncate_oldest', 'compact')
    long_msgs = [
        {"role": "user", "content": "OLD_MSG " + "x" * 400},
        {"role": "assistant", "content": "MID_MSG " + "y" * 400},
        {"role": "user", "content": "NEW_MSG " + "z" * 40},
    ]
    t_ov = await port.create_session(DESCRIPTOR, actor=ACTOR)
    with pytest.raises(RuntimeRequestError) as exc_info:
        await port.seed_context(t_ov, long_msgs, actor=ACTOR, overflow="reject", max_tokens=50)
    assert exc_info.value.code == "context_budget_exceeded"

    r_trunc = await port.seed_context(t_ov, long_msgs, actor=ACTOR, overflow="truncate_oldest", max_tokens=60)
    assert r_trunc["truncated"] is True and r_trunc["seeded_count"] < len(long_msgs)

    r_comp = await port.seed_context(
        t_ov, long_msgs, actor=ACTOR, overflow="compact", max_tokens=60,
        summary_fn=lambda _ms: "CUSTOM_COMPACT_SUMMARY",
    )
    assert r_comp["truncated"] is True and r_comp["seeded_count"] == 1


# ── 5. S09-R5/R6 + Probe 7 (2nd Takeover DELETE+INSERT) + M1c TURN_REPAIR Seam ──


async def test_m1b_takeover_s09_r5_r6_probe7_and_turn_repair_seam(registry_store):
    store, _ = registry_store
    hook_calls: list[dict[str, Any]] = []

    async def on_repair(session_id: str, dangling: list[dict[str, Any]]):
        hook_calls.append({"session_id": session_id, "dangling": dangling})
        return {d["id"]: f"<hook_repaired:{d['id']}>" for d in dangling}

    port = RuntimeInputStore(store, on_repair_interrupted_turn=on_repair)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    sid = target.session_id
    _, batch, _ = await port.accept_and_claim(
        target, "run tools", message_id="m1", actor=ACTOR, node_id="n1", boot_id="b1", ttl_s=1.0,
    )
    assert batch is not None and batch.send_index == 1

    # Simulate crash after writing assistant(round_index=1, tool_calls=[tc_1]) and setting pending_tool_seq
    asst_seq = batch.message_seq + 1
    await store._db.execute(
        f"INSERT INTO {store.t.messages} "
        "(session_id,seq,role,content,tool_calls_json,round_index,state,meta_json,send_index,created_at) "
        "VALUES (?,?,'assistant','calling',?,1,'active','{}',1,1000)",
        (sid, asst_seq, json.dumps([{"id": "tc_1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}])),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET pending_tool_seq=?,owner_lease_until_ms=1 WHERE session_id=?",
        (asst_seq, sid),
    )

    # First takeover: Defense 1 repairs tc_1 via on_repair_interrupted_turn seam, stamps S09-R5 & S09-R6
    reclaimed1 = await port.reclaim_running(target, node_id="n2", boot_id="b2")
    assert reclaimed1 is not None
    assert reclaimed1.is_reclaimed is True
    assert reclaimed1.reclaimed_has_assistant is True
    assert reclaimed1.reclaimed_next_round_index == 2  # MAX(round_index=1) + 1
    assert reclaimed1.reclaimed_pending_tool_seq == asst_seq
    assert len(hook_calls) == 1 and hook_calls[0]["dangling"][0]["id"] == "tc_1"

    # Verify S09-R5 meta_json._assistant_seq and next_seq = MAX(seq) + 1
    msgs = await store.load_active_messages(sid, send_index=1)
    tool_rows = [m for m in msgs if m.role == "tool" and m.tool_call_id == "tc_1"]
    assert len(tool_rows) == 1
    assert tool_rows[0].meta.get("_assistant_seq") == asst_seq
    assert tool_rows[0].meta.get("repair_source") == "host_recovered"
    assert tool_rows[0].content == "<hook_repaired:tc_1>"
    srow = await store._db.fetchone(f"SELECT next_seq,pending_tool_seq FROM {store.t.sessions} WHERE session_id=?", (sid,))
    assert int(srow["next_seq"]) == max(m.seq for m in msgs) + 1
    assert srow["pending_tool_seq"] is None

    # Probe 7: Crash AGAIN on the SAME send_index=1 with a 2nd dangling tool call (round_index=2)
    asst_seq_2 = int(srow["next_seq"])
    await store._db.execute(
        f"INSERT INTO {store.t.messages} "
        "(session_id,seq,role,content,tool_calls_json,round_index,state,meta_json,send_index,created_at) "
        "VALUES (?,?,'assistant','calling 2',?,2,'active','{}',1,2000)",
        (sid, asst_seq_2, json.dumps([{"id": "tc_2", "type": "function", "function": {"name": "echo", "arguments": "{}"}}])),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET pending_tool_seq=?,owner_lease_until_ms=1 WHERE session_id=?",
        (asst_seq_2, sid),
    )

    # 2nd takeover MUST succeed (same-tx DELETE+INSERT 'repaired' projection -> no unique key collision!)
    reclaimed2 = await port.reclaim_running(target, node_id="n3", boot_id="b3")
    assert reclaimed2 is not None
    assert reclaimed2.reclaimed_next_round_index == 3  # MAX(round_index=2) + 1
    assert reclaimed2.reclaimed_pending_tool_seq == asst_seq_2
    rep_projs = [
        p for p in await store.load_project_messages(sid)
        if p.send_index == 1 and p.kind == "repaired"
    ]
    assert len(rep_projs) == 1


# ── 6. Defense 2 (0-SQL Sanitizer) & Defense 3 (400 Retry Per-Original-send_index) (Probe 10) ──


async def test_m1b_defense2_sanitizer_and_defense3_per_send_index_400_retry(registry_store):
    store, _ = registry_store

    # Defense 2: pure in-memory 0-SQL sanitizer drops orphan tools, synthesizes missing tools, strips internal keys
    dirty = [
        {"role": "tool", "tool_call_id": "orphan_tc", "content": "drop me"},
        {"role": "user", "content": "hi", "_source_send_index": 1},
        {
            "role": "assistant", "content": "", "_source_send_index": 1,
            "tool_calls": [{"id": "missing_tc", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
        },
        {"role": "user", "content": "next", "_source_send_index": 2},
    ]
    clean = sanitize_request_messages(dirty)
    assert [m["role"] for m in clean] == ["user", "assistant", "tool", "user"]
    assert clean[2]["tool_call_id"] == "missing_tc"
    assert all("_source_send_index" not in m for m in clean)
    dangling = find_dangling_tool_calls(dirty)
    assert len(dangling) == 1 and dangling[0]["id"] == "missing_tc"
    assert (dangling[0].get("function") or {}).get("name") == "echo"

    # Defense 3 unit check: per-original-send_index folding never conflates sends into send_index - 1
    hist = [
        {"role": "user", "content": "q1", "_source_send_index": 1},
        {
            "role": "assistant", "content": "call1", "_source_send_index": 1,
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "name": "echo", "content": "r1", "_source_send_index": 1},
        {"role": "user", "content": "q2", "_source_send_index": 2},
        {
            "role": "assistant", "content": "call2", "_source_send_index": 2,
            "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t2", "name": "echo", "content": "r2", "_source_send_index": 2},
        {"role": "user", "content": "q3_live"},
    ]
    folded, per_send, _cur = fold_tool_rounds_by_send_index(
        hist, current_send_index=3, historical_end_idx=6,
    )
    assert set(per_send.keys()) == {1, 2}
    assert all(m.get("role") != "tool" and not m.get("tool_calls") for m in folded)

    # Defense 3 end-to-end: HTTP 400 format error triggers fold + per-send 'repaired' projection + single retry
    class _Format400Error(Exception):
        status_code = 400

        def __init__(self):
            super().__init__("invalid tool_call_id messages format")

    calls_seen: list[list[dict[str, Any]]] = []

    class _Defense3Model:
        def __init__(self):
            self.n = 0

        async def complete(self, request, **kwargs):
            self.n += 1
            msgs = request.messages if hasattr(request, "messages") else request
            calls_seen.append([dict(m) for m in msgs])
            if self.n == 1:
                raise _Format400Error()
            return LLMResponse(raw_text="recovered_after_400")

        async def close(self):
            return None

    agent = await RunningAgent(
        store, _Defense3Model(), _tool_registry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        sid = agent.target.session_id
        # Pre-populate send 1 & send 2 as 'repaired' projections containing structured tool_calls/tool rows
        for si in (1, 2):
            await store.upsert_repaired_projection(
                sid,
                send_index=si,
                content={
                    "messages": [
                        {"role": "user", "content": f"q{si}"},
                        {
                            "role": "assistant", "content": f"c{si}",
                            "tool_calls": [{"id": f"tc_{si}", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
                        },
                        {"role": "tool", "tool_call_id": f"tc_{si}", "name": "echo", "content": f"out_{si}"},
                    ],
                },
                rendered_text=f"turn {si}",
            )
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET send_index=2,next_seq=10 WHERE session_id=?", (sid,),
        )
        res = await agent.send("live turn 3")
        assert res.status == "completed" and res.final_text == "recovered_after_400"
        assert len(calls_seen) == 2
        # Retry payload has NO tool/tool_calls messages in history
        retry_non_sys = [m for m in calls_seen[1] if m.get("role") != "system"]
        assert all(m.get("role") != "tool" and not m.get("tool_calls") for m in retry_non_sys)
        # Both send 1 and send 2 have updated 'repaired' projections attributed to their OWN send_index!
        projs = await store.load_project_messages(sid)
        rep_by_si = {p.send_index: p for p in projs if p.kind == "repaired"}
        assert {1, 2}.issubset(rep_by_si.keys())
        assert rep_by_si[1].content.get("folded_on_400") is True
        assert rep_by_si[1].content.get("orig_send_index") == 1
        assert rep_by_si[2].content.get("folded_on_400") is True
        assert rep_by_si[2].content.get("orig_send_index") == 2
    finally:
        await agent.aclose()


# ── 7. Probe 4A / 4B / 4C: Stop Notice Key, release_claim & consume_pending_inbox Fences ──


async def test_m1b_probe4_stop_notice_release_claim_and_consume_pending_inbox(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)

    # Probe 4A: human single-target stop on child notifies root with sys_stop_{sid}_c{new_cancel_seq} (wake=0)
    root = await port.create_session(DESCRIPTOR, actor=ACTOR)
    child = await port.create_session(DESCRIPTOR, actor=ACTOR, parent=root)
    await port.accept(child, "m1", message_id="p4a_m1", actor=ACTOR)
    await port.cancel(child, actor=ACTOR, scope="self", reason="stop_1")
    await port.accept(child, "m2", message_id="p4a_m2", actor=ACTOR)
    await port.cancel(child, actor=ACTOR, scope="self", reason="stop_2")

    notices = await store._db.fetchall(
        f"SELECT item_id,kind,wake,status FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='system_notice' ORDER BY id ASC",
        (root.session_id,),
    )
    assert [r["item_id"] for r in notices] == [
        f"sys_stop_{child.session_id}_c1",
        f"sys_stop_{child.session_id}_c2",
    ]
    assert all(int(r["wake"]) == 0 for r in notices)
    root_row = await store._db.fetchone(
        f"SELECT execution_state,wake_pending_count,pending_count FROM {store.t.sessions} WHERE session_id=?",
        (root.session_id,),
    )
    assert root_row["execution_state"] == "idle"
    assert int(root_row["wake_pending_count"]) == 0 and int(root_row["pending_count"]) == 2

    # Probe 4B: release_claim fences on closing/closed, suspended, and open (cancel_by propagation)
    t_closed = await port.create_session(DESCRIPTOR, actor=ACTOR)
    _, b_closed, _ = await port.accept_and_claim(
        t_closed, "work", message_id="p4b_cl", actor=ACTOR, node_id="n1", boot_id="b1",
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='closed' WHERE session_id=?", (t_closed.session_id,),
    )
    await port.release_claim(b_closed)
    cl_state = await store._db.fetchone(
        f"SELECT lifecycle,execution_state,pending_count,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
        (t_closed.session_id,),
    )
    assert cl_state["lifecycle"] == "closed" and cl_state["execution_state"] == "idle"
    assert int(cl_state["pending_count"]) == 0 and int(cl_state["wake_pending_count"]) == 0
    cl_q = await store._db.fetchone(
        f"SELECT status,terminal_status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='p4b_cl'",
        (t_closed.session_id,),
    )
    assert cl_q == {"status": "void", "terminal_status": "cancelled"}

    # Suspended preserves pending input & propagates cancel_by, stays idle
    t_susp = await port.create_session(DESCRIPTOR, actor=ACTOR)
    _, b_susp, _ = await port.accept_and_claim(
        t_susp, "work", message_id="p4b_sp", actor=ACTOR, node_id="n1", boot_id="b1",
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='suspended',cancel_by='user' WHERE session_id=?",
        (t_susp.session_id,),
    )
    await port.release_claim(b_susp)
    sp_state = await store._db.fetchone(
        f"SELECT lifecycle,execution_state,pending_count,cancel_by FROM {store.t.sessions} WHERE session_id=?",
        (t_susp.session_id,),
    )
    assert sp_state["lifecycle"] == "suspended" and sp_state["execution_state"] == "idle"
    assert int(sp_state["pending_count"]) == 1 and sp_state["cancel_by"] == "user"
    sp_q = await store._db.fetchone(
        f"SELECT status,cancel_by FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='p4b_sp'",
        (t_susp.session_id,),
    )
    assert sp_q == {"status": "pending", "cancel_by": "user"}

    # Open returning to pending preserves & propagates cancel_by and clears ready_item_json
    t_open = await port.create_session(DESCRIPTOR, actor=ACTOR)
    _, b_open, _ = await port.accept_and_claim(
        t_open, "work", message_id="p4b_op", actor=ACTOR, node_id="n1", boot_id="b1",
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET cancel_by='subtree',ready_item_json='stale' WHERE session_id=?",
        (t_open.session_id,),
    )
    await port.release_claim(b_open)
    op_state = await store._db.fetchone(
        f"SELECT execution_state,cancel_by,ready_item_json FROM {store.t.sessions} WHERE session_id=?",
        (t_open.session_id,),
    )
    assert op_state == {"execution_state": "queued", "cancel_by": "subtree", "ready_item_json": None}
    op_q = await store._db.fetchone(
        f"SELECT status,cancel_by FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='p4b_op'",
        (t_open.session_id,),
    )
    assert op_q == {"status": "pending", "cancel_by": "subtree"}

    # Probe 4C: consume_pending_inbox permits suspended, voids cancel_by!=NULL rows & decrements counts
    t_mid = await port.create_session(DESCRIPTOR, actor=ACTOR)
    _, b_mid, _ = await port.accept_and_claim(
        t_mid, "init", message_id="p4c_init", actor=ACTOR, node_id="n1", boot_id="b1",
    )
    await port.accept(t_mid, "CANCELLED", message_id="p4c_bad", actor=ACTOR)
    await port.accept(t_mid, "VALID", message_id="p4c_good", actor=ACTOR)
    await store._db.execute(
        f"UPDATE {store.t.follow_up_queue} SET cancel_by='user' WHERE session_id=? AND item_id='p4c_bad'",
        (t_mid.session_id,),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='suspended' WHERE session_id=?",
        (t_mid.session_id,),
    )
    claimed = await port.consume_pending_inbox(b_mid)
    assert claimed is not None
    refs, _seq, _merged, _ver = claimed
    assert [r.message_id for r in refs] == ["p4c_good"]
    bad_row = await port.inspect_input(InputRef(store.store_id, t_mid.session_id, "p4c_bad"))
    assert bad_row["status"] == "void" and bad_row["terminal_status"] == "cancelled"
    mid_state = await store._db.fetchone(
        f"SELECT pending_count,wake_pending_count,pending_bytes FROM {store.t.sessions} WHERE session_id=?",
        (t_mid.session_id,),
    )
    assert int(mid_state["pending_count"]) == 0
    assert int(mid_state["wake_pending_count"]) == 0
    assert int(mid_state["pending_bytes"]) == 0


# ── 8. repair() Three Modes + transform & context_budget() ──────────────────


async def test_m1b_repair_three_modes_and_context_budget(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)

    # context_budget on RuntimeDescriptor (0 SQL!)
    from dataclasses import replace as dc_replace

    desc = dc_replace(
        DESCRIPTOR,
        definition_ref="budget.agent@215",
        config={
            "model": "gpt-test",
            "context_budget_tokens": 10_000,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant.",
        },
    )
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        profiled.records.clear()
        b_desc = await port.context_budget(desc)
        assert isinstance(b_desc, ContextBudget)
        assert b_desc.context_budget_tokens == 10_000
        assert b_desc.max_tokens == 1000
        assert b_desc.model == "gpt-test"
        assert b_desc.remaining_tokens == 10_000 - 1000 - b_desc.used_tokens
        assert profiled.records == []  # 0 SQL!
    finally:
        store._db = original

    # context_budget on TargetRef reflects fold_stats_json.tokens (1 SQL)
    target = await port.create_session(desc, actor=ACTOR)
    sid = target.session_id
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET send_index=1,fold_stats_json=? WHERE session_id=?",
        (json.dumps({"v": 3, "sends": 1, "rows": 2, "tokens": 500, "maximum": 1, "dedup": {}}), sid),
    )
    b_target = await port.context_budget(target, actor=ACTOR)
    assert b_target.used_tokens == b_desc.used_tokens + 500
    assert b_target.remaining_tokens == b_desc.remaining_tokens - 500

    # repair(mode='heal_broken') stitches dangling tool call and writes 'repaired' projection
    await store._db.execute(
        f"INSERT INTO {store.t.messages} "
        "(session_id,seq,role,content,tool_calls_json,round_index,state,meta_json,send_index,created_at) "
        "VALUES (?,1,'user','go',NULL,0,'active','{}',1,1000),"
        "(?,2,'assistant','calling',?,0,'active','{}',1,1001)",
        (sid, sid, json.dumps([{"id": "tc_heal", "type": "function", "function": {"name": "echo", "arguments": "{}"}}])),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET send_index=1,next_seq=3,pending_tool_seq=2 WHERE session_id=?",
        (sid,),
    )
    r_heal = await port.repair(target, mode="heal_broken", actor=ACTOR)
    assert r_heal.mode == "heal_broken"
    assert r_heal.repaired_messages == 1
    assert r_heal.repaired_projections == 1
    healed_msgs = await store.load_active_messages(sid)
    assert any(m.role == "tool" and m.tool_call_id == "tc_heal" and m.meta.get("_assistant_seq") == 2 for m in healed_msgs)

    # repair(mode='fold_tools_to_text') folds tool round into text 'repaired' projection
    r_fold = await port.repair(target, mode="fold_tools_to_text", actor=ACTOR)
    assert r_fold.mode == "fold_tools_to_text"
    assert r_fold.repaired_projections == 1

    # repair(mode='compact') compacts projections and clears dedup
    r_comp = await port.repair(target, mode="compact", actor=ACTOR)
    assert r_comp.mode == "compact"
    assert r_comp.send_index == 1 and r_comp.repaired_projections == 1
    assert await store.latest_project_compact(sid) is not None

    # repair(..., transform=...) rewrites messages cleanly
    r_tr = await port.repair(
        target,
        mode="heal_broken",
        transform=lambda _ms: [{"role": "user", "content": "rewritten", "send_index": 1}],
        actor=ACTOR,
    )
    assert r_tr.repaired_messages == 1
    final_msgs = await store.load_active_messages(sid)
    assert len(final_msgs) == 1 and final_msgs[0].content == "rewritten"

    # Refuses running session
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET execution_state='running',owner_node_id='n1',"
        "owner_boot_id='b1',owner_lease_until_ms=9999999999999 WHERE session_id=?",
        (sid,),
    )
    with pytest.raises(SessionBusy):
        await port.repair(target, mode="heal_broken", actor=ACTOR)


# ── 9. R01: repair(mode='compact') Snapshot Consistency & State Updates ──────


async def test_r01_compact_concurrent_seed_preserves_new_history_and_retry_visibility(registry_store):
    """R01: Pause repair(compact) right after paged read, seed send 24 with a unique marker,
    then resume repair. Verify conflict detection, untouched state, next-turn projection window
    visibility both before and after retrying repair(compact), and actual model prompt visibility.
    """
    import asyncio

    from power_loop.runtime.representation import ProjectedRepresentation

    store, _ = registry_store
    model = _RecordingModel([
        LLMResponse(raw_text="ack_after_retry_compact"),
    ])
    agent = await RunningAgent(
        store, model, ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        sid = agent.target.session_id
        # Populate 23 sends (> 1 page with page_size=20) so load_projections_paged pages across chunks.
        for si in range(1, 24):
            await store.upsert_project_message(
                sid, send_index=si, kind="user", content={"input": [f"q{si}"]}, rendered_text=f"q{si}",
            )
            await store.upsert_project_message(
                sid, send_index=si, kind="project", content={"final_text": f"a{si}"}, rendered_text=f"a{si}",
            )
        initial_stats = json.dumps({
            "v": 3, "sends": 23, "rows": 46, "tokens": 230, "maximum": 23,
            "compact_to_send": None, "dedup": {"abcdef012345": 23}, "revision": "rev_init",
        })
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET send_index=23,next_seq=47,fold_stats_json=? "
            "WHERE session_id=?",
            (initial_stats, sid),
        )

        paged_done = asyncio.Event()
        resume_repair = asyncio.Event()
        orig_paged = store.load_projections_paged

        async def paused_paged(*args, **kwargs):
            rows = await orig_paged(*args, **kwargs)
            paged_done.set()
            await resume_repair.wait()
            return rows

        marker = "R01_UNIQUE_SEED_MARKER_SEND_24"
        store.load_projections_paged = paused_paged  # type: ignore[method-assign]
        try:
            repair_task = asyncio.create_task(
                agent.runtime.client.repair(agent.target, mode="compact", actor=agent.actor)
            )
            await asyncio.wait_for(paged_done.wait(), timeout=5.0)
            seed_res = await agent.runtime.client.seed_context(
                agent.target,
                [{"role": "user", "content": marker}],
                actor=agent.actor,
            )
            assert seed_res["send_index"] == 24
            resume_repair.set()

            with pytest.raises(RuntimeRequestError) as exc_info:
                await repair_task
            assert exc_info.value.code == "request_conflict"
        finally:
            store.load_projections_paged = orig_paged  # type: ignore[method-assign]

        # Conflict must not write a stale compact or clobber fold_stats_json.
        assert await store.latest_project_compact(sid) is None
        srow_conflict = await store._db.fetchone(
            f"SELECT send_index,fold_stats_json FROM {store.t.sessions} WHERE session_id=?", (sid,),
        )
        assert int(srow_conflict["send_index"]) == 24
        assert srow_conflict["fold_stats_json"] == initial_stats

        # Before retrying repair, the next turn's projection window MUST see the seeded marker.
        projector = ProjectedRepresentation()
        win_before = await store.load_projection_window(sid, current_send_index=25)
        assert win_before.compact is None
        assert any(r.send_index == 24 and r.kind == "seed" for r in win_before.rows)
        rendered_before = projector.render(list(win_before.rows))
        assert any(marker in str(m.get("content") or "") for m in rendered_before)

        # Retry repair(mode='compact') without contention: summary & compact_to_send both reach 24.
        retry_res = await agent.runtime.client.repair(
            agent.target, mode="compact", actor=agent.actor,
        )
        assert retry_res.repaired_projections == 1
        assert retry_res.send_index == 24
        assert retry_res.summary is not None and marker in retry_res.summary

        compact_row = await store.latest_project_compact(sid)
        assert compact_row is not None
        assert compact_row.compact_from_send == 1
        assert compact_row.compact_to_send == 24
        assert marker in (compact_row.rendered_text or "")

        srow_after = await store._db.fetchone(
            f"SELECT fold_stats_json FROM {store.t.sessions} WHERE session_id=?", (sid,),
        )
        parsed_after = parse_state(srow_after["fold_stats_json"])
        assert parsed_after is not None
        assert parsed_after["compact_to_send"] == 24
        assert parsed_after["maximum"] == 24
        assert parsed_after["dedup"] == {}

        # Next turn's actual projection window AND real LLM request prompt MUST still see the marker.
        win_after = await store.load_projection_window(sid, current_send_index=25)
        assert win_after.compact is not None and win_after.compact.compact_to_send == 24
        assert win_after.rows == []
        rendered_after = projector.render([win_after.compact])
        assert any(marker in str(m.get("content") or "") for m in rendered_after)

        next_turn = await agent.send("follow up turn 25")
        assert next_turn.status == "completed" and next_turn.final_text == "ack_after_retry_compact"
        assert len(model.seen_messages) == 1
        assert any(marker in str(m.get("content") or "") for m in model.seen_messages[0])
    finally:
        await agent.aclose()


async def test_r01_compact_bounds_to_summarized_sends_preserves_pending_tool_seq_and_rolls_prior_compact(
    registry_store,
):
    """R01: Without contention, repair(compact):
    - Never advances compact_to_send past the highest summarized projection (e.g. unprojected send 3).
    - Preserves pending_tool_seq (does not clear unresolved dangling tool calls).
    - Keeps fold_stats_json.compact_to_send aligned with the written compact_to_send.
    - Rolls forward any prior compact summary into the new compact and removes superseded compacts.
    """
    from power_loop.runtime.representation import ProjectedRepresentation

    store, _ = registry_store
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    sid = target.session_id

    # Sends 1..2 projected; send 3 interrupted (no projection row yet, pending_tool_seq=5).
    for si in (1, 2):
        await store.upsert_project_message(
            sid, send_index=si, kind="user", content={"input": [f"turn_{si}_q"]}, rendered_text=f"turn_{si}_q",
        )
        await store.upsert_project_message(
            sid, send_index=si, kind="project", content={"final_text": f"turn_{si}_a"}, rendered_text=f"turn_{si}_a",
        )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET send_index=3,next_seq=6,pending_tool_seq=5,fold_stats_json=? "
        "WHERE session_id=?",
        (
            json.dumps({
                "v": 3, "sends": 2, "rows": 4, "tokens": 40, "maximum": 2,
                "compact_to_send": None, "dedup": {"112233445566": 2}, "revision": "r1",
            }),
            sid,
        ),
    )

    res1 = await port.repair(target, mode="compact", actor=ACTOR)
    assert res1.repaired_projections == 1
    assert res1.send_index == 2  # Must NOT advance to unprojected send 3!
    assert "turn_1_q" in (res1.summary or "") and "turn_2_a" in (res1.summary or "")

    c1 = await store.latest_project_compact(sid)
    assert c1 is not None and c1.compact_from_send == 1 and c1.compact_to_send == 2
    srow1 = await store._db.fetchone(
        f"SELECT pending_tool_seq,fold_stats_json FROM {store.t.sessions} WHERE session_id=?", (sid,),
    )
    assert int(srow1["pending_tool_seq"]) == 5  # Must NOT wipe unresolved pending_tool_seq!
    stats1 = parse_state(srow1["fold_stats_json"])
    assert stats1 is not None
    assert stats1["compact_to_send"] == 2
    assert stats1["maximum"] == 2
    assert stats1["dedup"] == {}

    # Now project send 3 and compact again: prior compact (1..2) must roll forward into (1..3).
    await store.upsert_project_message(
        sid, send_index=3, kind="user", content={"input": ["turn_3_q"]}, rendered_text="turn_3_q",
    )
    await store.upsert_project_message(
        sid, send_index=3, kind="project", content={"final_text": "turn_3_a"}, rendered_text="turn_3_a",
    )
    res2 = await port.repair(target, mode="compact", actor=ACTOR)
    assert res2.repaired_projections == 1
    assert res2.send_index == 3
    assert "turn_1_q" in (res2.summary or "") and "turn_3_a" in (res2.summary or "")

    compacts = [r for r in await store.load_project_messages(sid) if r.kind == "compact"]
    assert len(compacts) == 1
    assert compacts[0].compact_from_send == 1 and compacts[0].compact_to_send == 3
    win = await store.load_projection_window(sid, current_send_index=4)
    rendered = ProjectedRepresentation().render(([win.compact] if win.compact else []) + list(win.rows))
    joined = "\n".join(str(m.get("content") or "") for m in rendered)
    assert "turn_1_q" in joined and "turn_2_a" in joined and "turn_3_a" in joined
    srow2 = await store._db.fetchone(
        f"SELECT pending_tool_seq,fold_stats_json FROM {store.t.sessions} WHERE session_id=?", (sid,),
    )
    assert int(srow2["pending_tool_seq"]) == 5
    stats2 = parse_state(srow2["fold_stats_json"])
    assert stats2 is not None and stats2["compact_to_send"] == 3 and stats2["dedup"] == {}


async def test_r01_compact_stops_before_unprojected_active_gap_and_preserves_visibility(registry_store):
    """R01 rework: When an intermediate send has active audit messages but no projection yet
    (e.g. send 1 succeeded, send 2 failed on LLM error, send 3 seeded), uncontended
    repair(mode='compact') MUST NOT advance compact_to_send past the unprojected gap;
    turn 4's actual model request must still contain both send 2's input and send 3's seed.
    """
    store, _ = registry_store

    # Case 1: gap in the middle (send 1 ok, send 2 LLM failure -> no projection, send 3 seed)
    model = _RecordingModel([
        LLMResponse(raw_text="ok_1"),
        RuntimeError("simulated model failure"),
        LLMResponse(raw_text="ok_4"),
    ])
    agent = await RunningAgent(
        store, model, ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        sid = agent.target.session_id
        r1 = await agent.send("R01_SEND1_OK")
        assert r1.status == "completed"
        r2 = await agent.send("GAP_R01_SEND2")
        assert r2.status == "failed"
        seed3 = await agent.runtime.client.seed_context(
            agent.target, [{"role": "user", "content": "R01_SEED3"}], actor=agent.actor,
        )
        assert seed3["send_index"] == 3

        rep = await agent.runtime.client.repair(agent.target, mode="compact", actor=agent.actor)
        assert rep.repaired_projections == 1
        assert rep.send_index == 1  # Must stop before unprojected send 2!
        assert "R01_SEND1_OK" in (rep.summary or "")
        assert "R01_SEED3" not in (rep.summary or "")

        win = await store.load_projection_window(sid, current_send_index=4)
        assert win.compact is not None and win.compact.compact_to_send == 1
        assert win.window_lo == 2
        assert any(r.send_index == 3 and r.kind == "seed" for r in win.rows)

        srow = await store._db.fetchone(
            f"SELECT fold_stats_json FROM {store.t.sessions} WHERE session_id=?", (sid,),
        )
        stats = parse_state(srow["fold_stats_json"])
        assert stats is not None
        assert stats["compact_to_send"] == 1
        assert stats["maximum"] == 3
        assert stats["sends"] == 1
        assert stats.get("invalid") is True

        r4 = await agent.send("four")
        assert r4.status == "completed"
        prompt4 = str(model.seen_messages[-1])
        assert "R01_SEND1_OK" in prompt4
        assert "GAP_R01_SEND2" in prompt4
        assert "R01_SEED3" in prompt4
    finally:
        await agent.aclose()

    # Case 2: gap before the very first projected send (send 1 failed, send 2 seed) -> no-op compact
    model2 = _RecordingModel([
        RuntimeError("simulated model failure on turn 1"),
        LLMResponse(raw_text="ok_3"),
    ])
    agent2 = await RunningAgent(
        store, model2, ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        sid2 = agent2.target.session_id
        r1_fail = await agent2.send("GAP_R01_FIRST_SEND")
        assert r1_fail.status == "failed"
        seed2 = await agent2.runtime.client.seed_context(
            agent2.target, [{"role": "user", "content": "R01_SEED2_AFTER_FAIL"}], actor=agent2.actor,
        )
        assert seed2["send_index"] == 2

        rep2 = await agent2.runtime.client.repair(agent2.target, mode="compact", actor=agent2.actor)
        assert rep2.repaired_projections == 0
        assert await store.latest_project_compact(sid2) is None

        r3 = await agent2.send("three")
        assert r3.status == "completed"
        prompt3 = str(model2.seen_messages[-1])
        assert "GAP_R01_FIRST_SEND" in prompt3
        assert "R01_SEED2_AFTER_FAIL" in prompt3
    finally:
        await agent2.aclose()


# ── 10. R02: Re-compacting Across Page Boundary Preserves Prior Summary ──────


async def test_r02_recompact_across_page_boundary_preserves_first_and_avoids_duplication(
    registry_store,
):
    """R02: Populate >20 sends (default page_size=20) with FIRST in send 1, compact once,
    append AFTER in send 26, then compact again. Verify FIRST and AFTER both appear in the
    new context window and real model prompt, covered sends are not duplicated, and ordering
    plus the 300-char per-row truncation rule are preserved.
    """
    from power_loop.runtime.representation import ProjectedRepresentation

    store, _ = registry_store
    model = _RecordingModel([LLMResponse(raw_text="ack_r02_turn_27")])
    agent = await RunningAgent(
        store, model, ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        sid = agent.target.session_id
        for si in range(1, 26):
            q = "FIRST" if si == 1 else f"R02_TURN_{si}_Q"
            a = f"R02_TURN_{si}_A"
            await store.upsert_project_message(
                sid, send_index=si, kind="user", content={"input": [q]}, rendered_text=q,
            )
            await store.upsert_project_message(
                sid, send_index=si, kind="project", content={"final_text": a}, rendered_text=a,
            )
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET send_index=25,next_seq=51 WHERE session_id=?",
            (sid,),
        )

        # First compact across >1 page (25 sends > page_size 20).
        r1 = await agent.runtime.client.repair(agent.target, mode="compact", actor=agent.actor)
        assert r1.repaired_projections == 1
        assert r1.send_index == 25
        assert r1.summary is not None and "FIRST" in r1.summary
        c1 = await store.latest_project_compact(sid)
        assert c1 is not None and c1.compact_from_send == 1 and c1.compact_to_send == 25

        # Append AFTER (with a >300-char project reply to verify per-row 300-char cap).
        long_reply = "AFTER_REPLY_" + ("Z" * 350) + "_TAIL_DROPPED"
        await store.upsert_project_message(
            sid, send_index=26, kind="user", content={"input": ["AFTER"]}, rendered_text="AFTER",
        )
        await store.upsert_project_message(
            sid, send_index=26, kind="project",
            content={"final_text": long_reply}, rendered_text=long_reply,
        )
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET send_index=26,next_seq=53 WHERE session_id=?",
            (sid,),
        )

        # Second compact: load_projections_paged stops at compact@25 (page 1 = sends 7..26).
        r2 = await agent.runtime.client.repair(agent.target, mode="compact", actor=agent.actor)
        assert r2.repaired_projections == 1
        assert r2.send_index == 26

        compacts = [r for r in await store.load_project_messages(sid) if r.kind == "compact"]
        assert len(compacts) == 1
        assert compacts[0].compact_from_send == 1
        assert compacts[0].compact_to_send == 26

        win = await store.load_projection_window(sid, current_send_index=27)
        assert win.compact is not None and win.compact.compact_to_send == 26
        assert win.rows == []
        rendered = ProjectedRepresentation().render([win.compact])
        joined = "\n".join(str(m.get("content") or "") for m in rendered)

        assert joined.count("FIRST") == 1
        assert joined.count("AFTER_REPLY_") == 1
        assert "[send 26 user] AFTER" in joined
        assert "_TAIL_DROPPED" not in joined  # existing 300-char per-row cap preserved
        for si in range(2, 26):
            assert joined.count(f"R02_TURN_{si}_Q") == 1
            assert joined.count(f"R02_TURN_{si}_A") == 1
        # Projection ordering: send 1 -> send 25 -> send 26 user -> send 26 project.
        assert (
            joined.index("FIRST")
            < joined.index("R02_TURN_25_A")
            < joined.index("[send 26 user] AFTER")
            < joined.index("[send 26 project] AFTER_REPLY_")
        )

        # Idempotent re-compact with no new sends after boundary.
        r3 = await agent.runtime.client.repair(agent.target, mode="compact", actor=agent.actor)
        assert r3.repaired_projections == 0

        # Real next turn receives both FIRST and AFTER without duplicating covered turns.
        turn27 = await agent.send("turn 27 check")
        assert turn27.status == "completed" and turn27.final_text == "ack_r02_turn_27"
        prompt27 = "\n".join(str(m.get("content") or "") for m in model.seen_messages[0])
        assert prompt27.count("FIRST") == 1
        assert "[send 26 user] AFTER" in prompt27
        assert prompt27.count("R02_TURN_7_Q") == 1
        assert prompt27.count("R02_TURN_25_Q") == 1
    finally:
        await agent.aclose()


async def test_r02_recompact_starts_from_latest_valid_compact_across_pruned_and_multipage_tail(
    registry_store,
):
    """R02 edge cases:
    1) When old rows <= compact_to_send were pruned (FIRST exists only in the prior compact)
       and a malformed kind='compact' row with compact_to_send=NULL sits at a higher send_index,
       resolve/window/repair still start from the latest valid compact and ignore the invalid row.
    2) When >20 new sends accumulate after a prior compact, re-compacting pages back to that
       valid compact, preserves FIRST, and appends only sends strictly after compact_to_send.
    """
    from power_loop.runtime.representation import ProjectedRepresentation

    store, _ = registry_store
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    sid = target.session_id

    # Seed a valid compact covering sends 1..25 (FIRST only lives in this compact row).
    await store.upsert_project_message(
        sid, send_index=25, kind="compact",
        content={"summary": "[send 1 user] FIRST\n[send 25 project] R02_OLD_25", "from_send": 1, "to_send": 25},
        rendered_text="[send 1 user] FIRST\n[send 25 project] R02_OLD_25",
        compact_from_send=1, compact_to_send=25,
    )
    # Add an invalid compact row (compact_to_send=NULL) at a higher send_index plus >20 new sends.
    await store._db.execute(
        f"INSERT INTO {store.t.project_messages} ("
        "session_id,send_index,kind,content_json,rendered_text,"
        "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
        "projector_version,token_estimate,created_at,completed_through_send_index) "
        "VALUES (?,26,'compact','{\"summary\":\"INVALID\"}','INVALID',NULL,NULL,NULL,NULL,0,2,1000,26)",
        (sid,),
    )
    for si in range(26, 48):
        label = "AFTER" if si == 47 else f"R02_TAIL_{si}"
        await store.upsert_project_message(
            sid, send_index=si, kind="user", content={"input": [label]}, rendered_text=label,
        )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET send_index=47,next_seq=95 WHERE session_id=?",
        (sid,),
    )

    res = await port.repair(target, mode="compact", actor=ACTOR)
    assert res.repaired_projections == 1
    assert res.send_index == 47

    latest = await store.latest_project_compact(sid)
    assert latest is not None
    assert latest.compact_from_send == 1 and latest.compact_to_send == 47

    win = await store.load_projection_window(sid, current_send_index=48)
    assert win.compact is not None and win.compact.compact_to_send == 47
    assert win.rows == []
    joined = "\n".join(
        str(m.get("content") or "") for m in ProjectedRepresentation().render([win.compact])
    )
    assert joined.count("FIRST") == 1
    assert joined.count("R02_OLD_25") == 1
    assert joined.count("AFTER") == 1
    assert "INVALID" not in joined
    for si in range(26, 47):
        assert joined.count(f"R02_TAIL_{si}") == 1

    # Capped window also selects the latest valid compact cleanly.
    win_capped = await store.load_projection_window(sid, current_send_index=48, send_limit=5)
    assert win_capped.compact is not None and win_capped.compact.compact_to_send == 47
    assert "FIRST" in (win_capped.compact.rendered_text or "")
    assert "AFTER" in (win_capped.compact.rendered_text or "")


async def test_r06_seed_context_summary_callback_outside_transaction(registry_store):
    """R06: async summary_fn runs outside the DB write transaction without self-deadlock.

    Covers:
      1. Async summary_fn reads the same SessionStore via get_session() and returns a unique
         summary; seed_context completes and the next turn receives that summary.
      2. Pausing summary_fn on an asyncio.Event allows concurrent reads/writes on another
         session on the same SessionStore without blocking.
      3. Raising inside summary_fn leaves zero messages, projections, or session mutations.
      4. Unauthorized caller is rejected before summary_fn runs.
      5. Session becoming busy while summary_fn awaits raises SessionBusy on commit with no
         half-seed written; concurrent idle advance commits cleanly at the next send_index.
    """
    import asyncio

    from power_loop.agent.runtime_types import TrustedActor

    store, _ = registry_store
    port = RuntimeInputStore(store)
    long_msgs = [
        {"role": "user", "content": "OLD_CTX " + "x" * 400},
        {"role": "assistant", "content": "MID_CTX " + "y" * 400},
    ]

    # 1. Async summary_fn queries same SessionStore; next turn reads the returned unique summary.
    model = _RecordingModel([LLMResponse(raw_text="turn1_ok")])
    agent = await RunningAgent(
        store, model, ToolRegistry(),
        AgentLoopConfig(system_prompt="SYS", max_rounds=2, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        sid = agent.target.session_id
        unique_summary = f"R06_UNIQUE_SUMMARY_{sid}"

        async def _reading_summary(msgs):
            row = await store.get_session(sid)
            assert row is not None and row.session_id == sid
            assert len(msgs) == 2
            return unique_summary

        seed_res = await asyncio.wait_for(
            agent.runtime.client.seed_context(
                agent.target,
                long_msgs,
                actor=agent.actor,
                overflow="compact",
                max_tokens=50,
                summary_fn=_reading_summary,
            ),
            timeout=5.0,
        )
        assert seed_res["send_index"] == 0
        assert seed_res["truncated"] is True
        assert seed_res["seeded_count"] == 1

        turn1 = await agent.send("follow_up_after_r06_seed")
        assert turn1.status == "completed"
        seen = " ".join(str(m.get("content") or "") for m in model.seen_messages[0])
        assert unique_summary in seen and "follow_up_after_r06_seed" in seen
    finally:
        await agent.aclose()

    # 2. Pause summary_fn on an event and operate on another session on the same store concurrently.
    t_a = await port.create_session(DESCRIPTOR, actor=ACTOR)
    t_b = await port.create_session(DESCRIPTOR, actor=ACTOR)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _paused_summary(_msgs):
        row = await store.get_session(t_a.session_id)
        assert row is not None
        entered.set()
        await release.wait()
        return "R06_PAUSED_SUMMARY_A"

    task_a = asyncio.create_task(
        port.seed_context(
            t_a, long_msgs, actor=ACTOR, overflow="compact", max_tokens=50, summary_fn=_paused_summary,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        # Operate on session B while session A's callback is suspended.
        b_row = await asyncio.wait_for(store.get_session(t_b.session_id), timeout=3.0)
        assert b_row is not None
        b_res = await asyncio.wait_for(
            port.seed_context(t_b, [{"role": "user", "content": "R06_CONCURRENT_B"}], actor=ACTOR),
            timeout=3.0,
        )
        assert b_res["send_index"] == 0
        release.set()
        a_res = await asyncio.wait_for(task_a, timeout=5.0)
        assert a_res["send_index"] == 0 and a_res["truncated"] is True
    finally:
        release.set()
        if not task_a.done():
            task_a.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task_a

    prows_a = await store.load_project_messages(t_a.session_id)
    assert len(prows_a) == 1 and "R06_PAUSED_SUMMARY_A" in (prows_a[0].rendered_text or "")

    # 3. Callback error leaves zero half-seed writes and preserves session state.
    t_err = await port.create_session(DESCRIPTOR, actor=ACTOR)
    before_s = await store._db.fetchone(
        f"SELECT send_index,next_seq,inbox_version FROM {store.t.sessions} WHERE session_id=?",
        (t_err.session_id,),
    )

    async def _failing_summary(_msgs):
        assert await store.get_session(t_err.session_id) is not None
        raise RuntimeError("r06_summary_boom")

    with pytest.raises(RuntimeError, match="r06_summary_boom"):
        await port.seed_context(
            t_err, long_msgs, actor=ACTOR, overflow="compact", max_tokens=50, summary_fn=_failing_summary,
        )
    after_s = await store._db.fetchone(
        f"SELECT send_index,next_seq,inbox_version FROM {store.t.sessions} WHERE session_id=?",
        (t_err.session_id,),
    )
    assert dict(after_s) == dict(before_s)
    assert await store.load_all_messages(t_err.session_id) == []
    assert await store.load_project_messages(t_err.session_id) == []

    # 4. Unauthorized caller fails before summary_fn is invoked.
    cb_calls = 0

    async def _never_called(_msgs):
        nonlocal cb_calls
        cb_calls += 1
        return "SHOULD_NOT_RUN"

    intruder = TrustedActor("other-tenant", "other-user", None)
    with pytest.raises(RuntimeRequestError) as exc_forb:
        await port.seed_context(
            t_err, long_msgs, actor=intruder, overflow="compact", max_tokens=50, summary_fn=_never_called,
        )
    assert exc_forb.value.code == "forbidden"
    assert cb_calls == 0

    # 5. Session becoming busy during callback is rejected in final commit with no half-seed.
    t_busy = await port.create_session(DESCRIPTOR, actor=ACTOR)
    busy_entered = asyncio.Event()
    busy_release = asyncio.Event()

    async def _busy_race_summary(_msgs):
        busy_entered.set()
        await busy_release.wait()
        return "R06_LATE_SUMMARY"

    task_busy = asyncio.create_task(
        port.seed_context(
            t_busy, long_msgs, actor=ACTOR, overflow="compact", max_tokens=50, summary_fn=_busy_race_summary,
        )
    )
    try:
        await asyncio.wait_for(busy_entered.wait(), timeout=5.0)
        _, claimed, _ = await port.accept_and_claim(
            t_busy, "concurrent turn", message_id="r06-m1", actor=ACTOR,
            node_id="n1", boot_id="b1", ttl_s=30.0,
        )
        assert claimed is not None
        busy_release.set()
        with pytest.raises(SessionBusy):
            await asyncio.wait_for(task_busy, timeout=5.0)
    finally:
        busy_release.set()
        if not task_busy.done():
            task_busy.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task_busy

    seed_prows = [
        r for r in await store.load_project_messages(t_busy.session_id) if r.kind == "seed"
    ]
    assert seed_prows == []

    # 6. Concurrent idle seed advances send_index while callback awaits; final commit uses latest state.
    t_adv = await port.create_session(DESCRIPTOR, actor=ACTOR)
    adv_entered = asyncio.Event()
    adv_release = asyncio.Event()

    async def _adv_summary(_msgs):
        adv_entered.set()
        await adv_release.wait()
        return "R06_SECOND_AFTER_CONCURRENT_SEED"

    task_adv = asyncio.create_task(
        port.seed_context(
            t_adv, long_msgs, actor=ACTOR, overflow="compact", max_tokens=50, summary_fn=_adv_summary,
        )
    )
    try:
        await asyncio.wait_for(adv_entered.wait(), timeout=5.0)
        first_seed = await port.seed_context(
            t_adv, [{"role": "user", "content": "R06_FIRST_CONCURRENT_SEED"}], actor=ACTOR,
        )
        assert first_seed["send_index"] == 0
        adv_release.set()
        second_seed = await asyncio.wait_for(task_adv, timeout=5.0)
        assert second_seed["send_index"] == 1
    finally:
        adv_release.set()
        if not task_adv.done():
            task_adv.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task_adv

    adv_prows = await store.load_project_messages(t_adv.session_id)
    assert [(r.send_index, r.kind) for r in adv_prows] == [(0, "seed"), (1, "seed")]
    assert "R06_SECOND_AFTER_CONCURRENT_SEED" in (adv_prows[1].rendered_text or "")


async def test_r06_seed_context_budget_tightened_during_callback(registry_store):
    """R06 rework: final commit checks tightened session budget even when summary_fn ran.

    Covers:
      1. Session budget starts at 100, seed_context omits explicit max_tokens, and summary_fn
         tightens budget to 10 via configure_session before returning a summary with 10 < est < 100
         -> raises request_conflict, runs callback once, keeps budget=10, writes zero rows and
         does not advance send_index/next_seq.
      2. Same tightening (100 -> 10), but returned summary fits (est <= 10) -> commits cleanly.
      3. Explicit max_tokens keeps precedence over session config tightening, and unchanged
         session budget preserves existing behavior.
    """
    from dataclasses import replace

    from power_loop.runtime.budget import estimate_tokens

    store, _ = registry_store
    port = RuntimeInputStore(store)
    desc_100 = replace(DESCRIPTOR, config={**DESCRIPTOR.config, "context_budget_tokens": 100})
    desc_10 = replace(DESCRIPTOR, config={**DESCRIPTOR.config, "context_budget_tokens": 10})
    long_msgs = [
        {"role": "user", "content": "OLD_CTX " + "x" * 400},
        {"role": "assistant", "content": "MID_CTX " + "y" * 400},
    ]
    assert estimate_tokens(long_msgs) > 100

    oversized_summary = "R06_OVER_10_UNDER_100 " + "s" * 100
    oversized_est = estimate_tokens([{"role": "user", "content": oversized_summary}])
    assert 10 < oversized_est < 100

    fitting_summary = "R06_FIT_OK"
    fitting_est = estimate_tokens([{"role": "user", "content": fitting_summary}])
    assert fitting_est <= 10

    # 1. Tightened budget (100 -> 10) with summary exceeding new budget -> request_conflict.
    t_conflict = await port.create_session(desc_100, actor=ACTOR)
    before_s = await store._db.fetchone(
        f"SELECT send_index,next_seq,inbox_version FROM {store.t.sessions} WHERE session_id=?",
        (t_conflict.session_id,),
    )
    conflict_calls = 0

    async def _tighten_oversized(_msgs):
        nonlocal conflict_calls
        conflict_calls += 1
        await port.configure_session(t_conflict, actor=ACTOR, descriptor=desc_10)
        return oversized_summary

    with pytest.raises(RuntimeRequestError) as exc_conflict:
        await port.seed_context(
            t_conflict, long_msgs, actor=ACTOR, overflow="compact", summary_fn=_tighten_oversized,
        )
    assert exc_conflict.value.code == "request_conflict"
    assert conflict_calls == 1

    after_s = await store._db.fetchone(
        f"SELECT send_index,next_seq,inbox_version,descriptor_json FROM {store.t.sessions} "
        "WHERE session_id=?",
        (t_conflict.session_id,),
    )
    assert json.loads(after_s["descriptor_json"])["config"]["context_budget_tokens"] == 10
    assert int(after_s["send_index"]) == int(before_s["send_index"])
    assert int(after_s["next_seq"]) == int(before_s["next_seq"])
    assert int(after_s["inbox_version"]) == int(before_s["inbox_version"])
    assert await store.load_all_messages(t_conflict.session_id) == []
    assert await store.load_project_messages(t_conflict.session_id) == []

    # 2. Tightened budget (100 -> 10) with summary still fitting (est <= 10) -> succeeds.
    t_fit = await port.create_session(desc_100, actor=ACTOR)
    fit_calls = 0

    async def _tighten_fitting(_msgs):
        nonlocal fit_calls
        fit_calls += 1
        await port.configure_session(t_fit, actor=ACTOR, descriptor=desc_10)
        return fitting_summary

    fit_res = await port.seed_context(
        t_fit, long_msgs, actor=ACTOR, overflow="compact", summary_fn=_tighten_fitting,
    )
    assert fit_calls == 1
    assert fit_res["send_index"] == 0
    assert fit_res["truncated"] is True
    assert fit_res["token_estimate"] == fitting_est <= 10

    fit_row = await store._db.fetchone(
        f"SELECT send_index,next_seq,descriptor_json FROM {store.t.sessions} WHERE session_id=?",
        (t_fit.session_id,),
    )
    assert json.loads(fit_row["descriptor_json"])["config"]["context_budget_tokens"] == 10
    fit_msgs = await store.load_all_messages(t_fit.session_id)
    assert len(fit_msgs) == 1 and fit_msgs[0].content == fitting_summary
    fit_prows = await store.load_project_messages(t_fit.session_id)
    assert len(fit_prows) == 1 and fitting_summary in (fit_prows[0].rendered_text or "")

    # 3. Explicit max_tokens keeps precedence over session config tightening.
    t_explicit = await port.create_session(desc_100, actor=ACTOR)
    explicit_calls = 0

    async def _tighten_with_explicit(_msgs):
        nonlocal explicit_calls
        explicit_calls += 1
        await port.configure_session(t_explicit, actor=ACTOR, descriptor=desc_10)
        return oversized_summary

    explicit_res = await port.seed_context(
        t_explicit,
        long_msgs,
        actor=ACTOR,
        overflow="compact",
        max_tokens=100,
        summary_fn=_tighten_with_explicit,
    )
    assert explicit_calls == 1
    assert explicit_res["send_index"] == 0
    assert explicit_res["truncated"] is True
    assert explicit_res["token_estimate"] == oversized_est
    explicit_row = await store._db.fetchone(
        f"SELECT descriptor_json FROM {store.t.sessions} WHERE session_id=?",
        (t_explicit.session_id,),
    )
    assert json.loads(explicit_row["descriptor_json"])["config"]["context_budget_tokens"] == 10

    # 4. Unchanged session budget preserves existing behavior even if summary_fn exceeds budget.
    t_same = await port.create_session(desc_10, actor=ACTOR)
    same_calls = 0

    async def _unchanged_budget_summary(_msgs):
        nonlocal same_calls
        same_calls += 1
        return oversized_summary

    same_res = await port.seed_context(
        t_same, long_msgs, actor=ACTOR, overflow="compact", summary_fn=_unchanged_budget_summary,
    )
    assert same_calls == 1
    assert same_res["send_index"] == 0
    assert same_res["truncated"] is True
    assert same_res["token_estimate"] == oversized_est



