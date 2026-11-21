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
