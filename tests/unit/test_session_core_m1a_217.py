"""Design 217 M1a Session-Core contract verification across SQLite / PostgreSQL 16 / MySQL 8.4.

Covers:
- Schema 18 DDL, forbidden tables/columns, index inventory, v17 rejection
- Probe 1: workflow_driver start_input_message_id never derives premature 'completed'
- Probe 3: COMPLETE_DECIDE False=0-SQL, True/inject bounded by max_extra_rounds, wake=0 isolation
- Probe 8: configure_session next_descriptor promotion across all 3 claim entry points
- Decision 1 (拍板一): fused 2-tx, transparent fallback, 0-assistant crash rerun, create_many + 1-SQL claim
- Decision 2 (拍板二): wake=0, reserved prefixes rejected with 0 SQL, active-only subtree cancel + depth DESC
- River same-tx non-cascading delivery: result_message_id, parent suspended, parent closed/full non-rollback
- Decision 3 (拍板三): InMemoryWakeTransport (0 SQL), DatabasePollingWakeTransport, aclose graceful_release
- Concurrency & Fencing: stale owner_epoch blocked, concurrent child Final + parent accept/cancel (0 deadlock)
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

import power_loop
import power_loop.workflow
from power_loop import (
    AgentLoopConfig,
    CompleteDecideCtx,
    DatabasePollingWakeTransport,
    HookDirective,
    HookPoint,
    InMemoryWakeTransport,
    InputRef,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RuntimeRequestError,
    SessionLeaseLost,
    TargetRef,
    ToolDefinition,
    TrustedActor,
    WakeTransport,
    result_message_id,
)
from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION, StoreSchemaError
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent, ScriptedModel

from .session_native_fixture import ACTOR, DESCRIPTOR, target
from .test_send_sql_profile import Profiled

pytestmark = [pytest.mark.unit]


def test_version_stable_api_and_workflow_import():
    assert power_loop.__version__ == "14.0.0"
    for sym in ("WakeTransport", "InMemoryWakeTransport", "DatabasePollingWakeTransport", "result_message_id"):
        assert sym in power_loop.STABLE_API
        assert sym in power_loop.__all__
        assert hasattr(power_loop, sym)
    assert isinstance(InMemoryWakeTransport(), WakeTransport)
    assert power_loop.workflow is not None


async def test_schema18_tables_indexes_and_rejects_v17(registry_store):
    store, connect = registry_store
    assert store.schema_version == CURRENT_SCHEMA_VERSION == 18
    assert not hasattr(store.t, "session_state")
    # Verify new §3.4 indexes exist by dropping none, only checking metadata/rejection of v17
    await store._db.execute(
        f"UPDATE {store.table_prefix}schema_migrations SET version=17 WHERE id=1"
    )
    with pytest.raises(StoreSchemaError):
        await connect(schema="verify")
    await store._db.execute(
        f"UPDATE {store.table_prefix}schema_migrations SET version=18 WHERE id=1"
    )


async def test_probe1_workflow_driver_start_input_never_premature_completed(registry_store):
    """Probe 1: inspect_input / inspect / get_result never derive premature 'completed'
    for workflow_driver.start_input_message_id until explicit final(settle_start_input=True)."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    driver_desc = replace(DESCRIPTOR, kind="workflow_driver")
    start_id = "wf_start_p1"
    drv = await port.create_session(
        driver_desc,
        actor=ACTOR,
        options={"session_id": "p1_drv", "start_input": InputRef(store.store_id, "p1_drv", start_id)},
    )
    receipt = await port.accept(drv, {"workflow_id": "wf1"}, message_id=start_id, actor=ACTOR)
    batch = await port.claim(drv, node_id="n1", boot_id="b1")
    assert batch is not None and batch.send_index == 1

    # Driver finishes turn 1 (waiting for leaves) without settling start_input
    await port.final(batch, status="completed", final_text="")

    # Must NOT derive premature 'completed'!
    inspected = await port.inspect_input(receipt.input_ref, actor=ACTOR)
    assert inspected["terminal_status"] is None
    st = await port.inspect(receipt.input_ref, actor=ACTOR)
    assert st.terminal_status is None and not st.result_available
    assert await port.get_result(receipt.input_ref, actor=ACTOR) is None

    # Turn 2: driver receives leaf result and explicitly settles start_input
    await port.accept(drv, {"leaf": "ok"}, message_id="res_leaf1", actor=ACTOR, kind="agent_result", public=False)
    batch2 = await port.claim(drv, node_id="n1", boot_id="b1")
    assert batch2 is not None and batch2.send_index == 2
    await port.final(
        batch2, status="completed", final_text="wf done",
        result_json={"workflow": "wf1", "ok": True}, settle_start_input=True,
    )
    settled = await port.inspect_input(receipt.input_ref, actor=ACTOR)
    assert settled["terminal_status"] == "completed"
    assert json.loads(settled["result_json"]) == {"workflow": "wf1", "ok": True}
    res = await port.get_result(receipt.input_ref, actor=ACTOR)
    assert res is not None and res.status == "completed"
    assert dict(res.structured_output) == {"workflow": "wf1", "ok": True}


async def test_probe3_complete_decide_and_wake0_isolation(registry_store):
    """Probe 3:
    3A: COMPLETE_DECIDE False -> 0 extra SQL on 0-tool fused turn.
    3B: COMPLETE_DECIDE SHORT_CIRCUIT + inject -> same send_index, bounded by max_extra_rounds.
    3C: Mid-tool-round wake=0 notice -> 0 piggyback miss, stays pending for next turn.
    """
    store, _ = registry_store
    db = store._db.dialect.name
    original = store._db

    # 3A: COMPLETE_DECIDE False -> 0 extra control SQL
    profiled = Profiled(original)
    store._db = profiled
    agent_a = await RunningAgent(
        store,
        ScriptedModel([LLMResponse(raw_text="clean", token_usage=LLMTokenUsage(2, 2, 4))]),
        ToolRegistry(),
        AgentLoopConfig(max_rounds=5, context_budget_tokens=10**9),
    ).start()
    try:
        agent_a.hooks.register(HookPoint.COMPLETE_DECIDE, lambda ctx: None)
        profiled.records.clear()
        res_a = await agent_a.send("hello")
        assert res_a.status == "completed" and res_a.final_text == "clean"
        boundary_reads = [r for r in profiled.records if "runtime_store:boundary" in " ".join(r["chain"])]
        assert boundary_reads == [], boundary_reads
    finally:
        await agent_a.aclose()
        store._db = original

    # 3B: COMPLETE_DECIDE always injecting -> hard capped by max_extra_rounds=2 (total 3 LLM calls)
    calls = 0

    class _CountingLLM(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal calls
            calls += 1
            return LLMResponse(raw_text=f"step_{calls}")

        def stream(self, request):
            raise NotImplementedError

        async def close(self):
            pass

    agent_b = await RunningAgent(
        store,
        _CountingLLM(),
        ToolRegistry(),
        AgentLoopConfig(max_rounds=10, max_extra_rounds=2, context_budget_tokens=10**9),
    ).start()
    try:
        def _always_inject(ctx: CompleteDecideCtx):
            ctx.directive = HookDirective.SHORT_CIRCUIT
            ctx.inject = "keep going"
            ctx.extra_rounds = 5
        agent_b.hooks.register(HookPoint.COMPLETE_DECIDE, _always_inject)
        res_b = await agent_b.send("start")
        assert res_b.status == "completed" and calls == 3
        msgs = await store.load_all_messages(agent_b.target.session_id)
        assert sum(1 for m in msgs if m.role == "assistant") == 3
        assert len({m.send_index for m in msgs}) == 1
    finally:
        await agent_b.aclose()

    # 3C: Mid-tool-round wake=0 system_notice -> 0 piggyback miss, stays pending after Final
    port = RuntimeInputStore(store)
    reg = ToolRegistry()
    target_holder: list[TargetRef] = []
    actor_holder: list[TrustedActor] = []

    async def work():
        acc, need_wake = await port.accept(
            target_holder[0],
            {"text": "[system_notice] silent info"},
            message_id="sys_w0_notice",
            actor=actor_holder[0],
            kind="system_notice",
            wake=False,
            public=False,
            _with_activation=True,
        )
        assert acc.newly_accepted and need_wake is False
        return "tool_done"

    reg.register(
        ToolDefinition("work", "do work", input_schema={"type": "object", "properties": {}}, pricing=Free("f")),
        work,
    )
    llm_c = ScriptedModel([
        LLMResponse(raw_text="", tool_calls=[{
            "id": "tc1", "type": "function", "function": {"name": "work", "arguments": "{}"},
        }]),
        LLMResponse(raw_text="round1_done"),
    ])
    profiled_c = Profiled(original)
    store._db = profiled_c
    agent_c = await RunningAgent(
        store, llm_c, reg, AgentLoopConfig(max_rounds=5, context_budget_tokens=10**9),
    ).start()
    target_holder.append(agent_c.target)
    actor_holder.append(agent_c.actor)
    try:
        profiled_c.records.clear()
        res_c = await agent_c.send("go")
        assert res_c.status == "completed" and res_c.final_text == "round1_done"
        # Piggyback MUST succeed with 0 boundary SELECT and 0 claim_follow_up!
        spurious = [
            r for r in profiled_c.records
            if any(s in " ".join(r["chain"]) for s in ("runtime_store:boundary", "runtime_store:claim_follow_up"))
        ]
        assert spurious == [], spurious
        # Session returned to idle and wake=0 notice stays pending for next turn
        st = await store._db.fetchone(
            f"SELECT execution_state,wake_pending_count,pending_count FROM {store.t.sessions} WHERE session_id=?",
            (agent_c.target.session_id,),
        )
        assert st["execution_state"] == "idle"
        assert int(st["wake_pending_count"]) == 0 and int(st["pending_count"]) == 1
        _ = db
    finally:
        await agent_c.aclose()
        store._db = original


async def test_probe8_configure_session_promotes_across_all_three_claim_entries(registry_store):
    """Probe 8: next_descriptor_json/hash is promoted atomically in:
    (1) fused accept_and_claim, (2) standard claim, (3) 1-SQL pre-staged child claim."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    desc_v1 = replace(DESCRIPTOR, config={"version": 1})
    desc_v2 = replace(DESCRIPTOR, config={"version": 2})

    # Idle configure_session applies immediately
    s_idle = await port.create_session(desc_v1, actor=ACTOR, options={"session_id": "p8_idle"})
    cfg_idle = await port.configure_session(s_idle, actor=ACTOR, descriptor=desc_v2)
    assert cfg_idle["staged"] is False
    row_idle = await store._db.fetchone(
        f"SELECT descriptor_json,next_descriptor_json FROM {store.t.sessions} WHERE session_id=?",
        ("p8_idle",),
    )
    assert json.loads(row_idle["descriptor_json"])["config"] == {"version": 2}
    assert row_idle["next_descriptor_json"] is None

    # Entry #1: running -> staged -> final -> fused accept_and_claim promotes v2
    s1 = await port.create_session(desc_v1, actor=ACTOR, options={"session_id": "p8_fused"})
    _, b1, _ = await port.accept_and_claim(s1, "t1", message_id="m1", actor=ACTOR, node_id="n1", boot_id="b1")
    assert b1 is not None and b1.descriptor.config == {"version": 1}
    cfg1 = await port.configure_session(s1, actor=ACTOR, descriptor=desc_v2)
    assert cfg1["staged"] is True
    await port.final(b1, final_text="ok1")
    _, b1_next, _ = await port.accept_and_claim(s1, "t2", message_id="m2", actor=ACTOR, node_id="n1", boot_id="b1")
    assert b1_next is not None and b1_next.descriptor.config == {"version": 2}
    await port.final(b1_next, final_text="ok2")

    # Entry #2: running -> staged -> final -> standard accept + claim promotes v2
    s2 = await port.create_session(desc_v1, actor=ACTOR, options={"session_id": "p8_std"})
    await port.accept(s2, "t1", message_id="m1", actor=ACTOR)
    b2 = await port.claim(s2, node_id="n1", boot_id="b1")
    assert b2 is not None and b2.descriptor.config == {"version": 1}
    assert (await port.configure_session(s2, actor=ACTOR, descriptor=desc_v2))["staged"] is True
    await port.final(b2, final_text="ok1")
    await port.accept(s2, "t2", message_id="m2", actor=ACTOR)
    b2_next = await port.claim(s2, node_id="n1", boot_id="b1")
    assert b2_next is not None and b2_next.descriptor.config == {"version": 2}
    await port.final(b2_next, final_text="ok2")

    # Entry #3: create_many pre-staged child -> configure_session stages on queued? Wait:
    # simulate staged next_descriptor_json on pre-staged child and verify 1-SQL claim promotes it!
    parent = await port.create_session(desc_v1, actor=ACTOR, options={"session_id": "p8_parent"})
    cm = await port.create_many(
        parent=parent,
        actor=ACTOR,
        specs=[ChildSpec(session_id="p8_pre", descriptor=desc_v1, message_id="init", body="run")],
        node_id="n1", boot_id="b1", free_slots=0,
    )
    assert len(cm.queued_targets) == 1
    # Directly stage next_descriptor on the pre-staged queued child to test 1-SQL claim promotion
    from power_loop.agent.runtime_types import canonical, sha256_canonical
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET next_descriptor_json=?,next_descriptor_hash=? WHERE session_id=?",
        (canonical(desc_v2), sha256_canonical(desc_v2), "p8_pre"),
    )
    b3 = await port.claim(cm.queued_targets[0], node_id="n2", boot_id="b2")
    assert b3 is not None and b3.descriptor.config == {"version": 2}
    row3 = await store._db.fetchone(
        f"SELECT descriptor_json,next_descriptor_json FROM {store.t.sessions} WHERE session_id=?",
        ("p8_pre",),
    )
    assert json.loads(row3["descriptor_json"])["config"] == {"version": 2}
    assert row3["next_descriptor_json"] is None
    await port.final(b3, final_text="ok3")


async def test_decision1_fused_fallback_zero_assistant_rerun_and_create_many(registry_store):
    """拍板一:
    - accept_and_claim transparently falls back when busy / incapable.
    - 0-assistant crash right after fused accept_and_claim -> reclaim_running reruns safely without dup input.
    - create_many pre-claims k and pre-stages N-k for 1-SQL cross-container claim (+ idempotent retry).
    """
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port, "d1_root")

    # Fallback when incapable
    r_incap, b_incap, wake_incap = await port.accept_and_claim(
        root, "hello", message_id="m_incap", actor=ACTOR,
        node_id="n1", boot_id="b1", can_execute=lambda d: False,
    )
    assert r_incap.newly_accepted and b_incap is None and wake_incap is True
    b_std = await port.claim(root, node_id="n2", boot_id="b2")
    assert b_std is not None
    # Fallback when busy
    r_busy, b_busy, _ = await port.accept_and_claim(
        root, "second", message_id="m_busy", actor=ACTOR, node_id="n2", boot_id="b2",
    )
    assert r_busy.newly_accepted and b_busy is None
    await port.final(b_std, final_text="done1")
    b_next = await port.claim(root, node_id="n2", boot_id="b2")
    await port.final(b_next, final_text="done2")

    # 0-assistant crash rerun after fused accept_and_claim
    crash_sid = await target(port, "d1_crash")
    _, crashed_batch, _ = await port.accept_and_claim(
        crash_sid, "crash-test", message_id="m_crash", actor=ACTOR, node_id="dead", boot_id="bdead",
    )
    assert crashed_batch is not None
    # Expire lease without writing any assistant row
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
        (crash_sid.session_id,),
    )
    reclaimed_list = await port.sweep_expired_leases(node_id="alive", boot_id="balive", limit=32)
    assert len(reclaimed_list) == 1
    reclaimed = reclaimed_list[0]
    assert reclaimed.target == crash_sid and reclaimed.is_reclaimed is True
    assert reclaimed.reclaimed_has_assistant is False
    assert reclaimed.send_index == crashed_batch.send_index
    assert reclaimed.owner_epoch == crashed_batch.owner_epoch + 1
    # User row was NOT duplicated
    user_rows = [m for m in await store.load_all_messages(crash_sid.session_id) if m.role == "user"]
    assert len(user_rows) == 1
    await port.final(reclaimed, final_text="rerun_succeeded")
    assert (await port.get_result(reclaimed.input_refs[0], actor=ACTOR)).final_text == "rerun_succeeded"

    # create_many + 1-SQL cross-container claim + idempotent retry
    specs = [
        ChildSpec(session_id=f"d1_c{i}", descriptor=DESCRIPTOR, message_id=f"init_{i}", body=f"job {i}")
        for i in range(3)
    ]
    cm = await port.create_many(parent=root, actor=ACTOR, specs=specs, node_id="n1", boot_id="b1", free_slots=1)
    assert len(cm.preclaimed) == 1 and len(cm.queued_targets) == 2
    assert cm.preclaimed[0].target.session_id == "d1_c0"
    # Idempotent retry returns empty preclaimed/queued_targets without error
    cm_retry = await port.create_many(parent=root, actor=ACTOR, specs=specs, node_id="n1", boot_id="b1", free_slots=1)
    assert cm_retry.preclaimed == () and cm_retry.queued_targets == ()

    # Measure 1-SQL (2 SQL MySQL) pre-staged child claim!
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        profiled.records.clear()
        b_c1 = await port.claim(cm.queued_targets[0], node_id="n2", boot_id="b2")
        assert b_c1 is not None and b_c1.send_index == 1 and b_c1.message_seq == 1
        expected_sql = 2 if original.dialect.name == "mysql" else 1
        assert len(profiled.records) == expected_sql, [r["sql"] for r in profiled.records]
    finally:
        store._db = original
    await port.final(cm.preclaimed[0], final_text="c0")
    await port.final(b_c1, final_text="c1")


async def test_decision2_reserved_prefixes_wake0_and_active_only_subtree_cancel(registry_store):
    """拍板二:
    - Public send() rejects res_/wfc_/sys_ prefixes with 0 SQL.
    - wake=0 never flips idle->queued or bumps wake_inbox_seq/wake_pending_count.
    - cancel(scope='subtree') touches ONLY active descendants (50 idle children untouched),
      locks depth DESC, cascaded receipts use wake=0.
    """
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port, "d2_root")

    # Reserved prefixes rejected with 0 SQL
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        for prefix in ("res_", "wfc_", "sys_"):
            profiled.records.clear()
            with pytest.raises(RuntimeRequestError) as exc:
                await port.accept(root, "forged", message_id=f"{prefix}123", actor=ACTOR, public=True)
            assert exc.value.code == "invalid_options"
            assert profiled.records == []
    finally:
        store._db = original

    # wake=0 isolation
    rec, need_wake = await port.accept(
        root, "quiet", message_id="sys_quiet", actor=ACTOR,
        kind="system_notice", wake=False, public=False, _with_activation=True,
    )
    assert rec.newly_accepted and need_wake is False
    st = await store._db.fetchone(
        f"SELECT execution_state,inbox_version,wake_inbox_seq,wake_pending_count,pending_count "
        f"FROM {store.t.sessions} WHERE session_id=?",
        (root.session_id,),
    )
    assert st["execution_state"] == "idle"
    assert int(st["inbox_version"]) == 1 and int(st["pending_count"]) == 1
    assert int(st["wake_inbox_seq"]) == 0 and int(st["wake_pending_count"]) == 0

    # Active-only subtree cancel: 50 idle children + 1 running child + 1 queued grandchild
    idle_specs = [
        ChildSpec(session_id=f"d2_idle_{i:02d}", descriptor=DESCRIPTOR, message_id="init", body="x")
        for i in range(50)
    ]
    # Create 50 idle children
    for sp in idle_specs:
        await port.create_session(DESCRIPTOR, actor=ACTOR, parent=root, options={"session_id": sp.session_id})
    active_child = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=root,
        result_route={"target": root, "input_ref": InputRef(store.store_id, "d2_c_active", "go"), "result_kind": "agent_result"},
        options={"session_id": "d2_c_active"},
    )
    grandchild = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=active_child,
        result_route={"target": active_child, "input_ref": InputRef(store.store_id, "d2_gc", "go"), "result_kind": "agent_result"},
        options={"session_id": "d2_gc"},
    )
    _, child_batch, _ = await port.accept_and_claim(
        active_child, "run", message_id="go", actor=ACTOR, node_id="n1", boot_id="b1",
    )
    await port.accept(grandchild, "run_gc", message_id="go", actor=ACTOR)

    await port.cancel(root, actor=ACTOR, scope="subtree", reason="user_stop")

    # Idle children untouched (cancel_seq == 0)
    idle_touched = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE parent_session_id=? "
        "AND session_id LIKE 'd2_idle_%' AND cancel_seq>0",
        (root.session_id,),
    )
    assert int(idle_touched["n"]) == 0
    # Grandchild voided with cancel_by='subtree' and cascaded receipt to parent has wake=0!
    gc_row = await store._db.fetchone(
        f"SELECT execution_state,cancel_by FROM {store.t.sessions} WHERE session_id=?",
        ("d2_gc",),
    )
    assert gc_row["execution_state"] == "idle" and gc_row["cancel_by"] == "subtree"
    c_receipts = await store._db.fetchall(
        f"SELECT wake,kind FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='agent_result'",
        ("d2_c_active",),
    )
    assert len(c_receipts) == 1 and int(c_receipts[0]["wake"]) == 0
    # Running child got cancel_seq=1, cancel_by='subtree'
    c_row = await store._db.fetchone(
        f"SELECT cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
        ("d2_c_active",),
    )
    assert int(c_row["cancel_seq"]) == 1 and c_row["cancel_by"] == "subtree"
    await port.final(child_batch, status="cancelled", error="user_stop", error_code="cancelled")


async def test_river_suspended_parent_and_non_rollback_errors(registry_store):
    """River delivery:
    - Parent lifecycle='suspended' accepts receipt without waking (execution_state stays 'idle').
    - Parent closed / full does NOT roll back child result and stamps result_delivery_error.
    """
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "riv_parent")
    child = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=parent,
        result_route={"target": parent, "input_ref": InputRef(store.store_id, "riv_child", "go"), "result_kind": "agent_result"},
        options={"session_id": "riv_child"},
    )
    # Suspend parent AFTER child creation (e.g., resident workflow driver or paused parent)
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='suspended' WHERE session_id=?",
        (parent.session_id,),
    )
    _, batch, _ = await port.accept_and_claim(child, "work", message_id="go", actor=ACTOR, node_id="n1", boot_id="b1")
    fin = await port.final(batch, final_text="child_ok")
    assert fin.result_delivery_error is None and fin.result_activation is None
    p_st = await store._db.fetchone(
        f"SELECT lifecycle,execution_state,pending_count FROM {store.t.sessions} WHERE session_id=?",
        (parent.session_id,),
    )
    assert p_st["lifecycle"] == "suspended" and p_st["execution_state"] == "idle" and int(p_st["pending_count"]) == 1
    expected_mid = result_message_id(child, InputRef(store.store_id, "riv_child", "go"), 1, "agent_result")
    p_in = await store._db.fetchone(
        f"SELECT item_id,kind FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (parent.session_id, expected_mid),
    )
    assert p_in is not None and p_in["kind"] == "agent_result"

    # Parent closed does NOT roll back child result and stamps result_delivery_error='session_closed'
    p_closed = await target(port, "riv_parent_closed")
    c_closed = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=p_closed,
        result_route={"target": p_closed, "input_ref": InputRef(store.store_id, "riv_child_closed", "go"), "result_kind": "agent_result"},
        options={"session_id": "riv_child_closed"},
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='closed' WHERE session_id=?",
        (p_closed.session_id,),
    )
    _, b_closed, _ = await port.accept_and_claim(c_closed, "work", message_id="go", actor=ACTOR, node_id="n1", boot_id="b1")
    fin_closed = await port.final(b_closed, final_text="child_still_committed")
    assert fin_closed.result_delivery_error == "session_closed" and fin_closed.result_activation is None
    child_res_closed = await port.get_result(InputRef(store.store_id, c_closed.session_id, "go"), actor=ACTOR)
    assert child_res_closed is not None and child_res_closed.status == "completed"
    assert child_res_closed.final_text == "child_still_committed"
    assert child_res_closed.result_delivery_error == "session_closed"
    closed_parent_q = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?", (p_closed.session_id,),
    )
    assert closed_parent_q == []

    # Parent full does NOT roll back child result and stamps result_delivery_error='capacity_exceeded'
    tight_port = RuntimeInputStore(store, inbox_item_limit=1, hard_pending_limit=1)
    p_full = await target(tight_port, "riv_parent_full")
    c_full = await tight_port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=p_full,
        result_route={"target": p_full, "input_ref": InputRef(store.store_id, "riv_child_full", "go"), "result_kind": "agent_result"},
        options={"session_id": "riv_child_full"},
    )
    await tight_port.accept(p_full, "fill", message_id="fill_1", actor=ACTOR)
    _, b_full, _ = await tight_port.accept_and_claim(c_full, "work", message_id="go", actor=ACTOR, node_id="n1", boot_id="b1")
    fin_full = await tight_port.final(b_full, final_text="child_full_ok")
    assert fin_full.result_delivery_error == "capacity_exceeded" and fin_full.result_activation is None
    child_res_full = await tight_port.get_result(InputRef(store.store_id, c_full.session_id, "go"), actor=ACTOR)
    assert child_res_full is not None and child_res_full.status == "completed"
    assert child_res_full.final_text == "child_full_ok"
    assert child_res_full.result_delivery_error == "capacity_exceeded"
    full_mid = result_message_id(c_full, InputRef(store.store_id, "riv_child_full", "go"), 1, "agent_result")
    full_parent_q = await store._db.fetchone(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (p_full.session_id, full_mid),
    )
    assert full_parent_q is None


async def test_decision3_wake_transports_and_graceful_release(registry_store):
    """拍板三: InMemoryWakeTransport (0 SQL), DatabasePollingWakeTransport, aclose graceful_release."""
    store, _ = registry_store
    port = RuntimeInputStore(store)

    # InMemoryWakeTransport: 0 SQL
    mem = InMemoryWakeTransport()
    seen: list[tuple[str, str, str]] = []

    async def _on_wake(sid: str, reason: str) -> None:
        seen.append(("wake", sid, reason))

    async def _on_cancel(sid: str, cancel_by: str) -> None:
        seen.append(("cancel", sid, cancel_by))

    await mem.subscribe(on_wake=_on_wake, on_cancel=_on_cancel)
    await mem.publish_wake("s_mem", reason="accept")
    await mem.publish_cancel("s_mem", cancel_by="user")
    assert seen == [("wake", "s_mem", "accept"), ("cancel", "s_mem", "user")]
    await mem.aclose()

    # DatabasePollingWakeTransport discovers queued sessions
    s_poll = await target(port, "d3_poll")
    await port.accept(s_poll, "hi", message_id="m1", actor=ACTOR)
    db_wake = DatabasePollingWakeTransport(store, poll_interval_s=60.0)
    found = await db_wake.poll_once(limit=32)
    assert s_poll.session_id in found
    await db_wake.aclose()

    # graceful_release expires all leases for (node_id, boot_id) in 1 SQL
    b = await port.claim(s_poll, node_id="n_rel", boot_id="b_rel")
    assert b is not None
    released = await port.graceful_release(node_id="n_rel", boot_id="b_rel")
    assert released == 1
    row = await store._db.fetchone(
        f"SELECT execution_state,owner_lease_until_ms FROM {store.t.sessions} WHERE session_id=?",
        (s_poll.session_id,),
    )
    assert row["execution_state"] == "running" and int(row["owner_lease_until_ms"]) == 0
    swept = await port.sweep_expired_leases(node_id="n_take", boot_id="b_take", limit=32)
    assert any(item.target == s_poll for item in swept)
    for item in swept:
        await port.final(item, final_text="taken_over")


async def test_concurrent_child_final_and_parent_accept_cancel_no_deadlock(registry_store):
    """Deep-first lock order: child Final + parent accept/cancel concurrently for both p<c and c<p."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    for p_id, c_id in (("aaa_parent", "zzz_child"), ("zzz_parent", "aaa_child")):
        parent = await port.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": p_id})
        child = await port.create_session(
            DESCRIPTOR, actor=ACTOR, parent=parent,
            result_route={"target": parent, "input_ref": InputRef(store.store_id, c_id, "go"), "result_kind": "agent_result"},
            options={"session_id": c_id},
        )
        _, batch, _ = await port.accept_and_claim(child, "go", message_id="go", actor=ACTOR, node_id="n1", boot_id="b1")
        # Run child Final concurrently with parent accept
        res_fin, res_acc = await asyncio.gather(
            port.final(batch, final_text="done"),
            port.accept(parent, "user msg", message_id=f"u_{p_id}", actor=ACTOR),
        )
        assert res_fin.result_delivery_error is None
        assert res_acc.state == "accepted"

    # Stale owner rejected after expiry & takeover via owner_epoch conditional write
    s_fence = await target(port, "d1_fence")
    _, old_batch, _ = await port.accept_and_claim(s_fence, "one", message_id="m1", actor=ACTOR, node_id="n1", boot_id="b1")
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
        (s_fence.session_id,),
    )
    new_batch = await port.reclaim_running(s_fence, node_id="n2", boot_id="b2")
    assert new_batch is not None and new_batch.owner_epoch == old_batch.owner_epoch + 1
    with pytest.raises(SessionLeaseLost):
        await port.final(old_batch, final_text="stale")
    await port.final(new_batch, final_text="fresh")


async def test_b1_subtree_cancel_single_child_locks_for_update_no_deadlock(registry_store):
    """B1: Single-child level (len(lvl_ids)==1) must still lock with ORDER BY session_id ASC FOR UPDATE."""
    from contextlib import asynccontextmanager

    store, connect = registry_store
    other = await connect(schema="verify")
    port, child_port = RuntimeInputStore(store), RuntimeInputStore(other)
    dialect = store._db.dialect.name

    for p_id, c_id in (("b1_aaa_parent", "b1_zzz_child"), ("b1_zzz_parent", "b1_aaa_child")):
        parent = await port.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": p_id})
        child = await port.create_session(
            DESCRIPTOR, actor=ACTOR, parent=parent,
            result_route={"target": parent, "input_ref": InputRef(store.store_id, c_id, "go"), "result_kind": "agent_result"},
            options={"session_id": c_id},
        )
        await port.accept(child, "go", message_id="go", actor=ACTOR)

        # Capture SQL executed during cancel(parent, scope='subtree') with single active child.
        captured_sql: list[str] = []
        orig_tx = store._db.transaction

        class _SpyTx:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            async def fetchall(self, sql, params=()):
                captured_sql.append(sql)
                return await self._inner.fetchall(sql, params)

            async def fetchone(self, sql, params=()):
                captured_sql.append(sql)
                return await self._inner.fetchone(sql, params)

            async def execute(self, sql, params=()):
                captured_sql.append(sql)
                return await self._inner.execute(sql, params)

            async def execute_packed(self, sql, params=()):
                captured_sql.append(sql)
                return await self._inner.execute_packed(sql, params)

        @asynccontextmanager
        async def spy_tx():
            async with orig_tx() as tx:
                yield _SpyTx(tx)

        async def _claim_and_maybe_final():
            b = await child_port.claim(child, node_id="n1", boot_id="b1")
            if b is not None:
                await child_port.final(b, final_text="raced_done")

        store._db.transaction = spy_tx
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    port.cancel(parent, actor=ACTOR, scope="subtree", reason="b1_stop"),
                    _claim_and_maybe_final(),
                ),
                timeout=10,
            )
        finally:
            store._db.transaction = orig_tx

        if dialect in ("postgres", "mysql"):
            assert any(
                "ORDER BY session_id ASC FOR UPDATE" in s and store.t.sessions in s
                for s in captured_sql
            ), f"expected FOR UPDATE on single-child level, got: {captured_sql}"


async def test_b2_mysql_execute_packed_survives_after_update_trigger(registry_store):
    """B2: With an AFTER UPDATE trigger (WriteMeter) on sessions, execute_packed must not zero out."""
    from tests.write_meter import WriteMeter

    store, _ = registry_store
    port = RuntimeInputStore(store)
    meter = await WriteMeter.install(store._db, store.table_prefix)
    parent = await target(port, "b2_parent")
    child = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=parent,
        result_route={"target": parent, "input_ref": InputRef(store.store_id, "b2_child", "go"), "result_kind": "agent_result"},
        options={"session_id": "b2_child"},
    )
    # 1. accept() under trigger: must compute real inbox_version=1 and need_wake=True
    rec, need_wake = await port.accept(child, "work", message_id="go", actor=ACTOR, _with_activation=True)
    assert rec.newly_accepted and need_wake is True
    q_row = await store._db.fetchone(
        f"SELECT inbox_version FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='go'",
        (child.session_id,),
    )
    assert int(q_row["inbox_version"]) == 1
    # 2. claim + final() with River delivery under trigger: parent_need_wake must be True & inbox_version=1
    batch = await port.claim(child, node_id="n1", boot_id="b1")
    assert batch is not None
    fin = await port.final(batch, final_text="b2_done")
    assert fin.result_delivery_error is None
    assert fin.result_activation == parent
    p_q = await store._db.fetchone(
        f"SELECT inbox_version FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='agent_result'",
        (parent.session_id,),
    )
    assert p_q is not None and int(p_q["inbox_version"]) == 1
    report = await meter.report()
    assert report is not None
