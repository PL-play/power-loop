"""Design 217 M1c focused verification suite across SQLite, PostgreSQL 16, and MySQL 8.4.

Covers every M1c engineering item and adversarial audit probe:
  - set_lifecycle('open'|'suspended'|'closed', inflight, subtree) & deactivate thin alias
  - suspended(drain): tool round writes + Final complete without losing reply, child receipts
    buffer without waking, 'open' resumes & activates pending wakes (Probe 8A)
  - Strategy 3A: set_lifecycle(main, 'closed', inflight='abort', subtree='all') soft-closes
    without deleting rows, stops active descendants in depth DESC order without touching idle
    descendants, River 0-row intercept marks session_closed & fires RESULT_UNDELIVERABLE post-commit
  - rebind_children: 3 hard invariants (sync tool wait, tenant/principal + closed to_parent +
    cycle immunity), BFS recursive root_session_id/depth refresh including idle grandchildren,
    rewrite_host_context callback, atomic replay of session_closed orphan receipts (Probe 8B)
  - 4 HookPoints wired at real call sites: EXECUTION_LOST (immediate kill + DB fence loss),
    SESSION_TAKEOVER (post-commit with prev node/boot/epoch), TURN_REPAIR (takeover & repair),
    RESULT_UNDELIVERABLE (post-commit on closed & capacity_exceeded)
  - closing.py (no terminal_status IS NULL check) & retention.py (RetentionPage contract,
    purges expired delivered rows without final_settle_inbox, idx_sessions_lifecycle paging) (Probe 9)
  - P1-4: background_execution.py deletes 50ms DB boundary polling; stop_event + batch_renew signals
  - Concurrent stop + set_lifecycle race (0 deadlocks, consistent closed state)
"""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import fields, replace
from typing import Any

import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    HookPoint,
    LLMResponse,
    LLMService,
    RebindResult,
    ToolDefinition,
    TrustedActor,
)
from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import InputRef, RuntimeRequestError, TargetRef
from power_loop.runtime import background_execution
from power_loop.runtime.pricing import Free
from power_loop.runtime.stop_policy import StopPolicy
from power_loop.runtime.store import closing
from power_loop.runtime.store.retention import RetentionPage
from power_loop.tools import create_default_tool_registry
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent
from tests.unit.session_native_fixture import ACTOR, DESCRIPTOR, target
from tests.unit.test_send_sql_profile import Profiled

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _ScriptedModel(LLMService):
    def __init__(self, handler):
        self.handler = handler
        self.calls = 0

    async def complete(self, request, **kwargs):
        idx = self.calls
        self.calls += 1
        res = self.handler(idx, request)
        if inspect.isawaitable(res):
            return await res
        return res

    def stream(self, request):
        raise AssertionError("nonstream")

    async def close(self):
        return None


# ── 1. suspended(drain) + buffered child + resume on 'open' (Probe 8A) ──────


async def test_m1c_suspended_drain_tool_round_buffered_child_and_open_resume(registry_store):
    store, _ = registry_store
    tool_entered = asyncio.Event()
    tool_continue = asyncio.Event()
    seen_texts: list[str] = []

    async def slow_tool() -> str:
        tool_entered.set()
        await asyncio.wait_for(tool_continue.wait(), 5)
        return "slow_done"

    reg = ToolRegistry()
    reg.register(
        ToolDefinition("slow", "Slow tool", input_schema={"type": "object"}, pricing=Free("f")),
        slow_tool,
    )

    def model_fn(idx, request):
        msgs = request.messages
        if idx == 0:
            return LLMResponse(
                raw_text="calling slow",
                tool_calls=[{"id": "tc_dr", "type": "function", "function": {"name": "slow", "arguments": "{}"}}],
            )
        if idx == 1:
            return LLMResponse(raw_text="drained_ok")
        seen_texts.append(json.dumps(msgs, ensure_ascii=False))
        return LLMResponse(raw_text="resumed_after_child")

    agent = await RunningAgent(
        store, _ScriptedModel(model_fn), reg, AgentLoopConfig(context_budget_tokens=10**9),
    ).start()
    try:
        client = agent.runtime.client
        port = client.inputs
        actor = agent.actor
        child = await port.create_session(
            DESCRIPTOR,
            actor=actor,
            parent=agent.target,
            result_route={
                "target": agent.target,
                "input_ref": InputRef(store.store_id, "susp-child", "c1"),
                "result_kind": "agent_result",
            },
            options={"session_id": "susp-child", "start_input": InputRef(store.store_id, "susp-child", "c1")},
        )
        await port.accept(child, "child work", message_id="c1", actor=actor)
        child_batch = await port.claim(child, node_id="child-node", boot_id="child-boot")
        assert child_batch is not None

        send_task = asyncio.create_task(agent.send("run drain"))
        await asyncio.wait_for(tool_entered.wait(), 5)

        # Suspend with inflight='drain': must NOT bump cancel_seq or abort active turn!
        before_row = await store._db.fetchone(
            f"SELECT cancel_seq FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        susp_status = await client.set_lifecycle(
            agent.target, "suspended", actor=actor, inflight="drain",
        )
        assert susp_status.lifecycle == "suspended"
        assert not susp_status.cancel_requested
        mid_row = await store._db.fetchone(
            f"SELECT lifecycle,cancel_seq FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        assert mid_row["lifecycle"] == "suspended"
        assert int(mid_row["cancel_seq"]) == int(before_row["cancel_seq"])

        # Child finishes while parent is suspended: River buffers receipt, parent does NOT wake!
        child_final = await port.final(child_batch, final_text="child_buffered_payload")
        assert child_final.result_delivery_error is None
        assert child_final.result_activation is None

        # Unblock tool -> tool write + Final complete cleanly without losing reply!
        tool_continue.set()
        turn1 = await asyncio.wait_for(send_task, 5)
        await agent.drain()
        assert turn1.status == "completed"
        assert turn1.final_text == "drained_ok"

        after_drain = await store._db.fetchone(
            f"SELECT lifecycle,execution_state,wake_pending_count,pending_count "
            f"FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        assert after_drain["lifecycle"] == "suspended"
        assert after_drain["execution_state"] == "idle"
        assert int(after_drain["wake_pending_count"]) == 1
        assert len(seen_texts) == 0  # child receipt NOT yet scheduled while suspended!

        # Resume -> transitions idle -> queued with need_wake=True, worker drains child receipt!
        open_status = await client.set_lifecycle(agent.target, "open", actor=actor)
        assert open_status.lifecycle == "open"
        assert open_status.need_wake is True
        await agent.drain(after_input=child_final.result_ref)
        assert any("child_buffered_payload" in txt for txt in seen_texts)

        # suspended(inflight='abort') bumps cancel_seq & rejects closed -> open
        await client.set_lifecycle(agent.target, "suspended", actor=actor, inflight="abort", reason="pause_now")
        ab_row = await store._db.fetchone(
            f"SELECT lifecycle,cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        assert ab_row["lifecycle"] == "suspended" and ab_row["cancel_by"] == "suspended"
        assert int(ab_row["cancel_seq"]) == int(mid_row["cancel_seq"]) + 1

        closed_st = await client.set_lifecycle(agent.target, "closed", actor=actor, reason="done")
        assert closed_st.lifecycle == "closed"
        with pytest.raises(RuntimeRequestError, match="session_closed"):
            await client.set_lifecycle(agent.target, "open", actor=actor)
    finally:
        tool_continue.set()
        await agent.aclose()


# ── 2. Strategy 3A (closed + abort + subtree='all') + RESULT_UNDELIVERABLE ──


async def test_m1c_strategy_3a_closed_abort_subtree_and_result_undeliverable_hook(registry_store):
    store, connect = registry_store
    hooks = AgentHooks()
    undeliverable_events: list[dict[str, Any]] = []
    committed_seen_from_peer: list[dict[str, Any]] = []

    async def on_undeliverable(ctx):
        ev = dict(ctx.values)
        undeliverable_events.append(ev)
        # Prove hook fires strictly AFTER transaction commit by reading on a separate connection!
        peer = await connect(schema="verify")
        try:
            row = await peer._db.fetchone(
                f"SELECT terminal_status,result_delivery_error FROM {peer.t.follow_up_queue} "
                "WHERE session_id=? ORDER BY id ASC LIMIT 1",
                (ev["child_session_id"],),
            )
            committed_seen_from_peer.append(dict(row) if row is not None else {})
        finally:
            await peer.close()

    hooks.register(HookPoint.RESULT_UNDELIVERABLE, on_undeliverable)
    port = RuntimeInputStore(store, hooks=hooks, inbox_item_limit=2, hard_pending_limit=2)

    main = await target(port, "s3a-main")
    # Create 2 idle children (must NOT be touched by subtree='all' scan!)
    idle_c1 = await target(port, "s3a-idle-1", parent=main)
    idle_c2 = await target(port, "s3a-idle-2", parent=main)
    # Create active child c1 (depth 1) + active grandchild g1 (depth 2)
    c1_start = InputRef(store.store_id, "s3a-c1", "c1-in")
    c1 = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=main,
        result_route={"target": main, "input_ref": c1_start, "result_kind": "agent_result"},
        cancel_propagation="independent",  # subtree='all' stops independent too!
        options={"session_id": "s3a-c1", "start_input": c1_start},
    )
    g1_start = InputRef(store.store_id, "s3a-g1", "g1-in")
    g1 = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=c1,
        result_route={"target": c1, "input_ref": g1_start, "result_kind": "agent_result"},
        options={"session_id": "s3a-g1", "start_input": g1_start},
    )
    # Start main, c1, g1 (all running) and queue a pending input on main
    await port.accept(main, "main work", message_id="m1", actor=ACTOR)
    main_batch = await port.claim(main, node_id="n1", boot_id="b1")
    await port.accept(main, "main pending", message_id="m2", actor=ACTOR)
    await port.accept(c1, "c1 work", message_id="c1-in", actor=ACTOR)
    c1_batch = await port.claim(c1, node_id="n1", boot_id="b1")
    await port.accept(g1, "g1 work", message_id="g1-in", actor=ACTOR)
    g1_batch = await port.claim(g1, node_id="n1", boot_id="b1")
    assert main_batch and c1_batch and g1_batch

    # Execute DeepTalk Strategy 3A: set_lifecycle(main, 'closed', inflight='abort', subtree='all')
    st = await port.set_lifecycle(
        main, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="agent_uninstalled",
    )
    assert st.lifecycle == "closed"
    assert st.voided_pending_inputs == 1
    assert st.stopped_descendant_ids == ("s3a-g1", "s3a-c1")  # depth DESC!

    # Main row MUST NOT be deleted!
    main_row = await store._db.fetchone(
        f"SELECT lifecycle,cancel_seq,cancel_by,pending_count,wake_pending_count "
        f"FROM {store.t.sessions} WHERE session_id=?",
        (main.session_id,),
    )
    assert main_row is not None and main_row["lifecycle"] == "closed"
    assert main_row["cancel_by"] == "removed"
    assert int(main_row["pending_count"]) == 0 and int(main_row["wake_pending_count"]) == 0

    # Idle descendants were NOT scanned or modified!
    for idle_ref in (idle_c1, idle_c2):
        irow = await store._db.fetchone(
            f"SELECT cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
            (idle_ref.session_id,),
        )
        assert int(irow["cancel_seq"]) == 0 and irow["cancel_by"] is None

    # Main final produces 0 ghost reply (closed_raced -> cancelled, no assistant message written)
    main_fin = await port.final(main_batch, final_text="GHOST_REPLY_FORBIDDEN")
    assert main_fin.results[0].status == "cancelled"
    main_asst = await store._db.fetchall(
        f"SELECT * FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
        (main.session_id,),
    )
    assert main_asst == []

    # In-flight child c1 completes Final -> cancelled(subtree), River 0-row intercept -> session_closed
    # AND fires HookPoint.RESULT_UNDELIVERABLE strictly AFTER commit!
    c1_fin = await port.final(c1_batch, final_text="late child text")
    assert c1_fin.results[0].status == "cancelled"
    assert c1_fin.result_delivery_error == "session_closed"
    assert c1_fin.result_activation is None
    assert len(undeliverable_events) == 1
    assert undeliverable_events[0]["child_session_id"] == "s3a-c1"
    assert undeliverable_events[0]["parent_session_id"] == "s3a-main"
    assert undeliverable_events[0]["reason"] == "session_closed"
    assert committed_seen_from_peer == [{"terminal_status": "cancelled", "result_delivery_error": "session_closed"}]

    # Also verify capacity_exceeded triggers RESULT_UNDELIVERABLE post-commit + deactivate alias!
    full_parent = await target(port, "s3a-full-parent")
    await port.accept(full_parent, "p1", message_id="fp1", actor=ACTOR)
    await port.accept(full_parent, "p2", message_id="fp2", actor=ACTOR)
    fc_start = InputRef(store.store_id, "s3a-full-child", "fc1")
    full_child = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=full_parent,
        result_route={"target": full_parent, "input_ref": fc_start, "result_kind": "agent_result"},
        options={"session_id": "s3a-full-child", "start_input": fc_start},
    )
    await port.accept(full_child, "go", message_id="fc1", actor=ACTOR)
    fc_batch = await port.claim(full_child, node_id="n1", boot_id="b1")
    fc_fin = await port.final(fc_batch, final_text="child survived overflow")
    assert fc_fin.results[0].status == "completed"
    assert fc_fin.result_delivery_error == "capacity_exceeded"
    assert undeliverable_events[-1]["reason"] == "capacity_exceeded"
    assert committed_seen_from_peer[-1]["result_delivery_error"] == "capacity_exceeded"

    # deactivate thin alias soft-closes without deleting rows
    deact = await port.deactivate(full_parent, actor=ACTOR, stop_subtree=True, reason="deact_test")
    assert deact.lifecycle == "closed" and deact.voided_pending_inputs == 2
    assert await store.get_session(full_parent.session_id) is not None


# ── 3. rebind_children: 3 hard invariants, idle grandchildren, rewrite & replay (Probe 8B) ──


async def test_m1c_rebind_children_three_invariants_subtree_rewrite_and_replay(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)

    old_root = await target(port, "rb-old-root")
    new_top = await target(port, "rb-new-top")
    # Place new_parent at depth=1 under new_top so depth_delta = +1 (non-zero relative shift!)
    new_parent = await target(port, "rb-new-parent", parent=new_top)

    # Foreign tenant/principal target for negative invariant check
    foreign_actor = TrustedActor("other-tenant", "other-user", None)
    foreign_parent = await port.create_session(
        DESCRIPTOR, actor=foreign_actor, options={"session_id": "rb-foreign"},
    )
    closed_parent = await target(port, "rb-closed-target")
    await port.set_lifecycle(closed_parent, "closed", actor=ACTOR)

    # 3-level subtree under old_root: mid_child (depth 1) -> idle_grand (depth 2) -> idle_great (depth 3)
    mid_desc = replace(DESCRIPTOR, host_context={"owner": "old", "tier": "mid"})
    grand_desc = replace(DESCRIPTOR, host_context={"owner": "old", "tier": "grand"})
    great_desc = replace(DESCRIPTOR, host_context={"owner": "old", "tier": "great"})
    mid_start = InputRef(store.store_id, "rb-mid", "mid-in")
    mid = await port.create_session(
        mid_desc, actor=ACTOR, parent=old_root,
        result_route={"target": old_root, "input_ref": mid_start, "result_kind": "agent_result"},
        options={"session_id": "rb-mid", "start_input": mid_start},
    )
    idle_grand = await port.create_session(
        grand_desc, actor=ACTOR, parent=mid, options={"session_id": "rb-idle-grand"},
    )
    await port.create_session(
        great_desc, actor=ACTOR, parent=idle_grand, options={"session_id": "rb-idle-great"},
    )

    # Negative Invariant 1a: running from_parent with pending_tool_seq -> rejected!
    await port.accept(old_root, "sync wait", message_id="or1", actor=ACTOR)
    or_batch = await port.claim(old_root, node_id="n1", boot_id="b1")
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET pending_tool_seq=2 WHERE session_id=?",
        (old_root.session_id,),
    )
    with pytest.raises(RuntimeRequestError, match="request_conflict"):
        await port.rebind_children(old_root, new_parent, actor=ACTOR)

    # Negative Invariant 1b: running from_parent with dangling assistant tool_call in messages
    # even when pending_tool_seq IS NULL (Session-Core fenced fast path) -> ALSO rejected!
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET pending_tool_seq=NULL WHERE session_id=?",
        (old_root.session_id,),
    )
    await store.append_message_fenced(
        old_root.session_id, owner_epoch=or_batch.owner_epoch, seq=2, send_index=1,
        round_index=0, role="assistant", content="waiting tool",
        tool_calls=[{"id": "tc_sync", "type": "function", "function": {"name": "wait_child", "arguments": "{}"}}],
    )
    with pytest.raises(RuntimeRequestError, match="request_conflict"):
        await port.rebind_children(old_root, new_parent, actor=ACTOR)

    # Finish old_root cleanly before negative Invariant 2 tests
    await port.final(or_batch, final_text="done")

    # Negative Invariant 2a: cross-tenant/principal forbidden
    with pytest.raises(RuntimeRequestError, match="forbidden"):
        await port.rebind_children(old_root, foreign_parent, actor=ACTOR)
    # Negative Invariant 2b: closed to_parent rejected
    with pytest.raises(RuntimeRequestError, match="session_closed"):
        await port.rebind_children(old_root, closed_parent, actor=ACTOR)
    # Negative Invariant 2c: cycle detection (self & descendant!)
    with pytest.raises(RuntimeRequestError, match="request_conflict"):
        await port.rebind_children(old_root, old_root, actor=ACTOR)
    with pytest.raises(RuntimeRequestError, match="request_conflict"):
        await port.rebind_children(old_root, idle_grand, actor=ACTOR)

    # Positive Probe 8B: mid_child runs while old_root is deactivated -> session_closed orphan receipt!
    await port.accept(mid, "child task", message_id="mid-in", actor=ACTOR)
    mid_batch = await port.claim(mid, node_id="n1", boot_id="b1")
    await port.deactivate(old_root, actor=ACTOR, stop_subtree=False, reason="removed")
    mid_fin = await port.final(mid_batch, final_text="orphan_child_answer")
    assert mid_fin.results[0].status == "completed"
    assert mid_fin.result_delivery_error == "session_closed"

    rewritten_calls: list[tuple[str, str]] = []

    def rewrite_cb(child_row: dict[str, Any], parent_row: dict[str, Any]) -> dict[str, Any]:
        rewritten_calls.append((child_row["session_id"], parent_row["session_id"]))
        return {**child_row["host_context"], "owner": "rebound_host", "new_parent": parent_row["session_id"]}

    res = await port.rebind_children(
        old_root, new_parent, actor=ACTOR, replay_undelivered=True, rewrite_host_context=rewrite_cb,
    )
    assert isinstance(res, RebindResult)
    assert res.rebound_children == ("rb-mid",)
    assert set(res.updated_descendants) == {"rb-mid", "rb-idle-grand", "rb-idle-great"}
    assert res.new_root_session_id == "rb-new-top"
    assert res.depth_delta == 1
    assert res.replayed_results == 1
    assert res.need_wake is True
    assert {sid for sid, _ in rewritten_calls} == {"rb-mid", "rb-idle-grand", "rb-idle-great"}

    # Verify ALL descendants (including idle_grand and idle_great) updated root, depth, host_context!
    expected_depths = {"rb-mid": 2, "rb-idle-grand": 3, "rb-idle-great": 4}
    for sid, exp_depth in expected_depths.items():
        row = await store._db.fetchone(
            f"SELECT parent_session_id,root_session_id,depth,descriptor_json,result_route_json "
            f"FROM {store.t.sessions} WHERE session_id=?",
            (sid,),
        )
        assert row["root_session_id"] == "rb-new-top"
        assert int(row["depth"]) == exp_depth
        hc = json.loads(row["descriptor_json"])["host_context"]
        assert hc["owner"] == "rebound_host" and hc["new_parent"] == "rb-new-parent"
        if sid == "rb-mid":
            assert row["parent_session_id"] == "rb-new-parent"
            assert json.loads(row["result_route_json"])["target"]["session_id"] == "rb-new-parent"

    # Verify orphan receipt replayed atomically to new_parent and error marker cleared!
    mid_q = await store._db.fetchone(
        f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        ("rb-mid", "mid-in"),
    )
    assert mid_q["result_delivery_error"] is None

    np_row = await store._db.fetchone(
        f"SELECT execution_state,wake_pending_count,pending_count FROM {store.t.sessions} WHERE session_id=?",
        ("rb-new-parent",),
    )
    assert np_row["execution_state"] == "queued"
    assert int(np_row["wake_pending_count"]) == 1
    np_inbox = await store._db.fetchall(
        f"SELECT item_id,kind,content_json FROM {store.t.follow_up_queue} WHERE session_id=?",
        ("rb-new-parent",),
    )
    assert len(np_inbox) == 1
    assert np_inbox[0]["kind"] == "agent_result"
    body = json.loads(np_inbox[0]["content_json"])
    assert "orphan_child_answer" in body["final_text"]
    assert "rb-old-root" in body["final_text"]


# ── 4. 4 HookPoints real wiring: EXECUTION_LOST, SESSION_TAKEOVER, TURN_REPAIR ──


async def test_m1c_hooks_execution_lost_immediate_kill_and_db_fence(registry_store):
    """EXECUTION_LOST fires immediately on lease loss to kill a blocked sandbox tool,
    and also fires when a DB fence raises SessionLeaseLost."""
    store, connect = registry_store
    hooks = AgentHooks()
    tool_entered = asyncio.Event()
    sandbox_killed = asyncio.Event()
    lost_events: list[dict[str, Any]] = []

    async def on_lost(ctx):
        lost_events.append(dict(ctx.values))
        sandbox_killed.set()  # kills the blocked sandbox immediately!

    hooks.register(HookPoint.EXECUTION_LOST, on_lost)

    async def blocked_sandbox_tool() -> str:
        tool_entered.set()
        # Intentionally ignores ctx.stop until EXECUTION_LOST kills the sandbox!
        await asyncio.wait_for(sandbox_killed.wait(), 5)
        return "killed"

    reg = ToolRegistry()
    reg.register(
        ToolDefinition("sandbox", "Blocked sandbox", input_schema={"type": "object"}, pricing=Free("f")),
        blocked_sandbox_tool,
    )

    def model_fn(idx, _req):
        if idx == 0:
            return LLMResponse(
                raw_text="",
                tool_calls=[{"id": "tc_sb", "type": "function", "function": {"name": "sandbox", "arguments": "{}"}}],
            )
        return LLMResponse(raw_text="should_not_finish")

    agent = await RunningAgent(
        store, _ScriptedModel(model_fn), reg,
        AgentLoopConfig(context_budget_tokens=10**9), hooks=hooks,
    ).start()
    try:
        # Ensure runtime.hooks is bound (RunningAgent passes hooks to env; bind to runtime too)
        agent.runtime.hooks = hooks
        send_task = asyncio.create_task(agent.send("run sandbox"))
        await asyncio.wait_for(tool_entered.wait(), 5)

        # Another node steals the expired lease -> batch_renew detects epoch loss ->
        # EXECUTION_LOST fires IMMEDIATELY without waiting for blocked_sandbox_tool to finish!
        peer = await connect(schema="verify")
        try:
            await peer._db.execute(
                f"UPDATE {peer.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
                (agent.target.session_id,),
            )
            reclaimed = await RuntimeInputStore(peer).reclaim_running(
                agent.target, node_id="thief", boot_id="thief-boot",
            )
            assert reclaimed is not None
        finally:
            await peer.close()

        await agent.runtime.worker.batch_renew()
        await asyncio.wait_for(sandbox_killed.wait(), 3)
        await asyncio.wait_for(agent.drain(), 5)
        send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)

        assert len(lost_events) == 1
        assert lost_events[0]["session_id"] == agent.target.session_id
        assert lost_events[0]["owner_epoch"] >= 1
    finally:
        sandbox_killed.set()
        await agent.aclose()


async def test_m1c_hook_execution_lost_fires_on_db_fence_session_lease_lost(registry_store):
    """Even without a renewal tick, a DB-fence SessionLeaseLost triggers HookPoint.EXECUTION_LOST."""
    store, connect = registry_store
    hooks = AgentHooks()
    lost_events: list[dict[str, Any]] = []
    model_entered = asyncio.Event()
    model_continue = asyncio.Event()

    async def on_lost(ctx):
        lost_events.append(dict(ctx.values))

    hooks.register(HookPoint.EXECUTION_LOST, on_lost)

    async def model_fn(_idx, _req):
        model_entered.set()
        await asyncio.wait_for(model_continue.wait(), 5)
        return LLMResponse(raw_text="stale_reply")

    agent = await RunningAgent(
        store, _ScriptedModel(model_fn), ToolRegistry(),
        AgentLoopConfig(context_budget_tokens=10**9), hooks=hooks,
    ).start()
    try:
        agent.runtime.hooks = hooks
        send_task = asyncio.create_task(agent.send("fence test"))
        await asyncio.wait_for(model_entered.wait(), 5)

        # Steal the lease on another connection WITHOUT calling batch_renew on the old worker;
        # when the model returns, the old worker hits SessionLeaseLost at the DB fence!
        peer = await connect(schema="verify")
        try:
            await peer._db.execute(
                f"UPDATE {peer.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
                (agent.target.session_id,),
            )
            reclaimed = await RuntimeInputStore(peer).reclaim_running(
                agent.target, node_id="thief2", boot_id="thief2-boot",
            )
            assert reclaimed is not None
        finally:
            await peer.close()

        model_continue.set()
        await asyncio.wait_for(agent.drain(), 5)
        send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)

        assert len(lost_events) == 1
        assert lost_events[0]["session_id"] == agent.target.session_id
    finally:
        model_continue.set()
        await agent.aclose()


async def test_m1c_hooks_session_takeover_and_turn_repair(registry_store):
    """SESSION_TAKEOVER fires post-commit on takeover; TURN_REPAIR hooks into M1b Line 1 repair."""
    store, _ = registry_store
    hooks = AgentHooks()
    takeovers: list[dict[str, Any]] = []
    repairs_seen: list[dict[str, Any]] = []

    async def on_takeover(ctx):
        takeovers.append(dict(ctx.values))

    async def on_turn_repair(ctx):
        repairs_seen.append(dict(ctx.values))
        dangling = ctx.values["dangling"]
        rec = dict(ctx.values["recovered"])
        for tc in dangling:
            rec[str(tc["id"])] = f"HOST_FACT:{tc['id']}"
        ctx.values["recovered"] = rec

    hooks.register(HookPoint.SESSION_TAKEOVER, on_takeover)
    hooks.register(HookPoint.TURN_REPAIR, on_turn_repair)

    port = RuntimeInputStore(store, hooks=hooks)
    sess = await target(port, "hk-takeover")
    await port.accept(sess, "run tools", message_id="tk1", actor=ACTOR)
    batch = await port.claim(sess, node_id="dead-node", boot_id="dead-boot")
    assert batch is not None
    await store.append_message_fenced(
        sess.session_id, owner_epoch=batch.owner_epoch, seq=2, send_index=1, round_index=0,
        role="assistant", content="calling tool",
        tool_calls=[{"id": "tc_tk_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}],
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
        (sess.session_id,),
    )

    reclaimed_list = await port.sweep_expired_leases(node_id="alive-node", boot_id="alive-boot")
    assert len(reclaimed_list) == 1
    rec_batch = reclaimed_list[0]
    assert rec_batch.owner_epoch == batch.owner_epoch + 1

    # Verify TURN_REPAIR fired and persisted host_recovered tool message with _assistant_seq
    assert len(repairs_seen) == 1
    assert repairs_seen[0]["session_id"] == "hk-takeover"
    msgs = await store.load_all_messages("hk-takeover")
    tool_rows = [m for m in msgs if m.role == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].content == "HOST_FACT:tc_tk_1"
    assert tool_rows[0].meta["repair_source"] == "host_recovered"
    assert tool_rows[0].meta["_assistant_seq"] == 2

    # Verify SESSION_TAKEOVER fired with previous node/boot and new epoch
    assert len(takeovers) == 1
    assert takeovers[0]["session_id"] == "hk-takeover"
    assert takeovers[0]["previous_node_id"] == "dead-node"
    assert takeovers[0]["previous_boot_id"] == "dead-boot"
    assert takeovers[0]["new_node_id"] == "alive-node"
    assert takeovers[0]["owner_epoch"] == rec_batch.owner_epoch

    # Also verify explicit repair(mode='heal_broken') calls HookPoint.TURN_REPAIR!
    await port.final(rec_batch, final_text="recovered")
    sess2 = await target(port, "hk-repair")
    await port.accept(sess2, "q", message_id="r1", actor=ACTOR)
    b2 = await port.claim(sess2, node_id="n1", boot_id="b1")
    await store.append_message_fenced(
        sess2.session_id, owner_epoch=b2.owner_epoch, seq=2, send_index=1, round_index=0,
        role="assistant", content="dangling",
        tool_calls=[{"id": "tc_heal_2", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
    )
    await port.final(b2, status="interrupted", error="crash", error_code="interrupted")
    rep_res = await port.repair(sess2, mode="heal_broken", actor=ACTOR)
    assert rep_res.repaired_messages == 1
    msgs2 = await store.load_all_messages("hk-repair")
    assert any(m.role == "tool" and m.content == "HOST_FACT:tc_heal_2" for m in msgs2)


# ── 5. Probe 9: closing.py & retention.py without final_settle_inbox ────────


async def test_m1c_probe9_closing_and_retention_without_final_settle_inbox(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)

    # Static check: closing.py must NOT inspect terminal_status IS NULL
    closing_src = inspect.getsource(closing)
    assert "terminal_status IS NULL" not in closing_src
    assert "terminal_status" not in inspect.getsource(closing.busy)

    # Verify RetentionPage dataclass contract unchanged
    assert [f.name for f in fields(RetentionPage)] == ["deleted", "scanned", "fetched", "next_after"]

    # 9A: In-flight workflow_driver -> close_sessions_if_idle MUST return 'busy'
    root = await target(port, "p9-root")
    cm = await port.create_many(
        parent=root,
        actor=ACTOR,
        specs=[ChildSpec(
            session_id="p9-drv",
            descriptor=replace(DESCRIPTOR, kind="workflow_driver"),
            body={"wf": 1},
            message_id="p9-wf-init",
            result_kind="workflow_result",
        )],
        node_id="n1",
        boot_id="b1",
        can_execute=True,
        free_slots=1,
    )
    assert len(cm.preclaimed) == 1
    assert await store.close_sessions_if_idle(["p9-drv"]) == "busy"
    await port.final(cm.preclaimed[0], final_text="wf done", settle_start_input=True)
    assert await store.close_sessions_if_idle(["p9-drv"]) == "closed"
    assert await store.close_sessions_if_idle(["p9-drv"]) == "closed"  # idempotent!

    # 9B: Normal completed idle session (terminal_status IS NULL!) + wake=0 notice -> closes cleanly
    expired_ids = [f"p9-exp-{i}" for i in range(5)]
    old_ts = 1_000_000
    for idx, sid in enumerate(expired_ids):
        ref = await target(port, sid)
        await port.accept(ref, f"hello {idx}", message_id=f"m-{idx}", actor=ACTOR)
        b = await port.claim(ref, node_id="n1", boot_id="b1")
        await port.final(b, final_text=f"reply {idx}")
        # Confirm normal completion left terminal_status IS NULL!
        qrow = await store._db.fetchone(
            f"SELECT status,terminal_status FROM {store.t.follow_up_queue} WHERE session_id=?",
            (sid,),
        )
        assert qrow["status"] == "delivered" and qrow["terminal_status"] is None
        if idx == 0:
            # Add a wake=0 notice to verify close_sessions_if_idle only checks wake_pending_count==0
            await port.accept(
                ref, "silent", message_id="sys_notice_0", actor=ACTOR,
                kind="system_notice", wake=False, public=False,
            )
        assert await store.close_sessions_if_idle([sid]) == "closed"
        assert await store.close_sessions_if_idle([sid]) == "closed"
        async with store._db.transaction() as tx:
            await tx.execute(f"UPDATE {store.t.sessions} SET updated_at=? WHERE session_id=?", (old_ts + idx, sid))
            await tx.execute(
                f"UPDATE {store.t.follow_up_queue} SET delivered_at=?,created_at=? WHERE session_id=?",
                (old_ts, old_ts, sid),
            )
            await tx.execute(
                f"UPDATE {store.t.follow_up_queue} SET completed_at=? WHERE session_id=? AND completed_at IS NOT NULL",
                (old_ts, sid),
            )
            await tx.execute(f"UPDATE {store.t.messages} SET created_at=? WHERE session_id=?", (old_ts, sid))
            await tx.execute(f"UPDATE {store.t.project_messages} SET created_at=? WHERE session_id=?", (old_ts, sid))

    # Active running session & fresh closed session MUST survive retention
    fresh = await target(port, "p9-fresh")
    assert await store.close_sessions_if_idle([fresh.session_id]) == "closed"

    # 9C: Multi-page purge_expired_page cleans ALL 5 expired sessions (including terminal_status IS NULL inbox)
    purge_now = old_ts + 40 * 86_400_000
    total_deleted: dict[str, int] = {}
    cursor: str | None = None
    pages = 0
    for _ in range(20):
        page = await store.purge_expired_page(
            purge_now, limit=100, candidate_limit=2, after_session_id=cursor,
        )
        pages += 1
        assert isinstance(page, RetentionPage)
        assert 0 <= page.scanned <= page.fetched <= 2
        if page.next_after is not None:
            assert isinstance(page.next_after, str) and page.next_after != cursor
        for k, v in page.deleted.items():
            total_deleted[k] = total_deleted.get(k, 0) + v
        cursor = page.next_after
        if cursor is None:
            break
    assert pages >= 3  # paginated across 5 sessions with candidate_limit=2
    assert total_deleted.get("inbox", 0) >= 5
    assert total_deleted.get("sessions", 0) == 5
    for sid in expired_ids:
        assert await store.get_session(sid) is None
    assert await store.get_session(fresh.session_id) is not None

    # Verify candidate query uses idx_sessions_lifecycle on SQLite / PG / MySQL!
    dialect = store._db.dialect.name
    probe_sql = (
        f"SELECT session_id,lifecycle,updated_at FROM {store.t.sessions} "
        "WHERE lifecycle=? AND execution_state='idle' AND updated_at<? "
        "ORDER BY updated_at ASC,session_id ASC LIMIT ?"
    )
    if dialect == "sqlite":
        plan = await store._db.fetchall("EXPLAIN QUERY PLAN " + probe_sql, ("closed", purge_now, 10))
        plan_text = " ".join(str(dict(r)) for r in plan)
        assert "idx_sessions_lifecycle" in plan_text, plan_text
    elif dialect == "postgres":
        async with store._db.transaction() as tx:
            await tx.execute("SET LOCAL enable_seqscan = off")
            plan = await tx.fetchall("EXPLAIN " + probe_sql, ("closed", purge_now, 10))
        plan_text = " ".join(str(dict(r)) for r in plan)
        assert "idx_sessions_lifecycle" in plan_text, plan_text
    elif dialect == "mysql":
        plan = await store._db.fetchall("EXPLAIN " + probe_sql, ("closed", purge_now, 10))
        plan_text = " ".join(str(dict(r)) for r in plan)
        assert "idx_sessions_lifecycle" in plan_text, plan_text


# ── 6. P1-4: background_execution.py zero 50ms DB polling + stop signals ────


async def test_m1c_p1_4_background_execution_zero_db_polling_and_stop_signals(registry_store, tmp_path):
    store, connect = registry_store
    # Static guard: no 50ms boundary polling loop in background_execution.py
    bg_src = inspect.getsource(background_execution.run_session)
    assert "boundary(" not in bg_src
    assert "0.05" not in bg_src

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_bg() -> str:
        started.set()
        await release.wait()
        return "unexpected"

    reg = create_default_tool_registry(workspace_dir=str(tmp_path))
    reg.register(
        ToolDefinition(
            "slow_bg", "Blocking background tool", async_capable=True, interrupt="abort",
            pricing=Free("local"), input_schema={"type": "object"},
        ),
        slow_bg,
    )

    class _BgParent(LLMService):
        def __init__(self):
            self.sent = False

        async def complete(self, request, **kwargs):
            if not self.sent:
                self.sent = True
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "bg-call",
                        "type": "function",
                        "function": {
                            "name": "background_run",
                            "arguments": json.dumps({"action": "tool", "tool": "slow_bg", "args": {}}),
                        },
                    }],
                )
            return LLMResponse(raw_text="parent done")

        async def close(self):
            pass

    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    runner = await RunningAgent(
        store, _BgParent(), reg,
        AgentLoopConfig(context_budget_tokens=10**9, stop_policy=StopPolicy(abort_tool_s=0.01)),
    ).start()
    try:
        await runner.send("launch bg")
        await asyncio.wait_for(started.wait(), 5)
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in rows if m.tool_call_id == "bg-call"))
        bg_target = TargetRef(**receipt["target_ref"])
        bg_start = InputRef(**receipt["input_ref"])

        # Clear SQL records and sleep 0.25s while background tool is running -> MUST be 0 boundary SQL!
        profiled.records.clear()
        await asyncio.sleep(0.25)
        boundary_sql = [
            r for r in profiled.records
            if "boundary" in " ".join(r["chain"]) or "background_execution" in " ".join(r["chain"])
        ]
        assert boundary_sql == [], boundary_sql

        # Remote DB cancel (no local transport notification) -> batch_renew wakes stop_event!
        peer = await connect(schema="verify")
        try:
            await RuntimeInputStore(peer).cancel(bg_target, actor=runner.actor, reason="remote_stop")
        finally:
            await peer.close()

        await runner.runtime.worker.batch_renew()
        result = await runner.runtime.client.wait(
            bg_start, actor=runner.actor, deadline_ms=now_ms() + 5000,
        )
        assert result.status == "cancelled"
        assert not release.is_set()
    finally:
        release.set()
        await runner.aclose()
        store._db = original


async def test_m1c_p1_4_subtree_cancel_stops_background_child_without_polling(registry_store, tmp_path):
    """Subtree client.cancel(parent, scope='subtree') immediately wakes background child's stop_event."""
    store, _ = registry_store
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_bg() -> str:
        started.set()
        await release.wait()
        return "unexpected"

    reg = create_default_tool_registry(workspace_dir=str(tmp_path))
    reg.register(
        ToolDefinition(
            "slow_bg", "Blocking background tool", async_capable=True, interrupt="abort",
            pricing=Free("local"), input_schema={"type": "object"},
        ),
        slow_bg,
    )

    class _BgParent(LLMService):
        def __init__(self):
            self.sent = False

        async def complete(self, request, **kwargs):
            if not self.sent:
                self.sent = True
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "bg-sub",
                        "type": "function",
                        "function": {
                            "name": "background_run",
                            "arguments": json.dumps({"action": "tool", "tool": "slow_bg", "args": {}}),
                        },
                    }],
                )
            return LLMResponse(raw_text="parent done")

        async def close(self):
            pass

    runner = await RunningAgent(
        store, _BgParent(), reg,
        AgentLoopConfig(context_budget_tokens=10**9, stop_policy=StopPolicy(abort_tool_s=0.01)),
    ).start()
    try:
        await runner.send("launch bg")
        await asyncio.wait_for(started.wait(), 5)
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in rows if m.tool_call_id == "bg-sub"))
        bg_start = InputRef(**receipt["input_ref"])

        await runner.runtime.client.cancel(
            runner.target, actor=runner.actor, scope="subtree", reason="parent_stopped",
        )
        result = await runner.runtime.client.wait(
            bg_start, actor=runner.actor, deadline_ms=now_ms() + 5000,
        )
        assert result.status == "cancelled"
        assert not release.is_set()
    finally:
        release.set()
        await runner.aclose()


# ── 7. Concurrency: simultaneous cancel(subtree) + set_lifecycle('closed') ──


async def test_m1c_concurrent_stop_and_lifecycle_race(registry_store):
    store, connect = registry_store
    p1 = RuntimeInputStore(store)
    peer = await connect(schema="verify")
    p2 = RuntimeInputStore(peer)
    try:
        root = await target(p1, "race-root")
        c1 = await target(p1, "race-c1", parent=root)
        await p1.accept(root, "r", message_id="rm1", actor=ACTOR)
        await p1.claim(root, node_id="n1", boot_id="b1")
        await p1.accept(c1, "c", message_id="cm1", actor=ACTOR)
        await p1.claim(c1, node_id="n1", boot_id="b1")

        # Concurrent subtree cancel and Strategy 3A soft-close must never deadlock!
        results = await asyncio.gather(
            p1.cancel(root, actor=ACTOR, scope="subtree", reason="stop_race"),
            p2.set_lifecycle(root, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="close_race"),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        assert errors == [], errors

        rrow = await store._db.fetchone(
            f"SELECT lifecycle,cancel_seq FROM {store.t.sessions} WHERE session_id=?",
            ("race-root",),
        )
        crow = await store._db.fetchone(
            f"SELECT cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
            ("race-c1",),
        )
        assert rrow is not None and rrow["lifecycle"] == "closed"
        assert int(rrow["cancel_seq"]) >= 1
        assert crow is not None and crow["cancel_by"] == "subtree" and int(crow["cancel_seq"]) >= 1
    finally:
        await peer.close()
