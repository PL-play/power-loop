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
    ExecutionEnvironment,
    HookPoint,
    LLMResponse,
    LLMService,
    RebindResult,
    RegistryRuntimeFactory,
    Runtime,
    ToolDefinition,
    TrustedActor,
)
from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import (
    CapacityError,
    InputRef,
    RuntimeRequestError,
    TargetRef,
    sha256_canonical,
)
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


# ── 8. Rework B1–B4 Regressions ─────────────────────────────────────────────


async def test_m1c_b1_3a_queued_prestaged_background_never_executes(registry_store, tmp_path):
    """B1: Pre-staged queued background child stopped by 3A settles cancelled with 0 tool calls."""
    store, _ = registry_store
    tool_calls: list[str] = []

    async def bg_tool() -> str:
        tool_calls.append("ran")
        return "MUST_NOT_RUN"

    tool_def = ToolDefinition(
        "bg_tool", "Background tool", async_capable=True, interrupt="abort",
        pricing=Free("local"), input_schema={"type": "object"},
    )
    reg = create_default_tool_registry(workspace_dir=str(tmp_path))
    reg.register(tool_def, bg_tool)
    frozen = {
        "tool": "bg_tool",
        "arguments": {},
        "definition_hash": sha256_canonical(tool_def.to_openai_tool()),
        "call_ref": {"store_id": store.store_id, "session_id": "b1-main", "seq": 1},
        "call_id": "tc-b1",
    }

    factory = RegistryRuntimeFactory()
    model = _ScriptedModel(lambda _i, _r: LLMResponse(raw_text="unused"))

    async def build(_desc, _ctx):
        return ExecutionEnvironment(
            model, reg, AgentLoopConfig(context_budget_tokens=10**9, stop_policy=StopPolicy(abort_tool_s=0.01)),
        )

    factory.register("test.agent@215", build)
    port = RuntimeInputStore(store)
    main = await target(port, "b1-main")
    bg_desc = replace(DESCRIPTOR, kind="background", config={**dict(DESCRIPTOR.config), "background": frozen})
    cm = await port.create_many(
        parent=main,
        actor=ACTOR,
        specs=[ChildSpec(
            session_id="b1-bg-child",
            message_id="b1-bg-m1",
            body=frozen,
            descriptor=bg_desc,
            result_kind="background_result",
        )],
        node_id="n-creator",
        boot_id="b-creator",
        can_execute=False,
        free_slots=0,
    )
    assert len(cm.queued_targets) == 1
    bg_start = InputRef(store.store_id, "b1-bg-child", "b1-bg-m1")

    # Submit Strategy 3A BEFORE worker starts -> child is queued + cancel_by='subtree'
    st = await port.set_lifecycle(
        main, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="removed_3a",
    )
    assert "b1-bg-child" in st.stopped_descendant_ids

    rt = Runtime(
        store=store, factory=factory, node_id="n-worker", boot_id="b-worker",
        capacity=2, recovery_interval_s=None, timers_enabled=False,
    )
    await rt.worker.start()
    try:
        res = await rt.client.wait(bg_start, actor=ACTOR, deadline_ms=now_ms() + 5000)
        assert tool_calls == []
        assert res.status == "cancelled"
        assert res.result_delivery_error == "session_closed"
        parent_inbox = await store._db.fetchall(
            f"SELECT * FROM {store.t.follow_up_queue} WHERE session_id=?",
            ("b1-main",),
        )
        assert parent_inbox == []
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_m1c_b1_concurrent_create_many_and_3a_close_no_missed_child(registry_store):
    """B1: Two-connection interleave between 3A active-descendant scan and create_many never misses child."""
    store, connect = registry_store
    p1 = RuntimeInputStore(store)
    peer = await connect(schema="verify")
    p2 = RuntimeInputStore(peer)
    try:
        root = await target(p1, "b1-race-root")
        spec = ChildSpec(
            session_id="b1-race-child",
            message_id="b1-race-m1",
            body={"text": "work"},
            descriptor=DESCRIPTOR,
            result_kind="agent_result",
        )
        dialect = store._db.dialect.name
        if dialect in ("postgres", "mysql"):
            # Deterministically place create_many AFTER initial active-descendant scan and BEFORE root lock!
            orig_collect = p1._collect_active_descendants
            scan_done = asyncio.Event()
            spawn_done = asyncio.Event()
            call_count = 0

            async def hooked_collect(tx, **kwargs):
                nonlocal call_count
                call_count += 1
                rows = await orig_collect(tx, **kwargs)
                if call_count == 1:
                    scan_done.set()
                    await asyncio.wait_for(spawn_done.wait(), 5)
                return rows

            p1._collect_active_descendants = hooked_collect  # type: ignore[method-assign]

            async def run_spawn():
                await asyncio.wait_for(scan_done.wait(), 5)
                try:
                    return await p2.create_many(
                        parent=root, actor=ACTOR, specs=[spec],
                        node_id="n2", boot_id="b2", can_execute=False, free_slots=0,
                    )
                finally:
                    spawn_done.set()

            close_res, spawn_res = await asyncio.gather(
                p1.set_lifecycle(root, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="3a_race"),
                run_spawn(),
                return_exceptions=True,
            )
            assert not isinstance(close_res, Exception), close_res
            assert call_count >= 2  # recheck detected new child & retried cleanly!
        else:
            close_res, spawn_res = await asyncio.gather(
                p1.set_lifecycle(root, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="3a_race"),
                p2.create_many(
                    parent=root, actor=ACTOR, specs=[spec],
                    node_id="n2", boot_id="b2", can_execute=False, free_slots=0,
                ),
                return_exceptions=True,
            )
            assert not isinstance(close_res, Exception), close_res

        if isinstance(spawn_res, Exception):
            assert isinstance(spawn_res, RuntimeRequestError) and spawn_res.code in ("session_closed", "cancelled")
            assert await store.get_session("b1-race-child") is None
        else:
            # Child was accepted before root locked -> MUST be cancelled by retry (never left uncanceled!)
            assert "b1-race-child" in close_res.stopped_descendant_ids
            crow = await store._db.fetchone(
                f"SELECT cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
                ("b1-race-child",),
            )
            assert crow is not None and crow["cancel_by"] == "subtree" and int(crow["cancel_seq"]) >= 1
    finally:
        await peer.close()


@pytest.mark.parametrize("old_parent_state", ["open", "closed"])
async def test_m1c_b2_inflight_rebind_delivers_only_to_new_parent(registry_store, old_parent_state):
    """B2: Rebinding a running child routes its in-flight Final result ONLY to new_parent."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    old_p = await target(port, f"b2-old-{old_parent_state}")
    new_p = await target(port, f"b2-new-{old_parent_state}")
    child_sid = f"b2-ch-{old_parent_state}"
    cm = await port.create_many(
        parent=old_p, actor=ACTOR,
        specs=[ChildSpec(
            session_id=child_sid, message_id="m1", body={"text": "compute"},
            descriptor=DESCRIPTOR, result_kind="agent_result",
        )],
        node_id="n1", boot_id="b1", can_execute=False, free_slots=0,
    )
    child = cm.queued_targets[0]
    batch = await port.claim(child, node_id="n1", boot_id="b1")
    assert batch is not None

    if old_parent_state == "closed":
        await port.set_lifecycle(old_p, "closed", actor=ACTOR, inflight="abort", subtree="none")

    rb = await port.rebind_children(old_p, new_p, actor=ACTOR)
    assert rb.rebound_children == (child.session_id,)

    # Finalize using the ORIGINAL ClaimedBatch captured before rebind!
    fb = await port.final(batch, final_text="rebound inflight answer")
    assert fb.result_delivery_error is None
    assert fb.result_activation == new_p

    old_inbox = await store._db.fetchall(
        f"SELECT * FROM {store.t.follow_up_queue} WHERE session_id=?", (old_p.session_id,),
    )
    new_inbox = await store._db.fetchall(
        f"SELECT * FROM {store.t.follow_up_queue} WHERE session_id=?", (new_p.session_id,),
    )
    assert old_inbox == []
    assert len(new_inbox) == 1
    assert "rebound inflight answer" in new_inbox[0]["content_json"]
    new_srow = await store._db.fetchone(
        f"SELECT execution_state,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
        (new_p.session_id,),
    )
    assert new_srow["execution_state"] == "queued" and int(new_srow["wake_pending_count"]) == 1


@pytest.mark.parametrize("old_closed", [False, True])
async def test_m1c_b2_concurrent_final_and_rebind_no_deadlock(registry_store, old_closed):
    """B2: Concurrent Final (child->parent) and rebind_children unify deep-first lock order (0 deadlocks)."""
    store, connect = registry_store
    p1 = RuntimeInputStore(store)
    peer = await connect(schema="verify")
    p2 = RuntimeInputStore(peer)
    try:
        suffix = "closed" if old_closed else "open"
        old_p = await target(p1, f"b2-dl-old-{suffix}")
        new_p = await target(p1, f"b2-dl-new-{suffix}")
        cm = await p1.create_many(
            parent=old_p, actor=ACTOR,
            specs=[ChildSpec(
                session_id=f"b2-dl-ch-{suffix}", message_id="m1", body={"text": "go"},
                descriptor=DESCRIPTOR, result_kind="agent_result",
            )],
            node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        )
        batch = cm.preclaimed[0]
        child = batch.target
        if old_closed:
            await p1.set_lifecycle(old_p, "closed", actor=ACTOR, inflight="abort", subtree="none")

        if store._db.dialect.name in ("postgres", "mysql"):
            # Force Conn 1 (Final) to hold the child row lock before touching old_parent,
            # then launch Conn 2 (rebind_children) and verify deep-first wait with 0 deadlock!
            orig_river = p1._river_deliver_tx
            child_locked = asyncio.Event()
            release_final = asyncio.Event()

            async def paused_river(tx, **kwargs):
                child_locked.set()
                await asyncio.wait_for(release_final.wait(), 5)
                return await orig_river(tx, **kwargs)

            p1._river_deliver_tx = paused_river  # type: ignore[method-assign]

            final_task = asyncio.create_task(p1.final(batch, final_text="concurrent ok"))
            await asyncio.wait_for(child_locked.wait(), 5)
            rebind_task = asyncio.create_task(p2.rebind_children(old_p, new_p, actor=ACTOR))
            await asyncio.sleep(0.08)
            release_final.set()
            fb, rb = await asyncio.wait_for(asyncio.gather(final_task, rebind_task), 10)
        else:
            fb, rb = await asyncio.gather(
                p1.final(batch, final_text="concurrent ok"),
                p2.rebind_children(old_p, new_p, actor=ACTOR),
            )

        assert rb.rebound_children == (child.session_id,)
        if old_closed:
            # Regardless of order, new_p MUST receive the result (either via live route or orphan replay!)
            new_inbox = await store._db.fetchall(
                f"SELECT * FROM {store.t.follow_up_queue} WHERE session_id=?", (new_p.session_id,),
            )
            assert len(new_inbox) == 1, (fb, rb)
            assert "concurrent ok" in new_inbox[0]["content_json"]
    finally:
        await peer.close()


async def test_m1c_b2_replay_undelivered_enforces_capacity_and_rolls_back_atomically(registry_store):
    """B2: replay_undelivered respects item-count & byte limits, rolls back on overflow, succeeds after retry."""
    store, _ = registry_store
    port = RuntimeInputStore(store, inbox_item_limit=1, hard_pending_limit=1)
    old_p = await target(port, "b2-cap-old")
    new_p = await target(port, "b2-cap-new")
    cm = await port.create_many(
        parent=old_p, actor=ACTOR,
        specs=[ChildSpec(
            session_id="b2-cap-ch", message_id="cm1", body={"text": "work"},
            descriptor=DESCRIPTOR, result_kind="agent_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    cb = cm.preclaimed[0]
    child = cb.target
    await port.set_lifecycle(old_p, "closed", actor=ACTOR, inflight="abort", subtree="none")
    fb = await port.final(cb, final_text="orphan payload")
    assert fb.result_delivery_error == "session_closed"

    # Fill new_p to its hard limit of 1 pending item
    await port.accept(new_p, "blocker", message_id="blk1", actor=ACTOR)

    with pytest.raises(CapacityError):
        await port.rebind_children(old_p, new_p, replay_undelivered=True, actor=ACTOR)

    # Verify full rollback: new_p count still 1, child still under old_p, session_closed preserved!
    new_row = await store._db.fetchone(
        f"SELECT pending_count FROM {store.t.sessions} WHERE session_id=?", (new_p.session_id,),
    )
    ch_row = await store._db.fetchone(
        f"SELECT parent_session_id FROM {store.t.sessions} WHERE session_id=?", (child.session_id,),
    )
    ch_q = await store._db.fetchone(
        f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='cm1'",
        (child.session_id,),
    )
    assert int(new_row["pending_count"]) == 1
    assert ch_row["parent_session_id"] == old_p.session_id
    assert ch_q["result_delivery_error"] == "session_closed"

    # Also verify byte capacity rollback before freeing space
    byte_port = RuntimeInputStore(store, inbox_item_limit=10, hard_pending_limit=10, inbox_byte_limit=64)
    with pytest.raises(CapacityError):
        await byte_port.rebind_children(old_p, new_p, replay_undelivered=True, actor=ACTOR)
    ch_q2 = await store._db.fetchone(
        f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='cm1'",
        (child.session_id,),
    )
    assert ch_q2["result_delivery_error"] == "session_closed"

    # Free space in new_p -> retry succeeds cleanly and clears session_closed!
    nb = await port.claim(new_p, node_id="n1", boot_id="b1")
    await port.final(nb, final_text="cleared")
    rb = await port.rebind_children(old_p, new_p, replay_undelivered=True, actor=ACTOR)
    assert rb.replayed_results == 1
    new_row_after = await store._db.fetchone(
        f"SELECT pending_count FROM {store.t.sessions} WHERE session_id=?", (new_p.session_id,),
    )
    ch_q_after = await store._db.fetchone(
        f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='cm1'",
        (child.session_id,),
    )
    assert int(new_row_after["pending_count"]) == 1
    assert ch_q_after["result_delivery_error"] is None


async def test_m1c_b3_automatic_periodic_tick_stops_background_without_notification_and_meets_sql_budget(
    registry_store, tmp_path,
):
    """B3: Real Worker automatic periodic_tick runs O(1) 3/3/4 SQL for 2 active tasks and stops remote 3A."""
    store, connect = registry_store
    started1, started2 = asyncio.Event(), asyncio.Event()
    release = asyncio.Event()

    async def slow_bg(slot: str = "1") -> str:
        (started1 if slot == "1" else started2).set()
        await release.wait()
        return "unexpected"

    tool_def = ToolDefinition(
        "slow_bg", "Blocking background tool", async_capable=True, interrupt="abort",
        pricing=Free("local"),
        input_schema={"type": "object", "properties": {"slot": {"type": "string"}}},
    )
    reg = create_default_tool_registry(workspace_dir=str(tmp_path))
    reg.register(tool_def, slow_bg)

    factory = RegistryRuntimeFactory()
    model = _ScriptedModel(lambda _i, _r: LLMResponse(raw_text="unused"))

    async def build(_desc, _ctx):
        return ExecutionEnvironment(
            model, reg, AgentLoopConfig(context_budget_tokens=10**9, stop_policy=StopPolicy(abort_tool_s=0.01)),
        )

    factory.register("test.agent@215", build)

    original = store._db
    profiled = Profiled(original)
    store._db = profiled

    rt = Runtime(
        store=store, factory=factory, node_id="b3-node", boot_id="b3-boot",
        capacity=4, session_lease_ttl_s=0.9, recovery_interval_s=0.18, timers_enabled=False,
    )
    # Observe real automatic periodic_tick calls without calling periodic_tick manually!
    orig_tick = rt.worker.periodic_tick
    tick_slices: list[tuple[list[dict[str, Any]], int]] = []
    two_ticks_ready = asyncio.Event()
    measure_ticks = False

    async def observed_tick(*, limit: int = 32):
        if measure_ticks:
            start_rec = len(profiled.records)
            start_tx = profiled.transactions
            res = await orig_tick(limit=limit)
            tick_slices.append((list(profiled.records[start_rec:]), profiled.transactions - start_tx))
            if len(tick_slices) >= 2:
                two_ticks_ready.set()
            return res
        return await orig_tick(limit=limit)

    rt.worker.periodic_tick = observed_tick  # type: ignore[method-assign]
    await rt.worker.start()
    peer = await connect(schema="verify")
    try:
        main = await rt.client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "b3-main"})
        specs = []
        starts = []
        for idx, slot in enumerate(("1", "2"), start=1):
            frozen = {
                "tool": "slow_bg",
                "arguments": {"slot": slot},
                "definition_hash": sha256_canonical(tool_def.to_openai_tool()),
                "call_ref": {"store_id": store.store_id, "session_id": "b3-main", "seq": idx},
                "call_id": f"tc-b3-{slot}",
            }
            bg_desc = replace(DESCRIPTOR, kind="background", config={**dict(DESCRIPTOR.config), "background": frozen})
            sid = f"b3-bg-{slot}"
            mid = f"b3-m-{slot}"
            specs.append(ChildSpec(
                session_id=sid, message_id=mid, body=frozen,
                descriptor=bg_desc, result_kind="background_result",
            ))
            starts.append(InputRef(store.store_id, sid, mid))
        await rt.client.inputs.create_many(
            parent=main, actor=ACTOR, specs=specs,
            node_id="b3-node", boot_id="b3-boot", can_execute=False, free_slots=0,
        )
        # Wait for automatic discovery to launch BOTH background tasks
        await asyncio.wait_for(asyncio.gather(started1.wait(), started2.wait()), 5)

        # Phase 1: Measure 2 real automatic healthy maintenance ticks with BOTH sessions active!
        measure_ticks = True
        profiled.records.clear()
        profiled.transactions = 0
        await asyncio.wait_for(two_ticks_ready.wait(), 5)
        measure_ticks = False
        all_healthy_sql = list(profiled.records)

        expected_sql = 4 if original.dialect.name == "mysql" else 3
        assert len(tick_slices) >= 2
        for recs, txs in tick_slices[:2]:
            assert txs == 0, txs
            assert len(recs) == expected_sql, [r["sql"] for r in recs]
            for r in recs:
                chain_str = " ".join(r["chain"])
                assert "_db_now" not in chain_str, r
                assert "store_session_ownership" not in chain_str, r
                assert "discover_unfinished" not in chain_str, r
        # Total SQL across the measured healthy ticks must equal exact tick_count * (3/3/4) (no side renewals!)
        assert len(all_healthy_sql) == len(tick_slices) * expected_sql, [r["sql"] for r in all_healthy_sql]

        # Phase 2: Remote 3A via independent connection (NO shared transport notification) ->
        # real automatic periodic_tick MUST stop BOTH background tasks!
        await RuntimeInputStore(peer).set_lifecycle(
            main, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="remote_3a_b3",
        )
        results = await asyncio.gather(*(
            rt.client.wait(ref, actor=ACTOR, deadline_ms=now_ms() + 5000) for ref in starts
        ))
        assert [r.status for r in results] == ["cancelled", "cancelled"]
        assert not release.is_set()
    finally:
        release.set()
        await peer.close()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        store._db = original


async def test_m1c_b4_release_claim_fires_result_undeliverable_post_commit(registry_store):
    """B4: Factory build failure during 3A enters release_claim() and fires RESULT_UNDELIVERABLE post-commit."""
    store, connect = registry_store
    hooks = AgentHooks()
    build_entered = asyncio.Event()
    build_release = asyncio.Event()
    undeliv_events: list[dict[str, Any]] = []
    observer_seen: list[tuple[Any, Any]] = []

    observer = await connect(schema="verify")
    peer = await connect(schema="verify")

    async def on_undeliv(ctx):
        undeliv_events.append(dict(ctx.values))
        # Independent connection MUST already see committed terminal_status + result_delivery_error!
        row = await observer._db.fetchone(
            f"SELECT terminal_status,result_delivery_error FROM {observer.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            ("b4-child", "b4-m1"),
        )
        observer_seen.append((row["terminal_status"], row["result_delivery_error"]) if row else (None, None))

    hooks.register(HookPoint.RESULT_UNDELIVERABLE, on_undeliv)

    factory = RegistryRuntimeFactory()

    async def failing_build(_desc, _ctx):
        build_entered.set()
        await asyncio.wait_for(build_release.wait(), 5)
        raise RuntimeRequestError("resource_unavailable", "simulated build failure")

    factory.register("test.agent@215", failing_build)

    rt = Runtime(
        store=store, factory=factory, node_id="b4-node", boot_id="b4-boot",
        capacity=2, recovery_interval_s=None, timers_enabled=False, hooks=hooks,
    )
    try:
        main = await rt.client.inputs.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "b4-main"})
        await rt.client.inputs.create_many(
            parent=main, actor=ACTOR,
            specs=[ChildSpec(
                session_id="b4-child", message_id="b4-m1", body={"text": "run"},
                descriptor=DESCRIPTOR, result_kind="agent_result",
            )],
            node_id="b4-node", boot_id="b4-boot", can_execute=False, free_slots=0,
        )
        await rt.worker.start()
        await asyncio.wait_for(build_entered.wait(), 5)

        # Close parent tree while child is claimed and paused in factory.build
        await RuntimeInputStore(peer).set_lifecycle(
            main, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="3a_during_build",
        )
        build_release.set()

        child_ref = InputRef(store.store_id, "b4-child", "b4-m1")
        res = await rt.client.wait(child_ref, actor=ACTOR, deadline_ms=now_ms() + 5000)
        assert res.status == "cancelled"
        assert res.result_delivery_error == "session_closed"

        # Wait briefly for worker task teardown to finish
        for _ in range(50):
            if undeliv_events:
                break
            await asyncio.sleep(0.02)
        assert len(undeliv_events) == 1
        assert undeliv_events[0]["reason"] == "session_closed"
        assert undeliv_events[0]["child_session_id"] == "b4-child"
        assert undeliv_events[0]["parent_session_id"] == "b4-main"
        assert observer_seen == [("cancelled", "session_closed")]
    finally:
        build_release.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await observer.close()
        await peer.close()


# ── 9. Round-2 & Round-3 Rework Regressions ─────────────────────────────────


@pytest.mark.parametrize("close_subtree", ["none", "all"])
async def test_m1c_r3_running_driver_create_many_under_subtree_none_and_all(
    registry_store, close_subtree,
):
    """R3-Item2: Running Driver keeps creating leaves under subtree='none' (5 SQL) and stops under subtree='all'."""
    store, connect = registry_store
    p1 = RuntimeInputStore(store)
    peer = await connect(schema="verify")
    p2 = RuntimeInputStore(peer)
    try:
        root = await target(p1, f"r3-drv-root-{close_subtree}")
        drv_sid = f"r3-drv-{close_subtree}"
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        cm_drv = await p1.create_many(
            parent=root, actor=ACTOR,
            specs=[ChildSpec(
                session_id=drv_sid, message_id="dm1", body={"wf": "start"},
                descriptor=drv_desc, result_kind="workflow_result",
            )],
            node_id="n-drv", boot_id="b-drv", can_execute=True, free_slots=1,
        )
        drv_batch = cm_drv.preclaimed[0]
        driver = drv_batch.target
        leaf_sid = f"r3-leaf-{close_subtree}"
        leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "n1"}})
        leaf_spec = ChildSpec(
            session_id=leaf_sid, message_id="lm1", body={"text": "next wave"},
            descriptor=leaf_desc, result_kind="workflow_node_result",
        )

        if close_subtree == "none":
            # Explicit subtree='none': running/open Driver is NOT cancelled and MUST be able
            # to spawn its next wave of leaves with 0 ancestor-lookup SQL (exact 5 SQL)!
            await p1.set_lifecycle(
                root, "closed", actor=ACTOR, inflight="abort", subtree="none", reason="close_main_only",
            )
            drow = await store._db.fetchone(
                f"SELECT lifecycle,execution_state,cancel_by,cancel_seq FROM {store.t.sessions} WHERE session_id=?",
                (drv_sid,),
            )
            assert drow["lifecycle"] == "open" and drow["execution_state"] == "running"
            assert drow["cancel_by"] is None and int(drow["cancel_seq"]) == 0

            original = store._db
            profiled = Profiled(original)
            store._db = profiled
            try:
                cm_leaf = await p1.create_many(
                    parent=driver, actor=ACTOR, specs=[leaf_spec],
                    node_id="n-drv", boot_id="b-drv", can_execute=False, free_slots=0,
                    parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
                )
                # Exact 5 SQL: lock_state(driver) + _db_now + INSERT sessions + INSERT queue + INSERT messages
                assert len(profiled.records) == 5, [r["sql"] for r in profiled.records]
                assert not any("FOR SHARE" in r["sql"] for r in profiled.records)
            finally:
                store._db = original
            assert cm_leaf.queued_targets == (TargetRef(store.store_id, leaf_sid),)
            lrow = await store.get_session(leaf_sid)
            assert lrow is not None and lrow.parent_session_id == drv_sid
        else:
            # Strategy 3A (subtree='all'): concurrent create_many from running Driver vs 3A close on main.
            # New leaf MUST either be rejected ('cancelled') OR caught & cancelled with 0 execution!
            if store._db.dialect.name in ("postgres", "mysql"):
                orig_collect = p1._collect_active_descendants
                scan1_done = asyncio.Event()
                spawn_done = asyncio.Event()
                calls = 0

                async def hooked_collect(tx, **kwargs):
                    nonlocal calls
                    calls += 1
                    rows = await orig_collect(tx, **kwargs)
                    if calls == 1:
                        scan1_done.set()
                        await asyncio.wait_for(spawn_done.wait(), 5)
                    return rows

                p1._collect_active_descendants = hooked_collect  # type: ignore[method-assign]

                async def run_leaf_spawn():
                    await asyncio.wait_for(scan1_done.wait(), 5)
                    try:
                        return await p2.create_many(
                            parent=driver, actor=ACTOR, specs=[leaf_spec],
                            node_id="n-drv", boot_id="b-drv", can_execute=False, free_slots=0,
                            parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
                        )
                    finally:
                        spawn_done.set()

                close_res, spawn_res = await asyncio.gather(
                    p1.set_lifecycle(root, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="3a_drv"),
                    run_leaf_spawn(),
                    return_exceptions=True,
                )
                assert not isinstance(close_res, Exception), close_res
            else:
                close_res, spawn_res = await asyncio.gather(
                    p1.set_lifecycle(root, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="3a_drv"),
                    p2.create_many(
                        parent=driver, actor=ACTOR, specs=[leaf_spec],
                        node_id="n-drv", boot_id="b-drv", can_execute=False, free_slots=0,
                        parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
                    ),
                    return_exceptions=True,
                )
                assert not isinstance(close_res, Exception), close_res

            # Also verify a post-3A create_many from the now-cancelled running Driver is rejected!
            with pytest.raises(RuntimeRequestError) as post_exc:
                await p1.create_many(
                    parent=driver, actor=ACTOR,
                    specs=[ChildSpec(
                        session_id="r3-leaf-post3a", message_id="lm2", body={"text": "too late"},
                        descriptor=leaf_desc, result_kind="workflow_node_result",
                    )],
                    node_id="n-drv", boot_id="b-drv", can_execute=False, free_slots=0,
                    parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
                )
            assert post_exc.value.code == "cancelled"
            assert await store.get_session("r3-leaf-post3a") is None

            if isinstance(spawn_res, Exception):
                assert isinstance(spawn_res, RuntimeRequestError) and spawn_res.code in ("cancelled", "session_closed")
                assert await store.get_session(leaf_sid) is None
            else:
                assert leaf_sid in close_res.stopped_descendant_ids
                grow = await store._db.fetchone(
                    f"SELECT cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
                    (leaf_sid,),
                )
                assert grow is not None and grow["cancel_by"] == "subtree" and int(grow["cancel_seq"]) >= 1
                model_calls: list[str] = []
                factory = RegistryRuntimeFactory()
                model = _ScriptedModel(lambda _i, _r: (model_calls.append("ran") or LLMResponse(raw_text="NO")))

                async def _chk_build(_d, _c):
                    return ExecutionEnvironment(model, ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9))

                factory.register("test.agent@215", _chk_build)
                rt = Runtime(
                    store=store, factory=factory, node_id="n-chk", boot_id="b-chk",
                    capacity=2, recovery_interval_s=None, timers_enabled=False,
                )
                await rt.worker.start()
                try:
                    res = await rt.client.wait(
                        InputRef(store.store_id, leaf_sid, "lm1"), actor=ACTOR, deadline_ms=now_ms() + 5000,
                    )
                    assert res.status == "cancelled"
                    assert model_calls == []
                finally:
                    await rt.worker.shutdown(timeout_s=2, cancel_running=True)
                    await rt.client.aclose()
                    await rt.transport.aclose()
    finally:
        await peer.close()


async def test_m1c_b1_r2_3a_does_not_cancel_rebound_child(registry_store):
    """B1-R2 Scene 2: Child rebound to another tree after 3A scan 1 is NOT cancelled and completes cleanly."""
    store, connect = registry_store
    p1 = RuntimeInputStore(store)
    peer = await connect(schema="verify")
    p2 = RuntimeInputStore(peer)
    try:
        old_root = await target(p1, "b1r2-old-root")
        new_parent = await target(p1, "b1r2-new-parent")
        cm = await p1.create_many(
            parent=old_root, actor=ACTOR,
            specs=[ChildSpec(
                session_id="b1r2-moved-ch", message_id="m1", body={"text": "survive"},
                descriptor=DESCRIPTOR, result_kind="agent_result",
            )],
            node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        )
        batch = cm.preclaimed[0]
        child = batch.target

        if store._db.dialect.name in ("postgres", "mysql"):
            orig_collect = p1._collect_active_descendants
            scan1_done = asyncio.Event()
            rebind_done = asyncio.Event()
            calls = 0

            async def hooked_collect(tx, **kwargs):
                nonlocal calls
                calls += 1
                rows = await orig_collect(tx, **kwargs)
                if calls == 1:
                    scan1_done.set()
                    await asyncio.wait_for(rebind_done.wait(), 5)
                return rows

            p1._collect_active_descendants = hooked_collect  # type: ignore[method-assign]

            async def run_rebind():
                await asyncio.wait_for(scan1_done.wait(), 5)
                try:
                    return await p2.rebind_children(old_root, new_parent, actor=ACTOR)
                finally:
                    rebind_done.set()

            close_res, rb_res = await asyncio.gather(
                p1.set_lifecycle(old_root, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="3a_old"),
                run_rebind(),
            )
            assert calls >= 2
        else:
            # On SQLite (single writer lock), first commit rebind_children, then inject stale scan-1
            # output containing the moved child so Step B/C verifies locked ownership and retries cleanly!
            rb_res = await p2.rebind_children(old_root, new_parent, actor=ACTOR)
            orig_collect = p1._collect_active_descendants
            stale_injected = False

            async def sqlite_hook(tx, **kwargs):
                nonlocal stale_injected
                if not stale_injected:
                    stale_injected = True
                    return [(child.session_id, 1, False)]
                return await orig_collect(tx, **kwargs)

            p1._collect_active_descendants = sqlite_hook  # type: ignore[method-assign]
            close_res = await p1.set_lifecycle(
                old_root, "closed", actor=ACTOR, inflight="abort", subtree="all", reason="3a_old",
            )

        assert child.session_id not in close_res.stopped_descendant_ids
        ch_row = await store._db.fetchone(
            f"SELECT parent_session_id,root_session_id,cancel_seq,cancel_by "
            f"FROM {store.t.sessions} WHERE session_id=?",
            (child.session_id,),
        )
        assert ch_row["parent_session_id"] == new_parent.session_id
        assert ch_row["root_session_id"] == new_parent.session_id
        assert int(ch_row["cancel_seq"]) == 0 and ch_row["cancel_by"] is None

        # Moved child MUST complete normally and deliver ONLY to new_parent!
        fb = await p1.final(batch, final_text="moved child survived")
        assert fb.results[0].status == "completed"
        assert fb.result_delivery_error is None
        new_inbox = await store._db.fetchall(
            f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=?",
            (new_parent.session_id,),
        )
        assert len(new_inbox) == 1 and "moved child survived" in new_inbox[0]["content_json"]
        _ = rb_res
    finally:
        await peer.close()


@pytest.mark.parametrize("old_parent_state", ["open", "closed"])
async def test_m1c_b2_r2_mixed_batch_per_input_rebind_and_zero_extra_normal_sql(
    registry_store, old_parent_state,
):
    """B2-R2: Mixed batch (unrouted first, routed second) honors in-flight rebind; normal Final adds 0 SQL."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    old_p = await target(port, f"b2r2-old-{old_parent_state}")
    new_p = await target(port, f"b2r2-new-{old_parent_state}")
    # Child with NO session-level default result_route
    child = await target(port, f"b2r2-ch-{old_parent_state}", parent=old_p)

    # First input has NO reply_route; second input explicitly routes back to old_p
    await port.accept(child, "direct msg", message_id="direct", actor=ACTOR)
    await port.accept(
        child,
        "routed msg",
        message_id="routed",
        actor=ACTOR,
        reply_route={"target": old_p, "result_kind": "agent_result"},
    )
    batch = await port.claim(child, node_id="n1", boot_id="b1")
    assert batch is not None
    assert batch.result_route is None
    assert [r.message_id for r in batch.input_refs] == ["direct", "routed"]
    assert batch.reply_to_session_id == old_p.session_id
    assert batch.reply_to_item_id == "routed"

    if old_parent_state == "closed":
        await port.set_lifecycle(old_p, "closed", actor=ACTOR, inflight="abort", subtree="none")

    rb = await port.rebind_children(old_p, new_p, actor=ACTOR)
    assert rb.rebound_children == (child.session_id,)

    # Finalize using the ORIGINAL pre-rebind batch
    fb = await port.final(batch, final_text="mixed routed answer")
    assert fb.result_delivery_error is None
    assert fb.result_activation == new_p

    old_inbox = await store._db.fetchall(
        f"SELECT * FROM {store.t.follow_up_queue} WHERE session_id=?", (old_p.session_id,),
    )
    new_inbox = await store._db.fetchall(
        f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=?", (new_p.session_id,),
    )
    assert old_inbox == []
    assert len(new_inbox) == 1
    body = json.loads(new_inbox[0]["content_json"])
    assert body["final_text"] == "mixed routed answer"
    assert body["input_ref"]["message_id"] == "routed"

    # Also assert 0 extra route-lookup SQL on NORMAL (non-rebound) per-input & session-route Finals!
    if old_parent_state == "open":
        original = store._db
        profiled = Profiled(original)
        store._db = profiled
        try:
            # 1) Normal per-input routed Final (no rebind): must NOT query follow_up_queue for reply_to_session_id
            ch_norm = await target(port, "b2r2-norm-perinput", parent=new_p)
            await port.accept(ch_norm, "d1", message_id="d1", actor=ACTOR)
            await port.accept(
                ch_norm, "r1", message_id="r1", actor=ACTOR,
                reply_route={"target": new_p, "result_kind": "agent_result"},
            )
            nb = await port.claim(ch_norm, node_id="n1", boot_id="b1")
            assert nb is not None
            profiled.records.clear()
            await port.final(nb, final_text="ok per-input")
            assert not any(
                "reply_to_session_id" in r["sql"] and "item_id=?" in r["sql"] for r in profiled.records
            ), [r["sql"] for r in profiled.records]
            expected_pi = 7 if original.dialect.name == "mysql" else 6
            assert len(profiled.records) == expected_pi, [r["sql"] for r in profiled.records]
            # SQLite/PG: 5 SQL (max_seq + UPDATE sessions + SELECT delivered + River UPDATE + River INSERT + INSERT msg = 6 if write_message, 5 if result_seq given)
            # Check exact MySQL/SQLite/PG counts with pre-written result_seq=2 (matches real Worker!)
            cm_norm = await port.create_many(
                parent=new_p, actor=ACTOR,
                specs=[ChildSpec(
                    session_id="b2r2-norm-route", message_id="m1", body={"text": "go"},
                    descriptor=DESCRIPTOR, result_kind="agent_result",
                )],
                node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
            )
            cb = cm_norm.preclaimed[0]
            await store.append_message_fenced(
                cb.target.session_id, owner_epoch=cb.owner_epoch, seq=2, send_index=1,
                round_index=0, role="assistant", content="done",
                meta={"usage": {}, "structured_output": None, "error": None, "error_code": None, "result_delivery_error": None},
            )
            profiled.records.clear()
            await port.final(
                cb, consumed=cb.input_refs, status="completed", final_text="done",
                usage={}, result_seq=2, next_seq=3,
                assistant_meta={"usage": {}, "structured_output": None, "error": None, "error_code": None, "result_delivery_error": None},
            )
            # Normal child Final with session-level result_route: 0 diag SELECT on MySQL (3 SQL on all 3 DBs: UPDATE child + River UPDATE parent + River INSERT parent!)
            assert not any("SELECT next_seq,cancel_by" in r["sql"] for r in profiled.records), [
                r["sql"] for r in profiled.records
            ]
            assert len(profiled.records) == 3, [r["sql"] for r in profiled.records]
        finally:
            store._db = original


async def test_m1c_b3_r2_automatic_takeover_respects_capacity_limit(registry_store):
    """B3-R2: With capacity=1 and 3 expired 0-assistant tasks, automatic periodic takeover keeps concurrency == 1."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    in_flight = 0
    max_in_flight = 0
    started_order: list[str] = []
    gates = {f"b3r2-cap-{i}": asyncio.Event() for i in range(1, 4)}
    entered = {f"b3r2-cap-{i}": asyncio.Event() for i in range(1, 4)}

    class _CountingBlockingModel:
        async def complete(self, request, **_kwargs):
            nonlocal in_flight, max_in_flight
            from power_loop.core.execution_context import get_execution_context
            sid = get_execution_context().session_id
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            started_order.append(sid)
            entered[sid].set()
            try:
                await asyncio.wait_for(gates[sid].wait(), 5)
            finally:
                in_flight -= 1
            return LLMResponse(raw_text=f"done:{sid}")

        async def close(self):
            pass

    model = _CountingBlockingModel()
    factory = RegistryRuntimeFactory()

    async def _cap_build(_d, _c):
        return ExecutionEnvironment(model, ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9))

    factory.register("test.agent@215", _cap_build)

    refs: list[InputRef] = []
    for i in range(1, 4):
        sid = f"b3r2-cap-{i}"
        t = await target(port, sid)
        r = await port.accept(t, f"task-{i}", message_id=f"m{i}", actor=ACTOR)
        b = await port.claim(t, node_id="dead-node", boot_id="dead-boot", ttl_s=1.0)
        assert b is not None
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
            (sid,),
        )
        refs.append(r.input_ref)

    rt = Runtime(
        store=store, factory=factory, node_id="b3r2-node", boot_id="b3r2-boot",
        capacity=1, session_lease_ttl_s=1.5, recovery_interval_s=0.1, timers_enabled=False,
    )
    await rt.worker.start()
    try:
        # Wait for the 1st task to enter model execution, then let multiple automatic ticks run
        await asyncio.wait_for(
            asyncio.wait(
                [asyncio.create_task(ev.wait()) for ev in entered.values()],
                return_when=asyncio.FIRST_COMPLETED,
            ),
            5,
        )
        await asyncio.sleep(0.35)
        assert in_flight == 1, f"expected 1 in-flight model call, got {in_flight}"
        assert max_in_flight == 1, f"concurrency exceeded capacity=1: {max_in_flight}"
        assert len(started_order) == 1

        # Release tasks one by one and assert concurrency NEVER exceeds 1 across all 3 takeovers!
        first_sid = started_order[0]
        gates[first_sid].set()
        for expected_started in (2, 3):
            for _ in range(100):
                if len(started_order) >= expected_started:
                    break
                await asyncio.sleep(0.03)
            assert len(started_order) == expected_started
            await asyncio.sleep(0.25)
            assert max_in_flight == 1, f"concurrency exceeded capacity=1: {max_in_flight}"
            gates[started_order[-1]].set()

        results = await asyncio.gather(*(
            rt.client.wait(ref, actor=ACTOR, deadline_ms=now_ms() + 5000) for ref in refs
        ))
        assert [r.status for r in results] == ["completed", "completed", "completed"]
        assert max_in_flight == 1
        assert len(started_order) == 3
    finally:
        for ev in gates.values():
            ev.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_m1c_b3_r2_post_commit_hook_no_false_execution_lost_and_real_loss_fires(registry_store):
    """B3-R2: Committed Final awaiting post-commit RESULT_UNDELIVERABLE never fires EXECUTION_LOST; real loss does."""
    store, connect = registry_store
    hooks = AgentHooks()
    undeliv_entered = asyncio.Event()
    undeliv_release = asyncio.Event()
    undeliv_events: list[dict[str, Any]] = []
    lost_events: list[dict[str, Any]] = []

    async def on_undeliv(ctx):
        undeliv_events.append(dict(ctx.values))
        undeliv_entered.set()
        await asyncio.wait_for(undeliv_release.wait(), 5)

    async def on_lost(ctx):
        lost_events.append(dict(ctx.values))

    hooks.register(HookPoint.RESULT_UNDELIVERABLE, on_undeliv)
    hooks.register(HookPoint.EXECUTION_LOST, on_lost)

    # Part 1: Child finishes after parent closed; RESULT_UNDELIVERABLE spans >=2 automatic maintenance ticks!
    real_loss_entered = asyncio.Event()
    real_loss_release = asyncio.Event()

    class _ModeModel:
        async def complete(self, request, **_kwargs):
            from power_loop.core.execution_context import get_execution_context
            sid = get_execution_context().session_id
            if sid == "b3r2-loss-sess":
                real_loss_entered.set()
                await asyncio.wait_for(real_loss_release.wait(), 5)
            return LLMResponse(raw_text="child finished ok")

        async def close(self):
            pass

    factory = RegistryRuntimeFactory()
    mode_model = _ModeModel()

    async def _mode_build(_d, _c):
        return ExecutionEnvironment(mode_model, ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9))

    factory.register("test.agent@215", _mode_build)

    rt = Runtime(
        store=store, factory=factory, node_id="b3r2-hnode", boot_id="b3r2-hboot",
        capacity=2, session_lease_ttl_s=0.6, recovery_interval_s=0.1,
        hook_timeout_s=2.0, timers_enabled=False, hooks=hooks,
    )
    tick_count = 0
    orig_tick = rt.worker.periodic_tick

    async def counting_tick(*, limit: int = 32):
        nonlocal tick_count
        res = await orig_tick(limit=limit)
        tick_count += 1
        return res

    rt.worker.periodic_tick = counting_tick  # type: ignore[method-assign]
    await rt.worker.start()
    peer = await connect(schema="verify")
    try:
        parent_t = await rt.client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "b3r2-parent"})
        cm = await rt.client.inputs.create_many(
            parent=parent_t, actor=ACTOR,
            specs=[ChildSpec(
                session_id="b3r2-child", message_id="cm1", body={"text": "work"},
                descriptor=DESCRIPTOR, result_kind="agent_result",
            )],
            node_id="b3r2-hnode", boot_id="b3r2-hboot", can_execute=False, free_slots=0,
        )
        # Soft-close parent without cancelling child -> child commits completed / session_closed!
        await rt.client.inputs.set_lifecycle(parent_t, "closed", actor=ACTOR, inflight="abort", subtree="none")
        rt.worker._activate(cm.queued_targets[0].session_id)

        await asyncio.wait_for(undeliv_entered.wait(), 5)
        # Confirm child result is already durably committed while RESULT_UNDELIVERABLE is running
        row = await peer._db.fetchone(
            f"SELECT execution_state FROM {peer.t.sessions} WHERE session_id=?", ("b3r2-child",),
        )
        assert row["execution_state"] == "idle"
        qrow = await peer._db.fetchone(
            f"SELECT terminal_status,result_delivery_error FROM {peer.t.follow_up_queue} "
            "WHERE session_id=? AND item_id='cm1'",
            ("b3r2-child",),
        )
        assert (qrow["terminal_status"], qrow["result_delivery_error"]) == ("completed", "session_closed")

        # Let >= 2 automatic maintenance ticks run while RESULT_UNDELIVERABLE is still awaiting!
        ticks_before = tick_count
        for _ in range(100):
            if tick_count >= ticks_before + 2:
                break
            await asyncio.sleep(0.03)
        assert tick_count >= ticks_before + 2
        assert lost_events == [], f"false EXECUTION_LOST during post-commit hook: {lost_events}"
        undeliv_release.set()

        res = await rt.client.wait(
            InputRef(store.store_id, "b3r2-child", "cm1"), actor=ACTOR, deadline_ms=now_ms() + 5000,
        )
        assert res.status == "completed" and res.result_delivery_error == "session_closed"
        assert len(undeliv_events) == 1
        assert lost_events == []

        # Part 2: Genuine ownership loss MUST still fire EXECUTION_LOST immediately on next tick!
        loss_t = await rt.client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "b3r2-loss-sess"})
        await rt.client.send(loss_t, "block", message_id="lm1", actor=ACTOR)
        await asyncio.wait_for(real_loss_entered.wait(), 5)

        # Simulate another node taking over the lease (bumps owner_epoch & changes owner_node_id)
        await peer._db.execute(
            f"UPDATE {peer.t.sessions} SET owner_node_id='other-node',owner_boot_id='other-boot',"
            "owner_epoch=owner_epoch+1 WHERE session_id=?",
            ("b3r2-loss-sess",),
        )
        for _ in range(100):
            if lost_events:
                break
            await asyncio.sleep(0.03)
        assert len(lost_events) == 1
        assert lost_events[0]["session_id"] == "b3r2-loss-sess"
    finally:
        undeliv_release.set()
        real_loss_release.set()
        await peer.close()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_m1c_r3_single_level_delegation_blocks_subagent_and_workflow_leaf(registry_store):
    """R3-Item1: Only main may spawn subagents or start workflows; subagent & workflow_node are blocked at both exposure and runtime."""
    from power_loop.core.agent_context import get_current_loop
    from power_loop.runtime.execution_roles import require_orchestration, role_tools
    from power_loop.runtime.tool_contracts import resolve
    from power_loop.tools.spawn_agent import _handle_spawn_agent, register_spawn_agent
    from power_loop.workflow.tool import register_workflow_tools

    store, _ = registry_store
    reg = ToolRegistry()
    register_spawn_agent(reg)
    register_workflow_tools(reg)

    # 1) Tool exposure contracts: ONLY "main" sees spawn_agent and create_workflow!
    assert resolve(reg.get("spawn_agent")).runtime_contract.allowed_roles == ("main",)
    assert resolve(reg.get("create_workflow")).runtime_contract.allowed_roles == ("main",)
    assert "spawn_agent" in role_tools(reg.names(), "main", registry=reg)
    assert "create_workflow" in role_tools(reg.names(), "main", registry=reg)
    for restricted_role in ("subagent", "workflow_node"):
        exposed = role_tools(reg.names(), restricted_role, registry=reg)
        assert "spawn_agent" not in exposed, (restricted_role, exposed)
        assert "create_workflow" not in exposed, (restricted_role, exposed)

    # 2) Real Worker verification: main spawns subagent (allowed), Driver creates workflow leaf (allowed),
    #    both subagent and workflow leaf attempt spawn_agent + create_workflow (both rejected, 0 children created!)
    seen_tools_by_sid: dict[str, list[str]] = {}
    runtime_checks: dict[str, dict[str, str]] = {}
    calls_per_sid: dict[str, int] = {}

    class _DelegationProbeModel:
        async def complete(self, request, **_kwargs):
            from power_loop.core.execution_context import get_execution_context
            sid = get_execution_context().session_id
            calls_per_sid[sid] = calls_per_sid.get(sid, 0) + 1
            seen_tools_by_sid[sid] = [
                t["function"]["name"] for t in (request.tools or []) if isinstance(t, dict) and "function" in t
            ]
            # Also probe runtime permission checks directly inside the live loop context!
            if calls_per_sid[sid] == 1 and sid != "r3-perm-main":
                loop = get_current_loop()
                checks: dict[str, str] = {}
                for cap in ("subagent", "workflow"):
                    try:
                        await require_orchestration(loop, cap)
                        checks[cap] = "allowed"
                    except ValueError as exc:
                        checks[cap] = str(exc)
                checks["direct_spawn_handler"] = await _handle_spawn_agent(task="must fail")
                runtime_checks[sid] = checks
                # Also force tool calls for BOTH spawn_agent and create_workflow to prove runtime rejection!
                return LLMResponse(
                    raw_text="",
                    tool_calls=[
                        {
                            "id": f"tc-spawn-{sid}",
                            "type": "function",
                            "function": {"name": "spawn_agent", "arguments": json.dumps({"task": "illegal nested"})},
                        },
                        {
                            "id": f"tc-wf-{sid}",
                            "type": "function",
                            "function": {
                                "name": "create_workflow",
                                "arguments": json.dumps({
                                    "name": "nested_wf",
                                    "root": {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "hi"}},
                                }),
                            },
                        },
                    ],
                )
            if sid == "r3-perm-main" and calls_per_sid[sid] == 1:
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "tc-main-spawn",
                        "type": "function",
                        "function": {"name": "spawn_agent", "arguments": json.dumps({"task": "sub work"})},
                    }],
                )
            return LLMResponse(raw_text=f"done:{sid}")

        async def close(self):
            pass

    probe_model = _DelegationProbeModel()
    factory = RegistryRuntimeFactory()

    async def _build(_d, _c):
        local_reg = ToolRegistry()
        register_spawn_agent(local_reg)
        register_workflow_tools(local_reg)
        return ExecutionEnvironment(
            probe_model, local_reg, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("test.agent@215", _build)
    rt = Runtime(
        store=store, factory=factory, node_id="r3-perm-node", boot_id="r3-perm-boot",
        capacity=4, recovery_interval_s=None, timers_enabled=False,
    )
    await rt.worker.start()
    try:
        main = await rt.client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "r3-perm-main"})
        main_rcpt = await rt.client.send(main, "start sub", message_id="m1", actor=ACTOR)
        main_res = await rt.client.wait(main_rcpt.input_ref, actor=ACTOR, deadline_ms=now_ms() + 5000)
        assert main_res.status == "completed"
        assert "spawn_agent" in seen_tools_by_sid["r3-perm-main"]
        assert "create_workflow" in seen_tools_by_sid["r3-perm-main"]

        # Main's spawn_agent MUST have created exactly 1 subagent child (depth=1)
        sub_rows = await store._db.fetchall(
            f"SELECT session_id,start_input_message_id,depth FROM {store.t.sessions} WHERE parent_session_id=?",
            ("r3-perm-main",),
        )
        assert len(sub_rows) == 1 and int(sub_rows[0]["depth"]) == 1
        sub_sid = sub_rows[0]["session_id"]
        sub_res = await rt.client.wait(
            InputRef(store.store_id, sub_sid, sub_rows[0]["start_input_message_id"]),
            actor=ACTOR, deadline_ms=now_ms() + 5000,
        )
        assert sub_res.status == "completed"
        assert "spawn_agent" not in seen_tools_by_sid[sub_sid]
        assert "create_workflow" not in seen_tools_by_sid[sub_sid]
        assert runtime_checks[sub_sid]["subagent"] == "execution_role_forbids_subagent"
        assert runtime_checks[sub_sid]["workflow"] == "execution_role_forbids_workflow"
        assert "execution_role_forbids_subagent" in runtime_checks[sub_sid]["direct_spawn_handler"]
        # Subagent MUST NOT have created any children!
        sub_children = await store._db.fetchall(
            f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=?", (sub_sid,),
        )
        assert sub_children == []

        # Driver creates a workflow_node leaf (depth=2) -> ALLOWED; leaf trying spawn_agent/workflow -> BLOCKED!
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        cm_drv = await rt.client.inputs.create_many(
            parent=main, actor=ACTOR,
            specs=[ChildSpec(
                session_id="r3-perm-drv", message_id="dm1", body={"wf": 1},
                descriptor=drv_desc, result_kind="workflow_result",
            )],
            node_id="r3-perm-node", boot_id="r3-perm-boot", can_execute=True, free_slots=1,
        )
        drv_batch = cm_drv.preclaimed[0]
        leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "leaf1"}})
        cm_leaf = await rt.client.inputs.create_many(
            parent=drv_batch.target, actor=ACTOR,
            specs=[ChildSpec(
                session_id="r3-perm-leaf", message_id="lm1", body={"text": "run leaf"},
                descriptor=leaf_desc, result_kind="workflow_node_result",
            )],
            node_id="r3-perm-node", boot_id="r3-perm-boot", can_execute=False, free_slots=0,
            parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
        )
        assert cm_leaf.queued_targets == (TargetRef(store.store_id, "r3-perm-leaf"),)
        rt.worker._activate("r3-perm-leaf")
        leaf_res = await rt.client.wait(
            InputRef(store.store_id, "r3-perm-leaf", "lm1"), actor=ACTOR, deadline_ms=now_ms() + 5000,
        )
        assert leaf_res.status == "completed"
        assert "spawn_agent" not in seen_tools_by_sid["r3-perm-leaf"]
        assert "create_workflow" not in seen_tools_by_sid["r3-perm-leaf"]
        assert runtime_checks["r3-perm-leaf"]["subagent"] == "execution_role_forbids_subagent"
        assert runtime_checks["r3-perm-leaf"]["workflow"] == "execution_role_forbids_workflow"
        assert "execution_role_forbids_subagent" in runtime_checks["r3-perm-leaf"]["direct_spawn_handler"]
        # Workflow leaf MUST NOT have created any depth=3 child session!
        leaf_children = await store._db.fetchall(
            f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? OR depth>=3",
            ("r3-perm-leaf",),
        )
        assert leaf_children == []
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_m1c_r3_runtime_state_write_no_extra_route_sql_vs_real_rebind(registry_store):
    """R3-Item3: Ordinary set_runtime_state() adds 0 route-lookup SQL in Final; real rebind still reroutes mixed batch."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    p_old = await target(port, "r3-rt-old")
    p_new = await target(port, "r3-rt-new")

    # 1) Control: child with per-input route writes runtime state (runtime_version 0 -> 1), NO rebind!
    ch_kv = await target(port, "r3-rt-ch-kv", parent=p_old)
    await port.accept(ch_kv, "direct", message_id="d0", actor=ACTOR)
    await port.accept(
        ch_kv, "routed", message_id="r0", actor=ACTOR,
        reply_route={"target": p_old, "result_kind": "agent_result"},
    )
    b_kv = await port.claim(ch_kv, node_id="n1", boot_id="b1")
    assert b_kv is not None and b_kv.runtime_version == 0 and b_kv.reply_to_item_id == "r0"

    await store.set_runtime_state(ch_kv.session_id, "step", {"n": 1})
    s_kv = await store._db.fetchone(
        f"SELECT runtime_version,parent_session_id FROM {store.t.sessions} WHERE session_id=?",
        (ch_kv.session_id,),
    )
    assert int(s_kv["runtime_version"]) == 1 and s_kv["parent_session_id"] == p_old.session_id

    # Baseline: identical child WITHOUT set_runtime_state()
    ch_plain = await target(port, "r3-rt-ch-plain", parent=p_old)
    await port.accept(ch_plain, "direct", message_id="d0", actor=ACTOR)
    await port.accept(
        ch_plain, "routed", message_id="r0", actor=ACTOR,
        reply_route={"target": p_old, "result_kind": "agent_result"},
    )
    b_plain = await port.claim(ch_plain, node_id="n1", boot_id="b1")
    assert b_plain is not None

    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        profiled.records.clear()
        fb_plain = await port.final(b_plain, final_text="plain ok")
        plain_sql = list(profiled.records)

        profiled.records.clear()
        fb_kv = await port.final(b_kv, final_text="kv ok")
        kv_sql = list(profiled.records)
    finally:
        store._db = original

    assert fb_plain.result_delivery_error is None and fb_kv.result_delivery_error is None
    assert not any(
        "reply_to_session_id" in r["sql"] and "item_id=?" in r["sql"] for r in kv_sql
    ), [r["sql"] for r in kv_sql]
    expected_pi = 7 if original.dialect.name == "mysql" else 6
    assert len(kv_sql) == len(plain_sql) == expected_pi, (
        [r["sql"] for r in plain_sql], [r["sql"] for r in kv_sql],
    )

    # 2) Contrast: child writes runtime state AND actually rebinds (old_p -> new_p) with mixed batch!
    ch_rb = await target(port, "r3-rt-ch-rb", parent=p_old)
    await port.accept(ch_rb, "direct", message_id="d1", actor=ACTOR)
    await port.accept(
        ch_rb, "routed", message_id="r1", actor=ACTOR,
        reply_route={"target": p_old, "result_kind": "agent_result"},
    )
    b_rb = await port.claim(ch_rb, node_id="n1", boot_id="b1")
    assert b_rb is not None and b_rb.reply_to_item_id == "r1"
    await store.set_runtime_state(ch_rb.session_id, "step", {"n": 2})
    rb = await port.rebind_children(p_old, p_new, child_targets=[ch_rb], actor=ACTOR)
    assert rb.rebound_children == (ch_rb.session_id,)

    fb_rb = await port.final(b_rb, final_text="rebound after kv")
    assert fb_rb.result_delivery_error is None and fb_rb.result_activation == p_new
    new_rows = await store._db.fetchall(
        f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=?", (p_new.session_id,),
    )
    assert len(new_rows) == 1
    body = json.loads(new_rows[0]["content_json"])
    assert body["final_text"] == "rebound after kv"
    assert body["input_ref"]["message_id"] == "r1"


# ── 10. Round-4 Rework Regressions ──────────────────────────────────────────


@pytest.mark.parametrize("pass_consumed", [False, True])
async def test_m1c_r4_double_rebind_aba_mixed_batch_and_kv_zero_extra_sql(
    registry_store, pass_consumed,
):
    """R4-Item1: Double rebind A->B->A on cross-parent mixed batch delivers only to A; no-rebind B & KV stay exact."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        sfx = "cons" if pass_consumed else "none"
        main_a = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": f"r4-a-{sfx}"})
        main_b = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": f"r4-b-{sfx}"})
        child_c = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=main_a, options={"session_id": f"r4-c-{sfx}"},
        )

        # Public send: unrouted direct input + explicit route targeting B (cross-parent!)
        await client.send(child_c, "direct work", message_id="direct", actor=ACTOR)
        await client.send(
            child_c, "routed work", message_id="routed", actor=ACTOR,
            reply_route={"target": main_b, "result_kind": "agent_result"},
        )
        batch = await port.claim(child_c, node_id="n1", boot_id="b1")
        assert batch is not None
        assert batch.parent_session_id == main_a.session_id
        assert batch.reply_to_session_id == main_b.session_id
        assert batch.reply_to_item_id == "routed"

        # Double rebind: A -> B, then B -> A (parent_session_id returns to A, while routed input changed B -> A!)
        rb1 = await port.rebind_children(main_a, main_b, actor=ACTOR)
        assert rb1.rebound_children == (child_c.session_id,)
        rb2 = await port.rebind_children(main_b, main_a, actor=ACTOR)
        assert rb2.rebound_children == (child_c.session_id,)

        fb = await port.final(
            batch,
            consumed=(batch.input_refs if pass_consumed else None),
            final_text="double rebound answer",
        )
        assert fb.result_delivery_error is None
        assert fb.result_activation == main_a

        a_inbox = await store._db.fetchall(
            f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=?", (main_a.session_id,),
        )
        b_inbox = await store._db.fetchall(
            f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=?", (main_b.session_id,),
        )
        assert len(a_inbox) == 1
        assert b_inbox == []
        a_body = json.loads(a_inbox[0]["content_json"])
        assert a_body["final_text"] == "double rebound answer"
        assert a_body["input_ref"]["message_id"] == "routed"

        # Control (no rebind): explicit route to B still delivers to B, AND set_runtime_state() keeps 6/6/7 SQL!
        if not pass_consumed:
            child_ctrl = await client.create_session(
                DESCRIPTOR, actor=ACTOR, parent=main_a, options={"session_id": "r4-c-ctrl"},
            )
            await client.send(child_ctrl, "d", message_id="d_ctrl", actor=ACTOR)
            await client.send(
                child_ctrl, "r", message_id="r_ctrl", actor=ACTOR,
                reply_route={"target": main_b, "result_kind": "agent_result"},
            )
            b_ctrl = await port.claim(child_ctrl, node_id="n1", boot_id="b1")
            assert b_ctrl is not None
            await store.set_runtime_state(child_ctrl.session_id, "k", {"v": 1})

            original = store._db
            profiled = Profiled(original)
            store._db = profiled
            try:
                profiled.records.clear()
                fb_ctrl = await port.final(b_ctrl, final_text="stay on B")
                ctrl_sql = list(profiled.records)
            finally:
                store._db = original
            assert fb_ctrl.result_delivery_error is None
            assert fb_ctrl.result_activation == main_b
            expected_pi = 7 if original.dialect.name == "mysql" else 6
            assert len(ctrl_sql) == expected_pi, [r["sql"] for r in ctrl_sql]
            assert not any("reply_to_session_id" in r["sql"] and "item_id=?" in r["sql"] for r in ctrl_sql)
            b_rows = await store._db.fetchall(
                f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=?", (main_b.session_id,),
            )
            assert len(b_rows) == 1
            assert json.loads(b_rows[0]["content_json"])["input_ref"]["message_id"] == "r_ctrl"
    finally:
        await client.aclose()
        await transport.aclose()


async def test_m1c_r4_public_runtime_client_workflow_blocks_subagent_and_leaf(registry_store):
    """R4-Item2: Public RuntimeClient.workflow rejects subagent & workflow leaf; allows main & Driver leaf creation."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    wf_spec = {
        "name": "nested_probe",
        "root": {"type": "agent", "id": "step1", "spec": {"name": "s1", "system_prompt": "ok"}},
    }
    try:
        main = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "r4-wf-main"})
        sub_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "child_spec": {"name": "sub"}})
        sub = await client.create_session(
            sub_desc, actor=ACTOR, parent=main, options={"session_id": "r4-wf-sub"},
        )

        # Positive: main agent submits workflow via public RuntimeClient.workflow -> creates & queues Driver (depth=1)
        main_actor = TrustedActor(ACTOR.tenant_id, ACTOR.principal_id, main.session_id)
        handle = await client.workflow(main, wf_spec, message_id="wf-main-1", actor=main_actor)
        drv_row = await store._db.fetchone(
            f"SELECT session_id,parent_session_id,session_kind,depth,execution_state "
            f"FROM {store.t.sessions} WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert drv_row is not None
        assert drv_row["parent_session_id"] == main.session_id
        assert drv_row["session_kind"] == "workflow_driver"
        assert int(drv_row["depth"]) == 1
        assert drv_row["execution_state"] == "queued"

        # Running Driver creates a legitimate workflow leaf (depth=2) -> ALLOWED
        drv_batch = await port.claim(handle.target, node_id="n-wf", boot_id="b-wf")
        assert drv_batch is not None
        leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "step1"}})
        cm_leaf = await port.create_many(
            parent=handle.target, actor=ACTOR,
            specs=[ChildSpec(
                session_id="r4-wf-leaf", message_id="lm1", body={"text": "work"},
                descriptor=leaf_desc, result_kind="workflow_node_result",
            )],
            node_id="n-wf", boot_id="b-wf", can_execute=False, free_slots=0,
            parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
        )
        assert cm_leaf.queued_targets == (TargetRef(store.store_id, "r4-wf-leaf"),)
        leaf = cm_leaf.queued_targets[0]

        before_sessions = await store._db.fetchall(f"SELECT session_id FROM {store.t.sessions}")
        before_queue = await store._db.fetchall(f"SELECT session_id,item_id FROM {store.t.follow_up_queue}")

        # Negative 1: Ordinary subagent calling public RuntimeClient.workflow -> REJECTED!
        sub_actor = TrustedActor(ACTOR.tenant_id, ACTOR.principal_id, sub.session_id)
        with pytest.raises(RuntimeRequestError) as sub_exc:
            await client.workflow(sub, wf_spec, message_id="wf-sub-illegal", actor=sub_actor)
        assert sub_exc.value.code == "forbidden"
        assert "execution_role_forbids_workflow" in str(sub_exc.value)

        # Negative 2: Legitimate workflow leaf calling public RuntimeClient.workflow -> REJECTED!
        leaf_actor = TrustedActor(ACTOR.tenant_id, ACTOR.principal_id, leaf.session_id)
        with pytest.raises(RuntimeRequestError) as leaf_exc:
            await client.workflow(leaf, wf_spec, message_id="wf-leaf-illegal", actor=leaf_actor)
        assert leaf_exc.value.code == "forbidden"
        assert "execution_role_forbids_workflow" in str(leaf_exc.value)

        # Verify 0 new Driver sessions (depth=3 == 0) and 0 new start inputs were created!
        after_sessions = await store._db.fetchall(f"SELECT session_id FROM {store.t.sessions}")
        after_queue = await store._db.fetchall(f"SELECT session_id,item_id FROM {store.t.follow_up_queue}")
        assert len(after_sessions) == len(before_sessions)
        assert len(after_queue) == len(before_queue)
        deep = await store._db.fetchall(f"SELECT session_id FROM {store.t.sessions} WHERE depth>=3")
        assert deep == []
    finally:
        await client.aclose()
        await transport.aclose()


async def test_m1c_r4_main_spawn_agent_send_input_zero_role_query(registry_store, monkeypatch):
    """R4-Item3: Real Worker main agent spawn_agent(action='send') delivers to direct child with 0 role-lookup SQL."""
    from power_loop.runtime import child_tasks
    from power_loop.tools.spawn_agent import register_spawn_agent

    store, _ = registry_store
    original = store._db
    profiled = Profiled(original)
    store._db = profiled

    send_input_slices: list[list[dict[str, Any]]] = []
    orig_send_input = child_tasks.send_input

    async def metered_send_input(loop, sid, message, *, urgent=False):
        start_idx = len(profiled.records)
        res = await orig_send_input(loop, sid, message, urgent=urgent)
        send_input_slices.append(list(profiled.records[start_idx:]))
        return res

    monkeypatch.setattr(child_tasks, "send_input", metered_send_input)

    child_seen_messages: list[str] = []
    created_child_sid: list[str] = []
    main_step = [0]
    child_entered = asyncio.Event()
    child_release = asyncio.Event()

    class _SendInputModel:
        async def complete(self, request, **_kwargs):
            from power_loop.core.execution_context import get_execution_context
            sid = get_execution_context().session_id
            if sid == "r4-send-main":
                main_step[0] += 1
                if main_step[0] == 1:
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "tc-create-1",
                            "type": "function",
                            "function": {"name": "spawn_agent", "arguments": json.dumps({"task": "first work"})},
                        }],
                    )
                if main_step[0] == 2:
                    await asyncio.wait_for(child_entered.wait(), 5)
                    # Extract child session_id from previous tool output
                    tool_msg = next(m for m in reversed(request.messages) if m.get("role") == "tool")
                    payload = json.loads(tool_msg["content"])
                    created_child_sid.append(payload["session_id"])
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "tc-send-1",
                            "type": "function",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": json.dumps({
                                    "action": "send",
                                    "sub_session": payload["session_id"],
                                    "message": "follow up instruction",
                                }),
                            },
                        }],
                    )
                child_release.set()
                return LLMResponse(raw_text="main done")
            # Direct child session pauses during turn 1 so send_input SQL has 0 concurrent noise!
            if not child_entered.is_set():
                child_entered.set()
                await asyncio.wait_for(child_release.wait(), 5)
            for m in request.messages:
                if m.get("role") == "user" and m.get("content"):
                    child_seen_messages.append(str(m["content"]))
            return LLMResponse(raw_text="child done")

        async def close(self):
            pass

    model = _SendInputModel()
    factory = RegistryRuntimeFactory()

    async def _build(_d, _c):
        reg = ToolRegistry()
        register_spawn_agent(reg)
        return ExecutionEnvironment(model, reg, AgentLoopConfig(context_budget_tokens=10**9))

    factory.register("test.agent@215", _build)
    rt = Runtime(
        store=store, factory=factory, node_id="r4-send-node", boot_id="r4-send-boot",
        capacity=4, recovery_interval_s=None, timers_enabled=False,
    )
    await rt.worker.start()
    try:
        main = await rt.client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "r4-send-main"})
        rcpt = await rt.client.send(main, "delegate and follow up", message_id="m1", actor=ACTOR)
        res = await rt.client.wait(rcpt.input_ref, actor=ACTOR, deadline_ms=now_ms() + 5000)
        assert res.status == "completed"
        assert len(created_child_sid) == 1
        assert len(send_input_slices) == 1

        # Wait for the follow-up message to arrive in child's durable inbox / execution
        c_sid = created_child_sid[0]
        q_rows = await original.fetchall(
            f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=?", (c_sid,),
        )
        assert any("follow up instruction" in str(r["content_json"]) for r in q_rows), q_rows

        # Verify ZERO role-lookup SQL in send_input (SELECT parent_session_id,session_kind,descriptor_json)
        si_recs = send_input_slices[0]
        role_queries = [
            r["sql"] for r in si_recs
            if "session_kind" in r["sql"] and "descriptor_json" in r["sql"]
        ]
        assert role_queries == [], f"unexpected role-lookup SQL in send_input: {role_queries}"
        # Total send_input SQL stays at exact R2 count (3 SQLite/PG, <= 5 MySQL) with 0 role reads!
        max_si_sql = 5 if original.dialect.name == "mysql" else 3
        assert len(si_recs) <= max_si_sql, [r["sql"] for r in si_recs]
    finally:
        child_release.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        store._db = original
