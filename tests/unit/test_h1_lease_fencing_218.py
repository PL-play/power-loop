"""Design 218 H1 regression tests: P0-1~P0-5 lease fencing, store locks, and discover_queued."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    HookPoint,
    LLMResponse,
    SessionLeaseLost,
    TargetRef,
    ToolDefinition,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.runtime.pricing import Free
from power_loop.runtime.session_ownership import SessionOwnership
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent
from tests.unit.session_native_fixture import ACTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _ScriptedModel:
    def __init__(self, fn):
        self._fn = fn
        self._idx = 0

    async def complete(self, request, **kwargs):
        idx = self._idx
        self._idx += 1
        res = self._fn(idx, request)
        if asyncio.iscoroutine(res):
            return await res
        return res

    async def close(self):
        return None


def _single_tool_registry(name: str, fn):
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name, f"{name} tool", input_schema={"type": "object"}, pricing=Free("f")),
        fn,
    )
    return reg


# ── P0-1: batch_renew orphan lease release + fusing guard + 1-SQL hot path ──


async def test_h1_p0_1_batch_renew_releases_orphan_and_preserves_fusing_and_hot_path(registry_store):
    store, _ = registry_store
    tool_entered = asyncio.Event()
    tool_release = asyncio.Event()

    async def hold_tool() -> str:
        tool_entered.set()
        await asyncio.wait_for(tool_release.wait(), 5)
        return "held_ok"

    def model_fn(idx, _req):
        if idx == 0:
            return LLMResponse(
                raw_text="calling hold",
                tool_calls=[{"id": "tc_p01", "type": "function", "function": {"name": "hold", "arguments": "{}"}}],
            )
        return LLMResponse(raw_text="active_done")

    hooks = AgentHooks()
    fuse_renew_checked = {"ran": False, "lease_during_fuse": None}
    worker_box: list[Any] = []

    async def on_accepted(ctx):
        receipt = ctx.values["receipt"]
        if receipt.target.session_id == "h1-p01-fused" and worker_box:
            # Trigger batch_renew right after accept_and_claim committed in DB,
            # BEFORE RuntimeClient.send calls worker.dispatch_claimed(batch).
            w = worker_box[0]
            assert "h1-p01-fused" in w._fusing
            assert "h1-p01-fused" not in w._active
            await w.batch_renew()
            row = await store._db.fetchone(
                f"SELECT owner_lease_until_ms,execution_state FROM {store.t.sessions} WHERE session_id=?",
                ("h1-p01-fused",),
            )
            fuse_renew_checked["ran"] = True
            fuse_renew_checked["lease_during_fuse"] = int(row["owner_lease_until_ms"])

    hooks.register(HookPoint.MESSAGE_ACCEPTED, on_accepted)

    agent = await RunningAgent(
        store,
        _ScriptedModel(model_fn),
        _single_tool_registry("hold", hold_tool),
        AgentLoopConfig(context_budget_tokens=10**9),
        hooks=hooks,
        fuse_local_send=True,
        capacity=4,
    ).start()
    worker = agent.runtime.worker
    agent.runtime.client.hooks = hooks
    worker_box.append(worker)
    port = RuntimeInputStore(store)
    try:
        # 1. Start a genuinely active session and pause inside its tool.
        active_task = asyncio.create_task(agent.send("active work"))
        await asyncio.wait_for(tool_entered.wait(), 5)

        # Hot-path check: healthy active session executes ONLY batch_renew SQL (1 on SQLite/PG, 2 on MySQL).
        seen_sqls: list[str] = []
        orig_exec = store._db.execute
        orig_fetchall = store._db.fetchall

        async def _rec_exec(sql, params=()):
            seen_sqls.append(sql)
            return await orig_exec(sql, params)

        async def _rec_fetchall(sql, params=()):
            seen_sqls.append(sql)
            return await orig_fetchall(sql, params)

        store._db.execute = _rec_exec
        store._db.fetchall = _rec_fetchall
        try:
            await worker.batch_renew()
        finally:
            store._db.execute = orig_exec
            store._db.fetchall = orig_fetchall
        expected_hot = 2 if store._db.dialect.name == "mysql" else 1
        assert len(seen_sqls) == expected_hot, f"unexpected hot-path SQLs: {seen_sqls}"

        # 2. Construct an orphan running row owned by this same (node_id, boot_id)
        #    that is NOT in worker._active.
        orphan_target = await target(port, "h1-p01-orphan")
        await port.accept(orphan_target, "orphan work", message_id="m_orph", actor=ACTOR)
        orphan_batch = await port.claim(
            orphan_target,
            node_id=agent.runtime.node_id,
            boot_id=agent.runtime.boot_id,
            ttl_s=60.0,
        )
        assert orphan_batch is not None
        assert orphan_target.session_id not in worker._active

        await worker.batch_renew()
        orphan_row = await store._db.fetchone(
            f"SELECT owner_lease_until_ms,execution_state FROM {store.t.sessions} WHERE session_id=?",
            (orphan_target.session_id,),
        )
        assert orphan_row["execution_state"] == "running"
        assert int(orphan_row["owner_lease_until_ms"]) == 0

        # Active session's lease MUST remain healthy!
        active_row = await store._db.fetchone(
            f"SELECT owner_lease_until_ms FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        assert int(active_row["owner_lease_until_ms"]) > 0

        # Orphan is immediately reclaimable by another node without waiting for TTL!
        reclaimed = await port.reclaim_running(
            orphan_target,
            expected_epoch=orphan_batch.owner_epoch,
            node_id="other-node",
            boot_id="other-boot",
        )
        assert reclaimed is not None
        assert reclaimed.owner_epoch == orphan_batch.owner_epoch + 1

        # 3. Epoch-mismatch orphan: worker._active has epoch E, DB has epoch E+1 on same node/boot.
        mismatch_target = await target(port, "h1-p01-mismatch")
        await port.accept(mismatch_target, "mm work", message_id="m_mm", actor=ACTOR)
        mm_batch = await port.claim(
            mismatch_target,
            node_id=agent.runtime.node_id,
            boot_id=agent.runtime.boot_id,
            ttl_s=60.0,
        )
        assert mm_batch is not None
        # Advance DB owner_epoch on the same (node_id, boot_id) to simulate stale local ctx.
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_epoch=owner_epoch+1 WHERE session_id=?",
            (mismatch_target.session_id,),
        )
        stale_ownership = SessionOwnership(
            (store.store_id, mismatch_target.session_id),
            agent.runtime.node_id + "/" + agent.runtime.boot_id,
            time.monotonic() + 60.0,
            lambda: None,
            epoch=mm_batch.owner_epoch,
            node_id=agent.runtime.node_id,
            boot_id=agent.runtime.boot_id,
            installs_fence=True,
        )

        class _FakeCtx:
            closing = False
            finalizing = False
            finalized = False
            ownership = stale_ownership
            batch = mm_batch

        worker._active[mismatch_target.session_id] = _FakeCtx()
        try:
            await worker.batch_renew()
        finally:
            worker._active.pop(mismatch_target.session_id, None)
        assert stale_ownership.lost is True
        mm_row = await store._db.fetchone(
            f"SELECT owner_lease_until_ms FROM {store.t.sessions} WHERE session_id=?",
            (mismatch_target.session_id,),
        )
        assert int(mm_row["owner_lease_until_ms"]) == 0

        # 4. In-flight fused accept_and_claim MUST NOT be falsely released by concurrent batch_renew.
        fused_target = await agent.runtime.client.create_session(
            agent.descriptor, actor=agent.actor, options={"session_id": "h1-p01-fused"},
        )
        receipt = await agent.runtime.client.send(
            fused_target, "fused hello", message_id="m_fused", actor=agent.actor,
        )
        fused_res = await asyncio.wait_for(
            agent.runtime.client.wait(receipt.input_ref, actor=agent.actor), 5.0,
        )
        assert fuse_renew_checked["ran"] is True
        assert (fuse_renew_checked["lease_during_fuse"] or 0) > 0
        assert fused_res.status == "completed"
        assert fused_res.final_text == "active_done"

        # Finish original active session cleanly.
        tool_release.set()
        active_res = await asyncio.wait_for(active_task, 5)
        assert active_res.status == "completed"
        assert active_res.final_text == "active_done"
    finally:
        tool_release.set()
        await agent.aclose()


async def test_h1_p0_1_workflow_create_many_and_continue_protected_during_commit_to_dispatch(registry_store):
    """P0-1 rework: workflow create_many and workflow_continue accept_and_claim are protected
    from batch_renew between DB commit and worker.dispatch_claimed, while a crashed/un-dispatched
    claiming task still gets released as an orphan once that task finishes."""
    store, _ = registry_store
    loop_calls = 0

    def model_fn(_idx, request):
        nonlocal loop_calls
        sys_text = str(getattr(request, "system_prompt", "") or "")
        if sys_text.startswith("WF_LOOP"):
            loop_calls += 1
            return LLMResponse(raw_text="step1" if loop_calls == 1 else "done")
        return LLMResponse(raw_text="main_ack")

    agent = await RunningAgent(
        store,
        _ScriptedModel(model_fn),
        None,
        AgentLoopConfig(context_budget_tokens=10**9),
        fuse_local_send=True,
        capacity=4,
    ).start()
    worker = agent.runtime.worker
    orig_create_many = worker.inputs.create_many
    orig_accept_and_claim = worker.inputs.accept_and_claim
    observed: dict[str, list[tuple[str, int, int]]] = {"create_many": [], "workflow_continue": []}

    async def _wrapped_create_many(*args, **kwargs):
        cm = await orig_create_many(*args, **kwargs)
        if cm.preclaimed:
            # Real batch_renew right after create_many committed, BEFORE workflow runner dispatches.
            await worker.batch_renew()
            for b in cm.preclaimed:
                assert b.target.session_id not in worker._active
                row = await store._db.fetchone(
                    f"SELECT owner_epoch,owner_lease_until_ms FROM {store.t.sessions} WHERE session_id=?",
                    (b.target.session_id,),
                )
                observed["create_many"].append(
                    (b.target.session_id, int(row["owner_epoch"]), int(row["owner_lease_until_ms"]))
                )
        return cm

    async def _wrapped_accept_and_claim(*args, **kwargs):
        receipt, batch, activate = await orig_accept_and_claim(*args, **kwargs)
        if batch is not None and kwargs.get("kind") == "workflow_continue":
            # Real batch_renew right after workflow_continue committed, BEFORE runner dispatches.
            await worker.batch_renew()
            assert batch.target.session_id not in worker._active
            row = await store._db.fetchone(
                f"SELECT owner_epoch,owner_lease_until_ms FROM {store.t.sessions} WHERE session_id=?",
                (batch.target.session_id,),
            )
            observed["workflow_continue"].append(
                (batch.target.session_id, int(row["owner_epoch"]), int(row["owner_lease_until_ms"]))
            )
        return receipt, batch, activate

    worker.inputs.create_many = _wrapped_create_many
    worker.inputs.accept_and_claim = _wrapped_accept_and_claim
    try:
        spec = {
            "name": "h1-p01-wf",
            "input": "go",
            "root": {
                "type": "loop",
                "id": "L",
                "max_iterations": 3,
                "continue_nodes": ["LOOP"],
                "body": {
                    "type": "sequence",
                    "id": "S",
                    "steps": [
                        {
                            "type": "agent",
                            "id": "LOOP",
                            "spec": {"name": "LOOP", "system_prompt": "WF_LOOP", "tools": []},
                        },
                        {
                            "type": "agent",
                            "id": "STEP2",
                            "spec": {"name": "STEP2", "system_prompt": "WF_STEP2", "tools": []},
                        },
                    ],
                },
                "until": {"ref": "LOOP", "op": "eq", "value": "done"},
            },
        }
        handle = await agent.runtime.client.workflow(
            agent.target, spec, message_id="wf-h1-p01", actor=agent.actor,
        )
        res = await asyncio.wait_for(handle.wait(), 10.0)
        assert res.status == "completed", res
        assert res.results["LOOP"].text == "done"

        # 1. Both workflow paths were hit and kept positive leases through batch_renew!
        assert len(observed["create_many"]) >= 1
        assert all(ep == 1 and lease > 0 for _, ep, lease in observed["create_many"])
        cm_sids = {sid for sid, _, _ in observed["create_many"]}

        assert len(observed["workflow_continue"]) == 1
        wc_sid, wc_epoch, wc_lease = observed["workflow_continue"][0]
        assert wc_sid in cm_sids and wc_epoch == 2 and wc_lease > 0

        # 2. A claiming task that exits WITHOUT calling dispatch_claimed MUST become a
        #    true orphan and have its lease zeroed by the next batch_renew!
        worker.inputs.accept_and_claim = orig_accept_and_claim
        abandoned = await agent.runtime.client.create_session(
            agent.descriptor, actor=agent.actor, options={"session_id": "h1-p01-abandoned"},
        )

        async def _claim_and_die():
            _, b, _ = await worker.inputs.accept_and_claim(
                abandoned, "orphan task", message_id="m_die", actor=agent.actor,
                node_id=agent.runtime.node_id, boot_id=agent.runtime.boot_id,
                ttl_s=60.0,
            )
            assert b is not None
            return b

        dead_task = asyncio.create_task(_claim_and_die())
        abandoned_batch = await dead_task
        assert dead_task.done()
        await worker.batch_renew()
        ab_row = await store._db.fetchone(
            f"SELECT owner_lease_until_ms,owner_epoch FROM {store.t.sessions} WHERE session_id=?",
            (abandoned.session_id,),
        )
        assert int(ab_row["owner_epoch"]) == abandoned_batch.owner_epoch
        assert int(ab_row["owner_lease_until_ms"]) == 0
    finally:
        worker.inputs.create_many = orig_create_many
        worker.inputs.accept_and_claim = orig_accept_and_claim
        await agent.aclose()


# ── P0-2: _reclaim_single_session mandatory expiry check + discover <= ──────


async def test_h1_p0_2_reclaim_running_enforces_lease_expiry_with_expected_epoch(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    t = await target(port, "h1-p02-sess")
    await port.accept(t, "work", message_id="m1", actor=ACTOR)
    batch = await port.claim(t, node_id="n1", boot_id="b1", ttl_s=60.0)
    assert batch is not None
    orig_epoch = batch.owner_epoch

    # Healthy lease: reclaim_running with expected_epoch MUST fail!
    stolen = await port.reclaim_running(
        t, expected_epoch=orig_epoch, node_id="n2", boot_id="b2", ttl_s=30.0,
    )
    assert stolen is None

    # Simulate expiry discovery at boundary (owner_lease_until_ms == db_now -> discover <= finds it).
    db_now = await port._db_now(store._db)
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=? WHERE session_id=?",
        (db_now, t.session_id),
    )
    expired = await port.discover_expired_leases(limit=32)
    assert any(r["session_id"] == t.session_id and int(r["owner_epoch"]) == orig_epoch for r in expired)

    # Simulate original owner renewing its lease BEFORE n2 calls reclaim_running(expected_epoch=orig_epoch).
    # Note: batch_renew extends owner_lease_until_ms without bumping owner_epoch!
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=? WHERE session_id=?",
        (db_now + 60_000, t.session_id),
    )
    renewed_rows = await port.batch_renew(node_id="n1", boot_id="b1", ttl_s=60.0)
    assert any(r["session_id"] == t.session_id and int(r["owner_epoch"]) == orig_epoch for r in renewed_rows)

    # Stale reclaim using previously discovered expected_epoch MUST fail because lease is healthy!
    stolen_after_renew = await port.reclaim_running(
        t, expected_epoch=orig_epoch, node_id="n2", boot_id="b2", ttl_s=30.0,
    )
    assert stolen_after_renew is None
    row = await store._db.fetchone(
        f"SELECT owner_node_id,owner_boot_id,owner_epoch,owner_lease_until_ms "
        f"FROM {store.t.sessions} WHERE session_id=?",
        (t.session_id,),
    )
    assert row["owner_node_id"] == "n1"
    assert row["owner_boot_id"] == "b1"
    assert int(row["owner_epoch"]) == orig_epoch
    assert int(row["owner_lease_until_ms"]) > db_now

    # Once genuinely expired, reclaim_running(expected_epoch=orig_epoch) succeeds.
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
        (t.session_id,),
    )
    reclaimed = await port.reclaim_running(
        t, expected_epoch=orig_epoch, node_id="n2", boot_id="b2", ttl_s=30.0,
    )
    assert reclaimed is not None
    assert reclaimed.owner_epoch == orig_epoch + 1


# ── P0-3: append_message_fenced FOR SHARE dialect & cross-connection lock ──


async def test_h1_p0_3_append_message_fenced_for_share_and_serialization(registry_store):
    store, connect = registry_store
    dialect = store._db.dialect.name
    port = RuntimeInputStore(store)
    t = await target(port, "h1-p03-sess")
    await port.accept(t, "hello", message_id="m1", actor=ACTOR)
    batch = await port.claim(t, node_id="n1", boot_id="b1", ttl_s=60.0)
    assert batch is not None

    # 1. Verify exact SQL dialect suffix (PG/MySQL carry FOR SHARE, SQLite does not).
    captured: list[str] = []
    orig_exec = store._db.execute
    orig_fetchone = store._db.fetchone

    async def _cap_exec(sql, params=()):
        captured.append(sql)
        return await orig_exec(sql, params)

    async def _cap_fetchone(sql, params=()):
        captured.append(sql)
        return await orig_fetchone(sql, params)

    store._db.execute = _cap_exec
    store._db.fetchone = _cap_fetchone
    try:
        seq1, _ = await store.append_message_fenced(
            t.session_id,
            owner_epoch=batch.owner_epoch,
            seq=batch.message_seq + 1,
            role="assistant",
            content="a1",
            send_index=batch.send_index,
        )
        seq2, _ = await store.append_message_fenced(
            t.session_id,
            owner_epoch=batch.owner_epoch,
            seq=seq1 + 1,
            role="tool",
            content="t1",
            send_index=batch.send_index,
            boundary_wake_inbox_seq=batch.wake_inbox_seq,
            boundary_cancel_seq=batch.cancel_seq,
        )
    finally:
        store._db.execute = orig_exec
        store._db.fetchone = orig_fetchone

    assert len(captured) == 2
    for stmt in captured:
        if dialect in ("postgres", "mysql"):
            assert " FOR SHARE" in stmt, f"missing FOR SHARE on {dialect}: {stmt}"
            if dialect == "postgres":
                assert stmt.index(" FOR SHARE") < stmt.index(" RETURNING id")
        else:
            assert "FOR SHARE" not in stmt

    # 2. On PG16 & MySQL 8.4, verify real row-lock serialization between an uncommitted
    #    takeover UPDATE (exclusive lock on sessions) and old-owner append_message_fenced (FOR SHARE).
    if dialect in ("postgres", "mysql"):
        peer = await connect(schema="verify")
        try:
            lock_acquired = asyncio.Event()
            release_takeover = asyncio.Event()

            async def _takeover_tx():
                async with peer._db.transaction() as tx:
                    await tx.execute(
                        f"UPDATE {peer.t.sessions} SET owner_node_id='n2',owner_boot_id='b2',"
                        "owner_epoch=owner_epoch+1 WHERE session_id=? AND execution_state='running'",
                        (t.session_id,),
                    )
                    lock_acquired.set()
                    await asyncio.wait_for(release_takeover.wait(), 5)

            takeover_task = asyncio.create_task(_takeover_tx())
            await asyncio.wait_for(lock_acquired.wait(), 5)

            # Old owner tries to append with old epoch while takeover holds exclusive lock on sessions.
            stale_append = asyncio.create_task(
                store.append_message_fenced(
                    t.session_id,
                    owner_epoch=batch.owner_epoch,
                    seq=seq2 + 1,
                    role="assistant",
                    content="STALE_DIRTY_ROW",
                    send_index=batch.send_index,
                )
            )
            # FOR SHARE MUST block behind the uncommitted exclusive lock (no MVCC bypass!).
            await asyncio.sleep(0.15)
            assert not stale_append.done(), "append_message_fenced bypassed uncommitted sessions row lock"

            release_takeover.set()
            await asyncio.wait_for(takeover_task, 5)

            # After takeover commits new owner_epoch, old owner sees updated row and raises SessionLeaseLost.
            with pytest.raises(SessionLeaseLost):
                await asyncio.wait_for(stale_append, 5)

            dirty = await store._db.fetchall(
                f"SELECT seq,content FROM {store.t.messages} WHERE session_id=? AND content='STALE_DIRTY_ROW'",
                (t.session_id,),
            )
            assert dirty == []
        finally:
            release_takeover.set()
            await peer.close()


# ── P0-4: closed + drain persists messages & advances next_seq; closed + abort cancels ──


async def test_h1_p0_4_closed_drain_persists_messages_and_closed_abort_cancels(registry_store):
    store, _ = registry_store
    tool_entered = asyncio.Event()
    tool_continue = asyncio.Event()

    async def slow_tool() -> str:
        tool_entered.set()
        await asyncio.wait_for(tool_continue.wait(), 5)
        return "tool_output_during_drain"

    def model_fn(idx, _req):
        if idx == 0:
            return LLMResponse(
                raw_text="calling tool before drain",
                tool_calls=[{"id": "tc_cd", "type": "function", "function": {"name": "slow", "arguments": "{}"}}],
            )
        return LLMResponse(raw_text="closed_drain_final_reply")

    agent = await RunningAgent(
        store,
        _ScriptedModel(model_fn),
        _single_tool_registry("slow", slow_tool),
        AgentLoopConfig(context_budget_tokens=10**9),
    ).start()
    try:
        client = agent.runtime.client
        actor = agent.actor

        send_task = asyncio.create_task(agent.send("run closed drain"))
        await asyncio.wait_for(tool_entered.wait(), 5)

        before = await store._db.fetchone(
            f"SELECT cancel_seq FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        # Soft-close with inflight='drain': must NOT bump cancel_seq or abort the running turn!
        st = await client.set_lifecycle(
            agent.target, "closed", actor=actor, inflight="drain", reason="graceful_close",
        )
        assert st.lifecycle == "closed"
        assert not st.cancel_requested

        # Periodic batch_renew during closed+drain MUST NOT cancel the in-flight turn!
        await agent.runtime.worker.batch_renew()
        mid = await store._db.fetchone(
            f"SELECT lifecycle,cancel_seq FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        assert mid["lifecycle"] == "closed"
        assert int(mid["cancel_seq"]) == int(before["cancel_seq"])

        # Unblock tool -> tool message + final assistant reply MUST persist cleanly!
        tool_continue.set()
        res = await asyncio.wait_for(send_task, 5)
        assert res.status == "completed"
        assert res.final_text == "closed_drain_final_reply"
        assert res.result_seq is not None

        msgs = await store.load_all_messages(agent.target.session_id)
        roles = [m.role for m in msgs]
        seqs = [m.seq for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert seqs == sorted(seqs) and len(set(seqs)) == 4
        assert msgs[-2].content == "tool_output_during_drain"
        assert msgs[-1].seq == res.result_seq
        assert msgs[-1].content == "closed_drain_final_reply"

        srow = await store._db.fetchone(
            f"SELECT lifecycle,execution_state,next_seq FROM {store.t.sessions} WHERE session_id=?",
            (agent.target.session_id,),
        )
        assert srow["lifecycle"] == "closed"
        assert srow["execution_state"] == "idle"
        assert int(srow["next_seq"]) == msgs[-1].seq + 1
    finally:
        tool_continue.set()
        await agent.aclose()

    # Verify closed + abort still cancels in-flight turn and writes 0 ghost reply.
    abort_entered = asyncio.Event()
    abort_continue = asyncio.Event()

    async def abort_tool() -> str:
        abort_entered.set()
        await asyncio.wait_for(abort_continue.wait(), 5)
        return "tool_output_abort"

    def abort_model_fn(idx, _req):
        if idx == 0:
            return LLMResponse(
                raw_text="calling abort tool",
                tool_calls=[{"id": "tc_ca", "type": "function", "function": {"name": "slow", "arguments": "{}"}}],
            )
        return LLMResponse(raw_text="GHOST_SHOULD_NOT_COMPLETE")

    agent2 = await RunningAgent(
        store,
        _ScriptedModel(abort_model_fn),
        _single_tool_registry("slow", abort_tool),
        AgentLoopConfig(context_budget_tokens=10**9),
    ).start()
    try:
        send_task2 = asyncio.create_task(agent2.send("run closed abort"))
        await asyncio.wait_for(abort_entered.wait(), 5)
        st2 = await agent2.runtime.client.set_lifecycle(
            agent2.target, "closed", actor=agent2.actor, inflight="abort", reason="hard_close",
        )
        assert st2.lifecycle == "closed"
        assert st2.cancel_requested is True
        msgs2_before = await store.load_all_messages(agent2.target.session_id)
        assert [(m.role, m.seq) for m in msgs2_before] == [("user", 1), ("assistant", 2)]

        abort_continue.set()
        res2 = await asyncio.wait_for(send_task2, 5)
        assert res2.status == "cancelled"
        assert res2.result_seq is None
        msgs2 = await store.load_all_messages(agent2.target.session_id)
        # Closed + abort MUST block post-cancel tool message insertion (no seq=3 tool row)
        assert [(m.role, m.seq) for m in msgs2] == [("user", 1), ("assistant", 2)]
        assert all(m.content != "GHOST_SHOULD_NOT_COMPLETE" for m in msgs2)
        srow2 = await store._db.fetchone(
            f"SELECT lifecycle,execution_state,cancel_seq,next_seq FROM {store.t.sessions} WHERE session_id=?",
            (agent2.target.session_id,),
        )
        assert srow2["lifecycle"] == "closed"
        assert srow2["execution_state"] == "idle"
        assert int(srow2["cancel_seq"]) == 1
        assert int(srow2["next_seq"]) == 3
    finally:
        abort_continue.set()
        await agent2.aclose()


# ── P0-5: upsert_repaired_projection locking, fencing & fold invalidation ──


async def test_h1_p0_5_upsert_repaired_projection_fenced_and_invalidates_fold(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    t = await target(port, "h1-p05-sess")
    await port.accept(t, "turn 1", message_id="m1", actor=ACTOR)
    b1 = await port.claim(t, node_id="n1", boot_id="b1", ttl_s=60.0)
    assert b1 is not None

    # Seed fold_stats_json with a valid non-invalid snapshot.
    valid_stats = json.dumps({"v": 3, "sends": 1, "rows": 2, "tokens": 50, "maximum": 1, "dedup": {}, "revision": "r1"})
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET fold_stats_json=? WHERE session_id=?",
        (valid_stats, t.session_id),
    )

    old_ownership = SessionOwnership(
        (store.store_id, t.session_id), "n1/b1", time.monotonic() + 60.0, lambda: None,
        epoch=b1.owner_epoch, node_id="n1", boot_id="b1", installs_fence=True,
    )

    # Expire & reclaim so DB advances to owner_epoch = b1.owner_epoch + 1.
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
        (t.session_id,),
    )
    b2 = await port.reclaim_running(t, node_id="n2", boot_id="b2", ttl_s=60.0)
    assert b2 is not None and b2.owner_epoch == b1.owner_epoch + 1
    new_ownership = SessionOwnership(
        (store.store_id, t.session_id), "n2/b2", time.monotonic() + 60.0, lambda: None,
        epoch=b2.owner_epoch, node_id="n2", boot_id="b2", installs_fence=True,
    )

    # Stale owner calling upsert_repaired_projection MUST raise SessionLeaseLost and leave DB untouched!
    with pytest.raises(SessionLeaseLost):
        await store.upsert_repaired_projection(
            t.session_id,
            send_index=1,
            rendered_text="STALE_OVERWRITE",
            content={"folded_on_400": True, "text": "STALE_OVERWRITE"},
            ownership=old_ownership,
        )
    projs_after_stale = await store.load_project_messages(t.session_id)
    assert all(p.rendered_text != "STALE_OVERWRITE" for p in projs_after_stale)
    srow_after_stale = await store._db.fetchone(
        f"SELECT fold_stats_json FROM {store.t.sessions} WHERE session_id=?",
        (t.session_id,),
    )
    assert srow_after_stale["fold_stats_json"] == valid_stats

    # Locally expired ownership MUST also raise SessionLeaseLost.
    expired_ownership = SessionOwnership(
        (store.store_id, t.session_id), "n2/b2", time.monotonic() - 1.0, lambda: None,
        epoch=b2.owner_epoch, node_id="n2", boot_id="b2", installs_fence=True,
    )
    with pytest.raises(SessionLeaseLost):
        await store.upsert_repaired_projection(
            t.session_id,
            send_index=1,
            rendered_text="EXPIRED_OVERWRITE",
            content={"folded_on_400": True, "text": "EXPIRED_OVERWRITE"},
            ownership=expired_ownership,
        )

    # Current valid owner succeeds AND invalidates fold_stats_json!
    await store.upsert_repaired_projection(
        t.session_id,
        send_index=1,
        rendered_text="VALID_REPAIR",
        content={"folded_on_400": True, "text": "VALID_REPAIR"},
        ownership=new_ownership,
    )
    projs = await store.load_project_messages(t.session_id)
    rep = [p for p in projs if p.kind == "repaired" and p.send_index == 1]
    assert len(rep) == 1 and rep[0].rendered_text == "VALID_REPAIR"
    srow_after_valid = await store._db.fetchone(
        f"SELECT fold_stats_json FROM {store.t.sessions} WHERE session_id=?",
        (t.session_id,),
    )
    parsed_stats = json.loads(srow_after_valid["fold_stats_json"])
    assert parsed_stats.get("invalid") is True

    # End-to-end pipeline HTTP 400 path: when ownership is stolen right before 400 repair,
    # pipeline passes current ownership so stale repair is fenced out and never overwrites projections.
    class _Format400Error(Exception):
        status_code = 400

        def __init__(self):
            super().__init__("invalid tool_call_id messages format")

    peer = await connect(schema="verify")
    peer_port = RuntimeInputStore(peer)
    stolen_box: list[Any] = []

    class _StealOn400Model:
        async def complete(self, request, **kwargs):
            sid = agent.target.session_id
            await peer._db.execute(
                f"UPDATE {peer.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
                (sid,),
            )
            reclaimed = await peer_port.reclaim_running(
                TargetRef(store.store_id, sid), node_id="thief", boot_id="thief", ttl_s=60.0,
            )
            assert reclaimed is not None
            stolen_box.append(reclaimed)
            raise _Format400Error()

        async def close(self):
            return None

    agent = await RunningAgent(
        store,
        _StealOn400Model(),
        _single_tool_registry("echo", lambda: "ok"),
        AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        fuse_local_send=True,
    ).start()
    try:
        sid = agent.target.session_id
        # Seed send 1 repaired projection with tool_calls so 400 path enters upsert_repaired_projection.
        await store.upsert_repaired_projection(
            sid,
            send_index=1,
            content={
                "messages": [
                    {"role": "user", "content": "q1"},
                    {
                        "role": "assistant", "content": "c1",
                        "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
                    },
                    {"role": "tool", "tool_call_id": "tc_1", "name": "echo", "content": "out_1"},
                ],
            },
            rendered_text="ORIGINAL_PROJECTION",
        )
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET send_index=1,next_seq=5 WHERE session_id=?", (sid,),
        )
        receipt = await agent.runtime.client.send(agent.target, "live turn 2", message_id="m_p05", actor=agent.actor)
        # Wait until worker's local task exits on SessionLeaseLost.
        await asyncio.wait_for(agent.drain(), 5)
        assert len(stolen_box) == 1
        # Original projection MUST remain untouched (NOT overwritten with folded_on_400=True)!
        projs_e2e = await store.load_project_messages(sid)
        rep1 = [p for p in projs_e2e if p.kind == "repaired" and p.send_index == 1]
        assert len(rep1) == 1
        assert rep1[0].rendered_text == "ORIGINAL_PROJECTION"
        assert rep1[0].content.get("folded_on_400") is not True
        # Cleanly finalize the thief's reclaimed batch.
        await peer_port.final(stolen_box[0], final_text="thief_done")
        final_res = await asyncio.wait_for(
            agent.runtime.client.wait(receipt.input_ref, actor=agent.actor), 5.0,
        )
        assert final_res.final_text == "thief_done"
    finally:
        await agent.aclose()
        await peer.close()


# ── Index optimization: discover_queued includes owner_lease_until_ms IS NULL ──


async def test_h1_discover_queued_hits_recovery_index_predicate(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)

    t_idle = await target(port, "h1-dq-idle")
    t_queued_1 = await target(port, "h1-dq-q1")
    t_queued_2 = await target(port, "h1-dq-q2")
    t_running = await target(port, "h1-dq-run")
    t_susp = await target(port, "h1-dq-susp")

    await port.accept(t_queued_1, "q1", message_id="q1", actor=ACTOR)
    await port.accept(t_queued_2, "q2", message_id="q2", actor=ACTOR)
    await port.accept(t_running, "r1", message_id="r1", actor=ACTOR)
    await port.claim(t_running, node_id="n1", boot_id="b1", ttl_s=60.0)
    await port.accept(t_susp, "s1", message_id="s1", actor=ACTOR)
    await port.set_lifecycle(t_susp, "suspended", actor=ACTOR)

    seen_sqls: list[str] = []
    orig_fetchall = store._db.fetchall

    async def _cap_fetchall(sql, params=()):
        seen_sqls.append(sql)
        return await orig_fetchall(sql, params)

    store._db.fetchall = _cap_fetchall
    try:
        found = await port.discover_queued(limit=32)
    finally:
        store._db.fetchall = orig_fetchall

    assert len(seen_sqls) == 1
    assert "owner_lease_until_ms IS NULL" in seen_sqls[0]
    assert set(found) == {t_queued_1.session_id, t_queued_2.session_id}
    assert t_idle.session_id not in found
    assert t_running.session_id not in found
    assert t_susp.session_id not in found
