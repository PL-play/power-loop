"""Design 218 H4 regressions: control-plane concurrency, reply_route security, lock order, retention & 3-DB parity (P1-1 ~ P1-6)."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    RegistryRuntimeFactory,
    Runtime,
    SessionLeaseLost,
    TargetRef,
    ToolRegistry,
    TrustedActor,
)
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.contracts.hooks import HookPoint
from power_loop.core.hooks import AgentHooks
from power_loop.runtime.session_ownership import _CURRENT, SessionOwnership
from power_loop.runtime.store import closing
from power_loop.runtime.store.backends.mysql import MySQLDatabase
from power_loop.runtime.store.schema import StoreSchemaError, _acquire_provision_lock

from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# ── P1-1: Fused 2-tx pre-reservation, dispatch before notify, and _closing guard ──


async def test_h4_p1_1_concurrent_fused_send_respects_capacity_and_releases_slots(registry_store, monkeypatch):
    """P1-1: capacity+5 concurrent sends never exceed capacity; fallback/error/cancel never leak slots."""
    store, _ = registry_store
    capacity = 2
    total_sessions = capacity + 5

    in_flight = 0
    max_in_flight = 0
    capacity_reached = asyncio.Event()
    release_models = asyncio.Event()

    class _BarrierModel:
        async def complete(self, _request, **_kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            if in_flight > max_in_flight:
                max_in_flight = in_flight
            if in_flight >= capacity:
                capacity_reached.set()
            await asyncio.wait_for(release_models.wait(), 5)
            in_flight -= 1
            return LLMResponse(raw_text="ok")

        async def close(self):
            pass

    model = _BarrierModel()
    factory = RegistryRuntimeFactory()

    async def _build(_d, _c):
        return ExecutionEnvironment(model, ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9))

    factory.register("test.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="h4-p11-node",
        boot_id="h4-p11-boot",
        capacity=capacity,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    await rt.worker.start()
    try:
        targets = [
            await rt.client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": f"h4-p11-s{i}"})
            for i in range(total_sessions)
        ]

        # Fire capacity+5 sends concurrently into idle sessions.
        receipts = await asyncio.gather(*(
            rt.client.send(t, f"hello {i}", message_id=f"m-{i}", actor=ACTOR)
            for i, t in enumerate(targets)
        ))
        await asyncio.wait_for(capacity_reached.wait(), 5)
        # Allow any mistakenly dispatched extra tasks a tick to enter complete().
        await asyncio.sleep(0.05)
        assert max_in_flight <= capacity, f"fused concurrency exceeded capacity: {max_in_flight} > {capacity}"
        assert rt.worker._active_llm_count() <= capacity

        release_models.set()
        results = await asyncio.gather(*(
            rt.client.wait(r.input_ref, actor=ACTOR, deadline_ms=now_ms() + 10000)
            for r in receipts
        ))
        assert all(res.status == "completed" and res.final_text == "ok" for res in results)
        assert rt.worker._active == {}
        assert rt.worker._fusing == {}
        assert rt.worker._reserved_wakes == set()

        # Error path (missing session) must release pre-reserved slot.
        missing = TargetRef(store.store_id, "h4-p11-missing")
        with pytest.raises(RuntimeRequestError):
            await rt.client.send(missing, "boom", message_id="m-miss", actor=ACTOR)
        assert missing.session_id not in rt.worker._active
        assert missing.session_id not in rt.worker._fusing

        # Cancellation during accept_and_claim must release pre-reserved slot.
        cancel_target = await rt.client.create_session(
            DESCRIPTOR, actor=ACTOR, options={"session_id": "h4-p11-cancel"},
        )
        orig_aac = rt.client.inputs.accept_and_claim
        entered_aac = asyncio.Event()

        async def _blocking_aac(*args, **kwargs):
            entered_aac.set()
            await asyncio.sleep(10)
            return await orig_aac(*args, **kwargs)

        monkeypatch.setattr(rt.client.inputs, "accept_and_claim", _blocking_aac)
        send_task = asyncio.create_task(
            rt.client.send(cancel_target, "will cancel", message_id="m-cancel", actor=ACTOR)
        )
        await asyncio.wait_for(entered_aac.wait(), 5)
        assert cancel_target.session_id in rt.worker._active
        assert cancel_target.session_id in rt.worker._fusing
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        assert cancel_target.session_id not in rt.worker._active
        assert cancel_target.session_id not in rt.worker._fusing
    finally:
        release_models.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_h4_p1_1_dispatch_claimed_before_notify_and_closing_releases_lease(registry_store, monkeypatch):
    """P1-1: dispatch_claimed runs before MESSAGE_ACCEPTED; concurrent shutdown rejects dispatch & releases lease."""
    store, _ = registry_store
    hook_saw_active = {"checked": False, "in_active": False, "in_tasks": 0}
    model_calls: list[str] = []

    class _RecordingModel:
        async def complete(self, _request, **_kwargs):
            from power_loop.core.execution_context import get_execution_context
            model_calls.append(get_execution_context().session_id)
            return LLMResponse(raw_text="done")

        async def close(self):
            pass

    hooks = AgentHooks()
    worker_box: list[Any] = []

    async def _on_accepted(ctx):
        receipt = ctx.values["receipt"]
        if receipt.target.session_id == "h4-p11-order" and worker_box:
            w = worker_box[0]
            hook_saw_active["checked"] = True
            hook_saw_active["in_active"] = "h4-p11-order" in w._active
            hook_saw_active["in_tasks"] = len(w._tasks)

    hooks.register(HookPoint.MESSAGE_ACCEPTED, _on_accepted)

    factory = RegistryRuntimeFactory()

    async def _build(_d, _c):
        return ExecutionEnvironment(_RecordingModel(), ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9))

    factory.register("test.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="h4-p11-shut-node",
        boot_id="h4-p11-shut-boot",
        capacity=4,
        fuse_local_send=True,
        hooks=hooks,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    rt.client.hooks = hooks
    worker_box.append(rt.worker)
    await rt.worker.start()
    try:
        # Part A: dispatch_claimed runs BEFORE notify(MESSAGE_ACCEPTED).
        order_t = await rt.client.create_session(
            DESCRIPTOR, actor=ACTOR, options={"session_id": "h4-p11-order"},
        )
        rcpt = await rt.client.send(order_t, "hi", message_id="m-order", actor=ACTOR)
        res = await rt.client.wait(rcpt.input_ref, actor=ACTOR, deadline_ms=now_ms() + 5000)
        assert res.status == "completed"
        assert hook_saw_active["checked"] is True
        assert hook_saw_active["in_active"] is True
        assert hook_saw_active["in_tasks"] >= 1

        # Part B: shutdown() races between accept_and_claim commit and dispatch_claimed ->
        # dispatch_claimed refuses to start a task and releases the lease cleanly.
        race_t = await rt.client.create_session(
            DESCRIPTOR, actor=ACTOR, options={"session_id": "h4-p11-shut-race"},
        )
        orig_aac = rt.client.inputs.accept_and_claim
        aac_committed = asyncio.Event()
        allow_return = asyncio.Event()

        async def _paused_aac(*args, **kwargs):
            result = await orig_aac(*args, **kwargs)
            target_arg = args[0] if args else kwargs.get("target")
            if target_arg is not None and target_arg.session_id == "h4-p11-shut-race":
                aac_committed.set()
                await asyncio.wait_for(allow_return.wait(), 5)
            return result

        monkeypatch.setattr(rt.client.inputs, "accept_and_claim", _paused_aac)
        send_task = asyncio.create_task(
            rt.client.send(race_t, "racing shutdown", message_id="m-shut-race", actor=ACTOR)
        )
        await asyncio.wait_for(aac_committed.wait(), 5)

        # Concurrently shut down worker while fused accept_and_claim has committed in DB.
        shutdown_task = asyncio.create_task(rt.worker.shutdown(timeout_s=2, cancel_running=False))
        await asyncio.wait_for(shutdown_task, 5)
        assert rt.worker._closing is True

        # Now let send() resume and call dispatch_claimed(batch).
        allow_return.set()
        race_rcpt = await asyncio.wait_for(send_task, 5)
        assert race_rcpt.state == "accepted"

        # Worker MUST NOT have dispatched or run the model, and DB MUST have no orphan lease.
        assert "h4-p11-shut-race" not in model_calls
        assert rt.worker._active == {}
        assert rt.worker._fusing == {}
        row = await store._db.fetchone(
            f"SELECT execution_state,owner_node_id,owner_boot_id,owner_lease_until_ms,pending_count "
            f"FROM {store.t.sessions} WHERE session_id=?",
            ("h4-p11-shut-race",),
        )
        assert row is not None
        assert row["execution_state"] == "queued"
        assert row["owner_node_id"] is None
        assert row["owner_boot_id"] is None
        assert row["owner_lease_until_ms"] is None
        assert int(row["pending_count"]) == 1
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


# ── P1-2: Public send() reply_route ancestor/permission & privileged result_kind ──


async def test_h4_p1_2_public_send_reply_route_validation_and_privileged_result_kind(registry_store):
    """P1-2: public send rejects non-ancestor/same-depth/child/unauthorized reply_route & privileged result_kind."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        root_a = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12-root-a"})
        mid_a = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=root_a, options={"session_id": "p12-mid-a"},
        )
        sib_a = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=root_a, options={"session_id": "p12-sib-a"},
        )
        leaf_a = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=mid_a, options={"session_id": "p12-leaf-a"},
        )
        root_b = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12-root-b"})
        other_actor = TrustedActor("other-tenant", "other-principal", None)
        other_root = await client.create_session(
            DESCRIPTOR, actor=other_actor, options={"session_id": "p12-other-root"},
        )

        # 1. Same-tenant non-ancestor (cross-tree) rejected on both accept & accept_and_claim.
        for idx, (src, dst) in enumerate([
            (mid_a, root_b),       # cross-tree shallower
            (leaf_a, root_b),      # cross-tree shallower
            (mid_a, sib_a),        # same depth sibling
            (root_a, root_b),      # same depth root
            (mid_a, mid_a),        # self
            (root_a, mid_a),       # downward child (AB-BA deadlock vector)
            (root_a, leaf_a),      # downward grandchild
            (mid_a, leaf_a),       # downward child
            (leaf_a, other_root),  # cross-tenant
        ]):
            with pytest.raises(RuntimeRequestError) as exc_info:
                await client.send(
                    src, f"bad-{idx}", message_id=f"p12-bad-{idx}", actor=ACTOR,
                    reply_route={"target": dst, "result_kind": "agent_result"},
                )
            assert exc_info.value.code == "forbidden", (src.session_id, dst.session_id, exc_info.value)

            with pytest.raises(RuntimeRequestError) as aac_exc:
                await port.accept_and_claim(
                    src, f"bad-aac-{idx}", message_id=f"p12-bad-aac-{idx}", actor=ACTOR,
                    node_id="n1", boot_id="b1",
                    reply_route={"target": dst, "result_kind": "agent_result"},
                )
            assert aac_exc.value.code == "forbidden"

        # 2. Missing reply_route target rejected with not_found.
        with pytest.raises(RuntimeRequestError) as nf_exc:
            await client.send(
                leaf_a, "missing-dst", message_id="p12-missing-dst", actor=ACTOR,
                reply_route={"target": TargetRef(store.store_id, "p12-nonexistent"), "result_kind": "agent_result"},
            )
        assert nf_exc.value.code == "not_found"

        # 3. Child/mid agent actor cannot route above its own session scope (forbidden on reply_route target).
        mid_actor = TrustedActor(ACTOR.tenant_id, ACTOR.principal_id, mid_a.session_id)
        with pytest.raises(RuntimeRequestError) as perm_exc:
            await client.send(
                leaf_a, "escalate", message_id="p12-escalate", actor=mid_actor,
                reply_route={"target": root_a, "result_kind": "agent_result"},
            )
        assert perm_exc.value.code == "forbidden"

        # 4. Public send rejects all internal privileged result_kinds even for valid shallower ancestors.
        for bad_rk in ("workflow_node_result", "workflow_result", "background_result", "managed_effect_result"):
            with pytest.raises(RuntimeRequestError) as rk_exc:
                await client.send(
                    leaf_a, "priv", message_id=f"p12-rk-{bad_rk}", actor=ACTOR,
                    reply_route={"target": mid_a, "result_kind": bad_rk},
                )
            assert rk_exc.value.code == "invalid_options"

        # 5. Direct public accept with explicit result_kind rejected.
        with pytest.raises(RuntimeRequestError) as drk_exc:
            await port.accept(
                leaf_a, "priv-rk", message_id="p12-direct-rk", actor=ACTOR,
                result_kind="workflow_node_result", public=True,
            )
        assert drk_exc.value.code == "invalid_options"

        # 6. Valid shallower ancestors (direct parent and grandparent) with agent_result succeed!
        r_direct = await client.send(
            leaf_a, "ok direct", message_id="p12-ok-direct", actor=mid_actor,
            reply_route={"target": mid_a, "result_kind": "agent_result"},
        )
        assert r_direct.state == "accepted"
        r_gp = await client.send(
            leaf_a, "ok gp", message_id="p12-ok-gp", actor=ACTOR,
            reply_route={"target": root_a, "result_kind": "agent_result"},
        )
        assert r_gp.state == "accepted"

        # 7. Internal public=False workflow routing remains unharmed.
        r_internal = await port.accept(
            leaf_a, "internal wf", message_id="p12-internal-wf", actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": mid_a, "result_kind": "workflow_node_result"},
            public=False,
        )
        assert r_internal.state == "accepted"
    finally:
        await client.aclose()
        await transport.aclose()


# ── P1-3: closing.lock_sessions deep-first lock of unbound direct children ──


async def test_h4_p1_3_lock_sessions_locks_unbound_children_deep_first_and_no_deadlock(registry_store):
    """P1-3: lock_sessions locks unbound direct children in strict depth DESC, session_id ASC order; no PG/MySQL deadlock."""
    store, connect = registry_store
    port = RuntimeInputStore(store)

    # Part A: Verify exact depth DESC, session_id ASC lock ordering for both cascade=True and cascade=False.
    parent = await target(port, "p13-parent")
    linked = await target(port, "p13-c-linked", parent=parent, cancel_propagation="linked")
    indep_b = await target(port, "p13-b-indep", parent=parent, cancel_propagation="independent")
    indep_a = await target(port, "p13-a-indep", parent=parent, cancel_propagation="independent")
    indep_gc = await target(port, "p13-gc-indep", parent=linked, cancel_propagation="independent")

    async def _record_lock_batches(cascade: bool):
        batches: list[tuple[str, ...]] = []
        async with store._db.transaction() as tx:
            orig_fetchall = tx.fetchall

            async def _spy_fetchall(sql, params=()):
                if "WHERE session_id IN (" in sql and "ORDER BY session_id ASC" in sql:
                    batches.append(tuple(str(p) for p in params))
                return await orig_fetchall(sql, params)

            tx.fetchall = _spy_fetchall  # type: ignore[method-assign]
            rows = await closing.lock_sessions(store, tx, parent.session_id, cascade=cascade)
        return batches, [str(r["session_id"]) for r in rows]

    batches_cascade, returned_cascade = await _record_lock_batches(cascade=True)
    assert batches_cascade == [
        (indep_gc.session_id,),                                        # depth 2
        (indep_a.session_id, indep_b.session_id, linked.session_id),   # depth 1 (ASC)
        (parent.session_id,),                                          # depth 0
    ]
    assert returned_cascade == [parent.session_id, linked.session_id]

    batches_nocasc, returned_nocasc = await _record_lock_batches(cascade=False)
    assert batches_nocasc == [
        (indep_a.session_id, indep_b.session_id, linked.session_id),   # depth 1 (ASC)
        (parent.session_id,),                                          # depth 0
    ]
    assert returned_nocasc == [parent.session_id]

    # Part B: Concurrent delete_sessions vs child final/cancel on 2 connections (PG16/MySQL 8.4/SQLite).
    peer = await connect(schema="verify")
    peer_port = RuntimeInputStore(peer)

    # Case 1: cascade=True with independent child finishing final() concurrently.
    p1 = await target(port, "p13-race1-parent")
    ch1 = await target(port, "p13-race1-indep", parent=p1, cancel_propagation="independent")
    await peer_port.accept(
        ch1, "indep work", message_id="p13-m1", actor=ACTOR,
        reply_route={"target": p1, "result_kind": "agent_result"},
    )
    b1 = await peer_port.claim(ch1, node_id="n1", boot_id="b1")
    assert b1 is not None

    del_res, fin_res = await asyncio.gather(
        store.close_session_tree(p1.session_id),
        peer_port.final(b1, final_text="indep done"),
        return_exceptions=True,
    )
    # Neither may deadlock; delete either succeeds (unbinding ch1) or sees SessionBusy if final delivered to p1 first.
    assert not isinstance(fin_res, Exception), f"child final failed: {fin_res!r}"
    if isinstance(del_res, Exception):
        from power_loop.contracts.errors import SessionBusy
        assert isinstance(del_res, SessionBusy), f"unexpected delete error: {del_res!r}"
    else:
        assert p1.session_id in del_res
        saved_ch1 = await store.get_session(ch1.session_id)
        assert saved_ch1 is not None and saved_ch1.parent_session_id is None

    # Case 2: cascade=False with direct child running cancel() concurrently.
    p2 = await target(port, "p13-race2-parent")
    ch2 = await target(port, "p13-race2-child", parent=p2, cancel_propagation="linked")
    r2 = await peer_port.accept(ch2, "child work", message_id="p13-m2", actor=ACTOR)
    del_res2, cancel_res2 = await asyncio.gather(
        store.close_session(p2.session_id, cascade=False),
        peer_port.cancel(r2.input_ref, actor=ACTOR, reason="stop"),
    )
    assert del_res2 == 1
    assert await store.get_session(p2.session_id) is None
    assert cancel_res2.state in ("voided", "cancelled")
    saved_ch2 = await store.get_session(ch2.session_id)
    assert saved_ch2 is not None and saved_ch2.parent_session_id is None


# ── P1-4: retention.purge_expired_page single-session truncation cursor ──────


async def test_h4_p1_4_purge_expired_page_truncation_reentrant_and_terminates(registry_store):
    """P1-4: single session with 15 expired rows is cleaned in 4 pages at limit=5 without skipping or looping."""
    store, _ = registry_store
    port = RuntimeInputStore(store)

    # Part A: Single closed session with 15 expired messages + 1 sessions row, limit=5 -> 4 pages.
    s1 = await target(port, "p14-single")
    for i in range(15):
        await store.append_message(s1.session_id, role="user", content=f"msg-{i}")
    assert await store.close_if_idle(s1.session_id) == "closed"

    future_ms = 2**62
    pages = []
    cursor = None
    for _ in range(10):
        page = await store.purge_expired_page(future_ms, limit=5, after_session_id=cursor)
        pages.append(page)
        cursor = page.next_after
        if cursor is None:
            break

    assert len(pages) == 4, [(p.deleted, p.next_after) for p in pages]
    nonzero = lambda d: {k: v for k, v in d.items() if v}  # noqa: E731
    assert nonzero(pages[0].deleted) == {"messages": 5} and pages[0].next_after is not None
    assert nonzero(pages[1].deleted) == {"messages": 5} and pages[1].next_after is not None
    assert nonzero(pages[2].deleted) == {"messages": 5} and pages[2].next_after is not None
    assert nonzero(pages[3].deleted) == {"sessions": 1} and pages[3].next_after is None
    assert await store.get_session(s1.session_id) is None

    # Empty page returns next_after=None immediately.
    empty = await store.purge_expired_page(future_ms, limit=5)
    assert empty.deleted == {} and empty.scanned == 0 and empty.fetched == 0 and empty.next_after is None

    # Part B: Multi-session where s_a exhausts exact quota (4 messages + 1 session = 5) and s_b spans pages.
    s_a = await target(port, "p14-multi-a")
    s_b = await target(port, "p14-multi-b")
    for i in range(4):
        await store.append_message(s_a.session_id, role="user", content=f"a-{i}")
    for i in range(7):
        await store.append_message(s_b.session_id, role="user", content=f"b-{i}")
    assert await store.close_if_idle(s_a.session_id) == "closed"
    assert await store.close_if_idle(s_b.session_id) == "closed"

    totals: dict[str, int] = {}
    cursor = None
    multi_pages = 0
    for _ in range(10):
        page = await store.purge_expired_page(future_ms, limit=5, after_session_id=cursor)
        multi_pages += 1
        for k, v in page.deleted.items():
            if v:
                totals[k] = totals.get(k, 0) + v
        cursor = page.next_after
        if cursor is None:
            break
    assert cursor is None
    assert multi_pages <= 4
    assert totals == {"messages": 11, "sessions": 2}
    assert await store.get_session(s_a.session_id) is None
    assert await store.get_session(s_b.session_id) is None

    # Part C: Open session with exact limit=5 expired inbox rows terminates cleanly (no infinite loop).
    s_open = await target(port, "p14-open-inbox")
    for i in range(5):
        r = await port.accept(s_open, f"inp-{i}", message_id=f"p14-open-m{i}", actor=ACTOR)
        await port.cancel(r.input_ref, actor=ACTOR)
    cursor = None
    open_pages = []
    for _ in range(5):
        page = await store.purge_expired_page(
            future_ms, limit=5, categories={"inbox"}, after_session_id=cursor,
        )
        open_pages.append(page)
        cursor = page.next_after
        if cursor is None:
            break
    assert cursor is None
    assert sum(p.deleted.get("inbox", 0) for p in open_pages) == 5
    assert len(open_pages) == 2


# ── P1-5: store.append_message unified diagnostic & 3-DB exception parity ────


@pytest.mark.parametrize("with_expected_seq", [False, True])
async def test_h4_p1_5_append_message_uniform_error_classification_three_dbs(registry_store, with_expected_seq):
    """P1-5: SQLite, PG16 & MySQL 8.4 classify missing/lease-lost/pending_tool_calls_exist identically."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    tc = [{"id": "tc1", "type": "function", "function": {"name": "noop", "arguments": "{}"}}]

    @asynccontextmanager
    async def _claimed_scope(sid: str):
        t = TargetRef(store.store_id, sid)
        await port.accept(t, "go", message_id=f"p15-msg-{sid}", actor=ACTOR)
        batch = await port.claim(t, node_id="n1", boot_id="b1")
        assert batch is not None
        lease = SessionOwnership(
            (store.store_id, sid), "n1/b1", time.monotonic() + 60, lambda: None,
            epoch=batch.owner_epoch, node_id="n1", boot_id="b1", installs_fence=True,
        )
        token = _CURRENT.set(lease)
        try:
            async with store.send_read_scope(sid, ownership=lease) as reads:
                if with_expected_seq:
                    await store.get_state(sid)
                    assert reads.next_seq is not None
                else:
                    reads.next_seq = None
                yield t, batch, lease
        finally:
            lease.released = True
            _CURRENT.reset(token)

    # 1. Valid lease + pending_tool_seq non-null -> ValueError("pending_tool_calls_exist") on all 3 DBs.
    t1 = await target(port, f"p15-pending-{int(with_expected_seq)}")
    async with _claimed_scope(t1.session_id):
        await store.append_message(t1.session_id, role="assistant", content="c1", tool_calls=tc, pending_tool_calls=True)
        with pytest.raises(ValueError, match="pending_tool_calls_exist"):
            await store.append_message(
                t1.session_id, role="assistant", content="c2", tool_calls=tc, pending_tool_calls=True,
            )

    # 2. Lease lost + pending_tool_seq NULL + pending_tool_calls=True -> SessionLeaseLost on all 3 DBs.
    t2 = await target(port, f"p15-lost-null-{int(with_expected_seq)}")
    async with _claimed_scope(t2.session_id):
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_epoch=owner_epoch+1 WHERE session_id=?",
            (t2.session_id,),
        )
        with pytest.raises(SessionLeaseLost):
            await store.append_message(
                t2.session_id, role="assistant", content="stale", tool_calls=tc, pending_tool_calls=True,
            )

    # 3. Lease lost + pending_tool_seq NON-NULL + pending_tool_calls=True -> SessionLeaseLost takes precedence!
    t3 = await target(port, f"p15-lost-both-{int(with_expected_seq)}")
    async with _claimed_scope(t3.session_id):
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_epoch=owner_epoch+1, pending_tool_seq=1 WHERE session_id=?",
            (t3.session_id,),
        )
        with pytest.raises(SessionLeaseLost):
            await store.append_message(
                t3.session_id, role="assistant", content="stale-both", tool_calls=tc, pending_tool_calls=True,
            )

    # 4. Unknown session -> ValueError("unknown session: ...") on all 3 DBs.
    with pytest.raises(ValueError, match="unknown session"):
        await store.append_message(
            f"p15-unknown-{int(with_expected_seq)}", role="assistant", content="x",
            tool_calls=tc, pending_tool_calls=True,
        )


# ── P1-6: MySQL GET_LOCK assertion & read_snapshot isolation leak guard ────


@pytest.mark.parametrize("lock_return", [{"acquired": 0}, {"acquired": None}, None])
async def test_h4_p1_6_mysql_provision_lock_rejects_zero_and_none(lock_return):
    """P1-6(1): _acquire_provision_lock asserts GET_LOCK == 1 and raises StoreSchemaError on 0/None."""
    class _FakeDialect:
        name = "mysql"

    class _FakeDB:
        dialect = _FakeDialect()

    class _FakeTx:
        def __init__(self, ret):
            self.ret = ret
            self.queries: list[tuple[str, tuple]] = []

        async def fetchone(self, sql, params=()):
            self.queries.append((sql, params))
            return self.ret

    tx = _FakeTx(lock_return)
    with pytest.raises(StoreSchemaError, match="GET_LOCK"):
        await _acquire_provision_lock(tx, _FakeDB(), "pl_")
    assert len(tx.queries) == 1 and "GET_LOCK" in tx.queries[0][0]

    ok_tx = _FakeTx({"acquired": 1})
    await _acquire_provision_lock(ok_tx, _FakeDB(), "pl_")
    assert len(ok_tx.queries) == 1


@pytest.mark.parametrize("reset_fails", [False, True])
async def test_h4_p1_6_mysql_read_snapshot_cancel_unit_no_isolation_leak(reset_fails):
    """P1-6(2): Cancelled read_snapshot resets READ COMMITTED or closes conn so REPEATABLE READ never leaks."""
    class _FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, sql, _params=()):
            self.conn.executed.append(sql)
            if sql.startswith("START TRANSACTION"):
                raise asyncio.CancelledError("cancelled before START TRANSACTION")
            if sql == "SET TRANSACTION ISOLATION LEVEL READ COMMITTED" and reset_fails:
                raise RuntimeError("broken connection")

    class _FakeConn:
        def __init__(self):
            self.closed = False
            self.executed: list[str] = []
            self.rollbacks = 0

        def cursor(self, *_args, **_kwargs):
            return _FakeCursor(self)

        async def rollback(self):
            self.rollbacks += 1

        def close(self):
            self.closed = True

    class _FakePool:
        def __init__(self, conn):
            self.conn = conn

        @asynccontextmanager
        async def acquire(self):
            yield self.conn

    conn = _FakeConn()
    db = MySQLDatabase(_FakePool(conn))

    with pytest.raises(asyncio.CancelledError):
        async with db.read_snapshot():
            pass

    assert conn.rollbacks == 0
    assert conn.executed[:2] == [
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ",
        "START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY",
    ]
    assert conn.executed[2] == "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
    assert conn.closed is reset_fails


async def test_h4_p1_6_mysql_real_read_snapshot_cancel_does_not_leak_repeatable_read():
    """P1-6(2) real MySQL 8.4: mid-construction cancellation never leaks REPEATABLE READ to next write tx."""
    dsn = os.environ.get("POWER_LOOP_TASK_MYSQL_DSN")
    if not dsn:
        pytest.skip("POWER_LOOP_TASK_MYSQL_DSN not set")

    db = await MySQLDatabase.connect(dsn, minsize=1, maxsize=1)
    writer = await MySQLDatabase.connect(dsn, minsize=1, maxsize=1)
    tbl = "h4_p16_iso_leak_test"
    try:
        await writer.execute(f"DROP TABLE IF EXISTS {tbl}")
        await writer.execute(f"CREATE TABLE {tbl} (id INT PRIMARY KEY, val INT NOT NULL) ENGINE=InnoDB")
        await writer.execute(f"INSERT INTO {tbl} (id, val) VALUES (1, 10)")

        # Wrap the single pooled connection's cursor so START TRANSACTION raises CancelledError
        # right AFTER SET TRANSACTION ISOLATION LEVEL REPEATABLE READ executed on MySQL 8.4!
        orig_acquire = db._pool.acquire

        @asynccontextmanager
        async def _faulty_acquire():
            async with orig_acquire() as conn:
                orig_cursor = conn.cursor

                class _InterceptCursor:
                    def __init__(self, inner):
                        self._inner = inner

                    async def __aenter__(self):
                        self._cur = await self._inner.__aenter__()
                        return self

                    async def __aexit__(self, *exc):
                        return await self._inner.__aexit__(*exc)

                    async def execute(self, sql, params=()):
                        if isinstance(sql, str) and sql.startswith("START TRANSACTION"):
                            raise asyncio.CancelledError("injected cancel after SET TRANSACTION")
                        return await self._cur.execute(sql, params)

                conn.cursor = lambda *a, **kw: _InterceptCursor(orig_cursor(*a, **kw))
                try:
                    yield conn
                finally:
                    conn.cursor = orig_cursor

        db._pool.acquire = _faulty_acquire
        try:
            with pytest.raises(asyncio.CancelledError):
                async with db.read_snapshot():
                    pass
        finally:
            db._pool.acquire = orig_acquire

        # Next transaction on the SAME maxsize=1 pool MUST observe READ COMMITTED!
        async with db.transaction() as tx:
            before = await tx.fetchone(f"SELECT val FROM {tbl} WHERE id=1")
            assert before is not None and int(before["val"]) == 10
            # Commit an update from a second connection while tx is open.
            await writer.execute(f"UPDATE {tbl} SET val=20 WHERE id=1")
            # Under READ COMMITTED, tx MUST immediately see 20; leaked REPEATABLE READ would stay at 10!
            after = await tx.fetchone(f"SELECT val FROM {tbl} WHERE id=1")
            assert after is not None and int(after["val"]) == 20, (
                f"REPEATABLE READ leaked to pooled transaction! saw val={after['val']}"
            )
    finally:
        await writer.execute(f"DROP TABLE IF EXISTS {tbl}")
        await db.close()
        await writer.close()


# ── H4 Rework Regressions (P1-1 ~ P1-4) ────────────────────────────────────


async def test_h4_p1_1_cancel_send_during_closing_release_preserves_db_lease_release(registry_store, monkeypatch):
    """P1-1 rework: cancelling send() while closing release is in flight must not abort DB lease release."""
    store, _ = registry_store
    model_calls: list[str] = []

    class _NoRunModel:
        async def complete(self, _request, **_kwargs):
            from power_loop.core.execution_context import get_execution_context
            model_calls.append(get_execution_context().session_id)
            return LLMResponse(raw_text="should not run")

        async def close(self):
            pass

    factory = RegistryRuntimeFactory()

    async def _build(_d, _c):
        return ExecutionEnvironment(_NoRunModel(), ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9))

    factory.register("test.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="h4-p11-rw-node",
        boot_id="h4-p11-rw-boot",
        capacity=4,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    await rt.worker.start()
    try:
        t = await rt.client.create_session(
            DESCRIPTOR, actor=ACTOR, options={"session_id": "h4-p11-rw-cancel-release"},
        )
        orig_aac = rt.client.inputs.accept_and_claim
        orig_release_claim = rt.worker.inputs.release_claim
        aac_committed = asyncio.Event()
        allow_aac_return = asyncio.Event()
        release_started = asyncio.Event()
        allow_release_finish = asyncio.Event()

        async def _paused_aac(*args, **kwargs):
            result = await orig_aac(*args, **kwargs)
            aac_committed.set()
            await asyncio.wait_for(allow_aac_return.wait(), 5)
            return result

        async def _paused_release(batch, *args, **kwargs):
            release_started.set()
            await asyncio.wait_for(allow_release_finish.wait(), 5)
            return await orig_release_claim(batch, *args, **kwargs)

        monkeypatch.setattr(rt.client.inputs, "accept_and_claim", _paused_aac)
        monkeypatch.setattr(rt.worker.inputs, "release_claim", _paused_release)

        send_task = asyncio.create_task(
            rt.client.send(t, "cancel during closing release", message_id="m-rw-p11", actor=ACTOR)
        )
        await asyncio.wait_for(aac_committed.wait(), 5)

        await asyncio.wait_for(rt.worker.shutdown(timeout_s=2, cancel_running=False), 5)
        assert rt.worker._closing is True

        # Let send() proceed into dispatch_claimed -> closing release task, and pause inside release_claim.
        allow_aac_return.set()
        await asyncio.wait_for(release_started.wait(), 5)

        # Cancel send() while it is waiting on drain_closing_releases()!
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task

        # Slots must be released immediately, while the closing release task stays alive until finished.
        assert rt.worker._active == {}
        assert rt.worker._fusing == {}
        assert len(rt.worker._closing_release_tasks) == 1

        allow_release_finish.set()
        await asyncio.wait_for(rt.worker.drain_closing_releases(), 5)
        assert rt.worker._closing_release_tasks == set()
        assert model_calls == []

        row = await store._db.fetchone(
            f"SELECT execution_state,owner_node_id,owner_boot_id,owner_lease_until_ms,pending_count "
            f"FROM {store.t.sessions} WHERE session_id=?",
            (t.session_id,),
        )
        assert row is not None
        assert row["execution_state"] == "queued"
        assert row["owner_node_id"] is None
        assert row["owner_boot_id"] is None
        assert row["owner_lease_until_ms"] is None
        assert int(row["pending_count"]) == 1
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_h4_p1_2_workflow_leaf_public_reply_route_never_upgrades_to_workflow_node_result(registry_store):
    """P1-2 rework: public agent_result on a workflow leaf never upgrades to workflow_node_result on final or recover."""
    from dataclasses import replace

    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import ChildSpec
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        main = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12rw-main"})
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        leaf_desc = replace(DESCRIPTOR, config={"workflow_node": "n1"})

        cm_drv = await port.create_many(
            parent=main,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12rw-drv",
                message_id="drv-start",
                body={"text": "run wf"},
                descriptor=drv_desc,
                result_kind="workflow_result",
                cancel_propagation="independent",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
        )
        drv_batch = cm_drv.preclaimed[0]
        drv = drv_batch.target

        # Register a local driver waiter queue to prove public agent_result NEVER triggers driver workflow events.
        driver_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        port._driver_waiters[drv.session_id] = (driver_events, asyncio.Event(), object())

        cm_leaf = await port.create_many(
            parent=drv,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12rw-leaf",
                message_id="leaf-start",
                body={"text": "step 1"},
                descriptor=leaf_desc,
                result_kind="workflow_node_result",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
            parent_owner_epoch=drv_batch.owner_epoch,
            parent_cancel_seq=drv_batch.cancel_seq,
        )
        leaf_b1 = cm_leaf.preclaimed[0]
        leaf = leaf_b1.target

        # 1. Genuine internal leaf start settles as workflow_node_result and fires driver event.
        fin1 = await port.final(leaf_b1, final_text="internal step 1")
        assert fin1.result_delivery_error is None
        wf_rows_1 = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in wf_rows_1] == ["user", "workflow_node_result"]
        assert wf_rows_1[1]["result_kind"] == "workflow_node_result"
        assert not driver_events.empty()
        driver_events.get_nowait()

        # 2. Public send to finished workflow leaf with ordinary agent_result route targeting driver:
        #    final() MUST deliver agent_result (NOT workflow_node_result) and fire 0 driver workflow events!
        await client.send(
            leaf,
            "public ordinary send",
            message_id="pub-ord-1",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        leaf_b2 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert leaf_b2 is not None
        fin2 = await port.final(leaf_b2, final_text="public reply")
        assert fin2.result_delivery_error is None
        wf_rows_2 = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in wf_rows_2] == ["user", "workflow_node_result", "agent_result"]
        assert wf_rows_2[2]["result_kind"] == "agent_result"
        assert driver_events.empty(), "public agent_result must NOT trigger driver workflow_node_result event"

        # 3. Public send to workflow leaf with agent_result route that crashes -> recover() MUST emit
        #    0 workflow_node_result rows and 0 driver events!
        await client.send(
            leaf,
            "public crash send",
            message_id="pub-crash-1",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        leaf_b3 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert leaf_b3 is not None
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
            (leaf.session_id,),
        )
        rec_pub = await port.recover(leaf, node_id="n2", boot_id="b2")
        assert rec_pub is not None
        wf_rows_3 = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in wf_rows_3] == ["user", "workflow_node_result", "agent_result"]
        assert driver_events.empty()

        # 4. Internal workflow_continue (public=False, workflow_node_result) that crashes ->
        #    recover() STILL delivers workflow_node_result to driver!
        await port.accept(
            leaf,
            "internal continue crash",
            message_id="int-cont-crash",
            actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        leaf_b4 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert leaf_b4 is not None
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
            (leaf.session_id,),
        )
        rec_int = await port.recover(leaf, node_id="n2", boot_id="b2")
        assert rec_int is not None
        wf_rows_4 = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in wf_rows_4] == [
            "user", "workflow_node_result", "agent_result", "workflow_node_result",
        ]
        assert not driver_events.empty()
    finally:
        port._driver_waiters.pop("p12rw-drv", None)
        await client.aclose()
        await transport.aclose()


async def test_h4_p1_2_deep_ancestor_slow_path_rebind_and_stale_ancestor_retarget(registry_store, monkeypatch):
    """P1-2 rework: deep-ancestor slow path re-checks under tx lock; rebind retargets stale ancestor routes."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, connect = registry_store
    peer = await connect(schema="verify")
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    peer_port = RuntimeInputStore(peer)
    try:
        # Part A: Deep-ancestor slow path racing with rebind_children between unlocked pre-check and write tx.
        root_old = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12toctou-rold"})
        root_new = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12toctou-rnew"})
        mid = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=root_old, options={"session_id": "p12toctou-mid"},
        )
        leaf = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=mid, options={"session_id": "p12toctou-leaf"},
        )
        deep_actor = TrustedActor(ACTOR.tenant_id, ACTOR.principal_id, root_old.session_id)

        orig_tx = store._db.transaction
        rebound_once = False
        tx_count = 0

        @asynccontextmanager
        async def _rebind_before_slow_tx():
            nonlocal rebound_once, tx_count
            tx_count += 1
            # tx_count==1 is accept() fast-path UPDATE (0 rows for deep ancestor);
            # tx_count==2 is _accept_slow_path()'s write tx right after unlocked diag/validation!
            if tx_count == 2 and not rebound_once:
                rebound_once = True
                rb = await peer_port.rebind_children(root_old, root_new, child_targets=[mid], actor=ACTOR)
                assert rb.rebound_children == (mid.session_id,)
            async with orig_tx() as tx:
                yield tx

        monkeypatch.setattr(store._db, "transaction", _rebind_before_slow_tx)
        with pytest.raises(RuntimeRequestError) as exc_info:
            await client.send(
                leaf,
                "stale deep ancestor send",
                message_id="p12toctou-m1",
                actor=deep_actor,
                reply_route={"target": root_old, "result_kind": "agent_result"},
            )
        assert exc_info.value.code == "forbidden"
        assert rebound_once is True
        monkeypatch.setattr(store._db, "transaction", orig_tx)

        # Verify nothing was inserted into leaf's inbox.
        qrows = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?", (leaf.session_id,),
        )
        assert qrows == []

        # Part B: Rebind retargets active grandchild per-input route pointing to stale old ancestor.
        old_g = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12rebind-oldg"})
        new_g = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12rebind-newg"})
        p_mid = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=old_g, options={"session_id": "p12rebind-mid"},
        )
        g_leaf = await client.create_session(
            DESCRIPTOR, actor=ACTOR, parent=p_mid, options={"session_id": "p12rebind-leaf"},
        )
        await client.send(
            g_leaf,
            "route to old grandparent",
            message_id="p12rebind-m1",
            actor=ACTOR,
            reply_route={"target": old_g, "result_kind": "agent_result"},
        )
        rb2 = await port.rebind_children(old_g, new_g, child_targets=[p_mid], actor=ACTOR)
        assert rb2.rebound_children == (p_mid.session_id,)
        retargeted = await store._db.fetchone(
            f"SELECT reply_to_session_id FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id='p12rebind-m1'",
            (g_leaf.session_id,),
        )
        assert retargeted is not None and retargeted["reply_to_session_id"] == new_g.session_id
    finally:
        await client.aclose()
        await transport.aclose()


async def test_h4_p1_3_concurrent_linked_child_attached_between_discovery_and_direct_children(
    registry_store, monkeypatch,
):
    """P1-3 rework: linked child committed between _discover_linked_tree and direct_children triggers retry."""
    from power_loop.contracts.errors import SessionBusy

    store, connect = registry_store
    is_sqlite = store._db.dialect.name == "sqlite"
    peer = await connect(schema="verify")
    port = RuntimeInputStore(store)
    peer_port = RuntimeInputStore(peer)

    # Case 1: Idle linked child attached in the discovery window -> retry includes & deletes BOTH.
    p1 = await target(port, "p13rw-parent-1")
    if is_sqlite:
        # SQLite uses BEGIN IMMEDIATE (single writer); pre-create the linked child and omit it
        # from the first _discover_linked_tree result to simulate the exact PG/MySQL race window.
        await target(port, "p13rw-late-linked-1", parent=p1, cancel_propagation="linked")
    orig_discover = closing._discover_linked_tree
    injected_1 = False
    calls_1 = 0

    async def _discover_inject_idle(st, tx, seed_ids, cascade):
        nonlocal injected_1, calls_1
        calls_1 += 1
        res = await orig_discover(st, tx, seed_ids, cascade)
        if not injected_1 and p1.session_id in seed_ids:
            injected_1 = True
            if is_sqlite:
                res.pop("p13rw-late-linked-1", None)
            else:
                await target(peer_port, "p13rw-late-linked-1", parent=p1, cancel_propagation="linked")
        return res

    monkeypatch.setattr(closing, "_discover_linked_tree", _discover_inject_idle)
    deleted_1 = await store.close_session_tree(p1.session_id)
    assert calls_1 >= 2
    assert set(deleted_1) == {p1.session_id, "p13rw-late-linked-1"}
    assert await store.get_session(p1.session_id) is None
    assert await store.get_session("p13rw-late-linked-1") is None

    # Case 2: Busy linked child attached in the discovery window -> retry detects busy & raises SessionBusy,
    # leaving parent open (NEVER half-closed as closing with an open linked child).
    p2 = await target(port, "p13rw-parent-2")
    if is_sqlite:
        late_sq = await target(port, "p13rw-late-linked-2", parent=p2, cancel_propagation="linked")
        await port.accept(late_sq, "active work", message_id="p13rw-m2", actor=ACTOR)
    injected_2 = False
    calls_2 = 0

    async def _discover_inject_busy(st, tx, seed_ids, cascade):
        nonlocal injected_2, calls_2
        calls_2 += 1
        res = await orig_discover(st, tx, seed_ids, cascade)
        if not injected_2 and p2.session_id in seed_ids:
            injected_2 = True
            if is_sqlite:
                res.pop("p13rw-late-linked-2", None)
            else:
                late = await target(peer_port, "p13rw-late-linked-2", parent=p2, cancel_propagation="linked")
                await peer_port.accept(late, "active work", message_id="p13rw-m2", actor=ACTOR)
        return res

    monkeypatch.setattr(closing, "_discover_linked_tree", _discover_inject_busy)
    with pytest.raises(SessionBusy):
        await store.close_session_tree(p2.session_id)
    assert calls_2 >= 2
    p2_state = await store.get_session(p2.session_id)
    assert p2_state is not None and p2_state.lifecycle == "open"


async def test_h4_p1_4_raw_cursor_with_prefix_colliding_session_ids_advances_and_terminates(registry_store):
    """P1-4 rework: legal IDs starting with sidi:/sid:ge:/sid:gt:/lc: never loop under raw cursor pagination."""
    store, _ = registry_store
    port = RuntimeInputStore(store)

    # Create open sessions including prefix-colliding IDs after raw cursor 'r', plus trailing session 't'
    # with an expired inbox item to prove pagination actually reaches and cleans 't'.
    for sid in ("sid:ge:b", "sid:gt:c", "sidi:a", "t"):
        await target(port, sid)
    t_ref = TargetRef(store.store_id, "t")
    r_t = await port.accept(t_ref, "expired inbox", message_id="p14rw-t-m1", actor=ACTOR)
    await port.cancel(r_t.input_ref, actor=ACTOR)

    future_ms = 2**62
    cursor: str | None = "r"
    pages = []
    seen_cursors: set[str] = set()
    for _ in range(10):
        page = await store.purge_expired_page(
            future_ms,
            limit=5,
            candidate_limit=1,
            categories={"inbox"},
            after_session_id=cursor,
        )
        pages.append(page)
        cursor = page.next_after
        if cursor is None:
            break
        assert cursor not in seen_cursors, f"cursor loop detected: {cursor!r} in {seen_cursors!r}"
        seen_cursors.add(cursor)

    assert cursor is None, f"pagination did not terminate: {[(p.scanned, p.deleted, p.next_after) for p in pages]}"
    assert sum(p.deleted.get("inbox", 0) for p in pages) == 1
    assert len(pages) == 5  # sid:ge:b, sid:gt:c, sidi:a, t (deletes 1), final empty page -> None

    # Direct raw pagination starting from legal prefix-looking session IDs (sid:gt:z, sid:ge:z, lc:open:0:z)
    # must treat the caller input as a literal session_id and reach/clean successor session 't'!
    for raw_start in ("sid:gt:z", "sid:ge:z", "lc:open:0:z"):
        await target(port, raw_start)
        r_next = await port.accept(t_ref, f"exp-{raw_start}", message_id=f"p14rw2-{raw_start}", actor=ACTOR)
        await port.cancel(r_next.input_ref, actor=ACTOR)
        deleted_total = 0
        c2: str | None = raw_start
        for _ in range(10):
            pg = await store.purge_expired_page(
                future_ms,
                limit=5,
                candidate_limit=1,
                categories={"inbox"},
                after_session_id=c2,
            )
            deleted_total += pg.deleted.get("inbox", 0)
            c2 = pg.next_after
            if c2 is None:
                break
        assert c2 is None
        assert deleted_total == 1, f"failed to clean successor 't' starting from raw ID {raw_start!r}"


async def test_h4_p1_2_workflow_leaf_cancel_never_upgrades_public_reply_route(registry_store):
    """P1-2 rework 2: cancelling pending/finished public input on workflow leaf never emits workflow_node_result."""
    from dataclasses import replace

    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import ChildSpec
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        main = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12c-main"})
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        leaf_desc = replace(DESCRIPTOR, config={"workflow_node": "n1"})

        cm_drv = await port.create_many(
            parent=main,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12c-drv",
                message_id="drv-start",
                body={"text": "run wf"},
                descriptor=drv_desc,
                result_kind="workflow_result",
                cancel_propagation="independent",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
        )
        drv_batch = cm_drv.preclaimed[0]
        drv = drv_batch.target
        driver_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        port._driver_waiters[drv.session_id] = (driver_events, asyncio.Event(), object())

        cm_leaf = await port.create_many(
            parent=drv,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12c-leaf",
                message_id="leaf-start",
                body={"text": "step 1"},
                descriptor=leaf_desc,
                result_kind="workflow_node_result",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
            parent_owner_epoch=drv_batch.owner_epoch,
            parent_cancel_seq=drv_batch.cancel_seq,
        )
        leaf_b1 = cm_leaf.preclaimed[0]
        leaf = leaf_b1.target
        await port.final(leaf_b1, final_text="step 1 ok")
        driver_events.get_nowait()  # consume initial workflow_node_result event

        # 1. Cancel pending public input by InputRef (routed to main as agent_result):
        #    must deliver agent_result to main, 0 workflow_node_result to drv, 0 driver events!
        pub_r1 = await client.send(
            leaf, "pub pending 1", message_id="p12c-pub-1", actor=ACTOR,
            reply_route={"target": main, "result_kind": "agent_result"},
        )
        c1 = await port.cancel(pub_r1.input_ref, actor=ACTOR, reason="user stop 1")
        assert c1.state == "voided"
        main_rows_1 = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY id ASC",
            (main.session_id,),
        )
        assert [r["kind"] for r in main_rows_1] == ["agent_result"]
        drv_rows_1 = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in drv_rows_1] == ["user", "workflow_node_result"]
        assert driver_events.empty()

        # 2. Repeat cancel on already-finished public input (redeliver path) must NOT emit workflow_node_result!
        c1_again = await port.cancel(pub_r1.input_ref, actor=ACTOR, reason="repeat")
        assert c1_again.state == "already_finished"
        drv_rows_1b = await store._db.fetchall(
            f"SELECT kind FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in drv_rows_1b] == ["user", "workflow_node_result"]
        assert driver_events.empty()

        # 3. Cancel pending public input by session target (_void_pending_for_session_tx, routed to drv as agent_result):
        #    must deliver agent_result to drv (NOT workflow_node_result) and 0 driver workflow events!
        await client.send(
            leaf, "pub pending 2", message_id="p12c-pub-2", actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        await port.cancel(leaf, actor=ACTOR, reason="session stop")
        drv_rows_2 = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in drv_rows_2] == ["user", "workflow_node_result", "agent_result"]
        assert drv_rows_2[2]["result_kind"] == "agent_result"
        assert driver_events.empty()

        # 4. Genuine internal workflow_continue pending input cancelled by InputRef & by session
        #    STILL delivers workflow_node_result and fires driver events!
        int_r1 = await port.accept(
            leaf, "internal cont 1", message_id="p12c-int-1", actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        c_int1 = await port.cancel(int_r1.input_ref, actor=ACTOR, reason="wf stop 1")
        assert c_int1.state == "voided"
        assert not driver_events.empty()
        driver_events.get_nowait()

        await port.accept(
            leaf, "internal cont 2", message_id="p12c-int-2", actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        await port.cancel(leaf, actor=ACTOR, reason="wf stop 2")
        assert not driver_events.empty()
        driver_events.get_nowait()

        drv_rows_3 = await store._db.fetchall(
            f"SELECT kind FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert [r["kind"] for r in drv_rows_3] == [
            "user", "workflow_node_result", "agent_result", "workflow_node_result", "workflow_node_result",
        ]
    finally:
        port._driver_waiters.pop("p12c-drv", None)
        await client.aclose()
        await transport.aclose()


async def test_h4_p1_2_same_tree_rebind_retargets_direct_parent_and_preserves_shared_ancestor(registry_store):
    """P1-2 rework 2: same-tree rebind retargets direct-parent route to new parent while keeping other shared ancestors."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        # Case 1: Same-tree rebind (A -> B, A -> C; C routes to A; rebind C from A to B -> delivers & wakes B!)
        a = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12st-a"})
        b = await client.create_session(DESCRIPTOR, actor=ACTOR, parent=a, options={"session_id": "p12st-b"})
        c = await client.create_session(DESCRIPTOR, actor=ACTOR, parent=a, options={"session_id": "p12st-c"})
        await client.send(
            c, "work for parent", message_id="p12st-m1", actor=ACTOR,
            reply_route={"target": a, "result_kind": "agent_result"},
        )
        batch_c = await port.claim(c, node_id="n1", boot_id="b1")
        assert batch_c is not None and batch_c.reply_to_session_id == a.session_id

        rb = await port.rebind_children(a, b, child_targets=[c], actor=ACTOR)
        assert rb.rebound_children == (c.session_id,)
        c_state = await store.get_session(c.session_id)
        assert c_state is not None and c_state.parent_session_id == b.session_id

        fb = await port.final(batch_c, final_text="rebound to b")
        assert fb.result_delivery_error is None
        assert fb.result_activation == b
        a_rows = await store._db.fetchall(
            f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=?", (a.session_id,),
        )
        b_rows = await store._db.fetchall(
            f"SELECT kind,content_json FROM {store.t.follow_up_queue} WHERE session_id=?", (b.session_id,),
        )
        assert a_rows == []
        assert len(b_rows) == 1 and b_rows[0]["kind"] == "agent_result"

        # Case 2: Shared grandparent (G -> A2 -> C2, G -> B2; C2 explicitly routes to G; rebind A2 -> B2 keeps G!)
        g = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12sh-g"})
        a2 = await client.create_session(DESCRIPTOR, actor=ACTOR, parent=g, options={"session_id": "p12sh-a2"})
        b2 = await client.create_session(DESCRIPTOR, actor=ACTOR, parent=g, options={"session_id": "p12sh-b2"})
        c2 = await client.create_session(DESCRIPTOR, actor=ACTOR, parent=a2, options={"session_id": "p12sh-c2"})
        await client.send(
            c2, "work for shared gp", message_id="p12sh-m1", actor=ACTOR,
            reply_route={"target": g, "result_kind": "agent_result"},
        )
        batch_c2 = await port.claim(c2, node_id="n1", boot_id="b1")
        assert batch_c2 is not None and batch_c2.reply_to_session_id == g.session_id

        rb2 = await port.rebind_children(a2, b2, child_targets=[c2], actor=ACTOR)
        assert rb2.rebound_children == (c2.session_id,)
        fb2 = await port.final(batch_c2, final_text="kept on g")
        assert fb2.result_delivery_error is None
        assert fb2.result_activation == g
        g_rows = await store._db.fetchall(
            f"SELECT kind FROM {store.t.follow_up_queue} WHERE session_id=?", (g.session_id,),
        )
        b2_rows = await store._db.fetchall(
            f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=?", (b2.session_id,),
        )
        assert [r["kind"] for r in g_rows] == ["agent_result"]
        assert b2_rows == []
    finally:
        await client.aclose()
        await transport.aclose()


async def test_h4_p1_2_rebind_replay_undelivered_preserves_original_result_kind(registry_store):
    """P1-2 rework 3: rebind replay_undelivered preserves agent_result vs workflow_node_result on workflow leaves."""
    from dataclasses import replace

    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import ChildSpec
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        main = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12rp-main"})
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        leaf_desc = replace(DESCRIPTOR, config={"workflow_node": "n1"})

        # Case A: Workflow leaf settles a PUBLIC agent_result input after old driver closes ->
        #         rebind_children(replay_undelivered=True) MUST replay as agent_result (NOT workflow_node_result)!
        cm_drv1 = await port.create_many(
            parent=main,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12rp-drv1",
                message_id="rp-drv1-start",
                body={"text": "wf1"},
                descriptor=drv_desc,
                result_kind="workflow_result",
                cancel_propagation="independent",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
        )
        drv1_b = cm_drv1.preclaimed[0]
        drv1 = drv1_b.target
        new_parent1 = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12rp-newp1"})

        cm_leaf1 = await port.create_many(
            parent=drv1,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12rp-leaf1",
                message_id="rp-leaf1-start",
                body={"text": "init"},
                descriptor=leaf_desc,
                result_kind="workflow_node_result",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
            parent_owner_epoch=drv1_b.owner_epoch,
            parent_cancel_seq=drv1_b.cancel_seq,
        )
        leaf1_init = cm_leaf1.preclaimed[0]
        leaf1 = leaf1_init.target
        await port.final(leaf1_init, final_text="init ok")

        # Public send with ordinary agent_result route targeting drv1; claim, then close drv1 before final.
        pub_rcpt = await client.send(
            leaf1, "public task", message_id="rp-pub-1", actor=ACTOR,
            reply_route={"target": drv1, "result_kind": "agent_result"},
        )
        leaf1_pub = await port.claim(leaf1, node_id="n1", boot_id="b1")
        assert leaf1_pub is not None
        await port.set_lifecycle(drv1, "closed", actor=ACTOR, inflight="abort", subtree="none")
        fin_pub = await port.final(leaf1_pub, final_text="public undelivered answer")
        assert fin_pub.result_delivery_error == "session_closed"

        rb1 = await port.rebind_children(
            drv1, new_parent1, child_targets=[leaf1], replay_undelivered=True, actor=ACTOR,
        )
        assert rb1.replayed_results == 1
        np1_rows = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY id ASC",
            (new_parent1.session_id,),
        )
        assert len(np1_rows) == 1
        assert np1_rows[0]["kind"] == "agent_result"
        assert np1_rows[0]["result_kind"] == "agent_result"
        cleared1 = await store._db.fetchone(
            f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (leaf1.session_id, pub_rcpt.message_id),
        )
        assert cleared1 is not None and cleared1["result_delivery_error"] is None

        # Case B: Workflow leaf settles a genuine INTERNAL workflow_node_result input after old driver closes ->
        #         rebind_children(replay_undelivered=True) MUST preserve workflow_node_result!
        cm_drv2 = await port.create_many(
            parent=main,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12rp-drv2",
                message_id="rp-drv2-start",
                body={"text": "wf2"},
                descriptor=drv_desc,
                result_kind="workflow_result",
                cancel_propagation="independent",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
        )
        drv2_b = cm_drv2.preclaimed[0]
        drv2 = drv2_b.target
        new_drv2 = await client.create_session(drv_desc, actor=ACTOR, options={"session_id": "p12rp-newdrv2"})

        cm_leaf2 = await port.create_many(
            parent=drv2,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12rp-leaf2",
                message_id="rp-leaf2-start",
                body={"text": "internal step"},
                descriptor=leaf_desc,
                result_kind="workflow_node_result",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
            parent_owner_epoch=drv2_b.owner_epoch,
            parent_cancel_seq=drv2_b.cancel_seq,
        )
        leaf2_b = cm_leaf2.preclaimed[0]
        leaf2 = leaf2_b.target
        await port.set_lifecycle(drv2, "closed", actor=ACTOR, inflight="abort", subtree="none")
        fin_int = await port.final(leaf2_b, final_text="internal undelivered answer")
        assert fin_int.result_delivery_error == "session_closed"

        rb2 = await port.rebind_children(
            drv2, new_drv2, child_targets=[leaf2], replay_undelivered=True, actor=ACTOR,
        )
        assert rb2.replayed_results == 1
        np2_rows = await store._db.fetchall(
            f"SELECT kind,result_kind FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY id ASC",
            (new_drv2.session_id,),
        )
        assert len(np2_rows) == 1
        assert np2_rows[0]["kind"] == "workflow_node_result"
        assert np2_rows[0]["result_kind"] == "workflow_node_result"
        cleared2 = await store._db.fetchone(
            f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (leaf2.session_id, "rp-leaf2-start"),
        )
        assert cleared2 is not None and cleared2["result_delivery_error"] is None
    finally:
        await client.aclose()
        await transport.aclose()


async def test_h4_p1_4_truncated_cursor_advances_every_page_for_scheduler_budget(registry_store):
    """P1-4 rework 3: single-session multi-page truncation emits strictly advancing cursors matching scheduler checks."""
    from dataclasses import asdict

    store, _ = registry_store
    port = RuntimeInputStore(store)

    s1 = await target(port, "p14-sched-single")
    for i in range(15):
        await store.append_message(s1.session_id, role="user", content=f"msg-{i}")
    assert await store.close_if_idle(s1.session_id) == "closed"

    future_ms = 2**62
    batch = 5
    max_rows = 10

    async def _run_scheduler_day(start_after: str | None) -> tuple[int, str | None, bool]:
        """Mirror scheduler/app/jobs.py:276-298 exact page acceptance & budget enforcement."""
        fetched_total = 0
        deleted_total = 0
        cur_after = start_after
        hit_tail = False
        while fetched_total < max_rows and deleted_total < max_rows:
            candidate_limit = min(batch, max_rows - fetched_total)
            delete_limit = min(batch, max_rows - deleted_total)
            raw_page = await store.purge_expired_page(
                future_ms,
                limit=delete_limit,
                candidate_limit=candidate_limit,
                after_session_id=cur_after,
            )
            # Round-trip through JSON just like agent HTTP response -> scheduler httpx client.
            page = json.loads(json.dumps(asdict(raw_page)))
            deleted_vals = list(page["deleted"].values())
            scanned, fetched, next_after = page["scanned"], page["fetched"], page["next_after"]
            assert not any(type(c) is not int or c < 0 for c in deleted_vals)
            assert sum(deleted_vals) <= delete_limit
            assert 0 <= scanned <= fetched <= candidate_limit
            if next_after is not None:
                assert isinstance(next_after, str) and next_after
                assert next_after != cur_after, f"scheduler rejects repeated cursor: {next_after!r}"
                assert scanned > 0
            deleted_total += sum(deleted_vals)
            fetched_total += fetched
            cur_after = next_after
            if cur_after is None:
                hit_tail = True
                break
        return deleted_total, cur_after, hit_tail

    # Day 1: budget=10, batch=5 -> deletes exactly 10 messages (2 confirmed pages), stops at budget.
    day1_deleted, day1_cursor, day1_tail = await _run_scheduler_day(None)
    assert day1_deleted == 10
    assert day1_cursor is not None
    assert day1_tail is False
    remaining_msgs = await store._db.fetchall(
        f"SELECT id FROM {store.t.messages} WHERE session_id=?", (s1.session_id,),
    )
    assert len(remaining_msgs) == 5, f"expected 5 msgs remaining after day 1, got {len(remaining_msgs)}"

    # Day 2: resumes from day1_cursor -> deletes remaining 5 messages + 1 session row = 6, reaches tail.
    day2_deleted, day2_cursor, day2_tail = await _run_scheduler_day(day1_cursor)
    assert day2_deleted == 6
    assert day2_cursor is None
    assert day2_tail is True
    assert await store.get_session(s1.session_id) is None

    # Raw-cursor mode multi-page truncation on a single session also strictly advances every page.
    s_raw = await target(port, "p14-sched-raw")
    for i in range(11):
        await store.append_message(s_raw.session_id, role="user", content=f"raw-{i}")
    assert await store.close_if_idle(s_raw.session_id) == "closed"
    raw_cursor: str | None = "p14-sched-r"
    raw_seen: set[str] = set()
    raw_total = 0
    for _ in range(10):
        pg = await store.purge_expired_page(
            future_ms, limit=5, candidate_limit=5, after_session_id=raw_cursor,
        )
        raw_total += sum(pg.deleted.values())
        if pg.next_after is None:
            raw_cursor = None
            break
        assert pg.next_after != raw_cursor
        assert pg.next_after not in raw_seen
        raw_seen.add(pg.next_after)
        raw_cursor = pg.next_after
    assert raw_cursor is None
    assert raw_total == 12
    assert await store.get_session(s_raw.session_id) is None


@pytest.mark.parametrize("leaf_id_mode", ["short", "long60"])
async def test_h4_p1_2_rebind_replay_merged_batch_preserves_workflow_kind_for_short_and_long_ids(
    registry_store, leaf_id_mode,
):
    """P1-2 rework 4: merged ordinary + internal workflow_continue batch replays as 1 workflow_node_result for short & 60-char IDs."""
    from dataclasses import replace

    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import ChildSpec
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        sfx = "sh" if leaf_id_mode == "short" else "lg"
        main = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": f"p12mb-main-{sfx}"})
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        leaf_desc = replace(DESCRIPTOR, config={"workflow_node": "n1"})

        cm_drv = await port.create_many(
            parent=main,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id=f"p12mb-drv-{sfx}",
                message_id=f"mb-drv-start-{sfx}",
                body={"text": "wf"},
                descriptor=drv_desc,
                result_kind="workflow_result",
                cancel_propagation="independent",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
        )
        drv_b = cm_drv.preclaimed[0]
        drv = drv_b.target
        new_drv = await client.create_session(
            drv_desc, actor=ACTOR, options={"session_id": f"p12mb-newdrv-{sfx}"},
        )

        leaf_sid = f"leaf-{sfx}" if leaf_id_mode == "short" else ("p12mb_" + "x" * 54)
        assert leaf_id_mode != "long60" or len(leaf_sid) == 60

        cm_leaf = await port.create_many(
            parent=drv,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id=leaf_sid,
                message_id=f"mb-leaf-init-{sfx}",
                body={"text": "init"},
                descriptor=leaf_desc,
                result_kind="workflow_node_result",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
            parent_owner_epoch=drv_b.owner_epoch,
            parent_cancel_seq=drv_b.cancel_seq,
        )
        leaf_init = cm_leaf.preclaimed[0]
        leaf = leaf_init.target
        await port.final(leaf_init, final_text="init ok")

        # Queue ordinary public message FIRST (smaller id), then internal workflow_continue SECOND;
        # both are claimed into the SAME send_index=2.
        await client.send(
            leaf,
            "ordinary chat first",
            message_id=f"mb-pub-{sfx}",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        await port.accept(
            leaf,
            "internal workflow continue second",
            message_id=f"mb-wf-{sfx}",
            actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        merged_batch = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert merged_batch is not None
        assert [r.message_id for r in merged_batch.input_refs] == [f"mb-pub-{sfx}", f"mb-wf-{sfx}"]
        assert merged_batch.reply_result_kind == "workflow_node_result"

        # Close old driver before final so merged batch records result_delivery_error='session_closed' on BOTH rows.
        await port.set_lifecycle(drv, "closed", actor=ACTOR, inflight="abort", subtree="none")
        fin = await port.final(merged_batch, final_text="merged workflow answer")
        assert fin.result_delivery_error == "session_closed"

        rb = await port.rebind_children(
            drv, new_drv, child_targets=[leaf], replay_undelivered=True, actor=ACTOR,
        )
        assert rb.replayed_results == 1

        new_inbox = await store._db.fetchall(
            f"SELECT kind,result_kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (new_drv.session_id,),
        )
        assert len(new_inbox) == 1, f"expected exactly 1 replayed receipt, got {len(new_inbox)}: {new_inbox}"
        assert new_inbox[0]["kind"] == "workflow_node_result"
        assert new_inbox[0]["result_kind"] == "workflow_node_result"
        replayed_body = json.loads(new_inbox[0]["content_json"])
        assert replayed_body["input_ref"]["message_id"] == f"mb-wf-{sfx}"

        src_errors = await store._db.fetchall(
            f"SELECT item_id,result_delivery_error FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id IN (?,?) ORDER BY id ASC",
            (leaf.session_id, f"mb-pub-{sfx}", f"mb-wf-{sfx}"),
        )
        assert len(src_errors) == 2
        assert all(r["result_delivery_error"] is None for r in src_errors)
    finally:
        await client.aclose()
        await transport.aclose()


async def test_h4_p1_2_merged_batch_workflow_leaf_final_and_recover_reference_internal_input(registry_store):
    """P1-2 rework 5: merged ordinary + internal workflow_continue batch cites the internal input in final() and recover()."""
    from dataclasses import replace

    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import ChildSpec
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        main = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12mbsrc-main"})
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        leaf_desc = replace(DESCRIPTOR, config={"workflow_node": "n1"})

        cm_drv = await port.create_many(
            parent=main,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12mbsrc-drv",
                message_id="mbsrc-drv-start",
                body={"text": "wf"},
                descriptor=drv_desc,
                result_kind="workflow_result",
                cancel_propagation="independent",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
        )
        drv_b = cm_drv.preclaimed[0]
        drv = drv_b.target

        driver_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        port._driver_waiters[drv.session_id] = (driver_events, asyncio.Event(), object())

        cm_leaf = await port.create_many(
            parent=drv,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12mbsrc-leaf",
                message_id="mbsrc-leaf-init",
                body={"text": "init"},
                descriptor=leaf_desc,
                result_kind="workflow_node_result",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
            parent_owner_epoch=drv_b.owner_epoch,
            parent_cancel_seq=drv_b.cancel_seq,
        )
        leaf_init = cm_leaf.preclaimed[0]
        leaf = leaf_init.target
        await port.final(leaf_init, final_text="init ok")
        assert not driver_events.empty()
        init_evt = driver_events.get_nowait()
        assert init_evt["payload"]["input_ref"]["message_id"] == "mbsrc-leaf-init"

        # 1. Normal final() (consumed=None) on merged batch: ordinary chat first, internal next second.
        await client.send(
            leaf,
            "chat first",
            message_id="mbsrc-chat-1",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        await port.accept(
            leaf,
            "internal next second",
            message_id="mbsrc-next-1",
            actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        b1 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert b1 is not None
        assert [r.message_id for r in b1.input_refs] == ["mbsrc-chat-1", "mbsrc-next-1"]
        assert b1.reply_to_item_id == "mbsrc-next-1"
        assert b1.reply_result_kind == "workflow_node_result"

        fin1 = await port.final(b1, final_text="merged normal reply")
        assert fin1.result_delivery_error is None
        drv_rows_1 = await store._db.fetchall(
            f"SELECT kind,result_kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert len(drv_rows_1) == 3
        assert drv_rows_1[2]["kind"] == "workflow_node_result"
        assert drv_rows_1[2]["result_kind"] == "workflow_node_result"
        assert json.loads(drv_rows_1[2]["content_json"])["input_ref"]["message_id"] == "mbsrc-next-1"
        evt1 = driver_events.get_nowait()
        assert evt1["payload"]["input_ref"]["message_id"] == "mbsrc-next-1"

        # 2. Normal final(consumed=batch.input_refs) worker fast path on merged batch (0 extra SQL).
        await client.send(
            leaf,
            "chat first fast",
            message_id="mbsrc-chat-2",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        await port.accept(
            leaf,
            "internal next second fast",
            message_id="mbsrc-next-2",
            actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        b2 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert b2 is not None
        assert [r.message_id for r in b2.input_refs] == ["mbsrc-chat-2", "mbsrc-next-2"]
        assert b2.reply_to_item_id == "mbsrc-next-2"

        fin2 = await port.final(b2, final_text="merged fast reply", consumed=b2.input_refs)
        assert fin2.result_delivery_error is None
        drv_rows_2 = await store._db.fetchall(
            f"SELECT kind,result_kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert len(drv_rows_2) == 4
        assert drv_rows_2[3]["kind"] == "workflow_node_result"
        assert drv_rows_2[3]["result_kind"] == "workflow_node_result"
        assert json.loads(drv_rows_2[3]["content_json"])["input_ref"]["message_id"] == "mbsrc-next-2"
        evt2 = driver_events.get_nowait()
        assert evt2["payload"]["input_ref"]["message_id"] == "mbsrc-next-2"

        # 3. Lease-expired recover() on merged batch: ordinary chat first, internal next second.
        await client.send(
            leaf,
            "chat first crash",
            message_id="mbsrc-chat-3",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        await port.accept(
            leaf,
            "internal next second crash",
            message_id="mbsrc-next-3",
            actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        b3 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert b3 is not None
        assert [r.message_id for r in b3.input_refs] == ["mbsrc-chat-3", "mbsrc-next-3"]
        assert b3.reply_to_item_id == "mbsrc-next-3"

        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
            (leaf.session_id,),
        )
        rec = await port.recover(leaf, node_id="n2", boot_id="b2")
        assert rec is not None
        drv_rows_3 = await store._db.fetchall(
            f"SELECT kind,result_kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert len(drv_rows_3) == 5
        assert drv_rows_3[4]["kind"] == "workflow_node_result"
        assert drv_rows_3[4]["result_kind"] == "workflow_node_result"
        assert json.loads(drv_rows_3[4]["content_json"])["input_ref"]["message_id"] == "mbsrc-next-3"
        evt3 = driver_events.get_nowait()
        assert evt3["payload"]["input_ref"]["message_id"] == "mbsrc-next-3"
    finally:
        port._driver_waiters.pop("p12mbsrc-drv", None)
        await client.aclose()
        await transport.aclose()


async def test_h4_p1_2_mid_turn_claim_follow_up_internal_continue_delivers_workflow_node_result(registry_store):
    """P1-2 rework 6: mid-turn claim_follow_up of internal workflow_continue while handling ordinary chat delivers 1 workflow_node_result(next)."""
    from dataclasses import replace

    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import ChildSpec
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    port = client.inputs
    try:
        main = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "p12fup-main"})
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        leaf_desc = replace(DESCRIPTOR, config={"workflow_node": "n1"})

        cm_drv = await port.create_many(
            parent=main,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12fup-drv",
                message_id="fup-drv-start",
                body={"text": "wf"},
                descriptor=drv_desc,
                result_kind="workflow_result",
                cancel_propagation="independent",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
        )
        drv_b = cm_drv.preclaimed[0]
        drv = drv_b.target

        driver_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        port._driver_waiters[drv.session_id] = (driver_events, asyncio.Event(), object())

        cm_leaf = await port.create_many(
            parent=drv,
            actor=ACTOR,
            specs=[ChildSpec(
                session_id="p12fup-leaf",
                message_id="fup-leaf-init",
                body={"text": "init"},
                descriptor=leaf_desc,
                result_kind="workflow_node_result",
            )],
            node_id="n1",
            boot_id="b1",
            can_execute=True,
            free_slots=1,
            parent_owner_epoch=drv_b.owner_epoch,
            parent_cancel_seq=drv_b.cancel_seq,
        )
        leaf_init = cm_leaf.preclaimed[0]
        leaf = leaf_init.target
        await port.final(leaf_init, final_text="init ok")
        driver_events.get_nowait()

        # 1. Ordinary public chat (agent_result) claimed FIRST; internal workflow_continue arrives mid-turn
        #    and is drained by claim_follow_up(), then final(consumed=[chat, next]).
        await client.send(
            leaf,
            "public chat claimed first",
            message_id="fup-chat-1",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        b1 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert b1 is not None
        assert [r.message_id for r in b1.input_refs] == ["fup-chat-1"]
        assert b1.reply_result_kind == "agent_result"

        await port.accept(
            leaf,
            "internal continue mid-turn",
            message_id="fup-next-1",
            actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        fup1 = await port.claim_follow_up(b1)
        assert fup1 is not None
        fup1_refs, _, _, _ = fup1
        consumed_1 = [*b1.input_refs, *fup1_refs]
        assert [r.message_id for r in consumed_1] == ["fup-chat-1", "fup-next-1"]

        fin1 = await port.final(b1, final_text="merged mid-turn reply", consumed=consumed_1)
        assert fin1.result_delivery_error is None
        rows1 = await store._db.fetchall(
            f"SELECT kind,result_kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert len(rows1) == 3, f"expected [user, init wf, fup-next-1 wf], got {rows1}"
        assert rows1[2]["kind"] == "workflow_node_result"
        assert rows1[2]["result_kind"] == "workflow_node_result"
        assert json.loads(rows1[2]["content_json"])["input_ref"]["message_id"] == "fup-next-1"
        evt1 = driver_events.get_nowait()
        assert evt1["payload"]["input_ref"]["message_id"] == "fup-next-1"

        # 2. Unrouted human direct chat claimed FIRST; internal workflow_continue drained mid-turn via claim_follow_up().
        await client.send(leaf, "unrouted human chat", message_id="fup-human-2", actor=ACTOR)
        b2 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert b2 is not None and b2.reply_to_session_id is None

        await port.accept(
            leaf,
            "internal continue after human chat",
            message_id="fup-next-2",
            actor=ACTOR,
            kind="workflow_continue",
            reply_route={"target": drv, "result_kind": "workflow_node_result"},
            public=False,
        )
        fup2 = await port.claim_follow_up(b2)
        assert fup2 is not None
        consumed_2 = [*b2.input_refs, *fup2[0]]
        fin2 = await port.final(b2, final_text="merged after human", consumed=consumed_2)
        assert fin2.result_delivery_error is None
        rows2 = await store._db.fetchall(
            f"SELECT kind,result_kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert len(rows2) == 4
        assert rows2[3]["kind"] == "workflow_node_result"
        assert rows2[3]["result_kind"] == "workflow_node_result"
        assert json.loads(rows2[3]["content_json"])["input_ref"]["message_id"] == "fup-next-2"
        evt2 = driver_events.get_nowait()
        assert evt2["payload"]["input_ref"]["message_id"] == "fup-next-2"

        # 3. Negative control: ordinary public chat claimed FIRST + ordinary public follow-up drained mid-turn ->
        #    MUST remain ordinary agent_result and fire 0 workflow driver events!
        await client.send(
            leaf,
            "public chat A",
            message_id="fup-pub-a",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        b3 = await port.claim(leaf, node_id="n1", boot_id="b1")
        assert b3 is not None
        await client.send(
            leaf,
            "public chat B mid-turn",
            message_id="fup-pub-b",
            actor=ACTOR,
            reply_route={"target": drv, "result_kind": "agent_result"},
        )
        fup3 = await port.claim_follow_up(b3)
        assert fup3 is not None
        consumed_3 = [*b3.input_refs, *fup3[0]]
        fin3 = await port.final(b3, final_text="ordinary only reply", consumed=consumed_3)
        assert fin3.result_delivery_error is None
        rows3 = await store._db.fetchall(
            f"SELECT kind,result_kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? ORDER BY id ASC",
            (drv.session_id,),
        )
        assert len(rows3) == 5
        assert rows3[4]["kind"] == "agent_result"
        assert rows3[4]["result_kind"] == "agent_result"
        assert driver_events.empty(), "ordinary-only mid-turn follow-up must NOT trigger workflow_node_result event"
    finally:
        port._driver_waiters.pop("p12fup-drv", None)
        await client.aclose()
        await transport.aclose()
