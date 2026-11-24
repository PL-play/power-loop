"""Design 218 H4 regressions: control-plane concurrency, reply_route security, lock order, retention & 3-DB parity (P1-1 ~ P1-6)."""
from __future__ import annotations

import asyncio
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
