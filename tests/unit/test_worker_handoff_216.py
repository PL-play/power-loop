"""C03: real notifications, owner/final handoff and resource ordering on three DBs."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import pytest

from power_loop import AgentLoopConfig, InputRef, LLMResponse, ToolDefinition
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_transport import (
    PROTOCOL_VERSION,
    MemoryMessageTransport,
    PublishReceipt,
    TransportPacket,
)
from power_loop.contracts.errors import SessionLeaseLost
from power_loop.contracts.hooks import HookPoint
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent, ScriptedModel

from .session_native_fixture import ACTOR, target
from .test_send_sql_profile import Profiled, _counts

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Gate:
    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def hold(self):
        self.entered.set()
        await self.release.wait()

    async def wait(self):
        await asyncio.wait_for(self.entered.wait(), 10)


class Traffic(MemoryMessageTransport):
    """Delay packets without blocking the sender; ack proves the real listener ran."""
    def __init__(self):
        super().__init__()
        self.packets, self.held, self.seen = [], [], {}
        self.hold = lambda packet: False

    async def subscribe(self, binding):
        sub = await super().subscribe(binding)
        traffic = self

        class Observed:
            ready, aclose = sub.ready, sub.aclose

            def __aiter__(self):
                return self

            async def __anext__(self):
                delivery = await sub.__anext__()

                class Ack:
                    packet = delivery.packet

                    async def ack(self):
                        await delivery.ack()
                        traffic.seen.setdefault(id(self.packet), asyncio.Event()).set()

                return Ack()

        return Observed()

    async def publish(self, packet):
        self.packets.append(packet)
        if self.hold(packet):
            self.held.append(packet)
            return PublishReceipt(True)
        return await super().publish(packet)

    async def deliver(self, packet):
        await super().publish(packet)
        await asyncio.wait_for(self.seen.setdefault(id(packet), asyncio.Event()).wait(), 10)

    async def flush(self):
        packets, self.held = self.held, []
        for packet in packets:
            await self.deliver(packet)


@asynccontextmanager
async def handoff_rig(store, *, stage="model", tool=False, command=False):
    """Only first-run awaits are gated; all database work is the real worker path."""
    gate = Gate()
    traffic = Traffic()
    timeline, claims, stale, finals, tails = [], [], [], [], []

    class Model(ScriptedModel):
        def __init__(self):
            super().__init__()
            self.requests = []

        async def complete(self, request, **kwargs):
            self.requests.append(request)
            if len(self.requests) == 1:
                if stage == "model":
                    await gate.hold()
                if tool:
                    return LLMResponse(raw_text="", tool_calls=[{
                        "id": "gate-tool", "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    }])
            return LLMResponse(raw_text="done")

    registry = ToolRegistry()

    async def echo():
        return "tool done"

    registry.register(ToolDefinition("echo", "echo", input_schema={"type": "object"},
                                     pricing=Free("fixture")), echo)
    model = Model()
    agent = await RunningAgent(
        store, model, registry, AgentLoopConfig(max_rounds=5, context_budget_tokens=10**9),
        transport=traffic, command=("echo", {}) if command else None,
        fuse_local_send=False,
    ).start()
    worker = agent.runtime.worker
    claim, recover, final, has_pending = (
        worker.inputs.claim, worker._recover_stale, worker.inputs.final, worker._has_pending)
    build, notify = agent.runtime.factory.build, worker._notify_results

    async def observed_claim(*args, **kwargs):
        if stage == "claim" and not claims:
            await gate.hold()
        batch = await claim(*args, **kwargs)
        claims.append(batch)
        return batch

    async def observed_recover(*args, **kwargs):
        stale.append(True)
        return await recover(*args, **kwargs)

    async def observed_final(*args, **kwargs):
        if stage == "final_before" and not finals:
            await gate.hold()
        result = await final(*args, **kwargs)
        finals.append(result)
        return result

    async def observed_notify(batch):
        await notify(batch)
        if stage == "final_after" and len(finals) == 1:
            await gate.hold()

    async def observed_tail(sid):
        tails.append(sid)
        return await has_pending(sid)

    async def observed_build(descriptor, context):
        index = context.send_index
        timeline.append(f"build:{index}")
        if stage == "factory" and index == 1:
            await gate.hold()
        env = await build(descriptor, context)

        class Resources:
            async def aclose(self):
                timeline.append(f"close-start:{index}")
                if stage == "cleanup" and index == 1:
                    await gate.hold()
                timeline.append(f"close-end:{index}")

        env.resources = Resources()
        return env

    async def acquired(hook):
        context = hook.values["execution"]
        if stage == "acquired" and context.send_index == 1:
            await gate.hold()

    agent.runtime.hooks = agent.hooks
    agent.hooks.register(HookPoint.EXECUTION_ACQUIRED, acquired)
    agent.runtime.hook_timeout_s = 20
    worker.inputs.claim, worker.inputs.final = observed_claim, observed_final
    worker._recover_stale, worker._has_pending = observed_recover, observed_tail
    worker._notify_results, agent.runtime.factory.build = observed_notify, observed_build
    original = store._db
    store._db = profiled = Profiled(original)
    rig = type("Rig", (), {})()
    rig.agent, rig.worker, rig.traffic, rig.gate = agent, worker, traffic, gate
    rig.model, rig.timeline, rig.profiled = model, timeline, profiled
    rig.claims, rig.stale, rig.finals, rig.tails = claims, stale, finals, tails
    try:
        yield rig
    finally:
        gate.release.set()
        await agent.aclose()
        await traffic.aclose()
        store._db = original


def handoff_record(rig, scenario, elapsed_s, **extra):
    p = rig.profiled
    record = {
        "scenario": scenario, "db": p.dialect.name, "elapsed_s": elapsed_s,
        "total_sql": len(p.records), "transactions": p.transactions,
        "counts": _counts(p.records, p.tx_sites, p.tx_categories),
        "records": list(p.records), "tx_sites": list(p.tx_sites),
        "R": sum(batch is not None for batch in rig.claims),
        "M": len(rig.model.requests), "empty_claims": rig.claims.count(None),
        "recover_stale": len(rig.stale), "T1": len(rig.tails),
        "notifications": {channel: sum(p.channel == channel for p in rig.traffic.packets)
                          for channel in ("work", "activation", "output")},
        "timeline": list(rig.timeline), **extra,
    }
    if path := os.environ.get("C03_PROFILE_OUT"):
        with open(path, "a") as stream:
            stream.write(json.dumps(record) + "\n")
    return record


async def exercise_handoff(store, stage, action="duplicate", *, tool=False):
    async with handoff_rig(store, stage=stage, tool=tool) as rig:
        agent, worker, traffic, gate = rig.agent, rig.worker, rig.traffic, rig.gate
        # work starts the claim; the second original packet is delivered at the gate.
        traffic.hold = lambda p: p.channel == "activation" and p.message_id == "first"
        started = time.monotonic()
        first = await agent.runtime.client.send(
            agent.target, "first body", message_id="first", actor=agent.actor)
        await gate.wait()
        ctx = worker._active.get(agent.target.session_id)
        consumed = getattr(ctx, "consumed", None)
        snapshot = {"ctx_present": ctx is not None, "consumed_known": consumed is not None}
        await traffic.flush()
        refs = [first.input_ref]
        if action == "new":
            for mid in ("second", "third"):
                receipt = await agent.runtime.client.send(
                    agent.target, mid + " body", message_id=mid, actor=agent.actor)
                refs.append(receipt.input_ref)
                # queued accept emits both real packets; wait until listener handled them.
                for packet in tuple(traffic.packets):
                    if packet.message_id == mid and packet.channel in {"work", "activation"}:
                        await asyncio.wait_for(traffic.seen.setdefault(
                            id(packet), asyncio.Event()).wait(), 10)
        elif action in {"recover", "cancelled", "unknown", "duplicate-many"}:
            ref = None if action == "recover" else (
                InputRef(store.store_id, agent.target.session_id, "unknown")
                if action == "unknown" else first.input_ref)
            for _ in range(3):
                await traffic.deliver(TransportPacket(
                    PROTOCOL_VERSION, agent.runtime.namespace, "hint", agent.target, "work", ref,
                    {"cancelled": True} if action == "cancelled" else None))
        snapshot["wake_pending"] = getattr(ctx, "wake_pending", False)
        snapshot["pending_while_active"] = agent.target.session_id in worker._pending
        gate.release.set()
        await agent.drain(after_input=first.input_ref)
        # Include all worker tasks, including unsuccessful claims and stale-entry reads.
        statuses = [await worker.inputs.get_result(ref, actor=agent.actor) for ref in refs]
        return handoff_record(
            rig, f"{stage}/{action}/tool={tool}", time.monotonic() - started,
            snapshot=snapshot, results=[None if r is None else {
                "status": r.status, "send_index": r.send_index, "text": r.final_text,
            } for r in statuses],
            finals=[f.execution_state for f in rig.finals],
            user_texts=[[str(m.get("content")) for m in request.messages if m.get("role") == "user"]
                        for request in rig.model.requests],
        )


@pytest.mark.parametrize("stage", ("claim", "acquired", "factory", "model", "final_after", "cleanup"))
async def test_initial_second_packet_has_no_extra_claim_before_cleanup(registry_store, stage):
    record = await exercise_handoff(registry_store[0], stage)
    assert record["results"] == [{"status": "completed", "send_index": 1, "text": "done"}]
    # After pop even a known late packet is handled by the old pending/claim path.
    expected_empty = int(stage == "cleanup")
    assert (record["R"], record["empty_claims"], record["recover_stale"]) == (1, expected_empty, expected_empty)
    assert record["T1"] == 0


@pytest.mark.parametrize("stage", ("acquired", "factory", "model", "final_before", "final_after", "cleanup"))
async def test_real_new_inputs_finish_and_next_build_waits_for_close(registry_store, stage):
    record = await exercise_handoff(registry_store[0], stage, "new")
    assert len(record["results"]) == 3 and all(r["status"] == "completed" for r in record["results"])
    expected_runs = 1 if stage in {"acquired", "factory"} else 2
    assert record["R"] == expected_runs and record["empty_claims"] == 0
    assert record["T1"] == 0
    if stage in {"acquired", "factory"}:
        assert not record["snapshot"]["consumed_known"] and not record["snapshot"]["wake_pending"]
    if expected_runs == 2:
        assert record["timeline"].index("close-end:1") < record["timeline"].index("build:2")
    if stage == "final_after":
        assert record["snapshot"]["wake_pending"] and not record["snapshot"]["pending_while_active"]


async def test_boundary_extends_same_consumed_list_without_another_run(registry_store):
    record = await exercise_handoff(registry_store[0], "model", "new", tool=True)
    assert record["R"] == 1 and record["M"] == 2 and record["empty_claims"] == 0
    assert [r["send_index"] for r in record["results"]] == [1, 1, 1]
    assert all(r["status"] == "completed" for r in record["results"])
    assert "second body" in " ".join(record["user_texts"][1])
    assert "third body" in " ".join(record["user_texts"][1])


async def test_consumed_followup_packet_is_filtered_by_the_same_list(registry_store):
    store, _ = registry_store
    async with handoff_rig(store, tool=True) as rig:
        agent, worker = rig.agent, rig.worker
        second_model = Gate()
        complete = rig.model.complete

        async def gated_complete(*args, **kwargs):
            if len(rig.model.requests) == 1:
                await second_model.hold()
            return await complete(*args, **kwargs)

        rig.model.complete = gated_complete
        try:
            first = await agent.runtime.client.send(
                agent.target, "first", message_id="first", actor=agent.actor)
            await rig.gate.wait()
            ctx = worker._active[agent.target.session_id]
            consumed = ctx.consumed
            second = await agent.runtime.client.send(
                agent.target, "followup", message_id="second", actor=agent.actor)
            rig.gate.release.set()
            await second_model.wait()
            assert ctx.consumed is consumed and second.input_ref in consumed
            for channel in ("work", "activation"):
                await rig.traffic.deliver(TransportPacket(
                    PROTOCOL_VERSION, agent.runtime.namespace, "second", agent.target,
                    channel, second.input_ref))
            assert not ctx.wake_pending
            second_model.release.set()
            await agent.drain(after_input=first.input_ref)
            assert len(rig.claims) == 1 and rig.claims.count(None) == 0
            result = await worker.inputs.get_result(second.input_ref, actor=agent.actor)
            assert result.status == "completed" and result.send_index == 1
            handoff_record(rig, "consumed-followup-double-packet", 0)
        finally:
            second_model.release.set()


@pytest.mark.parametrize("action", ("recover", "cancelled", "duplicate-many", "unknown"))
async def test_active_hint_filter_and_full_empty_claim_cost(registry_store, action):
    record = await exercise_handoff(registry_store[0], "model", action)
    empty = int(action == "unknown")
    assert record["R"] == 1 and record["empty_claims"] == empty
    assert record["recover_stale"] == empty and record["T1"] == 0
    assert record["snapshot"]["wake_pending"] == bool(empty)
    assert not record["snapshot"]["pending_while_active"]
    assert record["results"][0]["status"] == "completed"


async def test_lost_work_after_final_needs_existing_recover_when_scan_disabled(registry_store):
    store, _ = registry_store
    async with handoff_rig(store, stage="final_after") as rig:
        agent, worker = rig.agent, rig.worker
        first = await agent.runtime.client.send(
            agent.target, "first", message_id="first", actor=agent.actor)
        await rig.gate.wait()
        rig.traffic.hold = lambda p: p.channel in {"work", "activation"}
        second = await agent.runtime.client.send(
            agent.target, "lost work", message_id="lost", actor=agent.actor)
        rig.gate.release.set()
        await agent.drain(after_input=first.input_ref)
        assert await worker.inputs.get_result(second.input_ref, actor=agent.actor) is None
        assert (await worker.inputs.inspect(agent.target, actor=agent.actor)).execution_state == "queued"
        assert len(rig.claims) == 1 and agent.runtime.recovery_interval_s is None
        await worker.recover()
        await agent.drain(after_input=second.input_ref)
        result = await worker.inputs.get_result(second.input_ref, actor=agent.actor)
        assert result.status == "completed" and result.send_index == 2
        assert rig.timeline.index("close-end:1") < rig.timeline.index("build:2")
        handoff_record(rig, "lost-work-explicit-recover", 0)


async def test_real_cancel_is_still_observed_at_boundary(registry_store):
    store, _ = registry_store
    async with handoff_rig(store) as rig:
        agent, worker = rig.agent, rig.worker
        receipt = await agent.runtime.client.send(
            agent.target, "go", message_id="cancel-me", actor=agent.actor)
        await rig.gate.wait()
        await agent.runtime.client.cancel(receipt.input_ref, actor=agent.actor, reason="stop now")
        for packet in rig.traffic.packets:
            if packet.channel in {"work", "activation"} and (packet.payload or {}).get("cancelled"):
                await asyncio.wait_for(rig.traffic.seen.setdefault(id(packet), asyncio.Event()).wait(), 10)
        assert not worker._active[agent.target.session_id].wake_pending
        rig.gate.release.set()
        await agent.drain(after_input=receipt.input_ref)
        result = await worker.inputs.get_result(receipt.input_ref, actor=agent.actor)
        assert result.status == "cancelled" and result.error == "stop now"
        assert rig.claims.count(None) == 0 and len(rig.claims) == 1
        assert rig.timeline[-1] == "close-end:1"
        handoff_record(rig, "cancel-active", 0)


async def test_shutdown_does_not_dispatch_a_pending_wake(registry_store):
    store, _ = registry_store
    async with handoff_rig(store, stage="final_after") as rig:
        agent, worker = rig.agent, rig.worker
        first = await agent.runtime.client.send(
            agent.target, "first", message_id="first", actor=agent.actor)
        await rig.gate.wait()
        second = await agent.runtime.client.send(
            agent.target, "next", message_id="next", actor=agent.actor)
        await asyncio.wait_for(rig.traffic.seen.setdefault(
            id(next(p for p in rig.traffic.packets if p.message_id == "next" and p.channel == "work")),
            asyncio.Event()).wait(), 10)
        closing = asyncio.Event()

        async def shutdown():
            closing.set()
            return await worker.shutdown(timeout_s=10)

        task = asyncio.create_task(shutdown())
        await closing.wait()
        assert worker._closing
        rig.gate.release.set()
        assert (await task).drained
        await agent.drain(after_input=first.input_ref)
        assert len(rig.claims) == 1 and not worker._pending
        assert (await worker.inputs.get_result(first.input_ref, actor=agent.actor)).status == "completed"
        assert await worker.inputs.get_result(second.input_ref, actor=agent.actor) is None
        assert rig.timeline[-1] == "close-end:1"
        handoff_record(rig, "shutdown-with-wake", 0)


async def test_other_node_loses_claim_and_late_idle_hint_includes_recovery_read(registry_store):
    store, connect = registry_store
    other = await connect(schema="verify")
    async with handoff_rig(store) as rig:
        agent = rig.agent
        peer = RunningAgent(other, ScriptedModel(), transport=rig.traffic)
        peer.runtime.node_id, peer.runtime.boot_id = "peer", "peer-boot"
        other_db = other._db
        other._db = profiled = Profiled(other_db)
        try:
            receipt = await agent.runtime.client.send(
                agent.target, "one owner", message_id="one", actor=agent.actor)
            await rig.gate.wait()
            # Two logical nodes, separate real DB connections. Enter below the local
            # session lock to model a peer process, which cannot share that Python lock.
            await peer.runtime.worker._run_session_owned(agent.target.session_id)
            first_cost = len(profiled.records)
            assert first_cost == 2 and not peer.executions
            rig.gate.release.set()
            await agent.drain(after_input=receipt.input_ref)
            result = await rig.worker.inputs.get_result(receipt.input_ref, actor=agent.actor)
            assert result.status == "completed" and result.send_index == 1
            await peer.runtime.worker._run_session_owned(agent.target.session_id)
            assert len(profiled.records) - first_cost == 2 and not peer.executions
            assert sum("agent.runtime_worker:_recover_stale" in r["chain"] for r in profiled.records) == 2
            handoff_record(rig, "peer-arbitration", 0, peer_records=profiled.records,
                           peer_transactions=profiled.transactions, peer_empty_claims=2,
                           peer_recover_stale=2)
        finally:
            await peer.aclose()
            other._db = other_db


async def test_discovery_states_limit_and_locked_recover(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    assert await port.discover_unfinished() == ()
    refs = {}
    for name in ("idle", "queued", "expired", "healthy"):
        refs[name] = ref = await target(port, name)
        if name == "idle":
            continue
        await port.accept(ref, name, message_id="m", actor=ACTOR)
        if name != "queued":
            await port.claim(ref, node_id="old", boot_id="old", ttl_s=300)
        if name == "expired":
            await store._db.execute(
                f"UPDATE {port.s} SET owner_lease_until_ms=1 WHERE session_id=?", (name,))
    assert await port.discover_unfinished() == tuple(refs[k] for k in ("queued", "expired"))
    assert await port.discover_unfinished(limit=1) == (refs["queued"],)
    assert len(await port.discover_unfinished(limit=1000)) == 2
    from power_loop.agent.runtime_types import RuntimeRequestError
    for limit in (0, 1001):
        with pytest.raises(RuntimeRequestError, match="limit"):
            await port.discover_unfinished(limit=limit)
    # Discovery does not authorize a takeover: recover checks the locked CURRENT expiry.
    await store._db.execute(
        f"UPDATE {port.s} SET owner_lease_until_ms=(SELECT {port._now_sql()})+300000 "
        "WHERE session_id=?", (refs["expired"].session_id,))
    assert await port.recover(refs["expired"], node_id="new", boot_id="new") is None
    assert await port.recover(refs["healthy"], node_id="new", boot_id="new") is None


async def _seed_discovery_states(store, rows):
    for offset in range(0, len(rows), 200):
        params = []
        batch = rows[offset:offset + 200]
        for sid, state, lease in batch:
            owner = ("node", "boot") if lease is not None else (None, None)
            wake_cnt = 1 if state == "queued" else 0
            params.extend((sid, sid, 0, "agent", "t", "p", "{}", "h" * 64, "c" * 64, "linked", state, wake_cnt, *owner, lease, 1, 1))
        await store._db.execute(
            f"INSERT INTO {store.t.sessions} (session_id,root_session_id,depth,session_kind,"
            "tenant_id,principal_id,descriptor_json,descriptor_hash,creation_identity_hash,"
            "cancel_propagation,execution_state,wake_pending_count,owner_node_id,owner_boot_id,"
            "owner_lease_until_ms,created_at,updated_at) VALUES "
            + ",".join("(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)" for _ in batch), params)


@pytest.mark.parametrize("candidates", ("mixed", "queued", "running"))
async def test_discovery_keeps_original_order_and_limits(registry_store, candidates):
    store, _ = registry_store
    db, port = store._db, RuntimeInputStore(store)
    assert await port.discover_unfinished() == ()
    rows = [("idle", "idle", None), ("healthy", "running", 9999999999999)]
    for state in ("queued", "running"):
        count = 6 if candidates == "mixed" else (1105 if state == candidates else 0)
        for i in range(count):
            lease = None if state == "queued" else 1 + i % 2
            rows.append((f"{state}-{count - i:04d}", state, lease))
    await _seed_discovery_states(store, rows)
    database_now = await port._db_now(db)
    for limit in (1, 10, 1000):
        expected = await db.fetchall(
            f"SELECT session_id FROM {port.s} WHERE execution_state='queued' "
            "OR (execution_state='running' AND owner_lease_until_ms<=?) "
            "ORDER BY execution_state,owner_lease_until_ms,session_id LIMIT ?",
            (database_now, limit))
        store._db = profiled = Profiled(db)
        try:
            actual = await port.discover_unfinished(limit=limit)
        finally:
            store._db = db
        assert [ref.session_id for ref in actual] == [row["session_id"] for row in expected]
        assert all(ref.store_id == store.store_id for ref in actual)
        assert len(profiled.records) == 1 and profiled.transactions == 0
        assert len(actual) == min(limit, len(rows) - 2)


async def test_discovery_sqlite_work_does_not_grow_with_unrelated_idle(tmp_path):
    from power_loop import SessionStore

    store = await SessionStore.open(str(tmp_path / "discovery-growth.db"))
    try:
        port = RuntimeInputStore(store)
        await _seed_discovery_states(store, [
            ("queued", "queued", None),
            ("expired", "running", 1),
            ("healthy", "running", 9999999999999),
        ])
        counts = []
        for low, high in ((0, 2000), (2000, 20000)):
            await _seed_discovery_states(store, [(f"idle-{i:05d}", "idle", None)
                                                  for i in range(low, high)])
            await store._db.execute(f"ANALYZE {port.s}")
            steps = [0]
            store._db._conn.set_progress_handler(
                lambda: steps.__setitem__(0, steps[0] + 1) or 0, 1)
            try:
                result = await port.discover_unfinished(limit=10)
            finally:
                store._db._conn.set_progress_handler(None, 0)
            assert {ref.session_id for ref in result} == {"queued", "expired"}
            counts.append(steps[0])
        assert counts[0] > 0 and counts[1] <= 2 * counts[0], counts
    finally:
        await store.close()


@pytest.mark.parametrize("mode", ("command", "factory-failed", "owner-lost", "final-error"))
async def test_paths_without_normal_final_keep_old_tail(registry_store, monkeypatch, mode):
    store, _ = registry_store
    async with handoff_rig(store, stage="none", command=mode == "command") as rig:
        worker, agent = rig.worker, rig.agent
        if mode == "factory-failed":
            failed = asyncio.Event()

            async def broken(*args):
                failed.set()
                raise RuntimeError("factory unavailable")
            monkeypatch.setattr(agent.runtime.factory, "build", broken)
        elif mode in {"owner-lost", "final-error"}:
            real_final = worker.inputs.final
            hit = False

            async def broken_final(*args, **kwargs):
                nonlocal hit
                if not hit:
                    hit = True
                    if mode == "owner-lost":
                        raise SessionLeaseLost(agent.target.session_id)
                    raise RuntimeError("injected before final")
                return await real_final(*args, **kwargs)

            monkeypatch.setattr(worker.inputs, "final", broken_final)
        receipt = await agent.runtime.client.send(
            agent.target, "go", message_id="one", actor=agent.actor)
        if mode == "factory-failed":
            await asyncio.wait_for(failed.wait(), 10)
            await agent.drain()
        else:
            await agent.drain(after_input=receipt.input_ref)
        result = await worker.inputs.get_result(receipt.input_ref, actor=agent.actor)
        assert len(rig.tails) == int(mode != "factory-failed")
        if mode == "command":
            assert result.status == "completed" and not rig.model.requests
        elif mode == "final-error":
            assert result.status == "failed" and result.error_code == "internal_error"
        else:
            assert result is None
        handoff_record(rig, mode, 0)
