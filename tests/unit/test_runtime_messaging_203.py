"""P1 input and transport contracts, including the real three database backends."""
import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    RequestOptions,
    Runtime,
    RuntimeClient,
    StoreError,
    WaitTimeout,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_transport import (
    DatabasePollingTransport,
    MemoryMessageTransport,
    SubscriptionSpec,
    TransportPacket,
)
from power_loop.agent.runtime_types import (
    Command,
    InputRef,
    JobRef,
    Result,
    RuntimeDescriptor,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    plain,
)
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def fixture(store):
    descriptor = RuntimeDescriptor(1, "agent", "test.agent@1", None, "builtin.empty@1",
                                   "builtin.none@1", "builtin.none@1", {"max_rounds": 2}, {})
    actor = TrustedActor("tenant", "principal", None)
    sid = await store.create_session(metadata={"tenant_id": actor.tenant_id,
        "principal_id": actor.principal_id, "runtime_descriptor": plain(descriptor)})
    return RuntimeInputStore(store), TargetRef(store.store_id, sid), actor


def command(target, mid, body="hello", **kwargs):
    return Command(mid, target, "send", {"body": body}, "new_turn", **kwargs)


async def accept(port, cmd, actor):
    async with port.db.transaction() as tx:
        return await port.accept_input(tx, cmd, actor)


async def owner(port, target):
    coordinator = MemorySessionOwnershipCoordinator()
    lease = await coordinator.acquire((target.store_id, target.session_id), owner_id="node/boot",
                                      ttl_s=90, on_lost=lambda: None)
    await port.db.execute(f"UPDATE {port.state} SET owner_epoch=1,owner_token=?,owner_node_id='node',owner_boot_id='boot' "
                          "WHERE session_id=?", (lease.token, target.session_id))
    return SimpleNamespace(epoch=1, lease=lease)


async def test_request_snapshot_retry_and_authorization(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    options = {"metadata": {"values": [1]}}
    cmd = command(target, "input", options=options)
    options["metadata"]["values"].append(2)
    first = await accept(port, cmd, actor)
    assert first.newly_accepted
    await store._db.execute(f"UPDATE {store.t.sessions} SET metadata_json=? WHERE session_id=?",
        ('{"tenant_id":"tenant","principal_id":"principal","runtime_descriptor":{}}', target.session_id))
    second = await accept(port, cmd, actor)
    assert not second.newly_accepted
    record = await port.input(port.db, first.input_ref)
    assert record.meta["options"]["max_rounds"] == 2
    assert record.meta["options"]["metadata"] == {"values": [1]}
    with pytest.raises(RuntimeRequestError, match="duplicate_input_conflict"):
        await accept(port, command(target, "input", "different"), actor)
    with pytest.raises(RuntimeRequestError, match="forbidden"):
        await accept(port, cmd, TrustedActor("tenant", "someone-else", None))
    assert (await store.inbox_counts(target.session_id))["pending"] == 1


async def test_claim_is_bounded_and_does_not_reclaim_by_age(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    for n in range(205):
        await accept(port, command(target, f"input-{n}"), actor)
    ownership = await owner(port, target)
    async with port.db.transaction() as tx:
        first = await port.claim_inputs(tx, target, ownership)
    assert len(first) == 100
    await port.db.execute(f"UPDATE {port.q} SET claimed_at=1 WHERE status='claimed'")
    async with port.db.transaction() as tx:
        second = await port.claim_inputs(tx, target, ownership)
    async with port.db.transaction() as tx:
        last = await port.claim_inputs(tx, target, ownership)
    assert (len(second), len(last)) == (100, 5)
    assert len({r.row["id"] for r in (*first, *second, *last)}) == 205
    assert all(r.row["claim_owner_epoch"] == 1 for r in first)


async def test_cancel_before_binding_and_binding_before_cancel(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    ownership = await owner(port, target)
    one = await accept(port, command(target, "one"), actor)
    two = await accept(port, command(target, "two"), actor)
    async with port.db.transaction() as tx:
        claims = await port.claim_inputs(tx, target, ownership)
        await port.bind_input_job(tx, one.input_ref, JobRef(target.store_id, "job-one", 1), ownership, claim_token=claims[0].row["claim_token"])
    async with port.db.transaction() as tx:
        await port.db.dialect.lock_state(tx, port.state, target.session_id)
        cancelled = await port.void_unbound(tx, two.input_ref, reason="cancelled")
        already_started = await port.void_unbound(tx, one.input_ref, reason="cancelled")
    assert cancelled.state == "void" and cancelled.job_ref is None
    assert already_started.job_ref == JobRef(target.store_id, "job-one", 1)
    with pytest.raises(RuntimeRequestError, match="input_not_startable"):
        async with port.db.transaction() as tx:
            await port.bind_input_job(tx, two.input_ref, JobRef(target.store_id, "job-two", 1), ownership, claim_token=claims[1].row["claim_token"])
    assert (await port.input(port.db, one.input_ref)).row["status"] == "claimed"


async def test_start_transaction_rollback_preserves_input(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    receipt = await accept(port, command(target, "request"), actor)
    ownership = await owner(port, target)
    with pytest.raises(RuntimeError, match="rollback"):
        async with port.db.transaction() as tx:
            claims = await port.claim_inputs(tx, target, ownership, limit=1)
            await port.bind_input_job(tx, receipt.input_ref, JobRef(target.store_id, "job", 1), ownership, claim_token=claims[0].row["claim_token"])
            raise RuntimeError("rollback")
    row = (await port.input(port.db, receipt.input_ref)).row
    assert row["status"] == "pending" and row["delivered_job_id"] is None


@pytest.mark.parametrize("transport_type", [MemoryMessageTransport, DatabasePollingTransport])
async def test_ready_control_ack_and_close(transport_type):
    target = TargetRef("store", "session")
    packet = TransportPacket(42, "test", "input", target, "control", InputRef("store", "session", "input"))
    class Source:
        items = []
        async def poll_notifications(self, binding, *, cursor, limit):
            items, self.items = self.items, []
            return items[:limit], None
    source = Source()
    transport = (transport_type() if transport_type is MemoryMessageTransport else
                 transport_type(source, interval_s=0.01, control_interval_s=0.01))
    subscription = await transport.subscribe(SubscriptionSpec("test", "store", "control", "node"))
    await subscription.ready()
    source.items.append(packet)
    await transport.publish(packet)
    delivery = await asyncio.wait_for(anext(subscription), 1)
    assert delivery.packet == packet
    await delivery.ack()
    await subscription.aclose()
    assert not transport._subscriptions
    await transport.aclose()


class GatedModel(LLMService):
    def __init__(self):
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.requests = []

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return LLMResponse(raw_text=f"result-{len(self.requests)}")

    def stream(self, request):
        raise NotImplementedError

    async def close(self):
        pass


def deadline():
    return int(time.time() * 1000) + 15_000


@asynccontextmanager
async def running(store, mode):
    model = GatedModel()
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(model, ToolRegistry(), AgentLoopConfig(compactor=None, max_rounds=2))
    factory.register("test.agent@1", build)
    transport = (MemoryMessageTransport() if mode == "memory" else DatabasePollingTransport(
        RuntimeInputStore(store), interval_s=0.01, control_interval_s=0.01))
    runtime = Runtime(store=store, factory=factory, node_id="worker", boot_id="boot", transport=transport,
                      capacity=1, recovery_interval_s=None)
    await runtime.worker.start()
    actor = TrustedActor("tenant", "principal", None)
    target = await runtime.open_session(RuntimeDescriptor(1, "agent", "test.agent@1", None,
        "builtin.empty@1", "builtin.none@1", "builtin.none@1", {"max_rounds": 2}, {}), actor=actor)
    try:
        yield runtime, model, target, actor
    finally:
        model.release.set()
        await runtime.worker.shutdown(timeout_s=2, cancel_running=True)
        await runtime.client.aclose()
        await transport.aclose()


@pytest.mark.parametrize("mode", ["memory", "polling"])
async def test_two_independent_results_and_frozen_options(registry_store, mode):
    store, _ = registry_store
    async with running(store, mode) as (runtime, model, target, actor):
        first = await runtime.client.send(target, "first", message_id="first", actor=actor,
                                          options=RequestOptions(temperature=0.2), deadline_ms=deadline())
        await asyncio.wait_for(model.started.wait(), 3)
        second = await runtime.client.send(target, "second", message_id="second", actor=actor,
                                           options=RequestOptions(temperature=0.8), deadline_ms=deadline())
        assert (await runtime.client.inspect(second.input_ref, actor=actor)).job_ref is None
        model.release.set()
        result1 = await runtime.client.wait(first.input_ref, actor=actor, deadline_ms=deadline())
        result2 = await runtime.client.wait(second.input_ref, actor=actor, deadline_ms=deadline())
        assert (result1.final_text, result2.final_text) == ("result-1", "result-2")
        assert result1.job_ref != result2.job_ref
        assert [request.temperature for request in model.requests] == [0.2, 0.8]
        detached = RuntimeClient(store=store, transport=runtime.transport)
        assert (await detached.wait(first.input_ref, actor=actor, deadline_ms=deadline())).job_ref == result1.job_ref
        await detached.aclose()


@pytest.mark.parametrize("mode", ["memory", "polling"])
async def test_queued_cancel_and_control_while_model_hangs(registry_store, mode):
    store, _ = registry_store
    async with running(store, mode) as (runtime, model, target, actor):
        first = await runtime.client.send(target, "first", message_id="first", actor=actor)
        await asyncio.wait_for(model.started.wait(), 3)
        second = await runtime.client.send(target, "second", message_id="second", actor=actor)
        output = await runtime.client.subscribe_output(second.input_ref, actor=actor)
        assert (await anext(output)).kind == 'gap'
        waiting = asyncio.create_task(anext(output))
        await asyncio.sleep(0)
        await runtime.client.cancel(second.input_ref, actor=actor)
        completed = await asyncio.wait_for(waiting, 3)
        assert completed.kind == 'completed' and completed.payload['status'] == 'cancelled'
        await output.aclose()
        void = await runtime.client.wait(second.input_ref, actor=actor, deadline_ms=deadline())
        assert void.status == "cancelled" and void.job_ref is None
        assert (await runtime.client.inspect(first.input_ref, actor=actor)).state == "running"
        await runtime.client.cancel(first.input_ref, actor=actor)
        result = await asyncio.wait_for(runtime.client.wait(first.input_ref, actor=actor, deadline_ms=deadline()), 3)
        assert result.status == "cancelled"
        assert len(model.requests) == 1


async def test_cancel_command_validates_target_and_preserves_original_reason(registry_store):
    store, _ = registry_store
    async with running(store, "memory") as (runtime, model, target, actor):
        client = runtime.client
        first = await client.send(target, "running", message_id="running", actor=actor)
        await asyncio.wait_for(model.started.wait(), 3)
        job = (await client.inspect(first.input_ref, actor=actor)).job_ref
        second = await client.send(target, "queued", message_id="queued", actor=actor)
        queued_cancel = Command("cancel-queued", target, "cancel",
            {"input_ref": plain(second.input_ref), "reason": "withdrawn"}, "control")
        with pytest.raises(RuntimeRequestError, match="forbidden"):
            await client.submit(queued_cancel, actor=TrustedActor("tenant", "intruder", None))
        with pytest.raises(RuntimeRequestError, match="invalid_options"):
            await client.submit(Command("bad-input", TargetRef(target.store_id, "another-session"), "cancel",
                {"input_ref": plain(second.input_ref), "reason": "bad"}, "control"), actor=actor)
        _, other, _ = await fixture(store)
        with pytest.raises(RuntimeRequestError, match="forbidden"):
            await client.submit(Command("bad-job", TargetRef(target.store_id, other.session_id, job.job_id),
                "cancel", {"reason": "bad"}, "control", expected_job_generation=job.generation), actor=actor)
        assert (await client.inspect(first.input_ref, actor=actor)).state == "running"
        assert (await client.inspect(second.input_ref, actor=actor)).state == "accepted"
        receipt = await client.submit(queued_cancel, actor=actor)
        assert receipt.message_id == "queued"
        assert (await client.wait(second.input_ref, actor=actor, deadline_ms=deadline())).error == "withdrawn"
        exact = TargetRef(target.store_id, target.session_id, job.job_id)
        for message_id, reason in (("cancel-job", "first reason"), ("retry-job", "new reason")):
            receipt = await client.submit(Command(message_id, exact, "cancel", {"reason": reason}, "control",
                expected_job_generation=job.generation), actor=actor)
            assert receipt.reason == "first reason"
            assert receipt.message_id == f"cancel:{job.job_id}:{job.generation}"
        assert (await client.wait(first.input_ref, actor=actor, deadline_ms=deadline())).status == "cancelled"
        assert len(model.requests) == 1


async def test_wait_timeout_client_close_and_output_authorization(registry_store):
    store, _ = registry_store
    async with running(store, "memory") as (runtime, model, target, actor):
        client = RuntimeClient(store=store, transport=runtime.transport)
        receipt = await client.send(target, "first", message_id="first", actor=actor)
        await asyncio.wait_for(model.started.wait(), 3)
        with pytest.raises(WaitTimeout):
            await client.wait(receipt.input_ref, actor=actor, deadline_ms=int(time.time()*1000)+20)
        assert not client._waiters
        intruder = TrustedActor("tenant", "intruder", None)
        for operation in (client.inspect, client.subscribe_output):
            with pytest.raises(RuntimeRequestError, match="forbidden"):
                await operation(receipt.input_ref, actor=intruder)
        await client.aclose()
        model.release.set()
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())
        assert result.status == "completed"
        output = await runtime.client.subscribe_output(receipt.input_ref, actor=actor)
        assert (await anext(output)).kind == "completed"
        assert not runtime.client._outputs


@pytest.mark.parametrize("mode", ["memory", "polling"])
async def test_append_joins_active_job_but_new_turn_waits(registry_store, mode):
    store, _ = registry_store
    async with running(store, mode) as (runtime, model, target, actor):
        first = await runtime.client.send(target, "first", message_id="first", actor=actor)
        await asyncio.wait_for(model.started.wait(), 3)
        second = await runtime.client.send(target, "second", message_id="second", actor=actor)
        appended = await runtime.client.append(target, "append", message_id="append", actor=actor)
        # Make delivery to the active execution deterministic before releasing the model.
        ctx = runtime.worker._active[target.session_id]
        for _ in range(100):
            if appended.input_ref in ctx.append_refs:
                break
            await asyncio.sleep(0.01)
        assert appended.input_ref in ctx.append_refs
        model.release.set()
        original = await runtime.client.wait(first.input_ref, actor=actor, deadline_ms=deadline())
        joined = await runtime.client.wait(appended.input_ref, actor=actor, deadline_ms=deadline())
        separate = await runtime.client.wait(second.input_ref, actor=actor, deadline_ms=deadline())
        assert joined.job_ref == original.job_ref
        assert separate.job_ref != original.job_ref
        assert joined.input_ref == appended.input_ref
        cancel = await runtime.client.cancel(appended.input_ref, actor=actor)
        assert cancel.reason == "already_applied"


async def test_request_deadline_cancels_running_job(registry_store):
    store, _ = registry_store
    async with running(store, "memory") as (runtime, model, target, actor):
        receipt = await runtime.client.send(target, "expire", message_id="expire", actor=actor,
                                            deadline_ms=int(time.time()*1000)+1500)
        await asyncio.wait_for(model.started.wait(), 3)
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())
        assert result.status == "cancelled"
        assert len(model.requests) == 1


async def test_unstarted_job_cancel_preserves_zero_generation_result(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    receipt = await accept(port, command(target, "start"), actor)
    manager = JobManager(store)
    created = await manager.register_job(JobActor(actor.tenant_id, actor.principal_id, target.session_id),
        kind="send", session_id=target.session_id, system_key="queued", request={"input_ref": plain(receipt.input_ref)})
    zero = JobRef(store.store_id, created.job.job_id, 0)
    await store._db.execute(f"UPDATE {port.q} SET delivered_job_id=?,delivered_job_generation=0 WHERE session_id=? AND item_id=?",
                            (zero.job_id, target.session_id, receipt.input_ref.message_id))
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    assert (await client.inspect(zero, actor=actor)).state == "accepted"
    await client.cancel(zero, actor=actor)
    result = await client.wait(zero, actor=actor, deadline_ms=deadline())
    assert result.job_ref == zero and result.input_ref is None and result.status == "cancelled"
    by_input = await client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())
    assert by_input.job_ref == zero and by_input.input_ref == receipt.input_ref
    count = await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {port.events} WHERE job_id=? AND event_type='execution'", (zero.job_id,))
    assert count["n"] == 0
    with pytest.raises(RuntimeRequestError, match="stale_generation"):
        await client.inspect(JobRef(store.store_id, zero.job_id, 1), actor=actor)
    await client.aclose()


async def test_first_result_and_input_do_not_follow_resume_generation(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    receipt = await accept(port, command(target, "start"), actor)
    manager = JobManager(store)
    created = await manager.register_job(JobActor(actor.tenant_id, actor.principal_id, target.session_id),
        kind="send", session_id=target.session_id, system_key="queued", request={"input_ref": plain(receipt.input_ref)})
    zero = JobRef(store.store_id, created.job.job_id, 0)
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    ownership = await owner(port, target)
    # The waiter resolves None while queued, before the first claim.
    waiter = asyncio.create_task(client.wait(JobRef(store.store_id, zero.job_id), actor=actor, deadline_ms=deadline()))
    await asyncio.sleep(0.05)
    for generation in (1, 2):
        actual = JobRef(store.store_id, zero.job_id, generation)
        async with manager.joined_write() as tx:
            await port.db.dialect.lock_state(tx, port.state, target.session_id)
            await tx.write_job(zero.job_id,
                f"UPDATE {port.jobs} SET generation=?,version=version+1,status='running',supervision_status='open' WHERE job_id=?",
                (generation, zero.job_id))
            if generation == 1:
                claims = await port.claim_inputs(tx, target, ownership, 1)
                await port.bind_input_job(tx, receipt.input_ref, actual, ownership, claim_token=claims[0].row["claim_token"])
            await port.commit_result_and_parent_ref(tx, actual,
                Result(actual, receipt.input_ref, "interrupted" if generation == 1 else "completed", f"g{generation}"), ownership)
    assert (await waiter).final_text == "g1"
    assert (await client.wait(zero, actor=actor, deadline_ms=deadline())).final_text == "g1"
    assert (await client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())).final_text == "g1"
    assert (await client.wait(JobRef(store.store_id, zero.job_id), actor=actor, deadline_ms=deadline())).final_text == "g2"
    await client.aclose()


@pytest.mark.parametrize("offset", [-1, 1])
async def test_current_schema_reopens_and_refuses_old_versions_without_conversion(registry_store, offset):
    from power_loop.runtime.store.schema import StoreSchemaError
    from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION, TASK_SCHEMA_VERSION
    store, connect = registry_store
    reopened = await connect()
    assert reopened.store_id == store.store_id
    sid = await store.create_session()
    await store.append_message(sid, role="user", content="unchanged")
    for table, column, original in (("schema_migrations", "version", TASK_SCHEMA_VERSION),
                                    ("task_store", "protocol_version", TASK_PROTOCOL_VERSION)):
        await store._db.execute(f"UPDATE {store.table_prefix}{table} SET {column}=? WHERE id=1", (original + offset,))
        try:
            with pytest.raises(StoreSchemaError):
                await connect()
            assert (await store.load_all_messages(sid))[0].content == "unchanged"
        finally:
            await store._db.execute(f"UPDATE {store.table_prefix}{table} SET {column}=? WHERE id=1", (original,))


async def test_shared_deployment_rejects_process_only_adapters(registry_store):
    store, _ = registry_store
    with pytest.raises(RuntimeRequestError, match="shared adapters"):
        Runtime(store=store, factory=RegistryRuntimeFactory(), node_id="worker", boot_id="boot", shared_execution=True)


async def test_ready_window_recovers_input_without_periodic_scan(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    model, factory, transport = GatedModel(), RegistryRuntimeFactory(), MemoryMessageTransport()
    model.release.set()
    async def build(descriptor, context):
        return ExecutionEnvironment(model, ToolRegistry(), AgentLoopConfig(compactor=None))
    factory.register('test.agent@1', build)
    runtime = Runtime(store=store, factory=factory, transport=transport, node_id='n', boot_id='b',
                      capacity=1, recovery_interval_s=None)
    ready, release = asyncio.Event(), asyncio.Event()
    original = transport.subscribe
    async def subscribe(binding):
        sub = await original(binding)
        if binding.channel == 'activation':
            original_ready = sub.ready
            async def held_ready():
                ready.set()
                await release.wait()
                await original_ready()
            sub.ready = held_ready
        return sub
    transport.subscribe = subscribe
    start = asyncio.create_task(runtime.worker.start())
    try:
        await asyncio.wait_for(ready.wait(), 2)
        receipt = await RuntimeClient(store=store, transport=transport).send(
            target, 'during ready', message_id='ready-window', actor=actor)
        release.set()
        await start
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())
        assert result.final_text == 'result-1'
    finally:
        release.set()
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await transport.aclose()


async def test_idle_handoff_and_reconnect_with_recovery_disabled(registry_store, monkeypatch):
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, target, actor):
        model.release.set()
        held, release = asyncio.Event(), asyncio.Event()
        original = runtime.worker.inputs.has_pending
        calls = 0
        async def pending(sid):
            nonlocal calls
            calls += 1
            if calls == 1:
                held.set()
                await release.wait()
            return await original(sid)
        monkeypatch.setattr(runtime.worker.inputs, 'has_pending', pending)
        first = await runtime.client.send(target, 'one', message_id='one', actor=actor)
        await asyncio.wait_for(held.wait(), 5)
        second = await runtime.client.send(target, 'two', message_id='two', actor=actor)
        release.set()
        assert (await runtime.client.wait(first.input_ref, actor=actor, deadline_ms=deadline())).status == 'completed'
        assert (await runtime.client.wait(second.input_ref, actor=actor, deadline_ms=deadline())).status == 'completed'
        # Persist while both ordinary subscribers are disconnected; ready +
        # bounded recovery must discover it even with the periodic scan disabled.
        await runtime.worker._subscriptions[0].aclose()
        await runtime.worker._subscriptions[1].aclose()
        third = await runtime.client.send(target, 'three', message_id='three', actor=actor)
        assert (await runtime.client.wait(third.input_ref, actor=actor, deadline_ms=deadline())).status == 'completed'
        assert len(model.requests) == 3


async def test_stream_final_result_and_reference_cleanup(registry_store):
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, target, actor):
        receipt = await runtime.client.send(target, 'stream', message_id='stream', actor=actor)
        await model.started.wait()
        stream = await runtime.client.subscribe_output(receipt.input_ref, actor=actor)
        assert (await anext(stream)).kind == 'gap'
        ctx = runtime.worker._active[target.session_id]
        await runtime.transport.publish(TransportPacket(42, runtime.namespace, 'stream', target,
            'output', receipt.input_ref, {'kind': 'delta', 'text': 'token', 'job_ref': plain(ctx.job_ref)}))
        assert (await anext(stream)).payload['text'] == 'token'
        model.release.set()
        item = await asyncio.wait_for(anext(stream), 5)
        assert item.kind == 'completed' and item.job_ref == ctx.job_ref
        assert not runtime.client._outputs and not runtime.client._waiters and runtime.client._slots == 0
        assert len(runtime.transport._subscriptions) == 3
        completed = await runtime.client.subscribe_output(receipt.input_ref, actor=actor)
        assert (await anext(completed)).kind == 'completed'
        await runtime.client.aclose()
        assert not runtime.client._outputs and runtime.client._slots == 0


async def test_hot_backlog_progresses_and_other_session_is_not_starved(registry_store):
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, target, actor):
        first = await runtime.client.send(target, 'first', message_id='first', actor=actor)
        await model.started.wait()
        receipts = [await runtime.client.send(target, f'queued {i}', message_id=f'q{i}', actor=actor)
                    for i in range(105)]
        _, cold, _ = await fixture(store)
        other = await runtime.client.send(cold, 'cold', message_id='cold', actor=actor)
        batches = []
        original = runtime.worker.inputs.claim_inputs
        async def claim(*args, **kwargs):
            rows = await original(*args, **kwargs)
            batches.append(len(rows))
            return rows
        runtime.worker.inputs.claim_inputs = claim
        model.release.set()
        assert (await runtime.client.wait(other.input_ref, actor=actor, deadline_ms=deadline())).status == 'completed'
        assert len(model.requests) < 106
        assert (await runtime.client.wait(receipts[-1].input_ref, actor=actor,
                                         deadline_ms=int(time.time()*1000)+60000)).status == 'completed'
        assert len(model.requests) == 107 and max(batches) == 1
        assert (await runtime.client.wait(first.input_ref, actor=actor, deadline_ms=deadline())).status == 'completed'
        assert (await store.inbox_counts(target.session_id))['pending'] == 0


async def test_definition_failure_before_first_execution_keeps_generation_zero(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    receipt = await accept(port, command(target, 'start'), actor)
    manager = JobManager(store)
    created = await manager.register_job(JobActor(actor.tenant_id, actor.principal_id, target.session_id),
        kind='send', session_id=target.session_id, system_key='queued', request={'input_ref': plain(receipt.input_ref)})
    zero = JobRef(store.store_id, created.job.job_id, 0)
    async with manager.joined_write() as tx:
        await port.db.dialect.lock_state(tx, port.state, target.session_id)
        await tx.execute(f"UPDATE {port.q} SET delivered_job_id=?,delivered_job_generation=0,status='void' "
                         'WHERE session_id=? AND item_id=?', (zero.job_id, target.session_id, 'start'))
        await tx.write_job(zero.job_id, f"UPDATE {port.jobs} SET status='failed',outcome='failed',resource_state='stopped',"
            "supervision_status='closed',version=version+1 WHERE job_id=? AND generation=0", (zero.job_id,))
        await port.commit_result_and_parent_ref(tx, zero,
            Result(zero, receipt.input_ref, 'failed', error='definition_unavailable'), None)
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    for ref in (zero, JobRef(store.store_id, zero.job_id), receipt.input_ref):
        status = await client.inspect(ref, actor=actor)
        result = await client.wait(ref, actor=actor, deadline_ms=deadline())
        assert status.state == 'failed' and status.result_available and status.resource_state == 'stopped'
        assert result.job_ref == zero and result.error == 'definition_unavailable'
        assert result.final_text == '' and not result.usage and result.structured_output is None
    await client.aclose()


async def test_notifications_fail_after_commit_and_original_id_retries(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    class FailedTransport(MemoryMessageTransport):
        async def publish(self, packet):
            raise ConnectionError('test notification outage')
    client = RuntimeClient(store=store, transport=FailedTransport())
    first = await client.send(target, 'saved', message_id='stable-id', actor=actor)
    again = await client.send(target, 'saved', message_id='stable-id', actor=actor)
    assert first.notification == again.notification == 'failed'
    assert first.state == again.state == 'accepted' and first.reason == 'wake_pending'
    assert not again.newly_accepted and first.input_ref == again.input_ref
    assert (await store.inbox_counts(target.session_id))['pending'] == 1
    await client.aclose()


async def test_model_callbacks_stream_through_worker_and_overflow_is_bounded(registry_store):
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, target, actor):
        entered, emit, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def complete(request, **callbacks):
            entered.set()
            await emit.wait()
            for i in range(600):
                callbacks['on_chunk_delta_text'](f'token-{i}')
            await finish.wait()
            return LLMResponse(raw_text='final')
        model.complete = complete
        receipt = await runtime.client.send(target, 'stream', message_id='stream-overflow', actor=actor)
        await asyncio.wait_for(entered.wait(), 5)
        stream = await runtime.client.subscribe_output(receipt.input_ref, actor=actor)
        try:
            assert (await anext(stream)).kind == 'gap'
            emit.set()
            assert (await asyncio.wait_for(anext(stream), 5)).kind == 'gap'
            ctx = runtime.worker._active[target.session_id]
            assert ctx.output.qsize() <= 256 and stream.subscription.queue.qsize() <= 256
            item = await asyncio.wait_for(anext(stream), 5)
            assert item.kind == 'delta' and item.payload['text'].startswith('token-')
            finish.set()
            async with asyncio.timeout(5):
                async for item in stream:
                    if item.kind == 'completed':
                        assert item.payload['final_text'] == 'final'
                        break
            assert stream.closed and runtime.client._slots == 0
        finally:
            emit.set()
            finish.set()
            await stream.aclose()


async def test_tool_identity_reads_only_self_and_logical_descendants(registry_store):
    from power_loop.runtime.jobs import JobError

    from .test_job_registry_192 import call
    store, _ = registry_store
    port, root, host = await fixture(store)
    metadata = (await store.get_session(root.session_id)).metadata
    child_sid = await store.create_session(parent_session_id=root.session_id, metadata=metadata)
    sibling_sid = await store.create_session(parent_session_id=root.session_id, metadata=metadata)
    child = TargetRef(store.store_id, child_sid)
    root_input = await accept(port, command(root, 'root'), host)
    child_input = await accept(port, command(child, 'child'), host)
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    actors = []
    for sid in (root.session_id, child_sid, sibling_sid):
        ref = await call(store, JobActor(host.tenant_id, host.principal_id, sid))
        actors.append(TrustedActor(host.tenant_id, host.principal_id, sid, ref, 'call-1'))
    assert (await client.inspect(child_input.input_ref, actor=actors[0])).state == 'accepted'
    assert (await client.inspect(child_input.input_ref, actor=actors[1])).state == 'accepted'
    for target_ref, actor in ((root_input.input_ref, actors[1]), (child_input.input_ref, actors[2])):
        with pytest.raises(RuntimeRequestError, match='forbidden'):
            await client.inspect(target_ref, actor=actor)
        with pytest.raises(RuntimeRequestError, match='forbidden'):
            await client.subscribe_output(target_ref, actor=actor)
        with pytest.raises(RuntimeRequestError, match='forbidden'):
            await client.wait(target_ref, actor=actor, deadline_ms=deadline())
    forged = TrustedActor(host.tenant_id, host.principal_id, root.session_id, root_input.input_ref, 'call-1')
    with pytest.raises((RuntimeRequestError, JobError)):
        await client.inspect(child_input.input_ref, actor=forged)
    assert not client._outputs and not client._waiters and client._slots == 0
    await client.aclose()


@pytest.mark.parametrize('committed', [False, True])
async def test_submit_commit_failure_preserves_original_input_id(registry_store, monkeypatch, committed):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    transaction = port.db.transaction
    @asynccontextmanager
    async def failed_commit():
        async with transaction() as tx:
            yield tx
            if not committed:
                raise ConnectionError('commit failed')
        raise ConnectionError('commit acknowledgement lost')
    with monkeypatch.context() as patch:
        patch.setattr(port.db, 'transaction', failed_commit)
        with pytest.raises(StoreError) as failure:
            await client.send(target, 'same request', message_id='original', actor=actor)
    ref = InputRef(store.store_id, target.session_id, 'original')
    assert failure.value.input_ref == ref
    receipt = await client.send(target, 'same request', message_id='original', actor=actor)
    assert receipt.input_ref == ref and receipt.newly_accepted is not committed
    assert (await store.inbox_counts(target.session_id))['pending'] == 1
    await client.aclose()


async def test_two_local_workers_serialize_same_session(registry_store):
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, target, actor):
        second = Runtime(store=store, factory=runtime.factory, node_id='other', boot_id='boot',
            transport=runtime.transport, coordinator=runtime.coordinator, capacity=1, recovery_interval_s=None)
        await second.worker.start()
        try:
            first = await runtime.client.send(target, 'one', message_id='one', actor=actor)
            await asyncio.wait_for(model.started.wait(), 5)
            next_input = await second.client.send(target, 'two', message_id='two', actor=actor)
            # Both workers receive work; the second can wait for the existing
            # session lock, but must not re-enter its active ownership period.
            await asyncio.sleep(.1)
            assert len(model.requests) == 1
            model.release.set()
            results = await asyncio.gather(*(runtime.client.wait(r.input_ref, actor=actor, deadline_ms=deadline())
                                            for r in (first, next_input)))
            assert all(result.status == 'completed' for result in results)
            assert results[0].job_ref != results[1].job_ref
            assert len(model.requests) == 2
        finally:
            model.release.set()
            await second.worker.shutdown(timeout_s=5)
            await second.client.aclose()


async def test_list_jobs_pages_and_subtree_cancel_use_stored_relations(registry_store):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    manager = JobManager(store)
    jobs = [await manager.register_job(JobActor(actor.tenant_id, actor.principal_id, target.session_id),
                kind='send', session_id=target.session_id, system_key=f'queued-{i}') for i in range(3)]
    root = jobs[0].job.job_id
    for job in jobs[1:]:
        await port.db.execute(f'UPDATE {port.jobs} SET parent_job_id=?,root_job_id=? WHERE job_id=?',
                              (root, root, job.job.job_id))
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    first = await client.list_jobs(target, actor=actor, limit=2)
    second = await client.list_jobs(target, actor=actor, cursor=first.next_cursor, limit=2)
    assert len(first.items) == 2 and len(second.items) == 1 and second.next_cursor is None
    assert {s.job_ref.job_id for s in (*first.items, *second.items)} == {j.job.job_id for j in jobs}
    only = await client.list_jobs(TargetRef(target.store_id, target.session_id, root), actor=actor)
    assert len(only.items) == 1
    await client.cancel(JobRef(store.store_id, root, 0), actor=actor, scope='subtree', reason='tree stop')
    assert not (await client.list_jobs(target, actor=actor)).items
    for job in jobs:
        result = await client.wait(JobRef(store.store_id, job.job.job_id, 0), actor=actor, deadline_ms=deadline())
        assert result.status == 'cancelled' and result.error == 'tree stop'
    await client.aclose()


async def test_failed_start_closes_subscriptions_and_reconnect_retries(registry_store):
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, target, actor):
        other = Runtime(store=store, factory=runtime.factory, node_id='other', boot_id='boot',
                        transport=MemoryMessageTransport(), capacity=1, recovery_interval_s=None)
        subscribe = other.transport.subscribe
        async def fail_control(binding):
            if binding.channel == 'control':
                raise ConnectionError('subscribe unavailable')
            return await subscribe(binding)
        other.transport.subscribe = fail_control
        with pytest.raises(ConnectionError):
            await other.worker.start()
        assert not other.transport._subscriptions and not other.client._accepting
        await other.client.aclose()
        await other.transport.aclose()
        # The active worker must recover even if the first resubscribe fails.
        subscribe = runtime.transport.subscribe
        failures = set()
        async def fail_once(binding):
            if binding.channel in {'activation', 'work'} and binding.channel not in failures:
                failures.add(binding.channel)
                raise ConnectionError('transient resubscribe failure')
            return await subscribe(binding)
        runtime.transport.subscribe = fail_once
        await runtime.worker._subscriptions[0].aclose()
        await runtime.worker._subscriptions[1].aclose()
        receipt = await runtime.client.send(target, 'recover', message_id='reconnect', actor=actor)
        model.release.set()
        assert (await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())).status == 'completed'
        assert failures == {'activation', 'work'}


async def parent_child_port_fixture(store):
    """Create real logical records; do not claim this as child-worker execution."""
    port, parent_target, actor = await fixture(store)
    manager = JobManager(store)
    parent_actor = JobActor(actor.tenant_id, actor.principal_id, parent_target.session_id)
    parent = await manager.register_job(parent_actor, kind='send', session_id=parent_target.session_id, system_key='parent')
    lease = await manager.claim_job(parent_actor, parent.job.job_id,
                                   executor_id='parent', boot_id='boot', expected_version=1)
    await accept(port, command(parent_target, 'parent-start', options={'max_rounds': 7}), actor)
    await port.db.execute(f"UPDATE {port.q} SET status='delivered',delivered_job_id=?,delivered_job_generation=1 "
                         "WHERE session_id=? AND item_id='parent-start'", (parent.job.job_id, parent_target.session_id))
    metadata = dict((await store.get_session(parent_target.session_id)).metadata, execution_role='subagent')
    child_sid = await store.create_session(parent_session_id=parent_target.session_id, metadata=metadata)
    child_target = TargetRef(store.store_id, child_sid)
    child = await manager.register_job(parent_actor, kind='subagent', parent=lease, session_id=child_sid,
                                      system_key='child', independent=True)
    receipt = await accept(port, command(child_target, 'child-start'), actor)
    zero = JobRef(store.store_id, child.job.job_id, 0)
    await port.db.execute(f'UPDATE {port.q} SET delivered_job_id=?,delivered_job_generation=0 '
                         'WHERE session_id=? AND item_id=?', (zero.job_id, child_sid, receipt.input_ref.message_id))
    return port, manager, actor, parent, child_target, zero, receipt


async def test_unstarted_cancel_saves_parent_reference_in_same_transaction(registry_store, monkeypatch):
    store, _ = registry_store
    port, manager, actor, parent, child, zero, receipt = await parent_child_port_fixture(store)
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    # Failure at the parent inbox write must also undo cancellation and result.
    from power_loop.runtime.session_controls import ControlTransaction
    original = ControlTransaction.execute
    async def fail_parent(self, sql, params=()):
        if port.q in sql and 'INSERT' in sql:
            raise ConnectionError('parent inbox unavailable')
        return await original(self, sql, params)
    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, 'execute', fail_parent)
        with pytest.raises(ConnectionError):
            await client.cancel(receipt.input_ref, actor=actor)
    assert (await client.inspect(zero, actor=actor)).state == 'accepted'
    assert (await port.input(port.db, receipt.input_ref)).row['status'] == 'pending'
    assert await port.db.fetchone(f"SELECT event_id FROM {port.events} WHERE job_id=? AND event_type='result'", (zero.job_id,)) is None
    # Parent generation changes must not prevent delivery of the original child.
    await port.db.execute(f'UPDATE {port.jobs} SET generation=7 WHERE job_id=?', (parent.job.job_id,))
    await client.cancel(receipt.input_ref, actor=actor, reason='cancel child')
    await client.cancel(zero, actor=actor, reason='ignored retry reason')
    key = 'result_' + zero.job_id
    mid = hashlib.sha256(('result:' + key).encode()).hexdigest()
    ref = InputRef(store.store_id, parent.job.session_id, mid)
    record = await port.input(port.db, ref)
    assert record.meta['operation'] == 'result_ready'
    assert record.meta['result_ref'] == {'result_id': key, 'job_id': zero.job_id, 'generation': 0}
    assert record.row['content'] == '' and record.row['status'] == 'pending'
    assert record.meta['descriptor'] is None
    assert (await client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())).error == 'cancel child'
    row = await port.db.fetchone(f'SELECT result_json FROM {port.jobs} WHERE job_id=?', (zero.job_id,))
    assert json.loads(row['result_json'])['job_ref']['generation'] == 0
    assert (await store.inbox_counts(parent.job.session_id))['pending'] == 1
    await client.aclose()


async def test_result_port_rolls_back_and_duplicate_delivery_keeps_one_reference(registry_store):
    store, _ = registry_store
    port, manager, actor, parent, child, zero, receipt = await parent_child_port_fixture(store)
    ownership = await owner(port, child)
    actual = JobRef(store.store_id, zero.job_id, 1)
    async with manager.joined_write() as tx:
        await port.db.dialect.lock_state(tx, port.state, child.session_id)
        await tx.write_job(zero.job_id, f"UPDATE {port.jobs} SET generation=1,status='running',version=version+1 WHERE job_id=?", (zero.job_id,))
        claimed = await port.claim_inputs(tx, child, ownership, 1)
        await port.bind_input_job(tx, receipt.input_ref, actual, ownership, claim_token=claimed[0].row['claim_token'])
    value = Result(actual, receipt.input_ref, 'completed', 'child original')
    with pytest.raises(ConnectionError):
        async with manager.joined_write() as tx:
            await port.commit_result_and_parent_ref(tx, actual, value, ownership)
            raise ConnectionError('transaction rolled back')
    assert (await store.inbox_counts(parent.job.session_id))['pending'] == 0
    assert await port.get_result(actual, actor) is None
    refs = []
    for _ in range(2):
        async with manager.joined_write() as tx:
            refs.append(await port.commit_result_and_parent_ref(tx, actual, value, ownership))
    assert refs[0] == refs[1] and refs[0].session_id == parent.job.session_id
    assert (await store.inbox_counts(parent.job.session_id))['pending'] == 1
    assert (await port.get_result(receipt.input_ref, actor)).final_text == 'child original'
    session = await store.get_session(parent.job.session_id)
    changed = dict(session.metadata['runtime_descriptor'])
    changed['definition_ref'] = 'test.agent@2'
    await port_update_metadata(store, parent.job.session_id, dict(session.metadata, runtime_descriptor=changed))
    rendered = await port.result_input(await port.input(port.db, refs[0]))
    assert rendered.meta['descriptor']['definition_ref'] == 'test.agent@1'
    assert rendered.meta['options']['max_rounds'] == 7


async def test_factory_resolves_frozen_representation_and_rejects_missing_tools(registry_store):
    from power_loop.runtime.representation import ProjectedRepresentation, VerbatimRepresentation
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, _, actor):
        selected = []
        async def build(descriptor, context):
            name = context.options.representation_ref
            selected.append(name)
            profiles = {'test.projected@1': ProjectedRepresentation, 'test.verbatim@1': VerbatimRepresentation}
            if name not in profiles:
                raise RuntimeRequestError('definition_unavailable', name)
            return ExecutionEnvironment(model, ToolRegistry(), AgentLoopConfig(
                compactor=None, representation=profiles[name]()))
        runtime.factory.register('profiles.agent@1', build)
        descriptor = RuntimeDescriptor(1, 'agent', 'profiles.agent@1', None,
            'builtin.empty@1', 'builtin.none@1', 'builtin.none@1',
            {'max_rounds': 2, 'representation_ref': 'test.projected@1'}, {})
        target = await runtime.open_session(descriptor, actor=actor)
        one = await runtime.client.send(target, 'one', message_id='one', actor=actor,
            options=RequestOptions(representation_ref='test.verbatim@1'))
        await asyncio.wait_for(model.started.wait(), 5)
        two = await runtime.client.send(target, 'two', message_id='two', actor=actor)
        metadata = (await store.get_session(target.session_id)).metadata
        changed = plain(descriptor)
        changed['config']['representation_ref'] = 'unavailable@1'
        await port_update_metadata(store, target.session_id, dict(metadata, runtime_descriptor=changed))
        model.release.set()
        for receipt in (one, two):
            assert (await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline())).status == 'completed'
        assert selected == ['test.verbatim@1', 'test.projected@1']
        projected = await store.load_project_messages(target.session_id)
        assert any(row.kind == 'project' and row.send_index == 2 for row in projected)
        missing = await runtime.client.send(target, 'missing tool', message_id='missing-tool', actor=actor,
            descriptor=descriptor, options=RequestOptions(tools=('not-registered',)))
        result = await runtime.client.wait(missing.input_ref, actor=actor, deadline_ms=deadline())
        assert result.status == 'failed' and 'missing tools: not-registered' in result.error
        assert len(model.requests) == 2


async def port_update_metadata(store, sid, metadata):
    await store._db.execute(f'UPDATE {store.t.sessions} SET metadata_json=? WHERE session_id=?',
                            (json.dumps(metadata), sid))


async def test_host_subscribes_to_its_input_before_send_and_cannot_cross_tenants(registry_store):
    store, _ = registry_store
    async with running(store, 'memory') as (runtime, model, target, actor):
        ref = InputRef(store.store_id, target.session_id, 'http-stream')
        for stranger in (TrustedActor('foreign', actor.principal_id, None),
                         TrustedActor(actor.tenant_id, actor.principal_id, target.session_id)):
            with pytest.raises(RuntimeRequestError, match='forbidden'):
                await runtime.client.subscribe_output(ref, actor=stranger, before_send=True)
        output = await runtime.client.subscribe_output(ref, actor=actor, before_send=True)
        assert not model.requests
        try:
            receipt = await runtime.client.send(target, 'stream', message_id=ref.message_id, actor=actor)
            assert receipt.input_ref == ref
            assert (await anext(output)).kind == 'gap'
            model.release.set()
            async with asyncio.timeout(5):
                async for item in output:
                    if item.kind == 'completed':
                        assert item.input_ref == ref and item.payload['final_text'] == 'result-1'
            assert runtime.client._slots == 0
        finally:
            model.release.set()
            await output.aclose()

class QueuedStreamModel(LLMService):

    def __init__(self):
        self.started = asyncio.Event()
        self.commands = asyncio.Queue()

    async def complete(self, request, **callbacks):
        self.started.set()
        while True:
            count, done = await self.commands.get()
            if count == 0:
                break
            for _ in range(count):
                callbacks['on_chunk_delta_text']('x')
                await asyncio.sleep(0.03)
            done.set()
        return LLMResponse(raw_text='42')

    async def close(self):
        pass

async def test_queued_subscription_ignores_other_request_before_read(registry_store):
    import os

    from tests.runtime_fixture import RunningAgent
    from tests.unit.test_send_sql_profile import Profiled
    store, _ = registry_store
    raw = store._db
    store._db = p = Profiled(raw)
    model = QueuedStreamModel()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(max_rounds=1, compactor=None)).start()
    out = None
    pending = None
    try:
        c = runner.runtime.client
        first = await c.send(runner.target, 'first', message_id='first', actor=runner.actor)
        await asyncio.wait_for(model.started.wait(), 10)
        second = InputRef(store.store_id, runner.target.session_id, 'second')
        out = await c.subscribe_output(second, actor=runner.actor, before_send=True)
        await c.send(runner.target, 'second', message_id='second', actor=runner.actor)
        assert (await anext(out)).kind == 'gap'
        pending = asyncio.create_task(anext(out))
        await asyncio.sleep(0.05)
        for n in [5, 20]:
            p.records.clear()
            p.transactions = 0
            p.tx_sites.clear()
            done = asyncio.Event()
            await model.commands.put((n, done))
            await asyncio.wait_for(done.wait(), 10)
            await asyncio.sleep(0.1)
            reads = [r for r in p.records if 'agent.runtime_client:__anext__' in r['chain']]
            tx = [x for x in p.tx_sites if 'agent.runtime_client:__anext__' in x]
            assert len(reads) == len(tx) == 0
            if path := os.environ.get('P5BR1_STREAM_OUT'):
                with open(path, 'a') as f:
                    f.write(json.dumps({'db': store._db.dialect.name, 'deltas': n, 'sql': len(reads), 'transactions': len(tx)}) + '\n')
        await c.cancel(second, actor=runner.actor)
        assert (await asyncio.wait_for(pending, 5)).kind == 'completed'
        await model.commands.put((0, None))
        await c.wait(first.input_ref, actor=runner.actor, deadline_ms=None)
        await runner.drain()
    finally:
        if pending and (not pending.done()):
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        if out:
            await out.aclose()
        await runner.aclose()
        store._db = raw
