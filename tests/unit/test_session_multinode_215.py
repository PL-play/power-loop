"""P2 worker discovery, capability filtering and cross-node takeover."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RegistryRuntimeFactory,
    Runtime,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import InputRef

from .session_native_fixture import ACTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class EchoModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text="done", tool_calls=None,
                           token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


def _factory(definition="test.agent@215"):
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        return ExecutionEnvironment(EchoModel(), None, AgentLoopConfig(max_rounds=4, compactor=None))

    factory.register(definition, build)
    return factory


async def test_incapable_node_does_not_claim_or_fail_input(registry_store):
    """design 215 §9.4: a node that cannot build the descriptor leaves the session for a
    capable node — it never claims and never terminalizes the input."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    # This factory registers a DIFFERENT definition: the node is incapable.
    runtime = Runtime(store=store, factory=_factory("other.agent@215"), node_id="incapable",
                      boot_id="b", recovery_interval_s=None, timers_enabled=False)
    try:
        await runtime.worker.start()
        await runtime.worker.recover()
        await asyncio.sleep(0.2)
        state = await store._db.fetchone(
            f"SELECT execution_state FROM {store.t.session_state} WHERE session_id=?", (root.session_id,))
        assert state["execution_state"] == "queued"  # never claimed
        status = await port.inspect(InputRef(store.store_id, root.session_id, "m1"), actor=ACTOR)
        assert status.terminal_status is None and status.input_state == "pending"
        assert "test.agent@215" in runtime.worker.capability_missing
    finally:
        await runtime.worker.shutdown(timeout_s=2)
        await runtime.client.aclose()
        await runtime.transport.aclose()


class CountingModel(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        return LLMResponse(raw_text="done", tool_calls=None,
                           token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


async def _assert_incapable_leaves_input_pending(store, factory, model):
    """A node that cannot execute must NOT claim/terminalize: input stays pending, session
    queued, capability_missing records the definition, and the model is never called."""
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    runtime = Runtime(store=store, factory=factory, node_id="incapable", boot_id="b",
                      recovery_interval_s=None, timers_enabled=False)
    try:
        await runtime.worker.start()
        await runtime.worker.recover()
        await asyncio.sleep(0.2)
        state = await store._db.fetchone(
            f"SELECT execution_state,active_send_index,owner_token FROM {store.t.session_state} "
            "WHERE session_id=?", (root.session_id,))
        assert state["execution_state"] == "queued"
        assert state["active_send_index"] is None and state["owner_token"] is None
        status = await port.inspect(InputRef(store.store_id, root.session_id, "m1"), actor=ACTOR)
        assert status.input_state == "pending" and status.terminal_status is None
        row = await store._db.fetchone(
            f"SELECT status,terminal_status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id='m1'", (root.session_id,))
        assert row["status"] == "pending" and row["terminal_status"] is None
        assert "test.agent@215" in runtime.worker.capability_missing
        assert model.calls == 0  # never executed
        # No merged user message was written (never claimed).
        messages = await store._db.fetchall(
            f"SELECT role FROM {store.t.messages} WHERE session_id=?", (root.session_id,))
        assert messages == []
    finally:
        await runtime.worker.shutdown(timeout_s=2)
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_command_factory_does_not_claim_or_fail(registry_store):
    """P2-R1-1: a factory returning a host command is incapable in P2; the worker must
    detect this BEFORE claim and leave the input pending/queued (not delivered/failed)."""
    store, _ = registry_store
    model = CountingModel()
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        return ExecutionEnvironment(model, None, AgentLoopConfig(max_rounds=4, compactor=None),
                                    command=("host_command", {}))

    factory.register("test.agent@215", build, can_execute=lambda descriptor: False)
    await _assert_incapable_leaves_input_pending(store, factory, model)


@pytest.mark.parametrize("code", ["capability_unavailable", "resource_unavailable"])
async def test_build_failure_does_not_claim_or_fail(registry_store, code):
    """P2-R1-1: a factory whose build raises capability/resource unavailability is detected
    BEFORE claim; the input stays pending/queued for a capable node, never failed."""
    from power_loop.agent.runtime_types import RuntimeRequestError
    store, _ = registry_store
    model = CountingModel()
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        raise RuntimeRequestError(code, "node cannot execute this descriptor")

    def can_execute(descriptor):
        raise RuntimeRequestError(code, "node cannot execute this descriptor")

    factory.register("test.agent@215", build, can_execute=can_execute)
    await _assert_incapable_leaves_input_pending(store, factory, model)


async def test_capable_node_completes_after_incapable_node_leaves_input(registry_store):
    """The input an incapable node left pending is still claimable by a capable node."""
    from power_loop.agent.runtime_types import RuntimeRequestError
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    incapable = RegistryRuntimeFactory()

    async def bad_build(descriptor, context):
        raise RuntimeRequestError("capability_unavailable", "no executor")

    def can_execute(descriptor):
        raise RuntimeRequestError("capability_unavailable", "no executor")

    incapable.register("test.agent@215", bad_build, can_execute=can_execute)
    node_a = Runtime(store=store, factory=incapable, node_id="A", boot_id="A",
                     recovery_interval_s=None, timers_enabled=False)
    try:
        await node_a.worker.start()
        await node_a.worker.recover()
        await asyncio.sleep(0.1)
        assert (await port.inspect(InputRef(store.store_id, root.session_id, "m1"),
                                   actor=ACTOR)).terminal_status is None
    finally:
        await node_a.worker.shutdown(timeout_s=2)
        await node_a.client.aclose()
        await node_a.transport.aclose()
    # A capable node now takes the still-pending input and completes it.
    node_b = Runtime(store=store, factory=_factory(), node_id="B", boot_id="B",
                     recovery_interval_s=None, timers_enabled=False)
    try:
        await node_b.worker.start()
        await node_b.worker.recover()
        result = await node_b.client.wait(InputRef(store.store_id, root.session_id, "m1"),
                                          actor=ACTOR, deadline_ms=None)
        assert result.status == "completed" and result.final_text == "done"
    finally:
        await node_b.worker.shutdown(timeout_s=2)
        await node_b.client.aclose()
        await node_b.transport.aclose()


async def test_build_uses_claimed_execution_context(registry_store):
    """The only build happens after claim and receives the real execution identity."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    model = CountingModel()
    contexts = []
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        contexts.append((context.input_ref, context.send_index,
                         context.owner_epoch, context.owner_token))
        return ExecutionEnvironment(
            model, None, AgentLoopConfig(max_rounds=4, compactor=None))

    factory.register("test.agent@215", build)
    runtime = Runtime(store=store, factory=factory, node_id="claimed", boot_id="b",
                      recovery_interval_s=None, timers_enabled=False)
    try:
        await runtime.worker.start()
        await runtime.worker.recover()
        ref = InputRef(store.store_id, root.session_id, "m1")
        result = await asyncio.wait_for(
            runtime.client.wait(ref, actor=ACTOR, deadline_ms=None), 10)
        assert result.status == "completed" and result.send_index == 1
        assert len(contexts) == 1
        built_ref, send_index, owner_epoch, owner_token = contexts[0]
        assert (built_ref, send_index, owner_epoch) == (ref, 1, 1)
        assert owner_token
        assert model.calls == 1
    finally:
        await runtime.worker.shutdown(timeout_s=2)
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_factory_requiring_input_ref_completes(registry_store):
    """A legitimate factory may require the InputRef allocated by claim."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    factory = RegistryRuntimeFactory()
    builds = 0

    async def build(descriptor, context):
        nonlocal builds
        builds += 1
        assert context.input_ref is not None
        return ExecutionEnvironment(
            CountingModel(), None, AgentLoopConfig(max_rounds=4, compactor=None))

    factory.register("test.agent@215", build)
    runtime = Runtime(store=store, factory=factory, node_id="input-ref", boot_id="b",
                      recovery_interval_s=None, timers_enabled=False)
    try:
        await runtime.worker.start()
        await runtime.worker.recover()
        ref = InputRef(store.store_id, root.session_id, "m1")
        result = await asyncio.wait_for(
            runtime.client.wait(ref, actor=ACTOR, deadline_ms=None), 10)
        assert result.status == "completed" and result.final_text == "done"
        assert builds == 1
        assert not runtime.worker.capability_missing
    finally:
        await runtime.worker.shutdown(timeout_s=2)
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_claim_release_after_build_failure(registry_store):
    """A build-only capability failure rolls back its claim for the next capable node."""
    from power_loop.agent.runtime_types import RuntimeRequestError
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    model = CountingModel()
    contexts = []
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        contexts.append((context.input_ref, context.send_index, context.owner_epoch))
        raise RuntimeRequestError("resource_unavailable", "only available on another node")

    factory.register("test.agent@215", build)
    node_a = Runtime(store=store, factory=factory, node_id="A", boot_id="A",
                     recovery_interval_s=None, timers_enabled=False)
    ref = InputRef(store.store_id, root.session_id, "m1")
    try:
        await node_a.worker.start()
        await node_a.worker.recover()
        while node_a.worker._tasks:
            await asyncio.wait(tuple(node_a.worker._tasks))
        assert contexts == [(ref, 1, 1)]
        status = await port.inspect(ref, actor=ACTOR)
        assert status.input_state == "pending" and status.terminal_status is None
        state = await store._db.fetchone(
            f"SELECT execution_state,active_send_index,owner_token FROM {store.t.session_state} "
            "WHERE session_id=?", (root.session_id,))
        assert state == {"execution_state": "queued", "active_send_index": None, "owner_token": None}
        messages = await store._db.fetchall(
            f"SELECT role FROM {store.t.messages} WHERE session_id=?", (root.session_id,))
        assert messages == []
        assert model.calls == 0
        assert "test.agent@215" in node_a.worker.capability_missing
    finally:
        await node_a.worker.shutdown(timeout_s=2)
        await node_a.client.aclose()
        await node_a.transport.aclose()

    node_b = Runtime(store=store, factory=_factory(), node_id="B", boot_id="B",
                     recovery_interval_s=None, timers_enabled=False)
    try:
        await node_b.worker.start()
        await node_b.worker.recover()
        result = await asyncio.wait_for(
            node_b.client.wait(ref, actor=ACTOR, deadline_ms=None), 10)
        assert result.status == "completed" and result.send_index == 2
    finally:
        await node_b.worker.shutdown(timeout_s=2)
        await node_b.client.aclose()
        await node_b.transport.aclose()


async def test_release_claim_honours_matching_cancel(registry_store):
    """A cancel accepted during claimed build terminalizes the whole batch on release."""
    from power_loop.agent.runtime_types import RuntimeRequestError
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    first = await port.accept(root, "first", message_id="m1", actor=ACTOR)
    second = await port.accept(root, "second", message_id="m2", actor=ACTOR)
    model_a = CountingModel()
    build_started = asyncio.Event()
    finish_build = asyncio.Event()
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        assert context.input_ref == first.input_ref
        assert context.send_index == 1 and context.owner_epoch == 1
        build_started.set()
        await finish_build.wait()
        raise RuntimeRequestError("resource_unavailable", "build failed after cancel")

    factory.register("test.agent@215", build)
    node_a = Runtime(store=store, factory=factory, node_id="A", boot_id="A",
                     recovery_interval_s=None, timers_enabled=False)
    third = None
    try:
        await node_a.worker.start()
        await node_a.worker.recover()
        await asyncio.wait_for(build_started.wait(), 5)
        receipt = await node_a.client.cancel(first.input_ref, actor=ACTOR, reason="stop build")
        assert receipt.state == "cancel_requested"
        third = await port.accept(root, "third", message_id="m3", actor=ACTOR)
        from .test_send_sql_profile import Profiled, _counts
        original = store._db
        profiled = Profiled(original)
        store._db = profiled
        try:
            finish_build.set()
            while node_a.worker._tasks:
                await asyncio.wait(tuple(node_a.worker._tasks))
            counts = _counts(profiled.records, profiled.tx_sites, profiled.tx_categories)
            assert profiled.transactions == 1
            assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
            if output := os.environ.get("P215_RELEASE_PROFILE_OUT"):
                with open(output, "a") as stream:
                    stream.write(json.dumps({
                        "db": original.dialect.name, "scenario": "release_claim_matching_cancel",
                        "sql": len(profiled.records), "counts": counts,
                        "transactions": profiled.transactions,
                    }) + "\n")
        finally:
            store._db = original

        for ref in (first.input_ref, second.input_ref):
            result = await port.get_result(ref, actor=ACTOR)
            assert result.status == "cancelled" and result.send_index == 1
            assert result.error_code == "cancelled" and result.error == "stop build"
            assert result.result_seq is None
        rows = await store._db.fetchall(
            f"SELECT item_id,status,terminal_status,error_code,error_detail,completed_at "
            f"FROM {store.t.follow_up_queue} WHERE session_id=? ORDER BY item_id",
            (root.session_id,))
        assert [row["item_id"] for row in rows] == ["m1", "m2", "m3"]
        for row in rows[:2]:
            assert row["status"] == "delivered" and row["terminal_status"] == "cancelled"
            assert row["error_code"] == "cancelled" and row["error_detail"] == "stop build"
            assert row["completed_at"] is not None
        assert rows[2]["status"] == "pending" and rows[2]["terminal_status"] is None
        state = await store._db.fetchone(
            f"SELECT execution_state,active_send_index,owner_token FROM {store.t.session_state} "
            "WHERE session_id=?", (root.session_id,))
        assert state == {"execution_state": "queued", "active_send_index": None, "owner_token": None}
        messages = await store._db.fetchall(
            f"SELECT role FROM {store.t.messages} WHERE session_id=?", (root.session_id,))
        assert messages == []
        assert model_a.calls == 0
    finally:
        finish_build.set()
        await node_a.worker.shutdown(timeout_s=2)
        await node_a.client.aclose()
        await node_a.transport.aclose()

    model_b = CountingModel()
    capable = RegistryRuntimeFactory()

    async def capable_build(descriptor, context):
        return ExecutionEnvironment(
            model_b, None, AgentLoopConfig(max_rounds=4, compactor=None))

    capable.register("test.agent@215", capable_build)
    node_b = Runtime(store=store, factory=capable, node_id="B", boot_id="B",
                     recovery_interval_s=None, timers_enabled=False)
    try:
        await node_b.worker.start()
        await node_b.worker.recover()
        result = await asyncio.wait_for(
            node_b.client.wait(third.input_ref, actor=ACTOR, deadline_ms=None), 10)
        assert result.status == "completed" and result.send_index == 2
        assert model_b.calls == 1
        for ref in (first.input_ref, second.input_ref):
            assert (await port.get_result(ref, actor=ACTOR)).status == "cancelled"
    finally:
        await node_b.worker.shutdown(timeout_s=2)
        await node_b.client.aclose()
        await node_b.transport.aclose()


async def test_worker_discovers_and_takes_over_expired_run(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "old", message_id="old", actor=ACTOR)
    # Node A claims then dies (lease expires); a new input arrives.
    batch_a = await port.claim(root, node_id="A", boot_id="A")
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?", (root.session_id,))
    await port.accept(root, "new", message_id="new", actor=ACTOR)
    # Node B's worker discovers the unfinished session and takes over.
    runtime = Runtime(store=store, factory=_factory(), node_id="B", boot_id="B",
                      recovery_interval_s=None, timers_enabled=False)
    try:
        await runtime.worker.start()
        await runtime.worker.recover()
        new_ref = InputRef(store.store_id, root.session_id, "new")
        result = await runtime.client.wait(new_ref, actor=ACTOR, deadline_ms=None)
        assert result.status == "completed" and result.send_index == batch_a.send_index + 1
        assert (await port.get_result(batch_a.input_refs[0], actor=ACTOR)).status == "interrupted"
    finally:
        await runtime.worker.shutdown(timeout_s=2)
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_two_process_worker_takeover(registry_store):
    """Real two processes on a shared server DB: A holds a claimed run, B's WORKER
    discovers the expired session and completes the new input (design 215 §9.3)."""
    store, _ = registry_store
    backend = store._db.dialect.name
    if backend == "sqlite":
        pytest.skip("real worker takeover uses a shared server DB")
    port = RuntimeInputStore(store)
    root = await target(port, session_id="worker-takeover")
    await port.accept(root, "old", message_id="old", actor=ACTOR)
    batch_a = await port.claim(root, node_id="A", boot_id="A")
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?", (root.session_id,))
    await port.accept(root, "new", message_id="new", actor=ACTOR)

    dsn = os.environ["POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    repo = Path(__file__).resolve().parents[2]
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.unit._session_worker_215",
        dsn, store.table_prefix, root.session_id, "worker_takeover", cwd=repo,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        line = await asyncio.wait_for(process.stdout.readline(), 30)
        assert line, (await process.stderr.read()).decode()[-3000:]
        event = json.loads(line)
        assert event["event"] == "completed" and event["status"] == "completed"
        assert event["send_index"] == batch_a.send_index + 1
        await asyncio.wait_for(process.wait(), 20)
        assert process.returncode == 0, (await process.stderr.read()).decode()[-3000:]
        assert (await port.get_result(batch_a.input_refs[0], actor=ACTOR)).status == "interrupted"
        statuses = await store._db.fetchall(
            f"SELECT item_id,terminal_status FROM {store.t.follow_up_queue} ORDER BY item_id")
        assert statuses == [
            {"item_id": "new", "terminal_status": "completed"},
            {"item_id": "old", "terminal_status": "interrupted"},
        ]
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
