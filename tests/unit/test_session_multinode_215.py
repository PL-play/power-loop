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

    factory.register("test.agent@215", build)
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

    factory.register("test.agent@215", build)
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

    incapable.register("test.agent@215", bad_build)
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
