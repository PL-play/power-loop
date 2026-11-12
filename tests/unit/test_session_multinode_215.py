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
