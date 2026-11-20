"""Session-native subagent creation, continuation, results and cancellation."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    MemoryMessageTransport,
    MemorySessionOwnershipCoordinator,
)
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import (
    InputRef,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    result_message_id,
)
from power_loop.runtime.spec import AgentSpec
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class ParentAndChild(LLMService):
    def __init__(self):
        self.requests = []
        self.spawned = False

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        if request.system_prompt.startswith("CHILD"):
            return LLMResponse(raw_text="child result")
        if not self.spawned:
            self.spawned = True
            return LLMResponse(raw_text="", tool_calls=[{
                "id": "spawn", "type": "function", "function": {
                    "name": "spawn_agent",
                    "arguments": json.dumps({
                        "task": "compute", "system_prompt": "CHILD", "tools": [],
                    }),
                },
            }])
        if any(
            isinstance(message.get("content"), str)
            and "agent_result" in message["content"]
            for message in request.messages if message.get("role") == "user"
        ):
            return LLMResponse(raw_text="parent handled child")
        return LLMResponse(raw_text="parent done")

    async def close(self):
        pass


async def test_create_retry_wait_result_and_continue(registry_store):
    store, _ = registry_store
    model, registry = ParentAndChild(), ToolRegistry()
    register_spawn_agent(registry)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        parent = await runner.send("delegate")
        assert parent.status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in rows if m.tool_call_id == "spawn"))
        child = InputRef(**receipt["input_ref"])
        from power_loop.agent.runtime_types import TargetRef
        child_target = TargetRef(**receipt["target_ref"])
        assert receipt["target_ref"]["session_id"] == child.session_id
        result = await runner.runtime.client.wait(
            child, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert result.status == "completed" and result.final_text == "child result"
        assert result.send_index == 1

        parent_result = InputRef(
            store.store_id,
            runner.target.session_id,
            result_message_id(child_target, child, 1, "agent_result"),
        )
        await runner.drain(after_input=parent_result)
        handled = await runner.runtime.client.wait(
            parent_result, actor=runner.actor, deadline_ms=now_ms() + 1000,
        )
        assert handled.final_text == "parent handled child"

        continued = await runner.runtime.client.send(
            child_target,
            "continue",
            message_id="continue-child",
            actor=runner.actor,
        )
        second = await runner.runtime.client.wait(
            continued.input_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert second.status == "completed" and second.send_index == 2
        second_parent_result = InputRef(
            store.store_id,
            runner.target.session_id,
            result_message_id(child_target, continued.input_ref, 2, "agent_result"),
        )
        assert second_parent_result.message_id != parent_result.message_id
    finally:
        await runner.aclose()


async def test_notifications_lost_still_runs_child_and_routes_result_by_polling(
    registry_store, monkeypatch,
):
    store, connect = registry_store
    other = await connect(schema="verify")
    coordinator = MemorySessionOwnershipCoordinator()
    transport = MemoryMessageTransport()
    parent = await RunningAgent(
        store, ParentAndChild(), ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9), capacity=3,
        coordinator=coordinator, transport=transport,
    ).start()
    child = await RunningAgent(
        other, ParentAndChild(), ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9), capacity=3,
        coordinator=coordinator, transport=transport,
    ).start()
    register_spawn_agent(parent.registry)
    register_spawn_agent(child.registry)

    async def lost_notify(*args, **kwargs):
        return None

    monkeypatch.setattr(parent.runtime.client, "_notify", lost_notify)
    monkeypatch.setattr(child.runtime.client, "_notify", lost_notify)
    try:
        initial = await parent.runtime.client.send(
            parent.target, "delegate", message_id="delegate", actor=parent.actor,
        )
        await parent.runtime.worker.recover()
        initial_result = await parent.runtime.client.wait(
            initial.input_ref, actor=parent.actor, deadline_ms=now_ms() + 10000,
        )
        assert initial_result.status == "completed"
        await parent.finish(initial.input_ref)
        rows = await store.load_all_messages(parent.target.session_id)
        receipt = json.loads(next(row.content for row in rows if row.tool_call_id == "spawn"))
        start = InputRef(**receipt["input_ref"])
        child_target = TargetRef(**receipt["target_ref"])
        assert (await child.runtime.client.inspect(start, actor=child.actor)).input_state == "pending"

        await child.runtime.worker.recover()
        child_result = await parent.runtime.client.wait(
            start, actor=parent.actor, deadline_ms=now_ms() + 10000,
        )
        assert child_result.status == "completed" and child_result.final_text == "child result"
        await child.drain()

        parent_ref = InputRef(
            store.store_id, parent.target.session_id,
            result_message_id(child_target, start, 1, "agent_result"),
        )
        await parent.runtime.worker.recover()
        await parent.drain(after_input=parent_ref)
        handled = await parent.runtime.client.wait(
            parent_ref, actor=parent.actor, deadline_ms=now_ms() + 10000,
        )
        assert handled.status == "completed" and handled.final_text == "parent handled child"
        parent_rows = await store._db.fetchall(
            f"SELECT item_id,status,delivered_send_index FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='agent_result'",
            (parent.target.session_id,),
        )
        assert len(parent_rows) == 1 and parent_rows[0]["item_id"] == parent_ref.message_id
        assert parent_rows[0]["status"] == "delivered" and parent_rows[0]["delivered_send_index"] is not None
        assert (await parent.runtime.client.inspect(parent_ref, actor=parent.actor)).terminal_status == "completed"
    finally:
        await child.aclose()
        await parent.aclose()
        await transport.aclose()


async def test_follow_up_idempotence_preserves_text_and_running_session(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    running = await port.accept(root, "initial", message_id="initial", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    body = "  NUDGED-42\n    keep indentation\n"
    first = await port.accept(root, body, message_id="nudge", actor=ACTOR)
    duplicate = await port.accept(root, body, message_id="nudge", actor=ACTOR)
    assert first.newly_accepted and not duplicate.newly_accepted
    assert duplicate.input_ref == first.input_ref

    state = await store._db.fetchone(
        f"SELECT execution_state,send_index,inbox_version FROM {store.t.sessions} "
        "WHERE session_id=?",
        (root.session_id,),
    )
    assert state == {"execution_state": "running", "send_index": 1, "inbox_version": 2}
    row = await store._db.fetchone(
        f"SELECT content_json,status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='nudge'",
        (root.session_id,),
    )
    assert json.loads(row["content_json"]) == body and row["status"] == "pending"
    assert await port.get_result(running.input_ref, actor=ACTOR) is None
    await port.final(batch, final_text="original completed")
    original = await port.get_result(running.input_ref, actor=ACTOR)
    assert original.status == "completed" and original.send_index == 1


async def test_same_tool_call_retries_one_child_and_grant_cannot_expand(registry_store):
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.core.agent_context import get_current_loop
    from power_loop.runtime.delegation import start
    from power_loop.runtime.pricing import Free
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract

    store, _ = registry_store
    seen = []

    async def twice(**_kwargs):
        loop = get_current_loop()
        spec = AgentSpec(name="child", system_prompt="CHILD", tools=[], lifecycle="linked")
        first = await start(loop, spec, "compute")
        second = await start(loop, spec, "compute")
        seen.append((first, second))
        try:
            await start(loop, AgentSpec(
                name="wide", system_prompt="CHILD", tools=["not-granted"], lifecycle="linked",
            ), "expand")
        except RuntimeRequestError as exc:
            assert exc.code == "forbidden"
        else:
            raise AssertionError("grant expansion was accepted")
        return json.dumps(first)

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="spawn_agent", description="delegate", input_schema={"type": "object"}, pricing=Free("test")),
        FunctionRuntimeTool(twice, ToolRuntimeContract("test.subagent", background_allowed=False)),
    )
    model = ParentAndChild()
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        await runner.send("delegate twice")
        [(first, second)] = seen
        assert first["target_ref"] == second["target_ref"]
        assert first["input_ref"] == second["input_ref"]
        assert first["newly_accepted"] is True and second["newly_accepted"] is False
        children = await runner.runtime.client.list_children(runner.target, actor=runner.actor)
        assert len(children.items) == 1
        await runner.runtime.client.wait(
            InputRef(**first["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
    finally:
        await runner.aclose()


async def test_subtree_cancel_only_linked_and_close_checks_linked_child(registry_store):
    from power_loop.contracts.errors import SessionBusy

    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "cancel-parent")
    parent_actor = TrustedActor("tenant", "host", parent.session_id)
    seq = await store.append_message(
        parent.session_id,
        role="assistant",
        content="",
        tool_calls=[{
            "id": "call-" + value, "type": "function",
            "function": {"name": "spawn_agent", "arguments": "{}"},
        } for value in ("linked", "independent")],
    )
    call_ref = await store.message_ref(parent.session_id, seq)
    children = {}
    for propagation in ("linked", "independent"):
        child, start, _ = await port.create_child(
            DESCRIPTOR,
            parent=parent,
            actor=parent_actor,
            call_ref=call_ref,
            tool_call_id="call-" + propagation,
            body="run",
            cancel_propagation=propagation,
        )
        children[propagation] = (child, start)
    receipt = await port.cancel(parent, actor=parent_actor, scope="subtree", reason="stop tree")
    assert receipt.state == "already_finished"
    linked = await port.get_result(children["linked"][1], actor=parent_actor)
    independent = await port.inspect(children["independent"][1], actor=parent_actor)
    assert linked.status == "cancelled" and linked.send_index is None
    assert independent.input_state == "pending" and independent.terminal_status is None
    with pytest.raises(SessionBusy):
        await port.close(parent, actor=parent_actor)


async def _launch_worker(store, parent_sid, coordinator_path, role):
    import os
    import sys
    from pathlib import Path

    backend = store._db.dialect.name
    dsn = os.environ["POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    return await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.unit._subagent_worker_215",
        dsn, store.table_prefix, parent_sid, str(coordinator_path), role,
        cwd=Path(__file__).resolve().parents[2],
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _worker_event(process):
    line = await asyncio.wait_for(process.stdout.readline(), 30)
    assert line, (await process.stderr.read()).decode()[-3000:]
    return json.loads(line)


async def _worker_command(process, payload):
    process.stdin.write((json.dumps(payload) + "\n").encode())
    await process.stdin.drain()
    return await _worker_event(process)


async def _quit_worker(process):
    if process.returncode is None:
        process.stdin.write(b"quit\n")
        await process.stdin.drain()
        await asyncio.wait_for(process.wait(), 20)
        assert process.returncode == 0, (await process.stderr.read()).decode()[-3000:]


async def test_real_parent_a_child_b_parent_c_and_subtree_cancel(registry_store, tmp_path):
    import asyncio

    store, _ = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("real parent/child processes use a shared server DB")
    port = RuntimeInputStore(store)
    parent = await port.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "real-parent"})
    coordinator = tmp_path / "coord.db"
    processes = []
    try:
        a = await _launch_worker(store, parent.session_id, coordinator, "A")
        processes.append(a)
        finished = await _worker_event(a)
        assert finished["event"] == "parent_finished" and finished["status"] == "completed"
        receipt = finished["receipt"]
        start = InputRef(**receipt["input_ref"])
        child_target = TargetRef(**receipt["target_ref"])

        b = await _launch_worker(store, parent.session_id, coordinator, "B")
        processes.append(b)
        assert (await _worker_event(b))["event"] == "ready"
        child = await _worker_command(b, {"action": "wait", "input_ref": receipt["input_ref"]})
        assert child["status"] == "completed" and child["final_text"] == "child result"
        await _quit_worker(a)

        c = await _launch_worker(store, parent.session_id, coordinator, "C")
        processes.append(c)
        assert (await _worker_event(c))["event"] == "ready"
        parent_ref = InputRef(
            store.store_id, parent.session_id,
            result_message_id(child_target, start, 1, "agent_result"),
        )
        handled = await _worker_command(c, {"action": "wait", "input_ref": {
            "store_id": parent_ref.store_id, "session_id": parent_ref.session_id,
            "message_id": parent_ref.message_id,
        }})
        assert handled["status"] == "completed" and handled["final_text"] == "parent handled"
        continued = await _worker_command(c, {
            "action": "send", "body": "continue", "message_id": "parent-continue",
        })
        assert continued["status"] == "completed" and continued["send_index"] >= 3
        await _quit_worker(b)
        again = await _worker_command(c, {
            "action": "send", "body": "delegate again", "message_id": "parent-again",
        })
        assert again["status"] == "completed"
        rows = await store._db.fetchall(
            f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='tool' "
            "AND tool_call_id LIKE 'spawn%' ORDER BY seq",
            (parent.session_id,),
        )
        second = json.loads(rows[-1]["content"])
        second_start = InputRef(**second["input_ref"])
        cancelled = await _worker_command(c, {"action": "cancel_subtree"})
        assert cancelled["event"] == "cancelled"
        for _ in range(300):
            result = await port.get_result(second_start, actor=ACTOR)
            if result is not None:
                break
            await asyncio.sleep(0.01)
        assert result.status == "cancelled" and result.send_index is None
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()


async def test_real_crash_after_child_creation_recovers_deterministic_refs(registry_store, tmp_path):
    import asyncio

    store, _ = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("real crash recovery uses a shared server DB")
    port = RuntimeInputStore(store)
    parent = await port.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "real-crash"})
    coordinator = tmp_path / "coord.db"
    processes = []
    try:
        crash = await _launch_worker(store, parent.session_id, coordinator, "crash")
        processes.append(crash)
        created = await _worker_event(crash)
        assert created["event"] == "child_created"
        await asyncio.wait_for(crash.wait(), 20)
        receipt = created["receipt"]

        b = await _launch_worker(store, parent.session_id, coordinator, "B")
        processes.append(b)
        assert (await _worker_event(b))["event"] == "ready"
        child = await _worker_command(b, {"action": "wait", "input_ref": receipt["input_ref"]})
        assert child["status"] == "completed"

        c = await _launch_worker(store, parent.session_id, coordinator, "C")
        processes.append(c)
        assert (await _worker_event(c))["event"] == "ready"
        repair = await _worker_command(c, {"action": "read_repair"})
        assert repair["content"]["creation_result_uncertain"] is True
        assert repair["content"]["input_ref"] == receipt["input_ref"]
        original = await port.get_result(
            InputRef(store.store_id, parent.session_id, "parent-start"), actor=ACTOR,
        )
        assert original.status == "interrupted"
        after = await _worker_command(c, {
            "action": "send", "body": "after repair", "message_id": "after-repair",
        })
        assert after["status"] == "completed"
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
