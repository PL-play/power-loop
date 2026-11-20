"""Session-native workflow graph, tool admission and recovery contracts."""
from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from dataclasses import replace

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    MemoryMessageTransport,
    RuntimeClient,
)
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import InputRef, RuntimeRequestError, TargetRef, plain
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import register_workflow_tool
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_model_tool_creates_and_finishes_workflow_without_job_authority(registry_store):
    store, _ = registry_store

    class Parent(Model):
        created = False

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                return LLMResponse(raw_text="leaf done")
            if not self.created:
                self.created = True
                return LLMResponse(raw_text="", tool_calls=[{
                    "id": "workflow-call", "type": "function", "function": {
                        "name": "workflow",
                        "arguments": json.dumps({
                            "action": "create",
                            "spec": {"name": "from model", "root": leaf("A")},
                        }),
                    },
                }])
            return LLMResponse(raw_text="parent finished")

    registry = ToolRegistry()
    register_workflow_tool(registry)
    runner = await RunningAgent(store, Parent(), registry, AgentLoopConfig(context_budget_tokens=10**9)).start()
    built = []
    original_build = runner.runtime.factory.build

    async def build(descriptor, execution):
        from power_loop.runtime.session_ownership import current_ownership
        lease = current_ownership(store, execution.session_id)
        assert lease is runner.runtime.worker._active[execution.session_id].ownership
        lease.require_owned()
        assert (lease.key, lease.epoch, lease.token) == (
            (execution.store_id, execution.session_id), execution.owner_epoch, execution.owner_token)
        built.append((descriptor.kind, lease))
        return await original_build(descriptor, execution)

    runner.runtime.factory.build = build
    try:
        parent = await asyncio.wait_for(runner.send("run"), 10)
        assert parent.status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(row.content for row in rows if row.tool_call_id == "workflow-call"))
        assert receipt["status"] == "accepted", receipt
        assert set(receipt) >= {"run_id", "target_ref", "input_ref"}
        assert not {"job_id", "job_ref", "operation_id"} & set(receipt)
        result = await asyncio.wait_for(runner.runtime.client.wait(
            InputRef(**receipt["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000,
        ), 12)
        assert result.status == "completed", result
        assert result.structured_output["results"]["A"]["text"] == "leaf done"
        await runner.runtime.worker.recover()
        try:
            await asyncio.wait_for(runner.drain(), 10)
        except TimeoutError:
            states = await store._db.fetchall(
                f"SELECT session_id,execution_state,active_send_index,send_index,owner_token "
                f"FROM {store.t.session_state}",
            )
            inbox = await store._db.fetchall(
                f"SELECT session_id,item_id,kind,status,terminal_status,delivered_send_index "
                f"FROM {store.t.follow_up_queue}",
            )
            active = {sid: (ctx.descriptor.kind, ctx.send_index)
                      for sid, ctx in runner.runtime.worker._active.items() if ctx is not None}
            raise AssertionError((states, inbox, active, runner.runtime.worker.last_error,
                                  runner.runtime.worker._pending))
        sessions = await store._db.fetchall(
            f"SELECT session_id,session_kind,parent_session_id FROM {store.t.sessions} ORDER BY created_at",
        )
        assert {row["session_kind"] for row in sessions} == {"agent", "workflow_driver"}
        driver = TargetRef(**receipt["target_ref"])
        children = await runner.runtime.client.list_children(driver, actor=runner.actor)
        assert len(children.items) == 1
        assert {kind for kind, _ in built} == {"agent", "workflow_driver"}
        assert all(lease.released and not lease.lost for _, lease in built)
    finally:
        await runner.aclose()


async def _expire(store, session_id):
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=0 WHERE session_id=?",
        (session_id,),
    )


async def _driver_task(runner, handle):
    async with asyncio.timeout(5):
        while handle.input_ref not in runner.executions:
            await asyncio.sleep(0.01)
    return runner.executions[handle.input_ref]


async def test_crash_after_node_create_before_journal_aligns_without_stopping_leaf(
    registry_store, monkeypatch,
):
    store, _ = registry_store
    from power_loop.workflow import journal

    started, release = asyncio.Event(), asyncio.Event()

    class Waiting(Model):
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                started.set()
                await release.wait()
                return LLMResponse(raw_text="healthy leaf")
            return LLMResponse(raw_text="parent acknowledged")

    original_put = journal.put
    crashed = False

    async def crash_after_create(*args):
        nonlocal crashed
        key = args[3]
        if key.startswith(journal.NODE_PREFIX) and not crashed:
            crashed = True
            raise asyncio.CancelledError("crash after node create")
        return await original_put(*args)

    monkeypatch.setattr(journal, "put", crash_after_create)
    runner = await RunningAgent(store, Waiting(), config=AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "create gap", "root": leaf("A")},
            message_id="create-gap", actor=runner.actor,
        )
        await asyncio.wait_for(started.wait(), 5)
        task = await _driver_task(runner, handle)
        await asyncio.gather(task, return_exceptions=True)
        saved = await journal.read(store, runner.target.session_id, handle.run_id)
        assert saved["nodes"] == []
        children = await runner.runtime.client.list_children(handle.target, actor=runner.actor)
        assert len(children.items) == 1
        node = children.items[0]
        status = await runner.runtime.client.inspect(handle.input_ref, actor=runner.actor)
        node_status = await runner.runtime.client.inspect(node, actor=runner.actor)
        assert status.terminal_status is None and node_status.execution_state == "running"
        assert not node_status.cancel_requested

        monkeypatch.undo()
        await _expire(store, handle.target.session_id)
        release.set()
        await runner.runtime.worker.recover()
        result = await runner.runtime.client.wait(
            handle.input_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert result.status == "completed" and result.final_text == "healthy leaf"
        assert result.send_index == 2
        saved = await journal.read(store, runner.target.session_id, handle.run_id)
        assert len(saved["nodes"]) == 1
        assert saved["nodes"][0]["input_ref"]["session_id"] == node.session_id
        assert saved["nodes"][0]["result"]["send_index"] == 1
        continuation = await store._db.fetchone(
            f"SELECT continuation_of,continuation_index,kind FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue'",
            (handle.target.session_id,),
        )
        assert continuation == {
            "continuation_of": handle.input_ref.message_id,
            "continuation_index": 1,
            "kind": "workflow_continue",
        }
        inputs = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='user'",
            (node.session_id,),
        )
        assert len(inputs) == 1
    finally:
        release.set()
        await runner.aclose()


async def _crash_before_aggregate(store, monkeypatch):
    from power_loop.workflow import runner as workflow_runner

    class OneLeaf(Model):
        calls = 0

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                self.calls += 1
                return LLMResponse(raw_text="journaled result")
            return LLMResponse(raw_text="parent acknowledged")

    async def crash(*args, **kwargs):
        raise asyncio.CancelledError("crash before aggregate")

    monkeypatch.setattr(workflow_runner, "_complete_driver", crash)
    model = OneLeaf()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(context_budget_tokens=10**9)).start()
    handle = await runner.runtime.client.workflow(
        runner.target, {"name": "aggregate gap", "root": leaf("A")},
        message_id="aggregate-gap", actor=runner.actor,
    )
    task = await _driver_task(runner, handle)
    await asyncio.gather(task, return_exceptions=True)
    return runner, handle, model


async def test_results_journaled_empty_inbox_recovery_uses_deterministic_continuation(
    registry_store, monkeypatch,
):
    store, _ = registry_store
    from power_loop.workflow import journal

    runner, handle, model = await _crash_before_aggregate(store, monkeypatch)
    try:
        saved = await journal.read(store, runner.target.session_id, handle.run_id)
        assert len(saved["nodes"]) == 1 and saved["nodes"][0]["result"] is not None
        pending = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND status='pending'",
            (handle.target.session_id,),
        )
        assert pending == []
        monkeypatch.undo()
        await _expire(store, handle.target.session_id)
        await runner.runtime.worker.recover()
        result = await runner.runtime.client.wait(
            handle.input_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert result.status == "completed" and result.final_text == "journaled result"
        assert result.send_index == 2 and model.calls == 1
        aggregate = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (runner.target.session_id,),
        )
        assert len(aggregate) == 1
    finally:
        await runner.aclose()


async def test_original_cancel_closes_queued_continuation_chain_without_recreating_it(
    registry_store, monkeypatch,
):
    store, _ = registry_store
    runner, handle, _model = await _crash_before_aggregate(store, monkeypatch)
    try:
        monkeypatch.undo()
        await _expire(store, handle.target.session_id)
        recovered = await runner.runtime.worker.inputs.recover(
            handle.target, node_id="recovery", boot_id="recovery", ttl_s=30,
        )
        assert recovered.execution_state == "queued"
        continuation = await store._db.fetchone(
            f"SELECT item_id,status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue'",
            (handle.target.session_id,),
        )
        assert continuation["status"] == "pending"
        receipt = await handle.cancel("stop queued continuation")
        assert receipt.state == "voided"
        result = await runner.runtime.client.wait(
            handle.input_ref, actor=runner.actor, deadline_ms=now_ms() + 3000,
        )
        assert result.status == "cancelled" and result.error == "stop queued continuation"
        await runner.runtime.worker.recover()
        rows = await store._db.fetchall(
            f"SELECT item_id,status,terminal_status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue'",
            (handle.target.session_id,),
        )
        assert len(rows) == 1 and rows[0]["status"] == "void" and rows[0]["terminal_status"] == "cancelled"
    finally:
        await runner.aclose()


async def test_interrupted_node_is_not_business_retried(registry_store):
    store, _ = registry_store
    calls = 0

    class Interrupted(Model):
        async def complete(self, request, **kwargs):
            nonlocal calls
            if request.system_prompt.startswith("A"):
                calls += 1
                raise asyncio.CancelledError("node process lost")
            return LLMResponse(raw_text="parent acknowledged")

    runner = await RunningAgent(store, Interrupted(), config=AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        node = leaf("A")
        node["retry"] = {"max_attempts": 2, "on": ["failed"]}
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "unknown", "root": node},
            message_id="unknown", actor=runner.actor,
        )
        async with asyncio.timeout(5):
            while calls != 1:
                await asyncio.sleep(0.01)
        children = await runner.runtime.client.list_children(handle.target, actor=runner.actor)
        assert len(children.items) == 1
        await _expire(store, children.items[0].session_id)
        await runner.runtime.worker.recover()
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert result.status == "failed" and result.results["A"].status == "interrupted"
        assert calls == 1
        inputs = await store._db.fetchall(
            f"SELECT item_id,result_send_index,terminal_status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='user'",
            (children.items[0].session_id,),
        )
        assert len(inputs) == 1 and inputs[0]["terminal_status"] == "interrupted"
    finally:
        await runner.aclose()


async def test_original_cancel_running_continuation_cancels_current_send(registry_store):
    store, _ = registry_store
    started, release = asyncio.Event(), asyncio.Event()

    class Waiting(Model):
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("B"):
                started.set()
                await release.wait()
                return LLMResponse(raw_text="too late")
            if request.system_prompt.startswith("A"):
                return LLMResponse(raw_text="A done")
            return LLMResponse(raw_text="parent acknowledged")

    runner = await RunningAgent(store, Waiting(), config=AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "cancel continuation", "root": {"type": "sequence", "steps": [leaf("A"), leaf("B")]}},
            message_id="cancel-continuation", actor=runner.actor,
        )
        await asyncio.wait_for(started.wait(), 5)
        task = await _driver_task(runner, handle)
        runner.runtime.worker._active[handle.target.session_id].shutdown_interrupted = True
        task.cancel("driver process crash")
        await asyncio.gather(task, return_exceptions=True)
        # A real process crash also drops this local registry entry.
        runner.runtime.worker._active.pop(handle.target.session_id, None)
        await _expire(store, handle.target.session_id)
        recovered = await runner.runtime.worker.inputs.recover(
            handle.target, node_id="recovery", boot_id="recovery", ttl_s=30,
        )
        assert recovered is not None and recovered.execution_state == "queued"
        await runner.runtime.worker.recover()
        try:
            async with asyncio.timeout(5):
                while True:
                    state = await store._db.fetchone(
                        f"SELECT active_send_index FROM {store.t.session_state} WHERE session_id=?",
                        (handle.target.session_id,),
                    )
                    if state["active_send_index"] == 2:
                        break
                    await asyncio.sleep(0.01)
        except TimeoutError:
            rows = await store._db.fetchall(
                f"SELECT session_id,execution_state,active_send_index,cancel_send_index FROM {store.t.session_state}",
            )
            inbox = await store._db.fetchall(
                f"SELECT session_id,kind,status,terminal_status,delivered_send_index FROM {store.t.follow_up_queue}",
            )
            tasks = [(t.get_name(), t.done(), [frame.f_lineno for frame in t.get_stack()])
                     for t in runner.runtime.worker._tasks]
            raise AssertionError((rows, inbox, runner.runtime.worker.last_error,
                                  runner.runtime.worker._active.keys(), tasks))
        receipt = await handle.cancel("stop running continuation")
        assert receipt.state == "cancel_requested" and receipt.send_index == 2
        release.set()
        result = await runner.runtime.client.wait(
            handle.input_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert result.status == "cancelled" and result.send_index == 2
        continuations = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_continue'",
            (handle.target.session_id,),
        )
        assert len(continuations) == 1
    finally:
        release.set()
        await runner.aclose()


async def test_halt_late_result_is_journaled_without_advancing_finished_graph(registry_store, monkeypatch):
    store, _ = registry_store
    await _exercise_halt_late_result(store, store._db, monkeypatch)


async def _exercise_halt_late_result(store, raw_db, monkeypatch, *, measure=nullcontext):
    from power_loop.workflow import journal
    from power_loop.workflow import runner as workflow_runner

    started, release = asyncio.Event(), asyncio.Event()

    class Halt(Model):
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                await started.wait()
                raise RuntimeError("business failure")
            if request.system_prompt.startswith("B"):
                started.set()
                await release.wait()
                return LLMResponse(raw_text="late B")
            return LLMResponse(raw_text="parent acknowledged")

    runner = await RunningAgent(store, Halt(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "late halt", "root": {
                "type": "parallel", "on_error": "halt", "branches": [leaf("A"), leaf("B")],
            }},
            message_id="late-halt", actor=runner.actor,
        )
        await asyncio.wait_for(started.wait(), 5)
        try:
            result = await handle.wait(deadline_ms=now_ms() + 10000)
        except TimeoutError:
            states = await raw_db.fetchall(
                f"SELECT session_id,execution_state,active_send_index,cancel_send_index FROM {store.t.session_state}",
            )
            raise AssertionError(states)
        assert result.name == "late halt" and result.status == "failed"
        await runner.finish(handle.input_ref)
        driver_sid = handle.target.session_id
        saved_before = await journal.get(store, raw_db, driver_sid, journal.run_key(handle.run_id))
        assert saved_before["status"] == "failed" and saved_before["aggregate_delivered"]

        async def read_nodes():
            return await journal.nodes(store, raw_db, driver_sid)

        async def read_aggregates():
            return await raw_db.fetchall(
                f"SELECT item_id,content_json FROM {store.t.follow_up_queue} "
                "WHERE session_id=? AND kind='workflow_result'",
                (runner.target.session_id,),
            )

        nodes_before = await read_nodes()
        [a] = [node for node in nodes_before if node["path"] == "A"]
        [before_b] = [node for node in nodes_before if node["path"] == "B"]
        b_ref = InputRef(**before_b["input_ref"])
        assert before_b["result"] is None and before_b["late_results"] == []
        [first_aggregate] = await read_aggregates()
        aggregate_ref = InputRef(store.store_id, runner.target.session_id, first_aggregate["item_id"])
        # Finish the failed graph's original parent delivery before resetting the write meter.
        parent = await runner.runtime.client.wait(
            aggregate_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert parent.status == "completed" and parent.input_ref == aggregate_ref
        await runner.finish(aggregate_ref)
        await runner.finish(InputRef(**a["input_ref"]))
        committed = asyncio.Event()
        commits = []
        finish_late = workflow_runner._finish_late

        async def observe_finish(worker, context, refs):
            final = await finish_late(worker, context, refs)
            if worker is runner.runtime.worker and context.session_id == driver_sid:
                commits.append(tuple(refs))
                committed.set()
            return final

        with monkeypatch.context() as patch:
            patch.setattr(workflow_runner, "_finish_late", observe_finish)
            async with measure():
                release.set()
                async with asyncio.timeout(10):
                    await committed.wait()
                    await runner.drain()

        [(late_ref,)] = commits
        late_row = await raw_db.fetchone(
            f"SELECT kind,terminal_status,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (driver_sid, late_ref.message_id),
        )
        assert late_ref.session_id == driver_sid
        assert late_row["kind"] == "workflow_node_result" and late_row["terminal_status"] == "completed"
        routed = json.loads(late_row["content_json"])
        assert routed["status"] == "cancelled" and routed["input_ref"] == plain(b_ref)
        [b] = [entry for entry in await read_nodes() if entry["path"] == "B"]
        [late] = b["late_results"]
        assert b["input_ref"] == late["input_ref"] == plain(b_ref)
        assert b["result"] is None and late["status"] == "cancelled"
        inputs = runner.runtime.client.inputs
        b_result = await inputs.get_result(b_ref, actor=runner.actor)
        assert b_result.input_ref == b_ref and b_result.status == "cancelled"
        persisted = await inputs.get_result(handle.input_ref, actor=runner.actor)
        assert persisted.input_ref == handle.input_ref and persisted.target == handle.target
        assert persisted.status == result.status and persisted.result_seq == saved_before["result_seq"]
        assert plain(persisted.structured_output) == saved_before["result"]
        aggregates = await read_aggregates()
        [aggregate] = aggregates
        assert aggregate == first_aggregate
        body = json.loads(aggregate["content_json"])
        assert body["input_ref"] == plain(handle.input_ref) and body["status"] == "failed"
        return {
            "b_input_ref_before_release": plain(b_ref), "b_late_before_release": len(before_b["late_results"]),
            "b_result": b["result"], "b_late_results": b["late_results"],
            "b_persisted_status": b_result.status, "driver_late_input_ref": plain(late_ref),
            "driver_late_terminal_status": late_row["terminal_status"],
            "driver_original_input_ref": plain(persisted.input_ref), "driver_original_status": persisted.status,
            "driver_original_result_seq": persisted.result_seq, "parent_aggregate_count": len(aggregates),
            "parent_aggregate_input_ref": plain(aggregate_ref),
        }
    finally:
        release.set()
        await runner.aclose()


@pytest.mark.parametrize("supplied", (True, False))
async def test_public_workflow_rejects_closed_parent_and_keeps_existing_driver(
    registry_store, supplied,
):
    """The public workflow entry uses the same creation check: an independent driver
    lets the parent close, then a second workflow call on the closed parent is rejected
    without touching the existing driver, its journal or start input."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "closed-workflow-parent")
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    kwargs = {"descriptor": DESCRIPTOR} if supplied else {}
    spec = {"name": "independent driver", "root": leaf("A")}
    first = await client.workflow(
        parent, spec, message_id="first-run", actor=ACTOR, **kwargs,
    )
    await port.close(parent, actor=ACTOR)
    # Deterministic retry of the existing driver is rejected by the parent lifecycle
    # before the creation identity/idempotency check.
    with pytest.raises(RuntimeRequestError) as retried:
        await client.workflow(parent, spec, message_id="first-run", actor=ACTOR, **kwargs)
    assert retried.value.code == "session_closed"
    # A new driver under the same closed parent is rejected as well.
    with pytest.raises(RuntimeRequestError) as rejected:
        await client.workflow(
            parent, {"name": "after close", "root": leaf("A")},
            message_id="second-run", actor=ACTOR, **kwargs,
        )
    assert rejected.value.code == "session_closed"
    drivers = await store._db.fetchall(
        f"SELECT session_id,lifecycle FROM {store.t.sessions} "
        "WHERE parent_session_id=? AND session_kind='workflow_driver' "
        "ORDER BY session_id", (parent.session_id,),
    )
    assert [row["session_id"] for row in drivers] == [first.target.session_id]
    assert drivers[0]["lifecycle"] == "open"
    start = await store._db.fetchone(
        f"SELECT status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (first.target.session_id, first.input_ref.message_id),
    )
    assert start == {"status": "pending"}
    journal = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.session_runtime_state} WHERE session_id=?",
        (first.target.session_id,),
    )
    assert int(journal["n"]) == 1


@pytest.mark.parametrize("supplied", (True, False))
async def test_submit_descriptor_facts_retry_conflict_and_authorization(registry_store, monkeypatch, supplied):
    store, _ = registry_store
    client = RuntimeClient(store=store, transport=MemoryMessageTransport())
    port = client.inputs
    parent = await target(port, "descriptor-parent")
    explicit = replace(DESCRIPTOR, config={"chosen": "explicit"})
    kwargs = {"descriptor": explicit} if supplied else {}
    outside_parent_reads = []
    original = port._session

    async def read(reader, sid):
        if reader is port.db and sid == parent.session_id:
            outside_parent_reads.append(sid)
        return await original(reader, sid)

    monkeypatch.setattr(port, "_session", read)
    spec = {"name": "descriptor facts", "root": leaf("A")}
    first = await client.workflow(parent, spec, message_id="same", actor=ACTOR, **kwargs)
    assert len(outside_parent_reads) == (0 if supplied else 1)
    row = await original(port.db, first.target.session_id)
    descriptor = json.loads(row["descriptor_json"])
    expected = explicit if supplied else DESCRIPTOR
    assert descriptor["kind"] == "workflow_driver"
    assert {k: v for k, v in descriptor["config"].items() if k != "workflow"} == dict(expected.config)
    assert descriptor["config"]["workflow"]["spec"]["name"] == spec["name"]
    retry = await client.workflow(parent, spec, message_id="same", actor=ACTOR, **kwargs)
    assert retry.target == first.target and retry.input_ref == first.input_ref
    with pytest.raises(RuntimeRequestError) as conflict:
        await client.workflow(parent, {**spec, "name": "changed"}, message_id="same", actor=ACTOR, **kwargs)
    assert conflict.value.code == "creation_conflict"
    with pytest.raises(RuntimeRequestError) as denied:
        await client.workflow(parent, spec, message_id="unauthorized",
                              actor=replace(ACTOR, tenant_id="other-tenant"), **kwargs)
    assert denied.value.code == "forbidden"
    children = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE parent_session_id=?", (parent.session_id,))
    assert int(children["n"]) == 1
