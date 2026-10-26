"""Active workflow node steering uses exact durable routes and original sessions."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop import LLMResponse, LLMTokenUsage, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import node_inputs
from power_loop.runtime.jobs import JobError
from power_loop.workflow import runner

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import leaf, managed_workflow, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def target(c):
    rows = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='node_session'")
    return [(r, json.loads(r["payload_json"])) for r in rows]


async def client(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, parent, state, connect = managed_workflow
    c = SimpleNamespace(store=store, manager=manager, runtime=runtime, loop=loop, model=model,
                        parent=parent, state=state, connect=connect, replies=[], seen=[])
    original = model.complete
    pending_call = False

    async def complete(request, **kwargs):
        nonlocal pending_call
        sid = get_session_id()
        if sid == parent and pending_call:
            pending_call = False
            return LLMResponse(raw_text="", tool_calls=[{"id": "node-nudge-"+str(len(c.replies)), "type": "function",
                "function": {"name": "nudge_node", "arguments": "{}"}}],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        if sid != parent:
            c.seen.append((sid, str(request.messages)))
        response = await original(request, **kwargs)
        # The boundary may steer after this first request was already constructed.
        if sid != parent and "NUDGED-43" in str(request.messages):
            response.raw_text = "NUDGED-43"
        return response

    monkeypatch.setattr(model, "complete", complete)

    async def issue(*, sid=None, node_id="BUILD", run_id=None, message="NUDGED-43", handler=None):
        nonlocal pending_call
        async def nudge():
            if handler:
                value = await handler()
            else:
                value = await runner.steer_node(run_id or c.run_id, node_id, sid or c.sid, message)
            c.replies.append(value)
            return json.dumps(value)
        loop.tool_registry.register(ToolDefinition(name="nudge_node", description="Correct an active node",
            input_schema={"type": "object"}), nudge, overwrite=True)
        pending_call = True
        return await loop.send("Correct the running node", parent)

    c.issue = issue
    return c


@pytest.fixture
async def live_node(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    store, manager, runtime, loop, model, parent, state, connect = managed_workflow
    await loop.send("start workflow", parent)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [(row, data)] = await target(c)
    c.sid, c.job_id = data["session_id"], row["job_id"]
    c.data = data
    c.run_id = state["handles"][0].run_id
    c.engine = runner._ENGINES[c.run_id]
    c.binding = c.engine._task_nodes[c.sid]
    yield c


@pytest.mark.parametrize("fault", ["none", "lost_ack", "signal", "duplicate"])
async def test_nudge_is_atomic_processed_in_original_node_and_precedes_review(live_node, managed_workflow, monkeypatch, fault):
    c = live_node
    original = c.manager.accept_nudge
    admissions = []
    async def accept(*args, **kwargs):
        admissions.append(1)
        result = await original(*args, **kwargs)
        if fault == "lost_ack":
            raise RuntimeError("committed; lost ack")
        return result
    monkeypatch.setattr(c.manager, "accept_nudge", accept)
    steer_event = c.loop.steer_event
    if fault == "signal":
        def failed_signal(sid):
            raise RuntimeError("local notification unavailable")
        monkeypatch.setattr(c.loop, "steer_event", failed_signal)
    async def handler():
        first = await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
        if fault == "duplicate":
            again = await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
            assert again["operation_id"] == first["operation_id"] and again["replayed"]
        return first
    result = await c.issue(handler=handler)
    assert result.status == "completed" and len(c.replies) == 1
    receipt = c.replies[0]
    assert receipt["job_id"] == c.job_id and receipt["session_id"] == c.sid and receipt["run_id"] == c.run_id
    assert receipt["status"] == "accepted" and admissions == [1]
    assert c.model.counts["REVIEW"] == 0
    inbox = await c.manager.db.fetchall(f"SELECT item_id FROM {c.store.t.follow_up_queue} WHERE session_id=?", (c.sid,))
    assert len(inbox) == 1 and inbox[0]["item_id"] == "nudge:" + receipt["operation_id"]
    # Notification failure is not an acceptance failure; do not break model's own signal reads.
    monkeypatch.setattr(c.loop, "steer_event", steer_event)
    await settle(managed_workflow)
    job = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
    assert job.outcome == "succeeded" and job.result["text"] == "NUDGED-43"
    assert c.model.counts["REVIEW"] == 1
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
    assert event["delivery_status"] == "handled"
    op = await c.manager.get_operation(c.runtime.actor(c.parent), event["event_id"])
    assert op.call_ref["id"] != op.receipt_ref["id"] and op.receipt_ref["tool_call_id"].startswith("node-nudge-")
    [decision] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE job_id=? AND event_type='processing_decision'", (c.job_id,))
    assert json.loads(decision["payload_json"])["response_ref"]["session_id"] == c.sid
    assert not c.engine._task_nodes
    assert len(await target(c)) == 2  # BUILD and REVIEW, no replacement node/session


@pytest.mark.parametrize("wrong", ["node_id", "run_id", "sid", "empty", "type", "nul"])
async def test_invalid_or_unowned_target_is_rejected_without_input(live_node, wrong):
    c = live_node
    params = {"node_id": "OTHER"} if wrong == "node_id" else {"run_id": "missing"} if wrong == "run_id" else {"sid": c.parent} if wrong == "sid" else {"message": {"empty": "", "type": 2, "nul": "bad\x00"}[wrong]}
    await c.issue(**params)
    assert not c.replies
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0


@pytest.mark.parametrize("field", ["workflow_run_id", "node_id", "route", "attempt", "session_id"])
async def test_durable_descriptor_must_match_every_route_field(live_node, monkeypatch, field):
    c = live_node
    original = c.manager.accept_nudge
    async def corrupt(*args, **kwargs):
        kwargs["node"] = {**kwargs["node"], field: "forged"}
        return await original(*args, **kwargs)
    monkeypatch.setattr(c.manager, "accept_nudge", corrupt)
    await c.issue()
    assert not c.replies
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("fault", ["target_cancel", "source_cancel", "old_generation", "expired", "closed", "role"])
async def test_nudge_rechecks_fenced_execution_inside_admission(live_node, monkeypatch, fault):
    c = live_node
    original = c.manager.accept_nudge
    from dataclasses import replace
    errors = []
    async def break_admission(actor, lease, **kwargs):
        if fault in ("target_cancel", "source_cancel"):
            await c.manager.request_cancel(actor, lease.job_id if fault == "target_cancel" else kwargs["caller"].job_id, system_key="test-node-nudge-cancel", reason="test")
        elif fault == "old_generation":
            lease = replace(lease, generation=lease.generation+1)
        elif fault == "expired":
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=0 WHERE job_id=?", (lease.job_id,))
        elif fault == "closed":
            await node_inputs.close(c.runtime, c.binding, c.sid)
        else:
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET execution_role='workflow_node' WHERE job_id=?", (kwargs["caller"].job_id,))
        try:
            return await original(actor, lease, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    monkeypatch.setattr(c.manager, "accept_nudge", break_admission)
    if fault == "source_cancel":
        with pytest.raises(JobError, match="cancel_requested"):
            await c.issue()
    else:
        await c.issue()
    expected = {"target_cancel": "cancel_requested", "source_cancel": "cancel_requested", "old_generation": "stale_execution",
                "expired": "execution_lease_expired", "closed": "node_input_closed", "role": "nudge_target_mismatch"}
    assert errors == [expected[fault]]
    assert not c.replies
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("point", ["operation", "inbox"])
async def test_admission_write_failure_rolls_back_both_records(live_node, monkeypatch, point):
    c = live_node
    original = c.manager.db.dialect.insert_ignore
    insert = c.manager._insert_operation
    async def failed_insert(*args, **kwargs):
        result = await insert(*args, **kwargs)
        if kwargs.get("action") == "nudge":
            raise RuntimeError("operation write failed")
        return result
    if point == "operation":
        monkeypatch.setattr(c.manager, "_insert_operation", failed_insert)
    else:
        def failed_inbox(table, cols):
            if table == c.store.t.follow_up_queue:
                raise RuntimeError("inbox unavailable")
            return original(table, cols)
        monkeypatch.setattr(c.manager.db.dialect, "insert_ignore", failed_inbox)
    await c.issue()
    assert not c.replies
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0


@pytest.mark.parametrize("order", ["input_first", "close_first"])
async def test_final_collection_and_input_closure_have_no_lost_acceptance(live_node, managed_workflow, monkeypatch, order):
    c = live_node
    original = node_inputs.finish_attempt
    entered, release = asyncio.Event(), asyncio.Event()
    async def pause(runtime, engine, raw):
        if raw.get("session_id") == c.sid:
            if order == "close_first":
                await original(runtime, engine, raw)
            entered.set()
            await release.wait()
        if order == "input_first" or raw.get("session_id") != c.sid:
            await original(runtime, engine, raw)
    monkeypatch.setattr(node_inputs, "finish_attempt", pause)
    c.model.release["BUILD"].set()
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await c.issue()
        assert bool(c.replies) == (order == "input_first")
    finally:
        release.set()
    await settle(managed_workflow)
    job = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
    assert job.outcome == "succeeded"
    assert job.result["text"] == ("NUDGED-43" if order == "input_first" else "PROOF-42")
    [closed] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='node_input_closed'", (c.job_id,))
    assert json.loads(closed["payload_json"])["session_id"] == c.sid


async def test_missing_current_send_cannot_nudge_even_with_a_live_engine(live_node):
    c = live_node
    with pytest.raises(JobError, match="task_send_required"):
        await c.engine.steer_node("BUILD", c.sid, "NUDGED-43")


async def test_new_call_after_completion_cannot_reopen_node(live_node, managed_workflow):
    c = live_node
    await settle(managed_workflow)
    before = c.model.counts.copy()
    await c.issue()
    assert not c.replies and c.model.counts["BUILD"] == before["BUILD"]
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("field", ["workflow_run_id", "workflow_node_id", "task_node_route", "attempt", "task_owner_job_id", "execution_role"])
async def test_mutated_physical_session_cannot_redirect_a_durable_binding(live_node, field):
    c = live_node
    session = await c.store.get_session(c.sid)
    await c.store.merge_session_metadata(c.sid, {**session.metadata, field: "forged"})
    await c.issue()
    assert not c.replies
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("missing", ["event", "session"])
async def test_missing_binding_or_recipient_fails_closed(live_node, missing):
    c = live_node
    if missing == "event":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='node_session' AND job_id=?", (c.job_id,))
    else:
        await c.manager.db.execute(f"DELETE FROM {c.store.t.sessions} WHERE session_id=?", (c.sid,))
    await c.issue()
    assert not c.replies
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("fault", ["wrong_job", "wrong_type", "invalid_json", "list", "wrong_sid", "extra_generation"])
async def test_corrupt_closure_never_admits_input_or_overwrites_evidence(live_node, managed_workflow, fault):
    c = live_node
    marker = node_inputs.closed_id(c.manager, c.job_id, c.sid)
    payload = {"session_id": "other-sid" if fault == "wrong_sid" else c.sid}
    if fault == "extra_generation":
        payload["generation"] = 1
    raw = "{" if fault == "invalid_json" else "[]" if fault == "list" else json.dumps(payload)
    await c.manager.db.execute(f"INSERT INTO {c.manager.events} "
        "(event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
        (marker, "other-job" if fault == "wrong_job" else c.job_id,
         "operation" if fault == "wrong_type" else "node_input_closed", 1, raw, "not_applicable", c.manager.now()))
    before = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (marker,))
    try:
        await c.issue()
        assert not c.replies
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
        assert (await c.store.inbox_counts(c.sid))["pending"] == 0
        assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (marker,)) == before
    finally:
        # Restore only our injected damage so fixture cleanup can finish normally.
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (marker,))
    await settle(managed_workflow)
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).outcome == "succeeded"


@pytest.mark.parametrize("change", ["message", "node", "sid", "run"])
async def test_replaying_same_call_with_different_target_or_payload_is_a_conflict(live_node, change):
    c = live_node
    async def handler():
        result = await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
        args = [c.run_id, "BUILD", c.sid, "NUDGED-43"]
        args[{"run": 0, "node": 1, "sid": 2, "message": 3}[change]] = "changed"
        with pytest.raises(JobError, match="operation_conflict"):
            # The same engine entry can recall even when the supplied run is changed.
            await node_inputs.steer(c.loop, c.engine, *args)
        return result
    await c.issue(handler=handler)
    assert len(c.replies) == 1
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")) == 1


async def test_completed_exact_admission_can_be_recalled_after_engine_removal(live_node, managed_workflow):
    c = live_node
    async def handler():
        first = await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
        await settle(managed_workflow)
        assert c.run_id not in runner._ENGINES
        repeated = await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
        assert repeated["operation_id"] == first["operation_id"] and repeated["replayed"]
        return first
    await c.issue(handler=handler)
    assert len(c.replies) == 1
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")) == 1


@pytest.mark.parametrize("fault", ["decision_write", "input_removed", "input_changed", "wrong_receiver"])
async def test_unprocessed_input_cannot_release_review(live_node, managed_workflow, monkeypatch, fault):
    c = live_node
    if fault == "decision_write":
        original = c.manager.commit_processing_decision
        async def fail(actor, *args, **kwargs):
            if actor.session_id == c.sid:
                raise RuntimeError("decision persistence unavailable")
            return await original(actor, *args, **kwargs)
        monkeypatch.setattr(c.manager, "commit_processing_decision", fail)
    else:
        original = c.manager.prepare_processing
        async def prepare(actor, *args, **kwargs):
            if actor.session_id == c.sid:
                if fault == "input_removed":
                    kwargs["messages"] = []
                elif fault == "input_changed":
                    kwargs["messages"] = [{**m, "content": "rewritten"} for m in kwargs["messages"]]
                else:
                    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
                    payload = json.loads(event["payload_json"])
                    payload["request"]["session_id"] = c.parent
                    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
                        (json.dumps(payload), event["event_id"]))
            return await original(actor, *args, **kwargs)
        monkeypatch.setattr(c.manager, "prepare_processing", prepare)
    await c.issue()
    assert len(c.replies) == 1
    await settle(managed_workflow)
    node = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
    wf = await c.manager.get_job(c.runtime.actor(c.parent), c.state["handles"][0].job_id)
    assert c.model.counts["REVIEW"] == 0
    if fault == "decision_write":
        # A failed processing commit leaves the original send unreviewed.
        # The node cannot publish a result until that responsibility resolves.
        # The graph can report failure while retaining its unknown resources.
        assert node.outcome is None and node.result_status == "none"
        assert node.supervision_status == wf.supervision_status == "open"
        assert wf.outcome == "failed" and wf.resource_state == "unknown"
        assert node.status == "settling" and node.resource_state == "unknown"
    else:
        assert node.outcome == "failed" and wf.outcome == "failed"
    [op] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
    assert op["delivery_status"] != "handled"
    # The controller's separate exit writer must finish before fixture teardown
    # drops its tables; this does not settle the unresolved processing decision.
    async with asyncio.timeout(5):
        while c.runtime._adapter_tasks:
            await asyncio.gather(*tuple(c.runtime._adapter_tasks))
        assert await c.runtime.flush_control_checkpoints() == {
            "controller_retirement_pending": 0, "controller_task_exit_pending": 0,
        }


async def test_close_is_idempotent_and_survives_store_reopen(live_node):
    c = live_node
    await node_inputs.close(c.runtime, c.binding, c.sid)
    await node_inputs.close(c.runtime, c.binding, c.sid)
    from power_loop.runtime.jobs import JobManager
    reopened = JobManager(await c.connect())
    assert await reopened.db.fetchone(f"SELECT event_id FROM {reopened.events} WHERE event_id=?", (node_inputs.closed_id(reopened, c.job_id, c.sid),))
    await c.issue()
    assert not c.replies


async def test_parallel_foreach_same_name_routes_are_independently_addressed(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"] = {"type": "foreach", "id": "items", "as": "item", "items": [1, 2],
        "body": leaf("BUILD"), "parallel": True}
    await c.loop.send("parallel workflow", c.parent)
    async with asyncio.timeout(4):
        while len(await target(c)) != 2:
            await asyncio.sleep(0.01)
    c.run_id = c.state["handles"][0].run_id
    bindings = await target(c)
    assert bindings[0][1]["route"] != bindings[1][1]["route"]
    c.sid = bindings[0][1]["session_id"]
    await c.issue()
    assert len(c.replies) == 1 and c.replies[0]["job_id"] == bindings[0][0]["job_id"]
    await settle(managed_workflow)
    for row, data in bindings:
        job = await c.manager.get_job(c.runtime.actor(c.parent), row["job_id"])
        assert job.outcome == "succeeded" and job.result["text"] == ("NUDGED-43" if data["session_id"] == c.sid else "PROOF-42")
    [event] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE action='nudge'")
    assert json.loads(event["payload_json"])["request"]["node"]["route"] == bindings[0][1]["route"]


async def test_old_retry_session_closed_while_current_attempt_remains_addressable(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.model.hold = set()
    c.state["spec"]["root"] = leaf("BUILD", retry={"max_attempts": 2, "on": ["empty"]})
    original = c.model.complete
    entered, release = asyncio.Event(), asyncio.Event()
    async def retry(request, **kwargs):
        build = "ROLE-BUILD" in (request.system_prompt or "")
        if build and c.model.counts["BUILD"] == 1:
            entered.set()
            await release.wait()
        result = await original(request, **kwargs)
        if build and c.model.counts["BUILD"] == 1:
            result.raw_text = ""
            result.tool_calls = [{"id": "empty-pass", "type": "function", "function": {"name": "pass_turn", "arguments": '{"reason":"no output"}'}}]
        return result
    monkeypatch.setattr(c.model, "complete", retry)
    try:
        await c.loop.send("retry workflow", c.parent)
        await asyncio.wait_for(entered.wait(), 3)
        bindings = sorted(await target(c), key=lambda pair: pair[1]["attempt"])
        assert [data["attempt"] for _, data in bindings] == [1, 2]
        assert bindings[0][0]["job_id"] == bindings[1][0]["job_id"]
        c.run_id = c.state["handles"][0].run_id
        c.sid = bindings[0][1]["session_id"]
        await c.issue()
        assert not c.replies
        c.sid = bindings[1][1]["session_id"]
        await c.issue()
        assert len(c.replies) == 1
    finally:
        release.set()
    await settle(managed_workflow)
    node = await c.manager.get_job(c.runtime.actor(c.parent), bindings[0][0]["job_id"])
    assert node.outcome == "succeeded" and node.result["text"] == "NUDGED-43"
    [operation] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE action='nudge'")
    assert json.loads(operation["payload_json"])["request"]["node"]["attempt"] == 2


async def test_waiting_node_accepts_input_until_required_subagent_returns(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.model.hold = {"AUDIT"}
    c.model.spawn_from_build = True
    c.state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    await c.loop.send("node waits for audit", c.parent)
    await asyncio.wait_for(c.model.entered["AUDIT"].wait(), 3)
    [(row, data)] = await target(c)
    c.run_id, c.sid = c.state["handles"][0].run_id, data["session_id"]
    assert c.model.counts["REVIEW"] == 0
    await c.issue()
    assert len(c.replies) == 1
    await settle(managed_workflow)
    node = await c.manager.get_job(c.runtime.actor(c.parent), row["job_id"])
    assert node.outcome == "succeeded" and node.result["text"] == "NUDGED-43"
    [audit] = await c.manager.db.fetchall(f"SELECT result_status FROM {c.manager.jobs} WHERE kind='subagent'")
    assert audit["result_status"] == "handled" and c.model.counts["REVIEW"] == 1


async def test_failed_original_with_unhandled_input_does_not_retry_business(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"] = leaf("BUILD", retry={"max_attempts": 2, "on": ["failed"]})
    original = c.model.complete
    async def fail(request, **kwargs):
        if "ROLE-BUILD" in (request.system_prompt or ""):
            c.model.entered["BUILD"].set()
            await c.model.release["BUILD"].wait()
            raise RuntimeError("original provider failed")
        return await original(request, **kwargs)
    monkeypatch.setattr(c.model, "complete", fail)
    await c.loop.send("no replacement attempts for accepted inputs", c.parent)
    await asyncio.wait_for(c.model.entered["BUILD"].wait(), 3)
    [(row, data)] = await target(c)
    c.run_id, c.sid = c.state["handles"][0].run_id, data["session_id"]
    await c.issue()
    assert len(c.replies) == 1
    await settle(managed_workflow)
    assert len(await target(c)) == 1
    node = await c.manager.get_job(c.runtime.actor(c.parent), row["job_id"])
    assert node.outcome == "failed"
    [op] = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE action='nudge'")
    assert op["delivery_status"] != "handled"


async def test_live_caller_without_actual_invocation_cannot_accept(live_node):
    c = live_node
    from power_loop.runtime.invocation import invocation_context
    async def handler():
        with invocation_context(None, None, None):
            with pytest.raises(JobError, match="persisted_tool_call_required"):
                await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
        return {"status": "rejected"}
    await c.issue(handler=handler)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("mismatch", ["owner", "principal", "tenant", "descriptor", "bool_attempt", "float_attempt"])
async def test_actual_input_receiver_cannot_borrow_another_node_binding(live_node, mismatch):
    c = live_node
    await c.issue()
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
    [send] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send' AND session_id=? ORDER BY created_at DESC LIMIT 1", (c.sid,))
    send = dict(send)
    from dataclasses import replace
    actor = c.runtime.actor(c.sid)
    if mismatch == "owner":
        send["control_owner_job_id"] = "other-node"
    elif mismatch in ("tenant", "principal"):
        actor = replace(actor, **{mismatch+"_id": "other"})
    else:
        event = dict(event)
        data = json.loads(event["payload_json"])
        if mismatch == "descriptor":
            data["request"]["node"]["route"] = ["other-iteration"]
        else:
            data["request"]["node"]["attempt"] = True if mismatch == "bool_attempt" else 1.0
        event["payload_json"] = json.dumps(data)
    with pytest.raises(JobError, match="processing_input_mismatch"):
        await node_inputs.input_target(c.manager, c.manager.db, actor, send, event)


@pytest.mark.parametrize("attempt", [2, True, 1.0])
async def test_scope_guard_rechecks_immutable_input_descriptor(live_node, managed_workflow, attempt):
    c = live_node
    await c.issue()
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
    data = json.loads(event["payload_json"])
    data["request"]["node"]["attempt"] = attempt
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), event["event_id"]))
    job = await c.manager._row(c.manager.db, c.job_id)
    assert await node_inputs.descriptor(c.manager, c.manager.db, job, c.sid) == c.data
    from power_loop.runtime.processing import require_inputs_handled
    with pytest.raises(JobError, match="required_input_unhandled"):
        async with c.manager._write() as tx:
            await require_inputs_handled(c.manager, tx, job)


@pytest.mark.parametrize("attempt", [True, 1.0])
async def test_input_admission_does_not_coerce_descriptor_number_types(live_node, monkeypatch, attempt):
    c = live_node
    original = c.manager.accept_nudge
    async def changed(*args, **kwargs):
        kwargs["node"] = {**kwargs["node"], "attempt": attempt}
        return await original(*args, **kwargs)
    monkeypatch.setattr(c.manager, "accept_nudge", changed)
    await c.issue()
    assert not c.replies
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0


@pytest.mark.parametrize("accepted", [True, False])
async def test_negative_attempt_closure_preserves_pending_input_and_original_evidence(live_node, accepted):
    c = live_node
    if accepted:
        await c.issue()
    raw = {"session_id": c.sid, "status": "failed", "final_text": "original partial output", "usage": {"total_tokens": 10}}
    if accepted:
        with pytest.raises(JobError, match="required_input_unhandled") as raised:
            await node_inputs.finish_attempt(c.runtime, c.engine, raw)
        assert raised.value.execution_result == raw
    else:
        await node_inputs.finish_attempt(c.runtime, c.engine, raw)
    assert await c.manager.db.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=?", (node_inputs.closed_id(c.manager, c.job_id, c.sid),))
    await node_inputs.finish_attempt(c.runtime, c.engine, {"status": "skipped"})


async def test_node_admission_freezes_descriptor_before_await_including_receipt(live_node, monkeypatch):
    c = live_node
    original = c.manager.accept_nudge
    write = c.manager._write
    from contextlib import asynccontextmanager
    async def accept(*args, **kwargs):
        mutable = kwargs["node"]
        @asynccontextmanager
        async def mutate():
            async with write() as tx:
                mutable["workflow_run_id"] = "caller-mutated-after-await"
                yield tx
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_write", mutate)
            return await original(*args, **kwargs)
    monkeypatch.setattr(c.manager, "accept_nudge", accept)
    await c.issue()
    [reply] = c.replies
    assert reply["run_id"] == c.run_id
    operation = await c.manager.get_operation(c.runtime.actor(c.parent), reply["operation_id"])
    assert operation.result["request"]["node"]["workflow_run_id"] == c.run_id
    assert operation.result["run_id"] == c.run_id
