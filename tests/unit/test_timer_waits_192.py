"""Atomic finite-wait evidence, original owner fencing and no premature success."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import LLMResponse, LLMTokenUsage, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import timer_waits as waits
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_query import list_jobs
from power_loop.runtime.task_runtime import _SEND

from .test_job_registry_192 import registry_store, runtime
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "runtime", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def waiting(runtime):
    store, manager, main, root, root_lease, connect, now = runtime
    child = await manager.register_job(main, kind="subagent", parent=root_lease, independent=True,
                                      system_key="wait-owner")
    sid = await store.create_session(parent_session_id=main.session_id,
        metadata={"execution_role": "subagent", "standalone_subagent": True})
    child = await manager.bind_session(main, child.job.job_id, sid)
    actor = replace(main, session_id=sid)
    owner = await manager.claim_job(actor, child.job_id, executor_id="child", boot_id="boot", expected_version=child.version)
    source = await manager.register_job(actor, kind="send", parent=owner, session_id=sid, system_key="wait-source",
                                        request={"send_index": 1})
    caller = await manager.claim_job(actor, source.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
    seq = await store.append_message(sid, role="assistant", send_index=1, round_index=0, tool_calls=[
        {"id": "arm-wait", "type": "function", "function": {"name": "wait", "arguments": "{}"}}])
    ref = await store.message_ref(sid, seq)
    c = SimpleNamespace(store=store, manager=manager, main=main, root=root, actor=actor,
        owner=owner, caller=caller, ref=ref, sid=sid, connect=connect, now=now,
        request=waits.WaitRequest(now[0] + 100, "  Original reason\n", "Continue the original task: 42."),
        policy=waits.WaitPolicy(now[0] + 10000))

    async def register(**changes):
        return await waits.register_wait(manager, changes.pop("actor", actor), caller=changes.pop("caller", caller),
            call_ref=changes.pop("call_ref", ref), tool_call_id=changes.pop("tool_call_id", "arm-wait"),
            request=changes.pop("request", c.request), policy=changes.pop("policy", c.policy), **changes)

    c.register = register
    return c


async def receipt(c, registration):
    seq = await c.store.append_message(c.sid, role="tool", name="wait", content="accepted, not completed",
        tool_call_id="arm-wait", send_index=1, round_index=0)
    return await c.manager.reconcile_receipt(c.actor, registration.operation.operation_id,
                                            await c.store.message_ref(c.sid, seq))


async def test_original_wait_is_required_and_visible_and_replays_after_source_closed(waiting):
    c = waiting
    registered = await c.register()
    await receipt(c, registered)
    job = registered.job
    assert job.kind == "timer" and job.blocks_completion and job.control_owner_job_id == c.owner.job_id
    assert job.completion_scope_job_id == c.owner.job_id and job.parent_job_id == c.caller.job_id
    saved = await waits.read_wait(c.manager, c.main, job.job_id)
    assert saved["binding"]["main_session_id"] == c.main.session_id
    assert saved["request"]["request"]["note"] == c.request.note
    assert saved["call_ref"]["id"] != saved["receipt_ref"]["id"]
    page = await list_jobs(c.manager, c.main, kinds=["timer"])
    assert [j["job_id"] for j in page["nodes"] if not j["context_only"]] == [job.job_id]
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
    again = await c.register()
    assert again.replayed and again.job.job_id == job.job_id
    assert await waits.read_wait(JobManager(await c.connect()), c.main, job.job_id) == saved
    c.now[0] += 101
    assert (await waits.prepare_due(c.manager, c.main, job.job_id))["status"] == "decision_pending"
    # Omitting the legacy opt-in flag cannot release this new required wait.
    with pytest.raises(JobError, match="required_execution_unsettled"):
        await c.manager.record_execution_result(c.actor, c.owner, outcome="succeeded", result={"text": "premature"},
                                                resource_state="stopped")
    assert (await c.manager.get_job(c.main, c.owner.job_id)).outcome is None


async def test_due_event_is_single_atomic_evidence_and_not_notification_or_completion(waiting):
    c = waiting
    registered = await c.register()
    await receipt(c, registered)
    armed = await waits.prepare_due(c.manager, c.main, registered.job.job_id)
    assert armed["status"] == "armed"
    c.now[0] += 101
    results = await asyncio.gather(*(waits.prepare_due(c.manager, c.main, registered.job.job_id) for _ in range(5)))
    assert all(r == results[0] for r in results)
    assert results[0]["status"] == "decision_pending" and not results[0]["execution_authority"]
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_due'")
    assert len(events) == 1 and events[0]["delivery_status"] == "pending_delivery"
    job = await c.manager.get_job(c.main, registered.job.job_id)
    assert job.version == 2 and job.generation == 0 and job.resource_state == "not_started" and job.outcome is None
    assert (await c.store.inbox_counts(c.main.session_id))["pending"] == 0


@pytest.mark.parametrize("fault", ["caller", "caller_bool", "caller_string", "call_id", "send_index", "actor", "owner_expired", "owner_cancelled", "main_role"])
async def test_bad_original_authority_or_call_cannot_create_wait(waiting, fault):
    c = waiting
    changes = {}
    if fault == "caller":
        changes["caller"] = replace(c.caller, generation=2)
    elif fault == "caller_bool":
        changes["caller"] = replace(c.caller, generation=True)
    elif fault == "caller_string":
        changes["caller"] = replace(c.caller, generation="1")
    elif fault == "call_id":
        changes["tool_call_id"] = "not-the-call"
    elif fault == "send_index":
        await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET send_index=2 WHERE session_id=? AND seq=?", (c.sid, c.ref.seq))
        changes["call_ref"] = await c.store.message_ref(c.sid, c.ref.seq)
    elif fault == "actor":
        changes["actor"] = c.main
    elif fault == "owner_expired":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=1 WHERE job_id=?", (c.owner.job_id,))
    elif fault == "owner_cancelled":
        await c.manager.request_cancel(c.main, c.owner.job_id, system_key="stop-wait", reason="stop")
    else:
        await c.store.merge_session_metadata(c.main.session_id, {"execution_role": "subagent"})
    with pytest.raises(JobError):
        await c.register(**changes)
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='timer'") == []


@pytest.mark.parametrize("changes", [dict(due_at_ms=1), dict(due_at_ms=10000000)])
async def test_due_time_must_fit_trusted_owner_deadline(waiting, changes):
    c = waiting
    with pytest.raises(JobError, match="wait_deadline_out_of_bounds"):
        await c.register(request=replace(c.request, **changes))


@pytest.mark.parametrize("field,value", [("due_at_ms", True), ("due_at_ms", 0), ("note", " "),
    ("note", "\x00"), ("continuation", ""), ("continuation", "x" * 64001), ("continuation", "\ud800")])
async def test_invalid_request_types_and_text_are_rejected_before_registration(field, value):
    with pytest.raises(JobError):
        waits.WaitRequest(**{**dict(due_at_ms=100, note="reason", continuation="continue"), field: value})


@pytest.mark.parametrize("point", ["insert_job", "insert_operation"])
async def test_failed_registration_rolls_back_required_child_and_evidence(waiting, monkeypatch, point):
    c = waiting
    async with c.manager._write() as tx:
        transaction_type = type(tx)
    original = transaction_type.execute

    async def fail(tx, query, params=()):
        result = await original(tx, query, params)
        if ((point == "insert_job" and query.startswith("INSERT INTO " + c.manager.jobs))
                or (point == "insert_operation" and query.startswith("INSERT INTO " + c.manager.events))):
            raise RuntimeError("wait registration write failed")
        return result

    monkeypatch.setattr(transaction_type, "execute", fail)
    with pytest.raises(RuntimeError, match="registration write failed"):
        await c.register()
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='timer'") == []


@pytest.mark.parametrize("fault", ["missing_receipt", "cancelled", "expired", "owner_changed", "late", "foreign"])
async def test_due_event_rechecks_current_state_and_does_not_revive_wait(waiting, fault):
    c = waiting
    registered = await c.register()
    if fault != "missing_receipt":
        await receipt(c, registered)
    c.now[0] += 101
    actor = c.main
    if fault == "cancelled":
        await c.manager.request_cancel(c.main, registered.job.job_id, system_key="stop-due", reason="stop")
    elif fault == "expired":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=1 WHERE job_id=?", (c.owner.job_id,))
    elif fault == "owner_changed":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (c.owner.job_id,))
    elif fault == "late":
        c.now[0] += 10000
    elif fault == "foreign":
        actor = replace(c.main, principal_id="foreign")
    with pytest.raises(JobError):
        await waits.prepare_due(c.manager, actor, registered.job.job_id)
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_due'") == []


async def test_changed_original_intent_conflicts_without_an_additional_wait(waiting):
    c = waiting
    first = await c.register()
    with pytest.raises(JobError, match="operation_conflict"):
        await c.register(request=replace(c.request, continuation="a replacement task"))
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='timer'")
    assert row["job_id"] == first.job.job_id


async def test_existing_due_event_cannot_authorize_delivery_after_wait_is_cancelled(waiting):
    c = waiting
    registered = await c.register()
    await receipt(c, registered)
    c.now[0] += 101
    first = await waits.prepare_due(c.manager, c.main, registered.job.job_id)
    await c.manager.request_cancel(c.main, registered.job.job_id, system_key="cancel-pending-decision", reason="stop")
    with pytest.raises(JobError):
        await waits.prepare_due(c.manager, c.main, registered.job.job_id)
    [event] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_due'")
    assert event["event_id"] == first["event_id"]
    assert (await c.store.inbox_counts(c.main.session_id))["pending"] == 0


@pytest.mark.parametrize("fault", ["missing_creation", "erased_wait_marker"])
async def test_missing_wait_creation_evidence_never_removes_the_required_success_barrier(waiting, fault):
    c = waiting
    registered = await c.register()
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
    if fault == "missing_creation":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (registered.operation.operation_id,))
    else:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
            (json.dumps({"job_id": registered.job.job_id, "status": "queued"}), registered.operation.operation_id))
    with pytest.raises(JobError):
        await c.manager.record_execution_result(c.actor, c.owner, outcome="succeeded", result={"text": "false success"},
                                                resource_state="stopped")
    assert (await c.manager.get_job(c.main, c.owner.job_id)).outcome is None


@pytest.mark.parametrize("stage", ["creation", "due"])
@pytest.mark.parametrize("fault", ["child_role", "standalone_flag", "main_missing", "main_marker"])
async def test_wait_rechecks_physical_child_and_main_identity(waiting, stage, fault):
    c = waiting
    registered = None
    if stage == "due":
        registered = await c.register()
        await receipt(c, registered)
    if fault == "child_role":
        await c.store.merge_session_metadata(c.sid, {"execution_role": "main"})
    elif fault == "standalone_flag":
        await c.store.merge_session_metadata(c.sid, {"standalone_subagent": False})
    elif fault == "main_missing":
        await c.store.close_session(c.main.session_id, cascade=False)
    else:
        await c.store.merge_session_metadata(c.main.session_id, {"standalone_subagent": True})
    with pytest.raises(JobError):
        if stage == "creation":
            await c.register()
        else:
            c.now[0] += 101
            await waits.prepare_due(c.manager, c.actor, registered.job.job_id)


@pytest.mark.parametrize("fault", ["bool_version", "unknown_version", "extra_field", "request_shape", "policy_field"])
async def test_untyped_or_extended_raw_wait_request_cannot_bypass_the_sdk(waiting, fault):
    from dataclasses import asdict
    c = waiting
    raw = {"managed_wait": 1, "request": asdict(c.request), "policy": asdict(c.policy)}
    if fault == "bool_version":
        raw["managed_wait"] = True
    elif fault == "unknown_version":
        raw["managed_wait"] = 2
    elif fault == "extra_field":
        raw["main_session_id"] = "forged-target"
    elif fault == "request_shape":
        raw["request"] = []
    else:
        raw["policy"]["infinite"] = True
    with pytest.raises(JobError):
        await c.manager.register_job(c.actor, kind="timer", title="untyped wait", session_id=c.sid,
            parent=c.caller, call_ref=c.ref, tool_call_id="arm-wait", request=raw)
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='timer'") == []


@pytest.mark.parametrize("field,value", [("deadline_at_ms", True), ("deadline_at_ms", 0),
    ("max_wait_ms", False), ("max_wait_ms", 0), ("max_wait_ms", 30 * 86400000 + 1)])
async def test_invalid_trusted_deadline_policy_is_not_coerced(field, value):
    with pytest.raises(JobError):
        waits.WaitPolicy(**{**dict(deadline_at_ms=10000, max_wait_ms=1000), field: value})


@pytest.mark.parametrize("fault", ["main", "source", "owner", "route", "request", "scope"])
async def test_rehashed_binding_cannot_redirect_original_wait_or_erase_its_scope(waiting, fault):
    c = waiting
    registered = await c.register()
    await receipt(c, registered)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_id=?",
                                         (registered.operation.operation_id,))
    data = json.loads(event["payload_json"])
    binding = data["wait_binding"]
    if fault == "main":
        binding["main_session_id"] = c.sid
    elif fault == "source":
        binding["source"]["generation"] = 2
    elif fault == "owner":
        binding["owner"]["boot_id"] = "different-boot"
    elif fault == "route":
        binding["route"]["kind"] = "main"
    elif fault == "request":
        data["request"]["request"]["continuation"] = "new work"
    else:
        binding["completion_scope_job_id"] = c.root.job.job_id
    from power_loop.runtime.jobs import _hash
    binding["evidence_hash"] = _hash({k: v for k, v in binding.items() if k != "evidence_hash"})
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
                               (json.dumps(data), event["event_id"]))
    with pytest.raises(JobError):
        await waits.read_wait(c.manager, c.main, registered.job.job_id)


@pytest.mark.parametrize("point", ["job_version", "due_event"])
async def test_due_event_write_failure_rolls_back_version_and_delivery_responsibility(waiting, monkeypatch, point):
    c = waiting
    registered = await c.register()
    await receipt(c, registered)
    c.now[0] += 101
    async with c.manager._write() as tx:
        transaction_type = type(tx)
    execute = transaction_type.execute

    async def fail(tx, sql, params=()):
        result = await execute(tx, sql, params)
        if ((point == "job_version" and sql.startswith("UPDATE " + c.manager.jobs))
                or (point == "due_event" and sql.startswith("INSERT INTO " + c.manager.events))):
            raise RuntimeError("due event write failed")
        return result

    monkeypatch.setattr(transaction_type, "execute", fail)
    with pytest.raises(RuntimeError, match="due event write failed"):
        await waits.prepare_due(c.manager, c.main, registered.job.job_id)
    assert (await c.manager.get_job(c.main, registered.job.job_id)).version == 1
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_due'") == []


async def _exercise_node_wait(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, parent, state, _ = managed_workflow
    state["spec"]["root"]["steps"][0]["spec"]["tools"] = ["arm_wait"]
    ready, release = asyncio.Event(), asyncio.Event()
    registrations = []
    original = model.complete

    async def arm():
        binding = _SEND.get()
        sid = get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, runtime.actor(sid), caller=binding.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 250, "Node decision", "Continue original BUILD"),
            policy=waits.WaitPolicy(now + 10000))
        registrations.append(registration)
        return "accepted, pending owner decision"

    async def complete(request, **kwargs):
        result = await original(request, **kwargs)
        if "ROLE-BUILD" in (request.system_prompt or ""):
            if model.counts["BUILD"] == 1:
                return LLMResponse(raw_text="", tool_calls=[{"id": "node-wait-call", "type": "function",
                    "function": {"name": "arm_wait", "arguments": "{}"}}],
                    token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
            ready.set()
            await release.wait()
        return result

    monkeypatch.setattr(model, "complete", complete)
    loop.tool_registry.register(ToolDefinition(name="arm_wait", description="Request a decision",
                                input_schema={"type": "object"}, pricing=Free("test_fixture")), arm)
    try:
        await loop.send("start workflow", parent)
        model.release["BUILD"].set()
        await asyncio.wait_for(ready.wait(), 5)
        if not registrations:
            bindings = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
            raw_messages = []
            for binding in bindings:
                raw_messages.extend([m.content for m in await store.load_all_messages(json.loads(binding["payload_json"])["session_id"]) if m.role == "tool"])
            pytest.fail("Wait admission was not accepted: " + repr(raw_messages))
        [registration] = registrations
        saved = await waits.read_wait(manager, runtime.actor(parent), registration.job.job_id)
        await asyncio.sleep(max(0, (saved["request"]["request"]["due_at_ms"] - manager.now()) / 1000) + .01)
        due = await waits.prepare_due(manager, runtime.actor(parent), registration.job.job_id)
        route = due["payload"]["wait"]["binding"]["route"]
        assert route["kind"] == "workflow_node" and route["node"]["node_id"] == "BUILD"
        assert route["node"]["session_id"] == registration.job.session_id
        assert route["workflow"]["job_id"] == state["handles"][0].job_id
        assert due["payload"]["wait"]["receipt_ref"]["tool_call_id"] == "node-wait-call"
        assert model.counts["REVIEW"] == 0 and not state["notifications"]
    finally:
        for registration in registrations:
            await manager.request_cancel(runtime.actor(parent), registration.job.job_id,
                                          system_key="cleanup-node-wait", reason="test cleanup")
            await manager.record_unstarted_result(runtime.actor(registration.job.created_by_session_id),
                registration.job.job_id, outcome="cancelled", result={"reason": "test cleanup"})
        release.set()


async def test_actual_workflow_node_wait_keeps_its_driver_route_and_blocks_review(managed_workflow, monkeypatch):
    await _exercise_node_wait(managed_workflow, monkeypatch)


@pytest.mark.parametrize("field", ["attempt", "workflow_generation"])
async def test_boolean_alias_does_not_match_the_frozen_integer_node_route(managed_workflow, monkeypatch, field):
    original = waits.prepare_due

    async def corrupt(manager, actor, job_id):
        [row] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND action='create'", (job_id,))
        data = json.loads(row["payload_json"])
        binding = data["wait_binding"]
        if field == "attempt":
            assert binding["route"]["node"]["attempt"] == 1
            binding["route"]["node"]["attempt"] = True
        else:
            assert binding["route"]["workflow"]["generation"] == 1
            binding["route"]["workflow"]["generation"] = True
        from power_loop.runtime.jobs import _hash
        binding["evidence_hash"] = _hash({k: v for k, v in binding.items() if k != "evidence_hash"})
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), row["event_id"]))
        try:
            with pytest.raises(JobError):
                await original(manager, actor, job_id)
        finally:
            await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (row["payload_json"], row["event_id"]))
        return await original(manager, actor, job_id)

    monkeypatch.setattr(waits, "prepare_due", corrupt)
    await _exercise_node_wait(managed_workflow, monkeypatch)
