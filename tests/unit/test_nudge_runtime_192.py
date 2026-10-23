"""Atomic nudge admission and actual original-executor continuation."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import LLMResponse, LLMTokenUsage
from power_loop.core.agent_context import get_session_id, reset_session_id, set_session_id
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobActor, JobError, JobManager

from .test_job_registry_192 import call, registry_store, runtime
from .test_task_runtime_192 import finish, jobs, managed, receipt
from .test_workflow_task_runtime_192 import leaf, managed_workflow, settle

__all__ = ["registry_store", "runtime", "managed", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def nudge_case(runtime):
    store, manager, actor, _, supervisor, connect, now = runtime
    reg = await manager.register_job(actor, kind="subagent", parent=supervisor, independent=True, system_key="nudge-target")
    source = await manager.register_job(actor, kind="send", session_id=actor.session_id,
        system_key="nudge-caller", request={"send_index": 1})
    caller = await manager.claim_job(actor, source.job.job_id, executor_id="parent", boot_id="boot", expected_version=1)
    sid = await store.create_session(parent_session_id=actor.session_id,
        metadata={"execution_role": "subagent", "standalone_subagent": True})
    bound = await manager.bind_session(actor, reg.job.job_id, sid)
    child_actor = replace(actor, session_id=sid)
    target = await manager.claim_job(child_actor, bound.job_id, executor_id="child", boot_id="boot", expected_version=bound.version)
    seq = await store.append_message(actor.session_id, role="assistant", send_index=1, round_index=0,
        tool_calls=[{"id": "nudge-call", "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}])
    origin = await store.message_ref(actor.session_id, seq)
    send = await manager.register_job(child_actor, kind="send", session_id=sid, parent=target,
                                     system_key="receiver-send", request={"send_index": 1})
    receiver = await manager.claim_job(child_actor, send.job.job_id, executor_id="child", boot_id="boot", expected_version=1)

    async def accept(**changes):
        params = {"caller": caller, "call_ref": origin, "tool_call_id": "nudge-call", "session_id": sid, "message": "NUDGED-42"}
        params.update(changes)
        return await manager.accept_nudge(actor, target, **params)

    async def consume(operation, *, complete=True):
        rows = await store.inbox_claim(sid)
        seq = await store.append_message(sid, role="user", content="NUDGED-42", send_index=1,
            inbox={"claim_token": rows[0]["claim_token"], "ids": [r["id"] for r in rows]})
        snapshot = await manager.prepare_processing(child_actor, receiver, round_index=0, send_index=1,
            messages=[{"seq": seq, "role": "user", "content": "NUDGED-42"}])
        seq = await store.append_message(sid, role="assistant", content="NUDGED-42", send_index=1, round_index=0)
        response = await store.message_ref(sid, seq)
        ids = await manager.commit_processing_decision(child_actor, receiver, snapshot_id=snapshot, response_ref=response, reason="processed input")
        if complete:
            await manager.close_model_execution(child_actor, receiver, evidence={"exit": "completed",
                "last_assistant_ref": response.to_dict(), "processing_decisions": ids})
        return ids

    return SimpleNamespace(store=store, manager=manager, actor=actor, caller=caller, supervisor=supervisor, target=target, sid=sid,
        child_actor=child_actor, receiver=receiver, origin=origin, accept=accept, consume=consume, connect=connect, now=now)


async def test_atomic_nudge_receipt_call_and_inbox_survive_reopen_and_original_caller_close(nudge_case):
    c = nudge_case
    operation, replayed = await c.accept()
    assert not replayed and operation.action == "nudge"
    assert operation.result["job_id"] == c.target.job_id and operation.result["session_id"] == c.sid
    seq = await c.store.append_message(c.actor.session_id, role="tool", content="accepted", tool_call_id="nudge-call", send_index=1)
    await c.manager.bind_receipt(c.actor, operation.operation_id, await c.store.message_ref(c.actor.session_id, seq), caller=c.caller)
    found = await c.manager.operation_for_call(c.actor, c.origin, "nudge-call")
    assert found.operation_id == operation.operation_id and found.call_ref["id"] != found.receipt_ref["id"]
    await c.manager.close_model_execution(c.actor, c.caller)
    reopened = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    again, replayed = await reopened.accept_nudge(c.actor, c.target, caller=c.caller,
        call_ref=c.origin, tool_call_id="nudge-call", session_id=c.sid, message="NUDGED-42")
    assert replayed and again.operation_id == operation.operation_id
    assert (await c.store.inbox_counts(c.sid))["pending"] == 1


@pytest.mark.parametrize("conflict", ["message", "session", "target_generation"])
async def test_same_origin_changed_nudge_is_rejected_without_second_enqueue(nudge_case, conflict):
    c = nudge_case
    await c.accept()
    kwargs = {"message": "different"} if conflict == "message" else {"session_id": "different"}
    with pytest.raises(JobError, match="operation_conflict"):
        if conflict == "target_generation":
            await c.manager.accept_nudge(c.actor, replace(c.target, generation=c.target.generation+1), caller=c.caller,
                call_ref=c.origin, tool_call_id="nudge-call", session_id=c.sid, message="NUDGED-42")
        else:
            await c.accept(**kwargs)
    assert (await c.store.inbox_counts(c.sid))["pending"] == 1


@pytest.mark.parametrize("message", ["", "   \n", "\x00", "x" * 64001, None])
async def test_invalid_nudge_has_no_durable_admission(nudge_case, message):
    c = nudge_case
    with pytest.raises(JobError, match="invalid_nudge_message"):
        await c.accept(message=message)
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0


@pytest.mark.parametrize("barrier", ["target_cancel", "caller_cancel", "target_expired", "caller_expired", "target_closed", "wrong_sid", "missing_session", "wrong_role", "target_unknown"])
async def test_nudge_fences_both_caller_and_target_and_original_session(nudge_case, barrier):
    c = nudge_case
    if barrier in ("target_cancel", "caller_cancel"):
        await c.manager.request_cancel(c.actor, c.target.job_id if barrier == "target_cancel" else c.caller.job_id,
                                       system_key="stop", reason="stop")
    elif barrier in ("target_expired", "caller_expired"):
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=? WHERE job_id=?",
            (c.now[0]-1, c.target.job_id if barrier == "target_expired" else c.caller.job_id))
    elif barrier == "target_closed":
        await c.manager.record_execution_result(c.child_actor, c.target, outcome="failed", result={}, resource_state="stopped")
    elif barrier == "missing_session":
        await c.store.close_session(c.sid, cascade=False)
    elif barrier == "wrong_role":
        await c.store.merge_session_metadata(c.sid, {"execution_role": "main"})
    elif barrier == "target_unknown":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET resource_state='unknown' WHERE job_id=?", (c.target.job_id,))
    with pytest.raises(JobError):
        await c.accept(**({"session_id": "wrong"} if barrier == "wrong_sid" else {}))
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0


@pytest.mark.parametrize("which", ["caller", "target", "tenant", "actor"])
async def test_stale_or_foreign_proof_cannot_admit_nudge(nudge_case, which):
    c = nudge_case
    actor, caller, target = c.actor, c.caller, c.target
    if which == "caller":
        caller = replace(caller, generation=caller.generation+1)
    elif which == "target":
        target = replace(target, boot_id="old-boot")
    elif which == "tenant":
        actor = replace(actor, tenant_id="foreign")
    else:
        actor = JobActor(actor.tenant_id, actor.principal_id, await c.store.create_session())
    with pytest.raises(JobError):
        await c.manager.accept_nudge(actor, target, caller=caller, call_ref=c.origin,
                                    tool_call_id="nudge-call", session_id=c.sid, message="NUDGED-42")


@pytest.mark.parametrize("fault", ["rollback", "lost_ack"])
async def test_admission_and_inbox_are_atomic_and_exact_replay_never_duplicates(nudge_case, monkeypatch, fault):
    c = nudge_case
    original = c.manager._write

    @asynccontextmanager
    async def fail():
        async with original() as tx:
            yield tx
            if fault == "rollback":
                raise RuntimeError("rollback operation and inbox")
        if fault == "lost_ack":
            raise RuntimeError("commit ack lost")

    monkeypatch.setattr(c.manager, "_write", fail)
    with pytest.raises(RuntimeError):
        await c.accept()
    assert (await c.store.inbox_counts(c.sid))["pending"] == (0 if fault == "rollback" else 1)
    monkeypatch.setattr(c.manager, "_write", original)
    op, replayed = await c.accept()
    assert replayed == (fault == "lost_ack")
    same, repeated = await c.accept()
    assert repeated and same.operation_id == op.operation_id and (await c.store.inbox_counts(c.sid))["pending"] == 1


async def test_pending_or_history_only_input_blocks_success_until_original_receiver_decision_and_exit(nudge_case):
    c = nudge_case
    op, _ = await c.accept()
    with pytest.raises(JobError, match="required_input_unhandled"):
        await c.manager.record_execution_result(c.child_actor, c.target, outcome="succeeded", result={}, resource_state="stopped", require_children_stopped=True)
    await c.consume(op, complete=False)
    assert (await c.manager.get_job(c.child_actor, c.target.job_id)).result_status == "none"
    with pytest.raises(JobError, match="required_input_unhandled"):
        await c.manager.record_execution_result(c.child_actor, c.target, outcome="succeeded", result={}, resource_state="stopped", require_children_stopped=True)
    [decision] = await c.manager.db.fetchall(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE event_type='processing_decision'")
    payload = json.loads(decision["payload_json"])
    await c.manager.close_model_execution(c.child_actor, c.receiver, evidence={"exit": "completed",
        "last_assistant_ref": payload["response_ref"], "processing_decisions": [decision["event_id"]]})
    await c.manager.record_execution_result(c.child_actor, c.target, outcome="succeeded", result={"answer": "NUDGED-42"},
                                            resource_state="stopped", require_children_stopped=True)
    saved = await c.manager.get_job(c.child_actor, c.target.job_id)
    assert saved.outcome == "succeeded" and saved.result_status == "pending_delivery"


@pytest.mark.parametrize("first", ["nudge", "close", "cancel"])
async def test_admission_serializes_with_target_close_and_cancel(nudge_case, first):
    c = nudge_case
    original = c.manager._write
    entered, release = asyncio.Event(), asyncio.Event()
    initial = True

    @asynccontextmanager
    async def hold():
        nonlocal initial
        async with original() as tx:
            if initial:
                initial = False
                entered.set()
                await release.wait()
            yield tx

    async def close():
        return await c.manager.record_execution_result(c.child_actor, c.target, outcome="succeeded", result={}, resource_state="stopped", require_children_stopped=True)

    async def cancel():
        return await c.manager.request_cancel(c.actor, c.target.job_id, system_key="race", reason="stop")

    # A completed model send is stopped, so only the nudge/close race remains.
    await c.manager.close_model_execution(c.child_actor, c.receiver)
    c.manager._write = hold
    try:
        one = asyncio.create_task(c.accept() if first == "nudge" else close() if first == "close" else cancel())
        await asyncio.wait_for(entered.wait(), 2)
        two = asyncio.create_task(close() if first == "nudge" else c.accept())
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(one, two, return_exceptions=True)
    finally:
        release.set()
        c.manager._write = original
    assert not isinstance(results[0], BaseException)
    assert isinstance(results[1], JobError)
    assert (await c.store.inbox_counts(c.sid))["pending"] == (1 if first == "nudge" else 0)


async def issue_nudge(fixture, monkeypatch, *, message="NUDGED-42", repeat=False):
    _, _, _, loop, model, parent, _ = fixture
    accepted = await receipt(fixture)
    original = model.complete
    sent = False

    async def complete(request, **kwargs):
        nonlocal sent
        if get_session_id() == parent and not sent:
            sent = True
            model.root_calls += 1
            return LLMResponse(raw_text="", tool_calls=[{"id": "actual-nudge", "type": "function", "function": {
                "name": "spawn_agent", "arguments": json.dumps({"action": "nudge", "sub_session": accepted["session_id"], "message": message})}}],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        result = await original(request, **kwargs)
        if get_session_id() != parent and "NUDGED-42" in json.dumps(request.messages, default=str):
            result.raw_text = "NUDGED-42"
        return result

    monkeypatch.setattr(model, "complete", complete)
    receipts = []
    if repeat:
        original_nudge = delegation.nudge

        async def twice(*args, **kwargs):
            receipts.append(await original_nudge(*args, **kwargs))
            receipts.append(await original_nudge(*args, **kwargs))
            return receipts[0]

        monkeypatch.setattr(delegation, "nudge", twice)
    result = await loop.send("add a correction to the same child", parent)
    return accepted, result, receipts


@pytest.mark.parametrize("repeat", [False, True])
async def test_actual_nudge_retains_job_run_session_and_waits_for_original_receiver(managed, monkeypatch, repeat):
    store, manager, runtime, loop, model, parent, connect = managed
    runtime.lease_ms = 30000  # receiver identity/handling, not lease-expiry timing
    assert (await loop.send("delegate", parent)).status == "completed"
    await asyncio.wait_for(model.entered.wait(), 2)
    accepted, result, repeats = await issue_nudge(managed, monkeypatch, repeat=repeat)
    assert result.status == "completed"
    source_tools = [row for row in await store.load_all_messages(parent) if row.role == "tool"]
    steered = json.loads(source_tools[-1].content)
    assert steered["status"] == "accepted" and steered["job_id"] == accepted["job_id"]
    assert steered["run_id"] == accepted["run_id"] and steered["session_id"] == accepted["session_id"]
    operation = await manager.get_operation(runtime.actor(parent), steered["operation_id"])
    assert operation.receipt_ref["tool_call_id"] == "actual-nudge"
    assert operation.call_ref["id"] != operation.receipt_ref["id"]
    if repeat:
        assert len(repeats) == 2 and repeats[0]["operation_id"] == repeats[1]["operation_id"]
        assert not repeats[0]["replayed"] and repeats[1]["replayed"]
    assert (await manager.get_job(runtime.actor(parent), accepted["job_id"])).result is None
    await finish(managed)
    reopened = JobManager(await connect())
    saved = await reopened.get_job(runtime.actor(parent), accepted["job_id"])
    assert saved.outcome == "succeeded" and saved.result["final_text"] == "NUDGED-42"
    assert len([j for j in await jobs(managed) if j.kind == "subagent"]) == 1
    inputs = await manager.db.fetchall(f"SELECT delivery_status FROM {manager.events} WHERE action='nudge'")
    assert len(inputs) == 1 and inputs[0]["delivery_status"] == "handled"


async def test_new_nudge_after_completion_does_not_resurrect_child(managed, monkeypatch):
    _, _, _, loop, model, parent, _ = managed
    await loop.send("delegate", parent)
    await finish(managed)
    previous = model.child_calls
    accepted, result, _ = await issue_nudge(managed, monkeypatch)
    assert result.status == "completed" and model.child_calls == previous
    store = managed[0]
    tool = [m for m in await store.load_all_messages(parent) if m.role == "tool"][-1]
    assert "resume_result_unhandled" in tool.content
    assert len([j for j in await jobs(managed) if j.kind == "subagent"]) == 1
    assert accepted["session_id"] in {j.session_id for j in await jobs(managed) if j.kind == "subagent"}


@pytest.mark.parametrize("fault", ["lost_ack", "signal", "inbox_failure"])
async def test_actual_admission_failures_preserve_exact_result_and_never_repeat_admission(managed, monkeypatch, fault):
    store, manager, runtime, loop, model, parent, _ = managed
    runtime.lease_ms = 30000  # keep the injected admission failure as the only fault
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    calls = []
    accept = manager.accept_nudge

    async def wrapped(*args, **kwargs):
        calls.append(1)
        operation = await accept(*args, **kwargs)
        if fault == "lost_ack":
            raise RuntimeError("ACK lost after canonical acceptance")
        return operation

    monkeypatch.setattr(manager, "accept_nudge", wrapped)
    if fault == "signal":
        signal = loop.steer_event

        def fail_signal(sid):
            if sid != parent:
                raise RuntimeError("local notification failed")
            return signal(sid)

        monkeypatch.setattr(loop, "steer_event", fail_signal)
    elif fault == "inbox_failure":
        dialect = type(manager.db.dialect)
        original = dialect.insert_ignore

        def broken(self, table, cols):
            if table == store.t.follow_up_queue:
                raise RuntimeError("inbox insert unavailable")
            return original(self, table, cols)

        monkeypatch.setattr(dialect, "insert_ignore", broken)
    accepted, _, _ = await issue_nudge(managed, monkeypatch)
    assert len(calls) == 1
    if fault == "inbox_failure":
        operations = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
        assert not operations and (await store.inbox_counts(accepted["session_id"]))["pending"] == 0
        monkeypatch.setattr(dialect, "insert_ignore", original)
    else:
        op = await manager.operation_for_call(runtime.actor(parent),
            await store.message_ref(parent, next(row.seq for row in await store.load_all_messages(parent)
                if any(call["id"] == "actual-nudge" for call in row.tool_calls or []))), "actual-nudge")
        assert op is not None and op.receipt_ref is not None
    if fault == "signal":
        monkeypatch.setattr(loop, "steer_event", signal)
    await finish(managed)
    saved = await manager.get_job(runtime.actor(parent), accepted["job_id"])
    assert saved.outcome == "succeeded"
    assert saved.result["final_text"] == ("CHILD-PROOF-42" if fault == "inbox_failure" else "NUDGED-42")


async def test_same_invocation_replay_after_target_finishes_is_read_only(managed, monkeypatch):
    _, manager, _, loop, model, parent, _ = managed
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    original = delegation.nudge
    results = []

    async def finish_then_replay(*args, **kwargs):
        results.append(await original(*args, **kwargs))
        await finish(managed)
        results.append(await original(*args, **kwargs))
        return results[0]

    monkeypatch.setattr(delegation, "nudge", finish_then_replay)
    await issue_nudge(managed, monkeypatch)
    assert len(results) == 2 and results[0]["operation_id"] == results[1]["operation_id"]
    assert results[1]["replayed"] and model.child_calls == 2
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")) == 1


@pytest.mark.parametrize("visibility", ["missing", "rewritten", "wrong_session"])
async def test_receiver_cannot_ack_an_input_not_carried_by_its_request(nudge_case, visibility):
    c = nudge_case
    await c.accept()
    rows = await c.store.inbox_claim(c.sid)
    seq = await c.store.append_message(c.sid, role="user", content="NUDGED-42", send_index=1,
        inbox={"claim_token": rows[0]["claim_token"], "ids": [r["id"] for r in rows]})
    if visibility == "wrong_session":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE action='nudge'",
            (json.dumps({"request": {"session_id": "wrong"}}),))
    messages = [] if visibility == "missing" else [{"seq": seq, "role": "user",
        "content": "different" if visibility == "rewritten" else "NUDGED-42"}]
    assert await c.manager.prepare_processing(c.child_actor, c.receiver, round_index=0, send_index=1, messages=messages) is None
    with pytest.raises(JobError, match="required_input_unhandled"):
        await c.manager.record_execution_result(c.child_actor, c.target, outcome="succeeded", result={}, resource_state="stopped", require_children_stopped=True)


async def test_originating_parent_cannot_handle_the_receivers_input(nudge_case):
    c = nudge_case
    operation, _ = await c.accept()
    await c.consume(operation)
    [request] = await c.manager.db.fetchall(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE event_type='processing_request'")
    [decision] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='processing_decision'")
    response = json.loads(decision["payload_json"])["response_ref"]
    ref = await c.store.message_ref(c.sid, int(response["seq"]))
    with pytest.raises(JobError, match="processing_snapshot_mismatch"):
        await c.manager.commit_processing_decision(c.actor, c.caller, snapshot_id=request["event_id"], response_ref=ref, reason="parent ack")


async def test_nudge_requires_current_model_send_and_persisted_invocation(managed):
    _, _, _, loop, _, parent, _ = managed
    with pytest.raises(ValueError, match="execution_role_unavailable"):
        await delegation.nudge(loop, parent, "no execution")
    token = set_session_id(parent)
    try:
        with pytest.raises(JobError, match="task_send_required"):
            await delegation.nudge(loop, parent, "no execution")
    finally:
        reset_session_id(token)


@pytest.mark.parametrize("first", ["accept", "finish"])
async def test_actual_terminal_window_never_loses_an_accepted_input_or_resurrects_closed_task(managed, monkeypatch, first):
    store, manager, runtime, loop, model, parent, _ = managed
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    reached, release = asyncio.Event(), asyncio.Event()
    original = delegation._save_result

    if first == "finish":
        async def paused(*args, **kwargs):
            reached.set()
            await release.wait()
            return await original(*args, **kwargs)
        monkeypatch.setattr(delegation, "_save_result", paused)
        model.release.set()
        await asyncio.wait_for(reached.wait(), 2)
        nudging = asyncio.create_task(issue_nudge(managed, monkeypatch))
        await asyncio.sleep(0)
        release.set()
        accepted, _, _ = await nudging
        await finish(managed)
        tool = [m for m in await store.load_all_messages(parent) if m.role == "tool"][-1]
        assert "resume_result_unhandled" in tool.content and "accepted" not in tool.content
    else:
        accepted, _, _ = await issue_nudge(managed, monkeypatch)
        await finish(managed)
    saved = await manager.get_job(runtime.actor(parent), accepted["job_id"])
    assert saved.outcome == "succeeded"
    assert saved.result["final_text"] == ("NUDGED-42" if first == "accept" else "CHILD-PROOF-42")
    assert len([j for j in await jobs(managed) if j.kind == "subagent"]) == 1


@pytest.mark.parametrize("case", ["source_scope", "source_subagent", "target_kind", "invalid_proof", "other_action"])
async def test_nudge_rejects_wrong_execution_roles_kinds_and_conflicting_action(nudge_case, case):
    c = nudge_case
    if case == "source_scope":
        with pytest.raises(JobError, match="nudge_caller_mismatch"):
            await c.accept(caller=c.target)
    elif case == "source_subagent":
        origin = await call(c.store, c.child_actor, "forbidden-child")
        with pytest.raises(JobError, match="role_forbids_delegation"):
            await c.manager.accept_nudge(c.child_actor, c.target, caller=c.receiver, call_ref=origin,
                tool_call_id="forbidden-child", session_id=c.sid, message="no permission")
    elif case == "target_kind":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET kind='background_tool' WHERE job_id=?", (c.target.job_id,))
        with pytest.raises(JobError, match="nudge_target_mismatch"):
            await c.accept()
    elif case == "invalid_proof":
        with pytest.raises(JobError, match="execution_lease_required"):
            await c.manager.accept_nudge(c.actor, None, caller=c.caller, call_ref=c.origin,
                tool_call_id="nudge-call", session_id=c.sid, message="no proof")
    else:
        await c.manager.register_job(c.actor, kind="background_tool", independent=True, parent=c.caller,
                                    call_ref=c.origin, tool_call_id="nudge-call")
        with pytest.raises(JobError, match="operation_conflict"):
            await c.accept()
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0


@pytest.mark.parametrize("proof_field", ["job_id", "generation", "executor_id", "boot_id"])
async def test_input_from_another_generation_cannot_be_acknowledged(nudge_case, proof_field):
    c = nudge_case
    op, _ = await c.accept()
    payload = op.result
    payload["request"]["target"][proof_field] = 999 if proof_field == "generation" else "forged"
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), op.operation_id))
    with pytest.raises(JobError, match="processing_input_mismatch"):
        await c.consume(op)
    event = await c.manager.db.fetchone(f"SELECT delivery_status FROM {c.manager.events} WHERE event_id=?", (op.operation_id,))
    assert event["delivery_status"] == "pending_delivery"


async def test_inbox_identity_collision_rolls_back_operation_and_keeps_original_item(nudge_case, monkeypatch):
    c = nudge_case
    operation_id = "op_" + "f" * 32
    await c.store.inbox_put(c.sid, [{"item_id": "nudge:" + operation_id, "content": "original orphan"}])
    import power_loop.runtime.jobs as module
    monkeypatch.setattr(module.secrets, "token_hex", lambda size: "f" * (size * 2))
    with pytest.raises(JobError, match="nudge_inbox_conflict"):
        await c.accept()
    assert await c.manager.db.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=?", (operation_id,)) is None
    assert (await c.store.inbox_counts(c.sid))["pending"] == 1


async def test_ambiguous_async_operations_cannot_bind_an_implicit_receipt(nudge_case):
    c = nudge_case
    await c.manager.register_job(c.actor, kind="background_tool", independent=True, parent=c.caller,
                                call_ref=c.origin, tool_call_id="nudge-call")
    await c.manager.register_operation(c.actor, c.target, caller=c.caller, action="steer", call_ref=c.origin,
        tool_call_id="nudge-call", request={"message": "other"}, result={"status": "accepted"})
    with pytest.raises(JobError, match="ambiguous_tool_operation"):
        await c.manager.operation_for_call(c.actor, c.origin, "nudge-call")


async def test_missing_original_invocation_does_not_admit_steering(managed, monkeypatch):
    _, manager, _, loop, model, parent, _ = managed
    from power_loop.runtime.invocation import invocation_context
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    original = delegation.nudge

    async def missing(*args, **kwargs):
        with invocation_context(None, None, None):
            return await original(*args, **kwargs)

    monkeypatch.setattr(delegation, "nudge", missing)
    await issue_nudge(managed, monkeypatch)
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
    await finish(managed)


@pytest.mark.parametrize("max_rounds", [3, 4])
async def test_workflow_node_can_nudge_its_own_subagent_without_changing_owner_or_role(managed_workflow, monkeypatch, max_rounds):
    store, manager, runtime, loop, model, parent, state, _ = managed_workflow
    runtime.lease_ms = 30000  # the tested limit is model rounds, not execution expiry
    model.hold = {"AUDIT"}
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    state["spec"]["root"]["steps"][0]["spec"]["max_rounds"] = max_rounds
    node_exited = asyncio.Event()
    original_close = manager.close_model_execution
    async def close_model(actor, *args, **kwargs):
        result = await original_close(actor, *args, **kwargs)
        session = await store.get_session(actor.session_id)
        if session and session.metadata.get("workflow_node_id") == "BUILD":
            node_exited.set()
        return result
    monkeypatch.setattr(manager, "close_model_execution", close_model)
    original = model.complete

    async def complete(request, **kwargs):
        if "ROLE-BUILD" in (request.system_prompt or "") and model.counts["BUILD"] == 1:
            source = get_session_id()
            [target] = [sid for (key, sid), running in delegation._RUNNING.items()
                        if key == id(store) and running.parent == source]
            model.counts["BUILD"] += 1
            return LLMResponse(raw_text="", tool_calls=[{"id": "node-nudge", "type": "function", "function": {
                "name": "spawn_agent", "arguments": json.dumps({"action": "nudge", "sub_session": target, "message": "NUDGED-42"})}}],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", complete)
    assert (await loop.send("run with node-owned audit", parent)).status == "completed"
    await asyncio.wait_for(model.entered["AUDIT"].wait(), 5)
    assert model.counts["REVIEW"] == 0
    await asyncio.wait_for(node_exited.wait(), 3)
    await settle(managed_workflow)
    wf = await manager.get_job(runtime.actor(parent), state["handles"][0].job_id)
    if max_rounds == 3:
        assert wf.outcome == "failed" and model.counts["REVIEW"] == 0, wf.result
        assert model.counts["BUILD"] == 3
        assert wf.result["results"]["BUILD"]["error"] == "completion settlement round budget exhausted"
        assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='node_session'")) == 1
        return
    assert wf.outcome == "succeeded" and model.counts["REVIEW"] == 1, wf.result
    [event] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE action='nudge'")
    [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='subagent'")
    assert child["execution_role"] == "subagent" and child["control_owner_job_id"] is not None
    assert event["job_id"] == child["job_id"] and event["delivery_status"] == "handled"
    call_ref = json.loads(event["call_ref_json"])
    assert call_ref["session_id"] == child["created_by_session_id"] != parent
    assert json.loads(event["receipt_ref_json"])["tool_call_id"] == "node-nudge"


async def test_actual_same_call_changed_message_conflicts_without_enqueuing_it(managed, monkeypatch):
    _, manager, _, loop, model, parent, _ = managed
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    original = delegation.nudge

    async def conflict(current, sid, message, **kwargs):
        result = await original(current, sid, message, **kwargs)
        with pytest.raises(JobError, match="operation_conflict"):
            await original(current, sid, "different correction", **kwargs)
        return result

    monkeypatch.setattr(delegation, "nudge", conflict)
    await issue_nudge(managed, monkeypatch)
    await finish(managed)
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")) == 1


async def test_actual_call_with_another_admitted_action_cannot_accept_nudge(managed, monkeypatch):
    store, manager, _, loop, model, parent, _ = managed
    from power_loop.runtime.invocation import current_invocation
    from power_loop.runtime.task_runtime import _SEND
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    original = delegation.nudge

    async def conflicting_action(*args, **kwargs):
        execution = _SEND.get()
        seq, call_id = current_invocation(parent)
        await manager.register_job(execution.actor, kind="background_tool", independent=True, parent=execution.lease,
            call_ref=await store.message_ref(parent, seq), tool_call_id=call_id)
        return await original(*args, **kwargs)

    monkeypatch.setattr(delegation, "nudge", conflicting_action)
    await issue_nudge(managed, monkeypatch)
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
    await finish(managed)


async def test_another_main_session_cannot_nudge_a_foreign_live_executor(managed, monkeypatch):
    store, manager, _, loop, model, parent, _ = managed
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    created = await receipt(managed)
    foreign = await loop.new_session()
    original = model.complete
    called = False

    async def complete(request, **kwargs):
        nonlocal called
        if get_session_id() != foreign:
            return await original(request, **kwargs)
        if called:
            return LLMResponse(raw_text="done")
        called = True
        return LLMResponse(raw_text="", tool_calls=[{"id": "foreign-nudge", "type": "function", "function": {
            "name": "spawn_agent", "arguments": json.dumps({"action": "nudge", "sub_session": created["session_id"], "message": "wrong owner"})}}])

    monkeypatch.setattr(model, "complete", complete)
    assert (await loop.send("nudge other session", foreign)).status == "completed"
    tool = next(m for m in await store.load_all_messages(foreign) if m.role == "tool")
    assert "nudge_target_mismatch" in tool.content
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
    await finish(managed)
    assert (await store.inbox_counts(foreign))["pending"] == 0 and (await store.inbox_counts(parent))["pending"] == 1


async def test_tool_preserves_original_nudge_text_including_indentation_and_newlines(managed, monkeypatch):
    store, manager, _, loop, model, parent, _ = managed
    await loop.send("delegate", parent)
    await asyncio.wait_for(model.entered.wait(), 2)
    text = "  NUDGED-42\n    preserve this indentation\n"
    accepted, _, _ = await issue_nudge(managed, monkeypatch, message=text)
    [event] = await manager.db.fetchall(f"SELECT event_id,payload_json FROM {manager.events} WHERE action='nudge'")
    assert json.loads(event["payload_json"])["request"]["message"] == text
    [inbox] = await manager.db.fetchall(f"SELECT content FROM {store.t.follow_up_queue} WHERE item_id=?", ("nudge:" + event["event_id"],))
    assert inbox["content"] == text
    await finish(managed)
    assert accepted["session_id"] in {j.session_id for j in await jobs(managed) if j.kind == "subagent"}


async def test_nudge_of_queued_child_is_durable_without_starting_a_second_executor(managed, monkeypatch):
    store, manager, runtime, loop, model, parent, _ = managed
    semaphore = asyncio.Semaphore(0)
    start = delegation.start

    async def queued(*args, **kwargs):
        return await start(*args, admission=semaphore, **kwargs)

    monkeypatch.setattr(delegation, "start", queued)
    await loop.send("delegate", parent)
    accepted, _, _ = await issue_nudge(managed, monkeypatch)
    assert model.child_calls == 0 and (await store.inbox_counts(accepted["session_id"]))["pending"] == 1
    running = delegation._RUNNING[(id(store), accepted["session_id"])]
    assert running.phase == "queued"
    semaphore.release()
    await finish(managed)
    saved = await manager.get_job(runtime.actor(parent), accepted["job_id"])
    assert saved.outcome == "succeeded" and saved.result["final_text"] == "NUDGED-42"
    assert model.child_calls == 1 and len([j for j in await jobs(managed) if j.kind == "subagent"]) == 1


async def test_nudge_after_model_close_during_finalizer_uses_original_execution_and_session(managed, monkeypatch):
    store, manager, runtime, loop, model, parent, _ = managed
    from power_loop.tools import spawn_agent as tool_module
    original = tool_module._finalize
    reached, release = asyncio.Event(), asyncio.Event()
    initial = True

    async def finalizer(*args, **kwargs):
        nonlocal initial
        if initial:
            initial = False
            reached.set()
            await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(tool_module, "_finalize", finalizer)
    try:
        await loop.send("delegate", parent)
        await asyncio.wait_for(model.entered.wait(), 2)
        model.release.set()
        await asyncio.wait_for(reached.wait(), 2)
        created = await receipt(managed)
        running = delegation._RUNNING[(id(store), created["session_id"])]
        prior_task = running.task
        accepted, _, _ = await issue_nudge(managed, monkeypatch)
        assert delegation._RUNNING[(id(store), accepted["session_id"])].task is prior_task
        release.set()
        await finish(managed)
        saved = await manager.get_job(runtime.actor(parent), accepted["job_id"])
        assert saved.outcome == "succeeded" and saved.result["final_text"] == "NUDGED-42"
    finally:
        release.set()


async def test_nudge_cannot_borrow_a_previous_sends_persisted_tool_call(nudge_case):
    c = nudge_case
    seq = await c.store.append_message(c.actor.session_id, role="assistant", send_index=0, round_index=0,
        tool_calls=[{"id": "old-call", "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}])
    with pytest.raises(JobError, match="nudge_caller_mismatch"):
        await c.manager.accept_nudge(c.actor, c.target, caller=c.caller,
            call_ref=await c.store.message_ref(c.actor.session_id, seq), tool_call_id="old-call", session_id=c.sid, message="old request")
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0


async def test_target_supervision_cancel_survives_a_fresh_caller_send(nudge_case):
    c = nudge_case
    await c.manager.request_cancel(c.actor, c.supervisor.job_id, system_key="stop-original-tree", reason="stop")
    with pytest.raises(JobError, match="cancel_requested"):
        await c.accept()
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0
