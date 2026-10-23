"""Result snapshots/decisions: real stores, rollback, fencing and request races."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.processing import require_handled
from power_loop.runtime.task_runtime import _SEND, prepare_model_processing

from .test_job_registry_192 import registry_store, runtime
from .test_workflow_task_runtime_192 import managed_workflow, records, settle

__all__ = ["registry_store", "runtime", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def processing_case(runtime):
    store, manager, actor, _, parent, connect, now = runtime
    handler = await manager.register_job(actor, kind="send", session_id=actor.session_id,
        system_key="processing-handler", request={"send_index": 2})
    caller = await manager.claim_job(actor, handler.job.job_id, executor_id="handler", boot_id="boot", expected_version=1)
    counter = 0

    async def result(*, kind="subagent", delivery="delivered"):
        nonlocal counter
        counter += 1
        reg = await manager.register_job(actor, kind=kind, parent=parent, independent=True, system_key=f"child-{counter}")
        lease = await manager.claim_job(actor, reg.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
        event_id = await manager.record_execution_result(actor, lease, outcome="succeeded", result={"value": counter}, resource_state="stopped")
        item_id = "subagent:" + reg.operation.operation_id[3:] if kind == "subagent" else event_id
        await store.inbox_put(actor.session_id, [{"item_id": item_id, "kind": "task_done", "content": f"result-{counter}"}])
        if delivery == "pending":
            return reg.job.job_id, None
        rows = await store.inbox_claim(actor.session_id)
        if delivery == "claimed":
            return reg.job.job_id, None
        text = f"result-{counter}"
        seq = await store.append_message(actor.session_id, role="user", content=text, send_index=2,
            inbox={"claim_token": rows[0]["claim_token"], "ids": [r["id"] for r in rows]})
        return reg.job.job_id, {"seq": seq, "role": "user", "content": text}

    async def prepare(messages, **kwargs):
        return await manager.prepare_processing(actor, caller, round_index=kwargs.get("round_index", 0),
                                                 send_index=kwargs.get("send_index", 2), messages=messages)

    async def response(**kwargs):
        seq = await store.append_message(actor.session_id, role=kwargs.get("role", "assistant"), content="processed",
            round_index=kwargs.get("round_index", 0), send_index=kwargs.get("send_index", 2), tool_calls=kwargs.get("tool_calls"))
        return await store.message_ref(actor.session_id, seq)

    async def commit(snapshot, ref, **kwargs):
        return await manager.commit_processing_decision(actor, caller, snapshot_id=snapshot, response_ref=ref,
            decision=kwargs.get("decision", "consumed"), reason=kwargs.get("reason", "Validated output and registered follow-up work."))

    return SimpleNamespace(store=store, manager=manager, actor=actor, caller=caller, result=result, prepare=prepare,
                           response=response, commit=commit, connect=connect, now=now)


@pytest.mark.parametrize("kind", ["subagent", "workflow"])
@pytest.mark.parametrize("decision", ["consumed", "rejected", "escalated"])
async def test_decision_and_result_ack_survive_reopen_and_replay_after_close(processing_case, kind, decision):
    c = processing_case
    job_id, row = await c.result(kind=kind)
    snapshot = await c.prepare([row])
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"
    ref = await c.response()
    ids = await c.commit(snapshot, ref, decision=decision)
    assert len(ids) == 1
    await c.manager.close_model_execution(c.actor, c.caller)
    await c.manager.request_cancel(c.actor, c.caller.job_id, system_key="late-stop", reason="stop")
    reopened = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    assert (await reopened.get_job(c.actor, job_id)).result_status == "handled"
    assert await reopened.commit_processing_decision(c.actor, c.caller, snapshot_id=snapshot, response_ref=ref,
        decision=decision, reason="Validated output and registered follow-up work.") == ids
    record = await reopened.db.fetchone(f"SELECT payload_json FROM {reopened.events} WHERE event_id=?", (ids[0],))
    payload = json.loads(record["payload_json"])
    assert payload["response_ref"] == ref.to_dict() and payload["handler_job_id"] == c.caller.job_id
    assert payload["event"]["job_id"] == job_id and payload["decision"] == decision


async def test_result_arriving_after_snapshot_is_not_acknowledged_by_an_unrelated_later_assistant(processing_case):
    c = processing_case
    first, row = await c.result()
    snapshot = await c.prepare([row])
    late, _ = await c.result()
    await c.commit(snapshot, await c.response())
    assert (await c.manager.get_job(c.actor, first)).result_status == "handled"
    assert (await c.manager.get_job(c.actor, late)).result_status == "pending_delivery"


@pytest.mark.parametrize("visibility", ["pending", "claimed", "missing", "rewritten", "wrong_seq", "wrong_role"])
async def test_unobserved_or_rewritten_result_never_enters_request_snapshot(processing_case, visibility):
    c = processing_case
    job_id, row = await c.result(delivery=visibility if visibility in ("pending", "claimed") else "delivered")
    rows = [] if row is None or visibility == "missing" else [{**row}]
    if visibility == "rewritten":
        rows[0]["content"] = "not the result"
    elif visibility == "wrong_seq":
        rows[0]["seq"] += 100
    elif visibility == "wrong_role":
        rows[0]["role"] = "assistant"
    assert await c.prepare(rows) is None
    await c.response()
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"


@pytest.mark.parametrize("barrier", ["expired", "cancelled", "stale", "foreign_actor", "foreign_tenant", "caller_closed"])
async def test_invalid_execution_cannot_acknowledge_results(processing_case, barrier):
    c = processing_case
    job_id, row = await c.result()
    snapshot = await c.prepare([row])
    ref = await c.response()
    actor, caller = c.actor, c.caller
    if barrier == "expired":
        c.now[0] += 30001
    elif barrier == "cancelled":
        await c.manager.request_cancel(actor, caller.job_id, system_key="stop", reason="stop")
    elif barrier == "stale":
        caller = replace(caller, generation=caller.generation + 1)
    elif barrier == "foreign_actor":
        actor = JobActor(actor.tenant_id, actor.principal_id, await c.store.create_session())
    elif barrier == "foreign_tenant":
        actor = replace(actor, tenant_id="wrong")
    else:
        await c.manager.close_model_execution(actor, caller)
    with pytest.raises(JobError):
        await c.manager.commit_processing_decision(actor, caller, snapshot_id=snapshot, response_ref=ref, reason="consume")
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"


@pytest.mark.parametrize("mismatch", ["store", "session", "id", "seq", "send", "round", "role", "snapshot", "nonterminal"])
async def test_response_must_match_exact_request_and_persisted_assistant(processing_case, mismatch):
    c = processing_case
    job_id, row = await c.result()
    snapshot = await c.prepare([row])
    ref = await c.response(send_index=3 if mismatch == "send" else 2,
                           round_index=1 if mismatch == "round" else 0, role="user" if mismatch == "role" else "assistant")
    changes = {"store": {"store_id": "wrong"}, "session": {"session_id": "wrong"},
               "id": {"id": ref.id + 1000}, "seq": {"seq": ref.seq + 1000}}
    if mismatch in changes:
        ref = replace(ref, **changes[mismatch])
    elif mismatch == "snapshot":
        snapshot = "request_missing"
    elif mismatch == "nonterminal":
        await c.response()
    with pytest.raises(JobError, match="processing_.*mismatch"):
        await c.commit(snapshot, ref)
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"


@pytest.mark.parametrize("pending", ["session", "receipt", "operation"])
async def test_unregistered_or_unresolved_follow_up_actions_prevent_ack(processing_case, pending):
    c = processing_case
    job_id, row = await c.result()
    snapshot = await c.prepare([row])
    call_seq = await c.store.append_message(c.actor.session_id, role="assistant", content="", send_index=2, round_index=0,
        tool_calls=[{"id": "follow-up", "type": "function", "function": {"name": "work", "arguments": "{}"}}])
    if pending == "session":
        await c.store.set_pending(c.actor.session_id, {"tool_call_ids": ["follow-up"]})
    elif pending == "operation":
        await c.store.append_message(c.actor.session_id, role="tool", content="queued", tool_call_id="follow-up", send_index=2)
        await c.manager.register_job(c.actor, kind="subagent", independent=True, parent=c.caller,
            call_ref=await c.store.message_ref(c.actor.session_id, call_seq), tool_call_id="follow-up")
    ref = await c.response()
    with pytest.raises(JobError, match="processing_tools_pending"):
        await c.commit(snapshot, ref)
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"


async def test_registered_follow_up_work_is_linked_and_remains_active_after_consumption(processing_case):
    c = processing_case
    job_id, row = await c.result()
    snapshot = await c.prepare([row])
    seq = await c.store.append_message(c.actor.session_id, role="assistant", content="", send_index=2, round_index=0,
        tool_calls=[{"id": "follow-up", "type": "function", "function": {"name": "work", "arguments": "{}"}}])
    reg = await c.manager.register_job(c.actor, kind="subagent", independent=True, parent=c.caller,
        call_ref=await c.store.message_ref(c.actor.session_id, seq), tool_call_id="follow-up")
    receipt = await c.store.append_message(c.actor.session_id, role="tool", content="queued", tool_call_id="follow-up", send_index=2)
    await c.manager.bind_receipt(c.actor, reg.operation.operation_id,
        await c.store.message_ref(c.actor.session_id, receipt), caller=c.caller)
    [decision] = await c.commit(snapshot, await c.response())
    saved = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (decision,))
    payload = json.loads(saved["payload_json"])
    assert payload["operation_ids"] == [reg.operation.operation_id] and payload["follow_up_job_ids"] == [reg.job.job_id]
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "handled"
    assert (await c.manager.get_job(c.actor, reg.job.job_id)).status == "queued"


@pytest.mark.parametrize("mutation", ["revision", "payload"])
async def test_old_snapshot_cannot_acknowledge_new_result_revision(processing_case, mutation):
    c = processing_case
    job_id, row = await c.result()
    snapshot = await c.prepare([row])
    field, value = ("job_version", 999) if mutation == "revision" else ("payload_json", '{"changed":true}')
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET {field}=? WHERE event_id=?", (value, "result_" + job_id))
    with pytest.raises(JobError, match="processing_revision_changed"):
        await c.commit(snapshot, await c.response())
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"


@pytest.mark.parametrize("fault", ["before_commit", "lost_ack"])
async def test_atomic_batch_and_retry_never_partial_ack_or_duplicate_decisions(processing_case, monkeypatch, fault):
    c = processing_case
    first, row1 = await c.result()
    second, row2 = await c.result()
    snapshot = await c.prepare([row1, row2])
    ref = await c.response()
    from contextlib import asynccontextmanager
    original = c.manager._write

    @asynccontextmanager
    async def fail_transaction():
        async with original() as tx:
            yield tx
            if fault == "before_commit":
                raise RuntimeError("rollback entire batch")
        if fault == "lost_ack":
            raise RuntimeError("commit ack lost")

    monkeypatch.setattr(c.manager, "_write", fail_transaction)
    with pytest.raises(RuntimeError):
        await c.commit(snapshot, ref)
    statuses = [(await c.manager.get_job(c.actor, j)).result_status for j in (first, second)]
    assert statuses == (["pending_delivery"] * 2 if fault == "before_commit" else ["handled"] * 2)
    monkeypatch.setattr(c.manager, "_write", original)
    ids = await c.commit(snapshot, ref)
    assert await c.commit(snapshot, ref) == ids
    decisions = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='processing_decision'")
    assert len(decisions) == 2


@pytest.mark.parametrize("change", ["response", "decision", "reason"])
async def test_committed_processing_decision_cannot_be_rewritten(processing_case, change):
    c = processing_case
    _, row = await c.result()
    snapshot, ref = await c.prepare([row]), await c.response()
    await c.commit(snapshot, ref)
    kwargs = {change: "rejected" if change == "decision" else "different"} if change != "response" else {}
    if change == "response":
        ref = await c.response()
    with pytest.raises(JobError, match="processing_decision_conflict"):
        await c.commit(snapshot, ref, **kwargs)


@pytest.mark.parametrize("first", ["cancel", "decision"])
async def test_cancel_and_decision_serialize_without_reviving_execution(processing_case, first):
    c = processing_case
    job_id, row = await c.result()
    snapshot, ref = await c.prepare([row]), await c.response()
    from contextlib import asynccontextmanager
    original = c.manager._write
    entered, release = asyncio.Event(), asyncio.Event()
    initial = True

    @asynccontextmanager
    async def hold_first():
        nonlocal initial
        async with original() as tx:
            if initial:
                initial = False
                entered.set()
                await release.wait()
            yield tx

    async def cancel():
        return await c.manager.request_cancel(c.actor, c.caller.job_id, system_key="race-stop", reason="stop")

    try:
        c.manager._write = hold_first
        one = asyncio.create_task(cancel() if first == "cancel" else c.commit(snapshot, ref))
        await asyncio.wait_for(entered.wait(), 2)
        two = asyncio.create_task(c.commit(snapshot, ref) if first == "cancel" else cancel())
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(one, two, return_exceptions=True)
    finally:
        release.set()
        c.manager._write = original
    if first == "cancel":
        assert isinstance(results[1], JobError) and results[1].code == "cancel_requested"
    else:
        assert not any(isinstance(r, BaseException) for r in results)
    assert (await c.manager.get_job(c.actor, job_id)).result_status == ("pending_delivery" if first == "cancel" else "handled")
    assert (await c.manager.get_job(c.actor, c.caller.job_id)).status == "cancelling"


async def test_snapshot_replay_is_frozen_and_conflicting_visibility_is_rejected(processing_case):
    c = processing_case
    _, row = await c.result()
    snapshot = await c.prepare([row])
    assert await c.prepare([row]) == snapshot
    _, other = await c.result()
    with pytest.raises(JobError, match="processing_snapshot_conflict"):
        await c.prepare([row, other])


@pytest.mark.parametrize("mode", ["included", "removed", "copied", "rewritten", "short_circuit"])
async def test_actual_request_assembly_does_not_credit_hidden_or_rewritten_rows(processing_case, mode):
    c = processing_case
    _, row = await c.result()
    message = {"role": row["role"], "content": row["content"]}
    pipeline = SimpleNamespace(session_id=c.actor.session_id, send_index=2, history=[message],
                               sink=SimpleNamespace(_history_seqs=[row["seq"]]))
    binding = SimpleNamespace(actor=c.actor, lease=c.caller, resource_evidence={},
                              runtime=SimpleNamespace(manager=c.manager))
    messages = [] if mode == "removed" else [dict(message) if mode == "copied" else message]
    if mode == "rewritten":
        message["content"] = "no result here"
    token = _SEND.set(binding)
    try:
        await prepare_model_processing(pipeline, messages, 0, provider_call=mode != "short_circuit")
    finally:
        _SEND.reset(token)
    assert bool(binding.resource_evidence["processing_snapshot_id"]) == (mode == "included")


@pytest.mark.parametrize("fault", ["snapshot", "decision", "response_write"])
@pytest.mark.parametrize("retry", [False, True])
async def test_actual_workflow_never_advances_when_processing_persistence_fails(managed_workflow, monkeypatch, fault, retry):
    store, manager, _, loop, model, sid, state, _ = managed_workflow
    from .test_workflow_task_runtime_192 import leaf
    model.hold = {"AUDIT"}
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    if retry:
        state["spec"]["root"]["steps"][0]["retry"] = {"max_attempts": 3, "on": ["failed"], "backoff_s": 0}
    await loop.send("run with audit", sid)
    await asyncio.wait_for(model.entered["AUDIT"].wait(), 3)
    if fault in ("snapshot", "decision"):
        target = "prepare_processing" if fault == "snapshot" else "commit_processing_decision"
        original = getattr(manager, target)

        async def fail(actor, *args, **kwargs):
            if actor.session_id != sid:
                raise RuntimeError("processing persistence unavailable")
            return await original(actor, *args, **kwargs)

        monkeypatch.setattr(manager, target, fail)
    else:
        original = store.append_message

        async def fail(session_id, **kwargs):
            if session_id != sid and kwargs.get("role") == "assistant" and model.counts["BUILD"] >= 2:
                raise RuntimeError("assistant response write unavailable")
            return await original(session_id, **kwargs)

        monkeypatch.setattr(store, "append_message", fail)
    await settle(managed_workflow)
    assert model.counts["REVIEW"] == 0
    wf = await manager.get_job(manager_actor := loop.task_runtime.actor(sid), state["handles"][0].job_id)
    assert wf.outcome == "failed"
    [audit] = [r for r in await records(managed_workflow) if r.kind == "subagent"]
    assert audit.result_status != "handled"
    assert (await manager.get_job(manager_actor, audit.job_id)).result is not None
    bindings = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='node_session'")
    assert len(bindings) == 1  # Never re-run business work to repair a failed ACK.


async def test_scope_guard_requires_decision_and_matching_successful_model_exit(processing_case):
    c = processing_case
    job_id, row = await c.result()
    snapshot, ref = await c.prepare([row]), await c.response()
    ids = await c.commit(snapshot, ref)
    child = await c.manager._row(c.manager.db, job_id)
    with pytest.raises(JobError, match="required_result_unhandled"):
        await require_handled(c.manager, c.manager.db, child)
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed",
        "last_assistant_ref": ref.to_dict(), "processing_decisions": ids})
    await require_handled(c.manager, c.manager.db, child)


@pytest.mark.parametrize("invalid", ["failed", "cancelled", "new_input", "wrong_ref", "missing_decision", "wrong_generation", "wrong_owner", "new_revision"])
async def test_scope_guard_cannot_use_incomplete_or_other_execution_proof(processing_case, invalid):
    c = processing_case
    job_id, row = await c.result()
    snapshot, ref = await c.prepare([row]), await c.response()
    ids = await c.commit(snapshot, ref)
    evidence = {"exit": "completed", "last_assistant_ref": ref.to_dict(), "processing_decisions": ids}
    if invalid in ("failed", "cancelled"):
        evidence["exit"] = invalid
    elif invalid == "new_input":
        evidence["new_input_after_pass"] = True
    elif invalid == "wrong_ref":
        evidence["last_assistant_ref"] = replace(ref, id=ref.id + 100).to_dict()
    elif invalid == "missing_decision":
        evidence["processing_decisions"] = []
    await c.manager.close_model_execution(c.actor, c.caller, evidence=evidence)
    if invalid == "wrong_generation":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=generation+1 WHERE job_id=?", (c.caller.job_id,))
    elif invalid == "wrong_owner":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET control_owner_job_id=? WHERE job_id=?", (job_id, c.caller.job_id))
    elif invalid == "new_revision":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET job_version=job_version+1 WHERE event_id=?", ("result_" + job_id,))
    child = await c.manager._row(c.manager.db, job_id)
    with pytest.raises(JobError, match="required_result_unhandled"):
        await require_handled(c.manager, c.manager.db, child)


@pytest.mark.parametrize("fault", ["lost_ack", "lost_ack_recall_failure"])
async def test_workflow_recalls_committed_decision_without_repeating_model_or_tools(managed_workflow, monkeypatch, fault):
    _, manager, _, loop, model, sid, state, _ = managed_workflow
    from .test_workflow_task_runtime_192 import leaf
    model.hold = {"AUDIT"}
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    await loop.send("run with audit", sid)
    await asyncio.wait_for(model.entered["AUDIT"].wait(), 3)
    commit, recall = manager.commit_processing_decision, manager.recall_processing_decision
    calls = []

    async def lose_ack(actor, *args, **kwargs):
        value = await commit(actor, *args, **kwargs)
        if actor.session_id != sid:
            calls.append(value)
            raise RuntimeError("decision committed but ack lost")
        return value

    async def fail_recall(actor, *args, **kwargs):
        if actor.session_id != sid:
            raise RuntimeError("durable decision lookup unavailable")
        return await recall(actor, *args, **kwargs)

    monkeypatch.setattr(manager, "commit_processing_decision", lose_ack)
    if fault == "lost_ack_recall_failure":
        monkeypatch.setattr(manager, "recall_processing_decision", fail_recall)
    await settle(managed_workflow)
    [audit] = [r for r in await records(managed_workflow) if r.kind == "subagent"]
    assert audit.result_status == "handled" and len(calls) == 1
    assert model.counts["AUDIT"] == 1 and model.counts["BUILD"] == 3
    wf = await manager.get_job(loop.task_runtime.actor(sid), state["handles"][0].job_id)
    assert wf.outcome == ("succeeded" if fault == "lost_ack" else "failed")
    assert model.counts["REVIEW"] == (1 if fault == "lost_ack" else 0)


async def test_snapshot_is_frozen_before_entering_transaction(processing_case, monkeypatch):
    c = processing_case
    _, row = await c.result()
    messages = [row]
    from contextlib import asynccontextmanager
    original = c.manager._write

    @asynccontextmanager
    async def mutate():
        row["content"] = "changed while waiting for control lock"
        messages.clear()
        async with original() as tx:
            yield tx

    monkeypatch.setattr(c.manager, "_write", mutate)
    snapshot = await c.prepare(messages)
    assert snapshot is not None and not messages


@pytest.mark.parametrize("invalid", ["decision", "empty_reason", "long_reason", "round", "bool_round", "send"])
async def test_processing_input_validation_cannot_change_result_state(processing_case, invalid):
    c = processing_case
    job_id, row = await c.result()
    if invalid in ("round", "bool_round", "send"):
        args = {"round_index": -1 if invalid == "round" else True} if invalid != "send" else {"send_index": 999}
        with pytest.raises(JobError, match="invalid_processing_round|processing_send_mismatch"):
            await c.prepare([row], **args)
    else:
        snapshot, ref = await c.prepare([row]), await c.response()
        args = {"decision": "unknown"} if invalid == "decision" else {"reason": "" if invalid == "empty_reason" else "x" * 1001}
        with pytest.raises(JobError, match="invalid_processing"):
            await c.commit(snapshot, ref, **args)
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"


async def test_read_only_recall_does_not_acknowledge_prepared_request_and_rejects_other_identity(processing_case):
    c = processing_case
    _, row = await c.result()
    snapshot, ref = await c.prepare([row]), await c.response()
    args = {"snapshot_id": snapshot, "response_ref": ref, "reason": "Validated output and registered follow-up work."}
    assert await c.manager.recall_processing_decision(c.actor, c.caller, **args) is None
    ids = await c.commit(snapshot, ref)
    assert await c.manager.recall_processing_decision(c.actor, c.caller, **args) == ids
    with pytest.raises(JobError, match="processing_decision_conflict"):
        await c.manager.recall_processing_decision(c.actor, c.caller, **{**args, "reason": "different"})
    with pytest.raises(JobError):
        await c.manager.recall_processing_decision(replace(c.actor, principal_id="other"), c.caller, **args)


@pytest.mark.parametrize("method", ["prepare", "commit", "recall"])
async def test_processing_requires_actual_model_send_not_just_an_authorized_scope(processing_case, method):
    c = processing_case
    _, row = await c.result()
    snapshot, ref = await c.prepare([row]), await c.response()
    reg = await c.manager.register_job(c.actor, kind="subagent", parent=c.caller, independent=True, system_key="not-a-model")
    lease = await c.manager.claim_job(c.actor, reg.job.job_id, executor_id="scope", boot_id="boot", expected_version=1)
    with pytest.raises(JobError, match="processing_caller_mismatch"):
        if method == "prepare":
            await c.manager.prepare_processing(c.actor, lease, round_index=0, send_index=2, messages=[row])
        else:
            call = c.manager.commit_processing_decision if method == "commit" else c.manager.recall_processing_decision
            await call(c.actor, lease, snapshot_id=snapshot, response_ref=ref, reason="consume")


async def test_unknown_and_mismatched_snapshot_generation_are_rejected(processing_case):
    c = processing_case
    _, row = await c.result()
    snapshot, ref = await c.prepare([row]), await c.response()
    with pytest.raises(JobError, match="processing_snapshot_mismatch"):
        await c.manager.recall_processing_decision(c.actor, c.caller, snapshot_id="unknown", response_ref=ref, reason="consume")
    event = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (snapshot,))
    payload = json.loads(event["payload_json"])
    payload["generation"] += 1
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), snapshot))
    with pytest.raises(JobError, match="processing_snapshot_mismatch"):
        await c.commit(snapshot, ref)


async def test_response_before_result_delivery_cannot_acknowledge_it(processing_case):
    c = processing_case
    ref = await c.response()
    _, row = await c.result()
    with pytest.raises(JobError, match="processing_response_mismatch"):
        await c.commit(await c.prepare([row]), ref)


@pytest.mark.parametrize("corrupt", ["store_id", "session_id", "seq", "role", "tool_call_id", "send_index"])
async def test_async_operation_requires_exact_real_receipt_reference(processing_case, corrupt):
    c = processing_case
    _, row = await c.result()
    snapshot = await c.prepare([row])
    seq = await c.store.append_message(c.actor.session_id, role="assistant", content="", send_index=2, round_index=0,
        tool_calls=[{"id": "follow-up", "type": "function", "function": {"name": "work", "arguments": "{}"}}])
    reg = await c.manager.register_job(c.actor, kind="subagent", independent=True, parent=c.caller,
        call_ref=await c.store.message_ref(c.actor.session_id, seq), tool_call_id="follow-up")
    seq = await c.store.append_message(c.actor.session_id, role="tool", content="queued", tool_call_id="follow-up", send_index=2)
    ref = await c.store.message_ref(c.actor.session_id, seq)
    await c.manager.bind_receipt(c.actor, reg.operation.operation_id, ref, caller=c.caller)
    wire = {**ref.to_dict(), corrupt: "9999" if corrupt in ("seq", "send_index") else "wrong"}
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=? WHERE event_id=?",
                               (json.dumps(wire), reg.operation.operation_id))
    with pytest.raises(JobError, match="processing_tools_pending"):
        await c.commit(snapshot, await c.response())


async def test_two_prepared_handlers_cannot_rewrite_the_same_result_decision(processing_case):
    c = processing_case
    _, row = await c.result()
    first = await c.prepare([row])
    reg = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id,
        system_key="second-handler", request={"send_index": 3})
    caller = await c.manager.claim_job(c.actor, reg.job.job_id, executor_id="second", boot_id="boot", expected_version=1)
    second = await c.manager.prepare_processing(c.actor, caller, round_index=0, send_index=3, messages=[row])
    ref = await c.response()
    await c.commit(first, ref)
    second_ref = await c.response(send_index=3)
    with pytest.raises(JobError, match="processing_decision_conflict"):
        await c.manager.commit_processing_decision(c.actor, caller, snapshot_id=second, response_ref=second_ref, reason="consume")


async def test_processing_links_steer_operation_without_fabricating_a_new_job(processing_case):
    c = processing_case
    _, row = await c.result()
    snapshot = await c.prepare([row])
    seq = await c.store.append_message(c.actor.session_id, role="assistant", content="", send_index=2, round_index=0,
        tool_calls=[{"id": "steer", "type": "function", "function": {"name": "work", "arguments": "{}"}}])
    reg = await c.manager.register_job(c.actor, kind="background_tool", independent=True, parent=c.caller, system_key="steer-target")
    target = await c.manager.claim_job(c.actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    operation = await c.manager.register_operation(c.actor, target, caller=c.caller, action="steer",
        call_ref=await c.store.message_ref(c.actor.session_id, seq), tool_call_id="steer", request={"text": "continue"}, result={"status": "accepted"})
    seq = await c.store.append_message(c.actor.session_id, role="tool", content="accepted", tool_call_id="steer", send_index=2)
    await c.manager.bind_receipt(c.actor, operation.operation_id, await c.store.message_ref(c.actor.session_id, seq), caller=c.caller)
    [decision] = await c.commit(snapshot, await c.response())
    event = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (decision,))
    payload = json.loads(event["payload_json"])
    assert payload["operation_ids"] == [operation.operation_id] and not payload["follow_up_job_ids"]
