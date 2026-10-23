"""Success-only standalone resume with immutable revisions and one dispatch."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation, execution_membership
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.result_revisions import result_id

from .test_job_registry_192 import registry_store
from .test_nudge_runtime_192 import issue_nudge
from .test_task_runtime_192 import finish, managed, receipt

__all__ = ["registry_store", "managed"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def ready(managed):
    store, manager, runtime, loop, model, parent, connect = managed
    await loop.send("delegate", parent)
    created = await receipt(managed)
    await finish(managed)
    await loop.flush_follow_ups(parent)
    first = await manager.get_job(runtime.actor(parent), created["job_id"])
    assert first.generation == 1 and first.result_status == "handled"
    [row] = await manager.db.fetchall(f"SELECT executor_id,boot_id,generation FROM {manager.jobs} WHERE job_id=?", (first.job_id,))
    from power_loop.runtime.jobs import JobLease
    old_lease = JobLease(first.job_id, row["executor_id"], row["boot_id"], row["generation"])
    model.release.clear()
    model.entered.clear()
    return SimpleNamespace(store=store, manager=manager, runtime=runtime, loop=loop, model=model, parent=parent,
        connect=connect, created=created, old=first, old_lease=old_lease, sid=created["session_id"], fixture=managed)


async def latest(c):
    return await c.manager.get_job(c.runtime.actor(c.parent), c.old.job_id)


async def nudge_receipt(c):
    rows = await c.store.load_all_messages(c.parent)
    return json.loads(next(r.content for r in reversed(rows) if r.role == "tool"))


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack", "replay"])
async def test_original_job_and_session_resume_once_with_distinct_results_and_usage(ready, monkeypatch, fault):
    c = ready
    counts = {"admit": 0, "dispatch": 0}
    for name, key in (("resume_subagent", "admit"), ("claim_resume_dispatch", "dispatch")):
        original = getattr(c.manager, name)
        async def wrapper(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            value = await _original(*args, **kwargs)
            if fault == ("admission_ack" if _key == "admit" else "dispatch_ack"):
                raise RuntimeError("committed; ACK lost")
            return value
        monkeypatch.setattr(c.manager, name, wrapper)
    _, sent, _ = await issue_nudge(c.fixture, monkeypatch, repeat=fault == "replay")
    assert sent.status == "completed"
    new = await nudge_receipt(c)
    assert new["status"] == "accepted" and new["mode"] == "continue" and new["generation"] == 2
    assert new["job_id"] == c.old.job_id and new["session_id"] == c.sid and new["run_id"] != c.created["run_id"]
    assert new["result_event_id"] == result_id(c.old.job_id, 2) and counts == {"admit": 1, "dispatch": 1}
    assert len(await c.store.list_children(c.parent)) == 1
    assert (await latest(c)).generation == 2
    old = await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1)
    assert old["delivery_status"] == "handled" and old["payload"]["result"] == c.old.result
    for method, kwargs in ((c.manager.check_execution, {}), (c.manager.record_execution_result, {
            "outcome": "failed", "result": {"status": "failed"}, "resource_state": "stopped"})):
        with pytest.raises(JobError, match="stale_execution"):
            await method(c.runtime.actor(c.sid), c.old_lease, **kwargs)
    await finish(c.fixture)
    saved = await latest(c)
    assert saved.result["final_text"] == "NUDGED-42" and saved.result["usage"]["total_tokens"] == 10
    assert saved.result["run_id"] == new["run_id"] and saved.outcome == "succeeded"
    for generation in (1, 2):
        snapshot = await c.manager.get_execution_membership(c.runtime.actor(c.parent), c.old.job_id, generation)
        assert snapshot["sealed"] and snapshot["scope"]["owner"] is None and len(snapshot["members"]) == 1
        assert await c.manager.execution_usage(c.runtime.actor(c.parent), c.old.job_id, generation=generation) == saved.result["usage"]
    lifetime = await c.manager.execution_usage(c.runtime.actor(c.parent), c.old.job_id)
    assert {key: lifetime[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "calls")} == {
        "prompt_tokens": 14, "completion_tokens": 6, "total_tokens": 20, "calls": 2}
    assert (await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1)) == old
    revised = await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 2)
    assert revised["payload"]["generation"] == 2 and revised["delivery_status"] == "pending_delivery"
    [input_event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
    assert input_event["delivery_status"] == "handled" and json.loads(input_event["receipt_ref_json"])["tool_call_id"] == "actual-nudge"
    [dispatch] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
    assert dispatch["event_id"] == "dispatch_"+new["operation_id"]
    await c.loop.flush_follow_ups(c.parent)
    assert (await latest(c)).result_status == "handled"
    assert (await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 2))["delivery_status"] == "handled"
    assert (await c.store.get_runtime_state(c.parent, "subagent:usage:total"))["total_tokens"] == 20
    reopened = JobManager(await c.connect())
    assert await reopened.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1) == old


@pytest.mark.parametrize("block", ["cancel", "unknown", "failed", "open", "unhandled", "generation", "role", "owner", "principal", "scope_missing", "seal_missing", "edge_missing"])
async def test_resume_admission_rejects_invalid_original_state_without_new_model(ready, monkeypatch, block):
    c = ready
    original = c.manager.resume_subagent
    errors = []
    async def invalid(actor, job_id, **kwargs):
        if block == "cancel":
            await c.manager.request_cancel(actor, job_id, system_key="test-cancel", reason="test")
        elif block == "generation":
            kwargs["expected_generation"] += 1
        elif block == "principal":
            actor = replace(actor, principal_id="other")
        elif block in ("scope_missing", "seal_missing", "edge_missing"):
            key = {"scope_missing": execution_membership.scope_id(c.manager, job_id, 1),
                "seal_missing": execution_membership.seal_id(c.manager, job_id, 1)}.get(block)
            if key:
                await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
            else:
                await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='execution_member'", (job_id,))
        else:
            field, value = {"unknown": ("resource_state", "unknown"), "failed": ("outcome", "failed"),
                "open": ("intake", "open"), "unhandled": ("result_status", "pending_delivery"),
                "role": ("execution_role", "workflow_node"), "owner": ("control_owner_job_id", "other")}[block]
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET {field}=? WHERE job_id=?", (value, job_id))
        try:
            return await original(actor, job_id, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    monkeypatch.setattr(c.manager, "resume_subagent", invalid)
    await issue_nudge(c.fixture, monkeypatch)
    expected = {"cancel": "cancel_requested", "unknown": "unresolved_prior_attempt", "failed": "resume_target_unsettled",
        "open": "resume_target_unsettled", "unhandled": "resume_result_unhandled", "generation": "resume_generation_changed",
        "role": "resume_target_mismatch", "owner": "resume_target_mismatch", "principal": "access_denied",
        "scope_missing": "execution_membership_missing", "seal_missing": "execution_membership_missing", "edge_missing": "execution_membership_corrupt"}
    assert errors == [expected[block]]
    assert c.model.child_calls == 1 and (await latest(c)).generation == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('dispatch','execution')")


async def test_three_generations_keep_history_and_only_latest_result_is_consumed(ready, monkeypatch):
    c = ready
    for generation in (2, 3):
        await issue_nudge(c.fixture, monkeypatch)
        new = await nudge_receipt(c)
        assert new["generation"] == generation
        await finish(c.fixture)
        assert (await latest(c)).result["usage"]["total_tokens"] == 10
        await c.loop.flush_follow_ups(c.parent)
        c.model.release.clear()
    assert (await latest(c)).generation == 3
    rows = await c.manager.db.fetchall(f"SELECT event_id,delivery_status FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (c.old.job_id,))
    assert len(rows) == 3 and all(r["delivery_status"] == "handled" for r in rows)
    assert (await c.store.get_runtime_state(c.parent, "subagent:usage:total"))["total_tokens"] == 30
    assert (await c.manager.execution_usage(c.runtime.actor(c.parent), c.old.job_id))["total_tokens"] == 30


@pytest.mark.parametrize("generation", [False, 0, -1, "2", 2, 9223372036854775808])
async def test_revision_read_is_explicit_and_rejects_missing_or_invalid_generations(ready, generation):
    c = ready
    with pytest.raises(JobError, match="result_revision_missing" if generation == 2 else "invalid_result_generation"):
        await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, generation)


@pytest.mark.parametrize("field,value,code", [
    ("message", "", "invalid_nudge_message"), ("message", "\x00bad", "invalid_nudge_message"),
    ("message", 7, "invalid_nudge_message"), ("message", "x"*64001, "invalid_nudge_message"),
    ("expected_generation", False, "invalid_result_generation"), ("expected_generation", 0, "invalid_result_generation"),
    ("expected_generation", 9223372036854775807, "invalid_result_generation"),
    ("executor_id", "", "invalid_executor"), ("boot_id", 1, "invalid_executor"),
    ("caller", None, "execution_lease_required"),
    ("lease_ms", True, "invalid_lease_duration"), ("lease_ms", 0, "invalid_lease_duration"),
])
async def test_invalid_admission_parameters_have_no_effect(ready, monkeypatch, field, value, code):
    c = ready
    original = c.manager.resume_subagent
    errors = []
    async def invalid(*args, **kwargs):
        kwargs[field] = value
        try:
            return await original(*args, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    monkeypatch.setattr(c.manager, "resume_subagent", invalid)
    await issue_nudge(c.fixture, monkeypatch)
    assert errors == [code] and (await latest(c)).generation == 1 and c.model.child_calls == 1


@pytest.mark.parametrize("fault", ["pending_tools", "pending_input", "session_role", "contract_changed", "source_role", "old_call"])
async def test_admission_rechecks_current_session_contract_and_actual_caller(ready, monkeypatch, fault):
    c = ready
    original = c.manager.resume_subagent
    errors = []
    async def change(actor, job_id, **kwargs):
        if fault == "pending_tools":
            await c.manager.db.execute(f"UPDATE {c.store.t.session_state} SET pending_json=? WHERE session_id=?", ('{"pending":true}', c.sid))
        elif fault == "pending_input":
            await c.store.inbox_put(c.sid, [{"item_id": "unprocessed", "content": "earlier input"}])
        elif fault == "session_role":
            await c.store.merge_session_metadata(c.sid, {"execution_role": "main"})
        elif fault == "contract_changed":
            await c.store.set_runtime_state(c.sid, "workflow_execution_contract", {"version": 999})
        elif fault == "source_role":
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET execution_role='workflow_node' WHERE job_id=?", (kwargs["caller"].job_id,))
        else:
            kwargs["call_ref"] = await c.store.message_ref(c.parent, c.created_call.seq)
            kwargs["tool_call_id"] = "actual-create"
        try:
            return await original(actor, job_id, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    rows = await c.store.load_all_messages(c.parent)
    c.created_call = next(r for r in rows if r.role == "assistant" and r.tool_calls)
    monkeypatch.setattr(c.manager, "resume_subagent", change)
    await issue_nudge(c.fixture, monkeypatch)
    code = {"pending_tools": "resume_tools_pending", "pending_input": "resume_input_pending", "session_role": "resume_session_unavailable",
        "contract_changed": "resume_contract_mismatch", "source_role": "resume_caller_mismatch", "old_call": "operation_conflict"}[fault]
    assert errors == [code] and (await latest(c)).generation == 1 and c.model.child_calls == 1


@pytest.mark.parametrize("point", ["operation", "inbox", "execution", "execution_scope"])
async def test_resume_failure_rolls_back_generation_operation_and_inbox(ready, monkeypatch, point):
    c = ready
    original = c.manager.resume_subagent
    write = c.manager._write
    hits = []
    from contextlib import asynccontextmanager
    async def fail(*args, **kwargs):
        @asynccontextmanager
        async def inject():
            async with write() as tx:
                execute = tx.execute
                async def faulty(sql, params=()):
                    value = await execute(sql, params)
                    match = (point == "operation" and sql.startswith("INSERT INTO "+c.manager.events) and "nudge" in params or
                        point == "inbox" and "INSERT" in sql and c.store.t.follow_up_queue in sql or
                        point in ("execution", "execution_scope") and sql.startswith("INSERT INTO "+c.manager.events) and point in params)
                    if match:
                        hits.append(point)
                        raise RuntimeError("injected write failure: "+point)
                    return value
                with monkeypatch.context() as patch:
                    patch.setattr(tx, "execute", faulty)
                    yield tx
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_write", inject)
            return await original(*args, **kwargs)
    monkeypatch.setattr(c.manager, "resume_subagent", fail)
    await issue_nudge(c.fixture, monkeypatch)
    assert hits == [point]
    assert (await latest(c)).generation == 1 and (await latest(c)).result == c.old.result
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge' OR event_type='execution'")
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0 and c.model.child_calls == 1


async def test_independent_duplicate_admission_and_dispatch_claims_never_duplicate_model_work(ready, monkeypatch):
    c = ready
    admit, claim = c.manager.resume_subagent, c.manager.claim_resume_dispatch
    async def admit_twice(*args, **kwargs):
        results = await asyncio.gather(admit(*args, **kwargs), admit(*args, **kwargs))
        assert sorted(r[1] for r in results) == [False, True]
        assert results[0][0].operation_id == results[1][0].operation_id
        return next(r for r in results if not r[1])
    async def claim_twice(*args, **kwargs):
        assert await claim(*args, **kwargs)
        assert await claim(*args, **kwargs)
        assert not await claim(*args, **{**kwargs, "token": "another-executor-token"})
        return True
    monkeypatch.setattr(c.manager, "resume_subagent", admit_twice)
    monkeypatch.setattr(c.manager, "claim_resume_dispatch", claim_twice)
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    assert (await latest(c)).generation == 2 and c.model.child_calls == 2
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")) == 1


@pytest.mark.parametrize("missing", [None, {"version": 999}])
async def test_missing_saved_child_contract_never_borrows_parent_configuration(ready, monkeypatch, missing):
    c = ready
    await c.store.set_runtime_state(c.sid, "workflow_execution_contract", missing)
    await issue_nudge(c.fixture, monkeypatch)
    rows = await c.store.load_all_messages(c.parent)
    assert "contract missing" in next(r.content for r in reversed(rows) if r.role == "tool")
    assert (await latest(c)).generation == 1 and c.model.child_calls == 1


@pytest.mark.parametrize("generation", [None, False, 0, -1, "2"])
async def test_usage_scope_generation_validation(ready, generation):
    c = ready
    if generation is None:
        assert (await c.manager.execution_usage(c.runtime.actor(c.parent), c.old.job_id))["total_tokens"] == 10
    else:
        with pytest.raises(JobError, match="invalid_result_generation"):
            await c.manager.execution_usage(c.runtime.actor(c.parent), c.old.job_id, generation=generation)


async def test_active_nudge_uses_resumed_execution_run_and_generation(ready, monkeypatch):
    c = ready
    await issue_nudge(c.fixture, monkeypatch)
    resumed = await nudge_receipt(c)
    await issue_nudge(c.fixture, monkeypatch)
    steered = await nudge_receipt(c)
    assert steered["mode"] == "steer" and steered["generation"] == 2 and steered["run_id"] == resumed["run_id"]
    await finish(c.fixture)
    assert (await latest(c)).result["run_id"] == resumed["run_id"]
    events = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE action='nudge'")
    assert len(events) == 2 and all(e["delivery_status"] == "handled" for e in events)


async def test_old_committed_processing_decision_cannot_acknowledge_new_result(ready, monkeypatch):
    c = ready
    [decision] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='processing_decision'")
    payload = json.loads(decision["payload_json"])
    handler = await c.manager._row(c.manager.db, payload["handler_job_id"])
    from power_loop.runtime.jobs import JobLease
    lease = JobLease(handler["job_id"], handler["executor_id"], handler["boot_id"], handler["generation"])
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    assert (await latest(c)).result_status == "pending_delivery"
    ids = await c.manager.commit_processing_decision(c.runtime.actor(c.parent), lease, snapshot_id=payload["snapshot_id"],
        response_ref=await c.store.message_ref(c.parent, int(payload["response_ref"]["seq"])), decision=payload["decision"], reason=payload["reason"])
    assert ids and (await latest(c)).result_status == "pending_delivery"


@pytest.mark.parametrize("stop_phase", ["queue", "model"])
async def test_new_generation_can_be_stopped_in_admission_queue_or_model(ready, monkeypatch, stop_phase):
    c = ready
    original = delegation.nudge
    semaphore = asyncio.Semaphore(0) if stop_phase == "queue" else None
    async def nudge(*args, **kwargs):
        return await original(*args, **{**kwargs, "admission": semaphore})
    monkeypatch.setattr(delegation, "nudge", nudge)
    await issue_nudge(c.fixture, monkeypatch)
    resumed = await nudge_receipt(c)
    if stop_phase == "model":
        await asyncio.wait_for(c.model.entered.wait(), 2)
    task = delegation._RUNNING[(id(c.store), c.sid)].task
    assert await delegation.stop(c.store, c.parent, resumed["run_id"], .01) in ("forced", "stopped")
    await asyncio.wait_for(asyncio.shield(task), 3)
    job = await latest(c)
    assert job.generation == 2 and job.outcome == "cancelled"
    assert (await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1))["delivery_status"] == "handled"
    assert (id(c.store), c.sid) not in delegation._RUNNING


@pytest.mark.parametrize("ending", ["recover", "empty", "failed", "repeat_pass"])
async def test_empty_completed_turn_has_one_bounded_original_session_final_answer_attempt(ready, monkeypatch, ending):
    c = ready
    from contextlib import contextmanager
    guarded = []
    @contextmanager
    def guard():
        guarded.append("enter")
        try:
            yield
        finally:
            guarded.append("exit")
    c.loop._child_run_guards = (("test", guard),)
    original = c.model.complete
    resumed_calls = 0
    async def empty(request, **kwargs):
        nonlocal resumed_calls
        result = await original(request, **kwargs)
        if c.model.root != get_session_id():
            resumed_calls += 1
            if resumed_calls == 1 or ending == "repeat_pass":
                result.raw_text = ""
                result.tool_calls = [{"id": "empty-pass-"+str(resumed_calls), "type": "function", "function": {
                    "name": "pass_turn", "arguments": '{"reason":"waiting"}'}}]
            elif ending == "empty":
                result.raw_text, result.tool_calls = "", None
            elif ending == "failed":
                raise RuntimeError("final answer provider failure")
        return result
    monkeypatch.setattr(c.model, "complete", empty)
    await issue_nudge(c.fixture, monkeypatch, message="finish the original calculation")
    await finish(c.fixture)
    # The existing core adds its read-only round-limit request after an invalid
    # repeated pass or empty reply; this is not a second resume/dispatch.
    assert resumed_calls == (3 if ending in ("repeat_pass", "empty") else 2)
    current = await latest(c)
    assert guarded == ["enter", "exit", "enter", "exit"]
    assert current.outcome == ("succeeded" if ending == "recover" else "failed")
    if ending == "recover":
        assert current.result["final_text"] == "CHILD-PROOF-42"
        assert current.result["usage"]["total_tokens"] == 20
    assert (await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1))["delivery_status"] == "handled"


@pytest.mark.parametrize("fault", ["invalid_token", "invalid_caller", "missing_operation", "wrong_caller", "wrong_mode", "source_role", "source_send", "cancel_target", "uncommitted"])
async def test_dispatch_faults_never_start_an_unclaimed_execution_and_remain_cancellable(ready, monkeypatch, fault):
    c = ready
    original = c.manager.claim_resume_dispatch
    errors = []
    async def fail(actor, op_id, **kwargs):
        if fault == "invalid_token":
            kwargs["token"] = ""
        elif fault == "invalid_caller":
            kwargs["caller"] = None
        elif fault == "missing_operation":
            op_id = "missing"
        elif fault == "wrong_caller":
            kwargs["caller"] = replace(kwargs["caller"], generation=kwargs["caller"].generation+1)
        elif fault == "wrong_mode":
            row = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (op_id,))
            data = json.loads(row["payload_json"])
            data["mode"] = "steer"
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), op_id))
        elif fault == "source_role":
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET execution_role='workflow_node' WHERE job_id=?", (kwargs["caller"].job_id,))
        elif fault == "source_send":
            row = await c.manager.db.fetchone(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE job_id=? AND action='create'", (kwargs["caller"].job_id,))
            data = json.loads(row["payload_json"])
            data["request"]["send_index"] = 999
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), row["event_id"]))
        elif fault == "cancel_target":
            await c.manager.request_cancel(actor, c.old.job_id, system_key="cancel-before-dispatch", reason="test")
        else:
            raise RuntimeError("dispatch not committed")
        try:
            return await original(actor, op_id, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    monkeypatch.setattr(c.manager, "claim_resume_dispatch", fail)
    if fault == "source_send":
        with pytest.raises(JobError, match="processing_send_mismatch"):
            await issue_nudge(c.fixture, monkeypatch)
    else:
        await issue_nudge(c.fixture, monkeypatch)
    expected = {"invalid_token": "invalid_dispatch_token", "invalid_caller": "execution_lease_required", "missing_operation": "resume_dispatch_mismatch",
        "wrong_caller": "resume_dispatch_mismatch", "wrong_mode": "resume_dispatch_mismatch",
        "source_role": "resume_caller_mismatch", "source_send": "resume_caller_mismatch", "cancel_target": "cancel_requested"}
    assert errors == ([] if fault == "uncommitted" else [expected[fault]])
    assert c.model.child_calls == 1 and (id(c.store), c.sid) not in delegation._RUNNING
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
    await c.manager.request_cancel(c.runtime.actor(c.parent), c.old.job_id, system_key="supervisor-stop", reason="stop undelivered execution")
    assert (await latest(c)).cancel_requested_at is not None


async def test_competing_dispatch_owner_only_returns_receipt_without_starting_another_model(ready, monkeypatch):
    c = ready
    original = c.manager.claim_resume_dispatch
    async def competing(*args, **kwargs):
        assert await original(*args, **{**kwargs, "token": "other-owner"})
        return await original(*args, **kwargs)
    monkeypatch.setattr(c.manager, "claim_resume_dispatch", competing)
    await issue_nudge(c.fixture, monkeypatch)
    reply = await nudge_receipt(c)
    assert reply["replayed"] and c.model.child_calls == 1 and (await latest(c)).generation == 2
    assert (id(c.store), c.sid) not in delegation._RUNNING


async def test_session_lock_prevents_uncoordinated_resume(ready, monkeypatch):
    c = ready
    async with c.loop._lock_for(c.sid):
        await issue_nudge(c.fixture, monkeypatch)
    rows = await c.store.load_all_messages(c.parent)
    assert "resume_session_busy" in next(r.content for r in reversed(rows) if r.role == "tool")
    assert (await latest(c)).generation == 1 and c.model.child_calls == 1


async def test_conflicting_low_level_resume_replay_is_not_a_second_generation(ready, monkeypatch):
    c = ready
    original = c.manager.resume_subagent
    async def twice(*args, **kwargs):
        first = await original(*args, **kwargs)
        with pytest.raises(JobError, match="operation_conflict"):
            await original(*args, **{**kwargs, "message": "different follow-up"})
        return first
    monkeypatch.setattr(c.manager, "resume_subagent", twice)
    await issue_nudge(c.fixture, monkeypatch)
    assert (await latest(c)).generation == 2


async def test_contract_is_frozen_before_admission_await(ready, monkeypatch):
    c = ready
    original, write = c.manager.resume_subagent, c.manager._write
    from contextlib import asynccontextmanager
    async def mutate(*args, **kwargs):
        contract = kwargs["contract"]
        @asynccontextmanager
        async def changed():
            async with write() as tx:
                contract["version"] = "changed-during-await"
                yield tx
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_write", changed)
            return await original(*args, **kwargs)
    monkeypatch.setattr(c.manager, "resume_subagent", mutate)
    await issue_nudge(c.fixture, monkeypatch)
    assert (await nudge_receipt(c))["generation"] == 2
    assert (await c.store.get_runtime_state(c.sid, "workflow_execution_contract"))["version"] == 1


@pytest.mark.parametrize("corruption", ["missing_execution", "old_pending", "old_unstarted"])
async def test_versioned_result_route_and_identity_fail_closed(ready, monkeypatch, corruption):
    c = ready
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    if corruption == "missing_execution":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='execution'", (c.old.job_id,))
        with pytest.raises(JobError, match="execution_revision_missing"):
            await c.loop.flush_follow_ups(c.parent)
        assert (await latest(c)).result_status == "pending_delivery"
    elif corruption == "old_pending":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (result_id(c.old.job_id, 1),))
        await c.loop.flush_follow_ups(c.parent)
        assert (await latest(c)).result_status == "handled"
        old = await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1)
        assert old["delivery_status"] == "pending_delivery"
    else:
        event = await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1)
        data = event["payload"]
        data["unstarted"] = True
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), event["event_id"]))
        with pytest.raises(JobError, match="result_revision_missing"):
            await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1)
    assert c.model.child_calls == 2


async def test_result_history_requires_original_supervisory_authority(ready):
    c = ready
    foreign = await c.loop.new_session()
    with pytest.raises(JobError, match="access_denied"):
        await c.manager.get_result_revision(c.runtime.actor(foreign), c.old.job_id, 1)


async def test_task_creation_failure_keeps_negative_new_revision_and_no_executor(ready, monkeypatch):
    c = ready
    original = asyncio.create_task
    hits = []
    def fail(coroutine, *args, **kwargs):
        if str(kwargs.get("name", "")).startswith("subagent-"):
            hits.append(1)
            raise RuntimeError("executor dispatch unavailable")
        return original(coroutine, *args, **kwargs)
    monkeypatch.setattr(asyncio, "create_task", fail)
    await issue_nudge(c.fixture, monkeypatch)
    current = await latest(c)
    assert hits == [1] and current.generation == 2 and current.outcome == "failed"
    assert c.model.child_calls == 1 and (id(c.store), c.sid) not in delegation._RUNNING
    assert (await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1))["payload"]["result"] == c.old.result


async def test_scope_entry_failure_is_finalized_without_model_dispatch(ready, monkeypatch):
    c = ready
    original = c.runtime.execution
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def refused(binding, **kwargs):
        if binding.lease.job_id == c.old.job_id and binding.lease.generation == 2:
            raise RuntimeError("new scope entry refused")
        async with original(binding, **kwargs):
            yield binding
    monkeypatch.setattr(c.runtime, "execution", refused)
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    assert (await latest(c)).outcome == "failed" and c.model.child_calls == 1
    assert (id(c.store), c.sid) not in delegation._RUNNING


@pytest.mark.parametrize("fault", ["missing_input", "empty_without_budget"])
async def test_unusable_child_result_cannot_be_reported_as_success(ready, monkeypatch, fault):
    c = ready
    from power_loop.runtime import spec
    original = spec.workflow_continuation_loop
    async def rebuilt(*args, **kwargs):
        child = await original(*args, **kwargs)
        flush = child.flush_follow_ups
        async def result(*a, **kw):
            if fault == "missing_input":
                return None
            value = await flush(*a, **kw)
            value.final_text, value.rounds = "", int(child.config.max_rounds)
            return value
        child.flush_follow_ups = result
        return child
    monkeypatch.setattr(spec, "workflow_continuation_loop", rebuilt)
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    current = await latest(c)
    assert current.outcome == "failed"
    assert ("resume_input_missing" if fault == "missing_input" else "resume_output_missing") in current.result["error"]


async def test_callback_and_host_guards_preserve_original_contract_and_order(ready, monkeypatch):
    c = ready
    from contextlib import contextmanager
    guards, callbacks = [], []
    @contextmanager
    def guard():
        guards.append("enter")
        try:
            yield
        finally:
            guards.append("exit")
    c.loop._child_run_guards = (("test", guard),)
    original = delegation.nudge
    async def completed(parent, result):
        assert parent == c.parent and result["run_id"] != c.created["run_id"]
        assert (await latest(c)).result == result
        assert (await c.store.inbox_counts(c.parent))["pending"] >= 1
        callbacks.append(result["run_id"])
    async def nudge(*args, **kwargs):
        return await original(*args, **{**kwargs, "on_complete": completed})
    monkeypatch.setattr(delegation, "nudge", nudge)
    contract = await c.store.get_runtime_state(c.sid, "workflow_execution_contract")
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    assert len(callbacks) == 1 and guards == ["enter", "exit"]
    assert await c.store.get_runtime_state(c.sid, "workflow_execution_contract") == contract


async def test_new_revision_delivery_failure_does_not_rerun_or_overwrite_success(ready, monkeypatch):
    c = ready
    original = c.loop.deliver
    async def fail(item, sid, **kwargs):
        if sid == c.parent and item.kind == "task_done":
            raise RuntimeError("owner inbox unavailable")
        return await original(item, sid, **kwargs)
    monkeypatch.setattr(c.loop, "deliver", fail)
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    current = await latest(c)
    assert current.outcome == "succeeded" and current.result_status == "pending_delivery" and c.model.child_calls == 2
    assert (await c.manager.get_result_revision(c.runtime.actor(c.parent), c.old.job_id, 1))["delivery_status"] == "handled"
    assert (id(c.store), c.sid) not in delegation._RUNNING


async def test_existing_inbox_identity_rolls_back_the_entire_resume_admission(ready, monkeypatch):
    c = ready
    import power_loop.runtime.jobs as module
    original = c.manager.resume_subagent
    errors = []
    async def collide(*args, **kwargs):
        identity = "nudge:op_"+"f"*32
        await c.store.inbox_put(c.sid, [{"item_id": identity, "content": "earlier delivered orphan"}])
        await c.manager.db.execute(f"UPDATE {c.store.t.follow_up_queue} SET status='delivered' WHERE session_id=? AND item_id=?", (c.sid, identity))
        with monkeypatch.context() as patch:
            patch.setattr(module.secrets, "token_hex", lambda n: "f"*(n*2))
            try:
                return await original(*args, **kwargs)
            except JobError as exc:
                errors.append(exc.code)
                raise
    monkeypatch.setattr(c.manager, "resume_subagent", collide)
    await issue_nudge(c.fixture, monkeypatch)
    assert errors == ["nudge_inbox_conflict"] and (await latest(c)).generation == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


async def test_resume_replay_returned_by_durable_admission_does_not_guess_dispatch(ready, monkeypatch):
    c = ready
    original = c.manager.resume_subagent
    async def competitor(*args, **kwargs):
        await original(*args, **kwargs)
        return await original(*args, **kwargs)
    monkeypatch.setattr(c.manager, "resume_subagent", competitor)
    await issue_nudge(c.fixture, monkeypatch)
    assert (await nudge_receipt(c))["replayed"]
    assert c.model.child_calls == 1 and (id(c.store), c.sid) not in delegation._RUNNING
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")


async def test_queued_resumption_releases_admission_slot_after_completion(ready, monkeypatch):
    c = ready
    original = delegation.nudge
    semaphore = asyncio.Semaphore(0)
    async def queued(*args, **kwargs):
        return await original(*args, **{**kwargs, "admission": semaphore})
    monkeypatch.setattr(delegation, "nudge", queued)
    await issue_nudge(c.fixture, monkeypatch)
    assert c.model.child_calls == 1
    semaphore.release()
    await finish(c.fixture)
    assert (await latest(c)).outcome == "succeeded" and semaphore._value == 1


async def test_nonexistent_original_task_cannot_be_replaced_by_a_new_child(ready, monkeypatch):
    c = ready
    original = delegation.nudge
    async def missing(loop, sid, message, **kwargs):
        return await original(loop, "missing-session", message, **kwargs)
    monkeypatch.setattr(delegation, "nudge", missing)
    await issue_nudge(c.fixture, monkeypatch)
    rows = await c.store.load_all_messages(c.parent)
    assert "task_adapter_unavailable:subagent_resume" in next(r.content for r in reversed(rows) if r.role == "tool")
    assert (await latest(c)).generation == 1 and c.model.child_calls == 1
