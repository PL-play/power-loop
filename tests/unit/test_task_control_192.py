"""Cancellation receipts are durable, fenced and distinct from stop confirmation."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop.runtime.jobs import JobActor, JobError, JobManager, _hash
from power_loop.runtime.task_control import recall_stop, request_stop, signal_local
from power_loop.runtime.task_runtime import Execution, TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def control(registry_store):
    store, connect = registry_store
    now = [1_000_000]
    m = JobManager(store, now_ms=lambda: now[0])
    sid = await store.create_session()
    actor = JobActor("control", "host", sid)
    root = await m.register_job(actor, kind="send", session_id=sid, system_key="root", request={"send_index": 1})
    caller = await m.claim_job(actor, root.job.job_id, executor_id="root", boot_id="boot", expected_version=1)
    child = await m.register_job(actor, kind="subagent", parent=caller, independent=True, system_key="child")
    seq = await store.append_message(sid, role="assistant", send_index=1, round_index=0,
        tool_calls=[{"id": "stop-child", "type": "function", "function": {"name": "jobs", "arguments": json.dumps({"action": "stop", "ids": [child.job.job_id]})}}])
    ref = await store.message_ref(sid, seq)
    async def stop(**kwargs):
        return await request_stop(m, actor, kwargs.pop("job_id", child.job.job_id), caller=kwargs.pop("caller", caller),
            call_ref=kwargs.pop("call_ref", ref), tool_call_id=kwargs.pop("tool_call_id", "stop-child"), **kwargs)
    from types import SimpleNamespace
    return SimpleNamespace(store=store, manager=m, actor=actor, root=root, caller=caller, child=child, ref=ref, now=now, stop=stop, connect=connect)


async def test_stop_atomic_receipt_survives_reopen_and_binds_distinct_raw_receipt(control):
    c = control
    op, repeated = await c.stop()
    assert not repeated and op.call_ref == c.ref.to_dict() and op.receipt_ref is None
    record = await c.manager.get_job(c.actor, c.child.job.job_id)
    assert record.status == "cancelled" and record.cancel_requested_at is not None and record.resource_state == "not_started"
    assert record.supervision_status == "open" and record.outcome is None
    assert op.result["response"]["control_pending"] and op.result["response"]["resource_stop_confirmed"] is False
    seq = await c.store.append_message(c.actor.session_id, role="tool", tool_call_id="stop-child", content="accepted", send_index=1)
    ref = await c.store.message_ref(c.actor.session_id, seq)
    await c.manager.bind_receipt(c.actor, op.operation_id, ref, caller=c.caller)
    reopened = JobManager(await c.connect(), now_ms=c.manager.now)
    saved = await reopened.get_operation(c.actor, op.operation_id)
    assert saved.call_ref["id"] != saved.receipt_ref["id"] and saved.receipt_ref == ref.to_dict()
    with pytest.raises(JobError, match="cancel|closed"):
        await c.manager.claim_job(c.actor, record.job_id, executor_id="never-dispatch", boot_id="boot", expected_version=record.version)


@pytest.mark.parametrize("state", ["running", "unknown"])
async def test_stopping_actual_or_unknown_execution_never_fabricates_stopped(control, state):
    c = control
    await c.manager.claim_job(c.actor, c.child.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    if state == "unknown":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET resource_state='unknown' WHERE job_id=?", (c.child.job.job_id,))
    op, _ = await c.stop(expected_generation=1)
    row = await c.manager.get_job(c.actor, c.child.job.job_id)
    assert row.status == "cancelling" and row.resource_state == state and row.result is None
    assert op.result["response"]["resource_state"] == state and not op.result["response"]["resource_stop_confirmed"]


async def test_stop_remains_available_after_actual_source_lease_expiry(control):
    c = control
    c.now[0] += 30001
    op, _ = await c.stop()
    assert op.result["response"]["cancel_requested"]
    with pytest.raises(JobError, match="expired"):
        await c.manager.check_execution(c.actor, c.caller)


async def test_closed_source_can_recall_committed_receipt_but_cannot_issue_a_new_stop(control):
    c = control
    first, _ = await c.stop()
    old = await c.manager.get_job(c.actor, c.child.job.job_id)
    await c.manager.close_model_execution(c.actor, c.caller)
    repeated, yes = await c.stop()
    assert yes and repeated == first
    assert await c.manager.get_job(c.actor, c.child.job.job_id) == old
    seq = await c.store.append_message(c.actor.session_id, role="assistant", send_index=1,
        tool_calls=[{"id": "new-stop", "type": "function", "function": {"name": "jobs", "arguments": "{}"}}])
    with pytest.raises(JobError, match="stop_caller_closed"):
        await c.stop(call_ref=await c.store.message_ref(c.actor.session_id, seq), tool_call_id="new-stop")
    with pytest.raises(JobError, match="operation_conflict"):
        await c.stop(reason="different intent")


@pytest.mark.parametrize("bad", ["self", "generation", "caller", "caller_bool", "ref", "call_id", "send_index", "tenant", "session"])
async def test_invalid_or_unauthorized_control_cannot_change_target(control, bad):
    c = control
    before = await c.manager.get_job(c.actor, c.child.job.job_id)
    kwargs = {}
    if bad == "self":
        kwargs["job_id"] = c.root.job.job_id
    elif bad == "generation":
        kwargs["expected_generation"] = 2
    elif bad == "caller":
        kwargs["caller"] = replace(c.caller, generation=2)
    elif bad == "caller_bool":
        kwargs["caller"] = replace(c.caller, generation=True)
    elif bad == "ref":
        kwargs["call_ref"] = replace(c.ref, id=c.ref.id + 100)
    elif bad == "call_id":
        kwargs["tool_call_id"] = "not-present"
    elif bad == "send_index":
        await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET send_index=2 WHERE id=?", (c.ref.id,))
        kwargs["call_ref"] = replace(c.ref, send_index=2)
    elif bad == "tenant":
        with pytest.raises(JobError, match="access_denied"):
            await request_stop(c.manager, replace(c.actor, tenant_id="other"), c.child.job.job_id,
                caller=c.caller, call_ref=c.ref, tool_call_id="stop-child")
        return
    else:
        other = await c.store.create_session()
        with pytest.raises(JobError, match="access_denied"):
            await request_stop(c.manager, replace(c.actor, session_id=other), c.child.job.job_id,
                caller=c.caller, call_ref=c.ref, tool_call_id="stop-child")
        return
    with pytest.raises(JobError):
        await c.stop(**kwargs)
    assert await c.manager.get_job(c.actor, c.child.job.job_id) == before
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'")


@pytest.mark.parametrize("port", ["update", "operation", "noop", "cancel", "ack"])
async def test_stop_transaction_rollback_and_lost_commit_ack(control, monkeypatch, port):
    c = control
    before = await c.manager.get_job(c.actor, c.child.job.job_id)
    original = c.manager._write
    fired = False
    @asynccontextmanager
    async def fault():
        nonlocal fired
        async with original() as tx:
            execute = tx.execute
            async def wrapped(sql, params=()):
                nonlocal fired
                hit = ("SET status=?,cancel_requested_at" in sql if port in ("update", "noop", "cancel") else "INSERT INTO" in sql and "job_events" in sql)
                if hit and not fired and port != "ack":
                    fired = True
                    if port == "noop":
                        return 1
                    raise asyncio.CancelledError() if port == "cancel" else RuntimeError("test statement fault")
                return await execute(sql, params)
            monkeypatch.setattr(tx, "execute", wrapped)
            yield tx
        if port == "ack" and not fired:
            fired = True
            raise RuntimeError("commit succeeded; ACK lost")
    monkeypatch.setattr(c.manager, "_write", fault)
    with pytest.raises((RuntimeError, JobError, asyncio.CancelledError)):
        await c.stop()
    assert fired
    monkeypatch.setattr(c.manager, "_write", original)
    if port == "ack":
        op, yes = await c.stop()
        assert yes and op.result["response"]["cancel_requested"]
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'")) == 1
    else:
        assert await c.manager.get_job(c.actor, c.child.job.job_id) == before
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'")


async def test_two_managers_replay_one_stop_and_repaired_receipt_corruption_is_rejected(control):
    c = control
    second = JobManager(await c.connect(), now_ms=c.manager.now)
    results = await asyncio.gather(c.stop(), request_stop(second, c.actor, c.child.job.job_id,
        caller=c.caller, call_ref=c.ref, tool_call_id="stop-child"))
    assert results[0][0] == results[1][0] and sorted(x[1] for x in results) == [False, True]
    op = results[0][0]
    data = op.result
    data["response"]["resource_stop_confirmed"] = True
    data["evidence_hash"] = _hash({k: v for k, v in data.items() if k != "evidence_hash"})
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), op.operation_id))
    with pytest.raises(JobError, match="stop_receipt_mismatch"):
        await c.stop()


async def test_local_cancel_deduplicates_nested_task_and_never_recancels_cleanup(control):
    c = control
    runtime = TaskRuntime(c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id, executor_id="worker", boot_id="boot", lease_ms=300)
    child_sid = await c.store.create_session(parent_session_id=c.actor.session_id, metadata={"execution_role": "subagent"})
    bound = await c.manager.bind_session(c.actor, c.child.job.job_id, child_sid)
    lease = await c.manager.claim_job(c.actor, c.child.job.job_id, executor_id="worker", boot_id="boot", expected_version=bound.version, lease_ms=300)
    binding = Execution(runtime, c.actor, lease)
    subactor = runtime.actor(child_sid)
    send = await c.manager.register_job(subactor, kind="send", session_id=child_sid, parent=lease, system_key="actual-child-send", request={"send_index": 1})
    send_lease = await c.manager.claim_job(subactor, send.job.job_id, executor_id="worker", boot_id="boot", expected_version=1, lease_ms=300)
    nested = Execution(runtime, subactor, send_lease)
    entered, stopping, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancelled = []
    async def worker():
        async with runtime.execution(binding, owner=True), runtime.execution(nested):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(1)
                stopping.set()
                await finish.wait()
    task = asyncio.create_task(worker())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert len(runtime._local_executions) == 2
        op, _ = await c.stop(expected_generation=1)
        assert await signal_local(runtime, c.actor, op) == 1
        await asyncio.wait_for(stopping.wait(), 3)
        assert await signal_local(runtime, c.actor, op) == 0 and cancelled == [1] and not task.done()
        await asyncio.sleep(0.25)
        assert not task.done() and cancelled == [1]
        assert (await c.manager.get_job(c.actor, c.child.job.job_id)).resource_state == "running"
        finish.set()
        await asyncio.wait_for(task, 3)
        assert not runtime._local_executions
    finally:
        finish.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_missing_local_handle_never_clears_persistent_control(control):
    c = control
    runtime = TaskRuntime(c.manager, tenant_id="control", principal_id="host", executor_id="new-process", boot_id="new-boot")
    op, _ = await c.stop()
    assert await signal_local(runtime, c.actor, op) == 0
    current = await c.manager.get_job(c.actor, c.child.job.job_id)
    assert current.cancel_requested_at is not None and current.supervision_status == "open"


@pytest.mark.parametrize("mode", ["new_generation", "settling", "forged_job"])
async def test_old_stop_cannot_cancel_new_generation_or_interrupt_known_exit_settlement(control, mode):
    c = control
    runtime = TaskRuntime(c.manager, tenant_id="control", principal_id="host", executor_id="worker", boot_id="boot", lease_ms=300000)
    lease = await c.manager.claim_job(c.actor, c.child.job.job_id, executor_id="worker", boot_id="boot", expected_version=1, lease_ms=300000)
    binding = Execution(runtime, c.actor, lease)
    entered, finish = asyncio.Event(), asyncio.Event()
    async def worker():
        async with runtime.execution(binding, owner=True):
            if mode == "settling":
                binding.stop_renewal.set()
                await c.manager.record_execution_result(c.actor, lease, outcome="succeeded", result={"final_text": "Saved"}, resource_state="stopped")
            entered.set()
            await finish.wait()
    task = asyncio.create_task(worker())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        old = await c.manager.get_result_revision(c.actor, lease.job_id, 1) if mode == "settling" else None
        op, _ = await c.stop(expected_generation=1)
        if mode == "new_generation":
            # Explicitly model a foreign takeover after the original intent.
            # This does not grant the fixture a real second-generation executor.
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (lease.job_id,))
        if mode == "forged_job":
            await c.manager.request_cancel(c.actor, c.root.job.job_id, system_key="host-stop-root", reason="host control")
            with pytest.raises(JobError, match="stop_receipt_mismatch"):
                await signal_local(runtime, c.actor, replace(op, job_id=c.root.job.job_id))
        else:
            assert await signal_local(runtime, c.actor, op) == 0
        assert not task.done()
        if old:
            assert await c.manager.get_result_revision(c.actor, lease.job_id, 1) == old
        finish.set()
        await asyncio.wait_for(task, 3)
        assert not runtime._local_executions
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_local_cancel_skips_other_authorized_principals_session_tasks(control):
    c = control
    runtime = TaskRuntime(c.manager, tenant_id="control", principal_id="host", executor_id="worker", boot_id="boot", lease_ms=300000)
    own = await c.manager.claim_job(c.actor, c.child.job.job_id, executor_id="worker", boot_id="boot", expected_version=1, lease_ms=300000)
    other_sid = await c.store.create_session()
    other_actor = replace(c.actor, session_id=other_sid)
    other = await c.manager.register_job(other_actor, kind="send", session_id=other_sid, system_key="other-root", request={"send_index": 1})
    foreign = await c.manager.claim_job(other_actor, other.job.job_id, executor_id="worker", boot_id="boot", expected_version=1, lease_ms=300000)
    finish, entered_own, entered_other, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event(), []
    async def work(binding, event, label):
        async with runtime.execution(binding):
            event.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.append(label)
                await finish.wait()
    tasks = [asyncio.create_task(work(Execution(runtime, c.actor, own), entered_own, "own")),
        asyncio.create_task(work(Execution(runtime, other_actor, foreign), entered_other, "other"))]
    try:
        await asyncio.wait_for(asyncio.gather(entered_own.wait(), entered_other.wait()), 3)
        op, _ = await c.stop()
        assert await signal_local(runtime, c.actor, op) == 1
        await asyncio.sleep(0)
        assert cancelled == ["own"] and not tasks[1].done()
        assert (await c.manager.get_job(other_actor, other.job.job_id)).cancel_requested_at is None
    finally:
        finish.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert not runtime._local_executions


async def test_readonly_stop_reconciliation_never_admits_missing_intent_and_survives_source_close(control):
    c = control
    args = dict(caller=c.caller, call_ref=c.ref, tool_call_id="stop-child")
    assert await recall_stop(c.manager, c.actor, c.child.job.job_id, **args) is None
    assert (await c.manager.get_job(c.actor, c.child.job.job_id)).cancel_requested_at is None
    op, _ = await c.stop()
    await c.manager.close_model_execution(c.actor, c.caller)
    before = await c.manager.get_job(c.actor, c.child.job.job_id)
    assert await recall_stop(c.manager, c.actor, c.child.job.job_id, **args) == op
    assert await c.manager.get_job(c.actor, c.child.job.job_id) == before


@pytest.mark.parametrize("kwargs", [{"reason": ""}, {"reason": "x"*1001}, {"expected_generation": True}, {"expected_generation": -1},
    {"job_id": ""}, {"caller": None}, {"call_ref": None}, {"tool_call_id": True}])
async def test_invalid_stop_parameters_have_no_side_effects(control, kwargs):
    c = control
    with pytest.raises(JobError):
        await c.stop(**kwargs)
    assert (await c.manager.get_job(c.actor, c.child.job.job_id)).cancel_requested_at is None
