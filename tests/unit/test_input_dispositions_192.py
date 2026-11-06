"""Stopped-negative task input is disposed, never fabricated as consumed."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop.runtime import input_dispositions as inputs
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_terminal as terminal
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.task_runtime import Execution, TaskRuntime

from .test_job_registry_192 import registry_store, runtime
from .test_nudge_runtime_192 import nudge_case
from .test_timer_waits_192 import waiting
from .test_wait_dispositions_192 import exited
from .test_wait_effects_192 import child_loop, decide
from .test_wait_handoff_192 import handoff
from .test_wait_terminal_192 import consume_owner

__all__ = ['registry_store', 'runtime', 'waiting', 'handoff', 'nudge_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("delivery", ["pending", "claimed", "delivered", "mixed"])
async def test_enqueued_continue_has_negative_proof_without_fabricating_consumption(handoff, delivery):
    h, c = handoff, handoff.c
    await decide(h, "continue")
    original = await effects.apply(c.manager, c.main, h.event_id)
    delivered = None
    if delivery == "claimed":
        await c.store.inbox_claim(c.sid)
    elif delivery in ("delivered", "mixed"):
        delivered = await _deliver(c, mixed=delivery == "mixed")
    await exited(h)
    before = await c.store.load_all_messages(c.sid)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    [proof] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='input_disposition'")
    payload = json.loads(proof["payload_json"])
    assert payload["disposition"] == ("delivered_unprocessed" if delivered else "withdrawn_before_delivery")
    assert payload["owner"]["job_id"] == c.owner.job_id
    assert payload["owner_outcome"] == "cancelled"
    assert (
        payload["delivered_message"]["ref"] == delivered.to_dict()
        if delivered
        else payload["delivered_message"] is None
    )
    queue = {r["item_id"]: r for r in await c.store.inbox_items(c.sid)}
    assert queue["nudge:" + original["operation_id"]]["status"] == ("delivered" if delivered else "void")
    if delivery == "mixed":
        assert queue["other-original-input"]["status"] == "delivered"
        assert "Other original input" in (await c.store.load_all_messages(c.sid))[-1].content
    assert await c.store.load_all_messages(c.sid) == before
    replay = await effects.apply(c.manager, c.main, h.event_id)
    assert replay["status"] == "owner_terminated_before_input_processed"
    assert replay["input_was_enqueued"] and not replay["execution_authority"] and not replay["billing_settled"]
    assert replay["operation_id"] == original["operation_id"]
    from power_loop.runtime.task_events import read_events

    listing = await read_events(c.manager, c.main, c.owner.job_id, view="operations")
    [event] = [e for e in listing["events"] if e["event_id"] == original["operation_id"]]
    assert event["input_disposition"]["disposition"] == payload["disposition"]
    detail = await read_events(c.manager, c.main, c.owner.job_id, view="operations", event_id=original["operation_id"])
    assert json.loads(detail["text"])["input_disposition"]["handled_as"] == "negative_terminal_disposition"
    assert "executor_id" not in detail["text"] and "boot_id" not in detail["text"]
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    await consume_owner(h)
    assert all(
        r["supervision_status"] == "closed"
        for r in await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}")
    )


async def _deliver(c, *, mixed=False):
    if mixed:
        await c.store.inbox_put(
            c.sid, [{"item_id": "other-original-input", "kind": "relay", "content": "Other original input"}]
        )
    rows = await c.store.inbox_claim(c.sid)
    seq = await c.store.append_message(
        c.sid,
        role="user",
        name="follow_up",
        send_index=1,
        content="\n".join(r["content"] for r in rows),
        inbox={"ids": [r["id"] for r in rows], "claim_token": rows[0]["claim_token"]},
        meta={"inbox": {"item_ids": [r["item_id"] for r in rows], "kinds": ["relay"]}},
    )
    return await c.store.message_ref(c.sid, seq)


async def _applied(h):
    await decide(h, "continue")
    result = await effects.apply(h.c.manager, h.c.main, h.event_id)
    return await h.c.manager.db.fetchone(
        f"SELECT * FROM {h.c.manager.events} WHERE event_id=?", (result["operation_id"],)
    )


async def test_consumed_input_keeps_its_actual_processing_proof_after_later_owner_failure(handoff):
    h, c = handoff, handoff.c
    op = await _applied(h)
    await attempts.enable(Execution(h.loop.task_runtime, c.actor, c.caller))
    loop, model, binding = await child_loop(h)
    async with loop.task_runtime.execution(binding, owner=True):
        assert (await loop.flush_follow_ups(c.sid)).status == "completed"
    await c.manager.request_cancel(c.main, c.owner.job_id, system_key="later-stop", reason="later stop")
    await c.manager.record_execution_result(
        c.actor, c.owner, outcome="cancelled", result={"status": "cancelled"}, resource_state="stopped"
    )
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='input_disposition'"
    )
    replay = await effects.apply(c.manager, c.main, h.event_id)
    assert replay["status"] == "owner_terminated_after_input_processed" and replay["input_was_processed"]
    [queue] = await c.store.inbox_items(c.sid)
    assert queue["status"] == "delivered" and model.calls == 1
    current = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (op["event_id"],))
    async with c.manager._read() as tx:
        assert await inputs.reconcile_tx(c.manager, tx, c.main, current) is None
    await consume_owner(h)


@pytest.mark.parametrize("fault", ["active", "unknown", "missing_snapshot", "missing_receipt", "foreign_actor"])
async def test_unproven_exit_or_origin_cannot_void_input(handoff, fault):
    h, c = handoff, handoff.c
    op = await _applied(h)
    if fault != "active":
        await exited(h, resource="unknown" if fault == "unknown" else "stopped", snapshot=fault != "missing_snapshot")
    actor = replace(c.main, tenant_id="foreign") if fault == "foreign_actor" else c.main
    if fault == "missing_receipt":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=NULL WHERE action='wait_decision'")
    before = await c.store.inbox_items(c.sid)
    with pytest.raises(JobError):
        async with c.manager._write() as tx:
            await inputs.reconcile_tx(c.manager, tx, actor, op)
    assert await c.store.inbox_items(c.sid) == before
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='input_disposition'"
    )


@pytest.mark.parametrize("point", ["void", "input_proof", "handled", "wait_proof", "supervision"])
async def test_full_supervision_transaction_rolls_back_disposition_and_queue(handoff, monkeypatch, point):
    h, c = handoff, handoff.c
    await _applied(h)
    await exited(h)
    async with c.manager._write() as tx:
        tx_type, original = type(tx), type(tx)._execute
    before_jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id")
    before_events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    before_queue = await c.store.inbox_items(c.sid)
    fault_hits = []

    async def fail(tx, sql, params=()):
        count = await original(tx, sql, params)
        inserted = params[2] if sql.startswith("INSERT INTO " + c.manager.events) else None
        if (
            (point == "void" and "SET status='void'" in sql)
            or (point == "input_proof" and inserted == "input_disposition")
            or (point == "handled" and sql.startswith("UPDATE " + c.manager.events + " SET delivery_status=?")
                and params[0] == "handled")
            or (point == "wait_proof" and inserted == "wait_disposition")
            or (point == "supervision" and inserted == "supervision_closed")
        ):
            fault_hits.append(point)
            raise OSError("original transaction failed")
        return count

    with monkeypatch.context() as patch:
        patch.setattr(tx_type, "_execute", fail)
        with pytest.raises(OSError, match="original transaction failed"):
            await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    assert fault_hits == [point]
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id") == before_jobs
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id") == before_events
    assert await c.store.inbox_items(c.sid) == before_queue
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)


async def test_committed_lost_ack_concurrent_replay_and_reopen_never_redeliver(handoff, monkeypatch):
    h, c = handoff, handoff.c
    await _applied(h)
    await exited(h)
    original = type(c.manager)._write

    @asynccontextmanager
    async def lost_ack(manager):
        async with original(manager) as tx:
            yield tx
        raise OSError("committed acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(type(c.manager), "_write", lost_ack)
        with pytest.raises(OSError, match="acknowledgement lost"):
            await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    restored = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    results = await asyncio.gather(
        *(terminal.terminate(m, c.main, h.registration.job.job_id) for m in (c.manager, restored, c.manager, restored))
    )
    assert all(r == results[0] for r in results)
    assert (
        len(
            await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='input_disposition'")
        )
        == 1
    )
    assert await c.store.inbox_claim(c.sid) == []
    assert (await effects.apply(restored, c.main, h.event_id))["status"] == "owner_terminated_before_input_processed"


@pytest.mark.parametrize(
    "fault", ["proof_hash", "proof_rehash", "queue", "operation", "raw_call", "raw_receipt", "snapshot", "fork"]
)
async def test_closed_negative_proof_revalidates_original_history(handoff, fault):
    h, c = handoff, handoff.c
    op = await _applied(h)
    await exited(h)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    [proof] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='input_disposition'")
    original_status = await c.manager.db.fetchone(f"SELECT delivery_status FROM {c.manager.events} WHERE event_id=?", (op['event_id'],))
    if fault.startswith("proof_"):
        data = json.loads(proof["payload_json"])
        data["owner_outcome"] = "failed"
        await c.manager.db.execute(
            f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
            (_json(data), _hash(data) if fault == "proof_rehash" else proof["request_hash"], proof["event_id"]),
        )
    elif fault == "queue":
        await c.manager.db.execute(
            f"UPDATE {c.store.t.follow_up_queue} SET content='replacement' WHERE session_id=?", (c.sid,)
        )
    elif fault == "operation":
        await c.manager.db.execute(
            f"UPDATE {c.manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (op["event_id"],)
        )
    elif fault in ("raw_call", "raw_receipt"):
        [decision] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='wait_decision'")
        ref = json.loads(decision["call_ref_json" if fault == "raw_call" else "receipt_ref_json"])
        await c.manager.db.execute(
            f"UPDATE {c.store.t.messages} SET content='replacement' WHERE session_id=? AND id=?",
            (ref["session_id"], int(ref["id"])),
        )
    elif fault == "snapshot":
        await c.manager.db.execute(
            f"DELETE FROM {c.manager.events} WHERE event_type='model_attempt_tracking' AND job_id=?", (c.caller.job_id,)
        )
    else:
        await c.manager.db.execute(
            f"INSERT INTO {c.manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "fork-proof",
                proof["job_id"],
                "input_disposition",
                proof["job_version"],
                proof["request_hash"],
                proof["payload_json"],
                "not_applicable",
                c.now[0],
            ),
        )
    try:
        with pytest.raises(JobError):
            await effects.apply(c.manager, c.main, h.event_id)
        if fault == "operation":
            assert (await c.manager.db.fetchone(f"SELECT delivery_status FROM {c.manager.events} WHERE event_id=?",
                (op['event_id'],)))['delivery_status'] == 'pending_delivery'
    finally:
        if fault == "operation":
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET delivery_status=? WHERE event_id=?",
                                      (original_status['delivery_status'], op['event_id']))


@pytest.mark.parametrize("existing", [False, True])
async def test_later_generation_cannot_receive_historical_disposition_writes(handoff, existing):
    h, c = handoff, handoff.c
    op = await _applied(h)
    await exited(h)
    if existing:
        await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
        op = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (op["event_id"],))
    fields = ('generation', 'status', 'resource_state', 'intake', 'outcome', 'result_json')
    original = await c.manager.db.fetchone(f"SELECT {','.join(fields)} FROM {c.manager.jobs} WHERE job_id=?", (c.owner.job_id,))
    await c.manager.db.execute(
        f"UPDATE {c.manager.jobs} SET generation=2,status='running',resource_state='running',intake='open',outcome=NULL,result_json=NULL WHERE job_id=?",
        (c.owner.job_id,),
    )
    try:
        before = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.owner.job_id,))
        async with c.manager._read() as tx:
            if existing:
                assert await inputs.read_tx(c.manager, tx, op, required=True)
            else:
                with pytest.raises(JobError, match="historical_read_only"):
                    await inputs.reconcile_tx(c.manager, tx, c.main, op)
        assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.owner.job_id,)) == before
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET " + ','.join(field+'=?' for field in fields) + ' WHERE job_id=?',
            (*[original[field] for field in fields], c.owner.job_id))


@pytest.mark.parametrize("delivery", ["pending", "claimed", "delivered", "missing_receipt"])
async def test_ordinary_nudge_retains_call_receipt_and_never_claims_model_consumption(nudge_case, delivery):
    c = nudge_case
    accepted, _ = await c.accept()
    if delivery != "pending" and delivery != "missing_receipt":
        [item] = await c.store.inbox_claim(c.sid)
        if delivery == "delivered":
            await c.store.append_message(
                c.sid,
                role="user",
                content=item["content"],
                send_index=1,
                inbox={"claim_token": item["claim_token"], "ids": [item["id"]]},
                meta={"inbox": {"item_ids": [item["item_id"]], "kinds": ["relay"]}},
            )

    async def bind_receipt():
        from power_loop.runtime.nudges import receipt

        seq = await c.store.append_message(
            c.actor.session_id,
            role="tool",
            name="spawn_agent",
            content=_json(receipt(accepted, False)),
            tool_call_id="nudge-call",
            send_index=1,
            round_index=0,
        )
        await c.manager.bind_receipt(
            c.actor, accepted.operation_id, await c.store.message_ref(c.actor.session_id, seq), caller=c.caller
        )

    if delivery != "missing_receipt":
        await bind_receipt()
    runtime = TaskRuntime(
        c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id, executor_id="child", boot_id="boot"
    )
    await attempts.enable(Execution(runtime, c.child_actor, c.receiver))
    await c.manager.request_cancel(
        c.actor, c.target.job_id, system_key="original-manual-stop", reason="external cancellation"
    )
    await c.manager.close_model_execution(c.child_actor, c.receiver, evidence={"exit": "cancelled"})
    await c.manager.record_execution_result(
        c.child_actor, c.target, outcome="cancelled", result={"status": "cancelled"}, resource_state="stopped"
    )
    op = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (accepted.operation_id,))
    if delivery == "missing_receipt":
        with pytest.raises(JobError, match="receipt_missing"):
            async with c.manager._write() as tx:
                await inputs.reconcile_tx(c.manager, tx, c.actor, op)
        [queue] = await c.store.inbox_items(c.sid)
        assert queue["status"] == "pending"
        await bind_receipt()
        op = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (accepted.operation_id,))
    before = await c.store.load_all_messages(c.sid)
    async with c.manager._write() as tx:
        proof = await inputs.reconcile_tx(c.manager, tx, c.actor, op)
    data = json.loads(proof["payload_json"])
    assert data["raw_origin"]["call"]["ref"]["id"] != data["raw_origin"]["receipt"]["ref"]["id"]
    assert data["disposition"] == ("delivered_unprocessed" if delivery == "delivered" else "withdrawn_before_delivery")
    assert await c.store.load_all_messages(c.sid) == before
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='processing_decision'"
    )
    updated = await c.manager.db.fetchone(
        f"SELECT * FROM {c.manager.events} WHERE event_id=?", (accepted.operation_id,)
    )
    from power_loop.runtime.processing import require_event_handled

    async with c.manager._write() as tx:
        await require_event_handled(c.manager, tx, updated, session_id=c.sid, owner_job_id=c.target.job_id)








async def test_closed_owner_revalidates_ordinary_input_raw_call_even_without_a_wait_binding(handoff):
    h, c = handoff, handoff.c
    await _applied(h)
    caller = await c.manager.register_job(
        c.main,
        kind="send",
        session_id=c.main.session_id,
        system_key="ordinary-input-source",
        request={"send_index": 99},
    )
    lease = await c.manager.claim_job(
        c.main, caller.job.job_id, executor_id="main-source", boot_id="boot", expected_version=1
    )
    seq = await c.store.append_message(
        c.main.session_id,
        role="assistant",
        send_index=99,
        round_index=0,
        tool_calls=[
            {"id": "ordinary-call", "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}
        ],
    )
    call = await c.store.message_ref(c.main.session_id, seq)
    op, replayed = await c.manager.accept_nudge(
        c.main,
        c.owner,
        caller=lease,
        call_ref=call,
        tool_call_id="ordinary-call",
        session_id=c.sid,
        message="Additional original input",
    )
    from power_loop.runtime.nudges import receipt

    seq = await c.store.append_message(
        c.main.session_id,
        role="tool",
        name="spawn_agent",
        content=_json(receipt(op, replayed)),
        tool_call_id="ordinary-call",
        send_index=99,
        round_index=0,
    )
    await c.manager.bind_receipt(
        c.main, op.operation_id, await c.store.message_ref(c.main.session_id, seq), caller=lease
    )
    await c.manager.close_model_execution(c.main, lease, evidence={"exit": "completed"})
    await exited(h)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    await consume_owner(h)
    owner = await c.manager.get_job(c.main, c.owner.job_id)
    assert owner.supervision_status == "closed"
    assert await c.manager.finalize_supervision(c.main, owner.job_id, generation=1)
    await c.manager.db.execute(
        f"UPDATE {c.store.t.messages} SET content='changed original raw call' WHERE session_id=? AND id=?",
        (call.session_id, call.id),
    )
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.main, owner.job_id, generation=1)


@pytest.mark.parametrize("fault", ["text", "metadata", "queue_reference"])
async def test_closed_delivered_input_retains_and_revalidates_the_actual_raw_message(handoff, fault):
    h, c = handoff, handoff.c
    await _applied(h)
    delivered = await _deliver(c)
    await exited(h)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    await consume_owner(h)
    if fault == "text":
        await c.manager.db.execute(
            f"UPDATE {c.store.t.messages} SET content='replacement raw input' WHERE session_id=? AND id=?",
            (c.sid, delivered.id),
        )
    elif fault == "metadata":
        await c.manager.db.execute(
            f"UPDATE {c.store.t.messages} SET meta_json='{{}}' WHERE session_id=? AND id=?", (c.sid, delivered.id)
        )
    else:
        await c.manager.db.execute(
            f"UPDATE {c.store.t.follow_up_queue} SET delivered_seq=1 WHERE session_id=?", (c.sid,)
        )
    with pytest.raises(JobError):
        await effects.apply(c.manager, c.main, h.event_id)


@pytest.mark.parametrize("ending", ["failed", "timed_out", "budget_exceeded", "interrupted"])
async def test_each_actual_negative_outcome_is_retained_in_input_proof(handoff, ending):
    h, c = handoff, handoff.c
    await _applied(h)
    await exited(h, ending=ending)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    [proof] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='input_disposition'")
    assert json.loads(proof["payload_json"])["owner_outcome"] == ending
    assert (await c.manager.get_job(c.main, c.owner.job_id)).outcome == ending
    await consume_owner(h)


async def test_input_proof_revision_cannot_precede_its_original_operation(handoff):
    h, c = handoff, handoff.c
    op = await _applied(h)
    await exited(h)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    proof = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (inputs.identity(c.manager, op["event_id"]),))
    current = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (op["event_id"],))
    await c.manager.db.execute(
        f"UPDATE {c.manager.events} SET job_version=? WHERE event_id=?", (op["job_version"], proof["event_id"])
    )
    with pytest.raises(JobError, match="input_disposition_evidence_corrupt"):
        async with c.manager._read() as tx:
            await inputs.read_tx(c.manager, tx, current, required=True)
