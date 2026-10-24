"""Real negative-owner facts retire intent without inventing an applied effect."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import timer_waits as waits
from power_loop.runtime import wait_decisions as decisions
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_reviews as reviews
from power_loop.runtime import wait_terminal as terminal
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.task_runtime import Execution

from .test_wait_effects_192 import decide, handoff, registry_store, runtime, waiting
from .test_wait_reviews_192 import advance, defer
from .test_wait_terminal_192 import consume_owner

__all__ = ["handoff", "registry_store", "runtime", "waiting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def exited(h, *, ending="cancelled", snapshot=True, resource="stopped"):
    c = h.c
    if snapshot:
        await attempts.enable(Execution(h.loop.task_runtime, c.actor, c.caller))
    if ending == "cancelled":
        await c.manager.request_cancel(c.main, c.owner.job_id, system_key="external-original-stop", reason="external cancellation")
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": ending}, resource_state=resource)
    await c.manager.record_execution_result(c.actor, c.owner, outcome=ending, result={"status": ending}, resource_state=resource)


@pytest.mark.parametrize("choice", ["continue", "defer", "cancel"])
async def test_actual_negative_owner_closes_unapplied_intent_without_executing_it(handoff, choice):
    h, c = handoff, handoff.c
    await decide(h, choice)
    await exited(h)
    before_ops = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action IN ('nudge','stop') ORDER BY event_id")
    before = h.model.calls
    timer = h.registration.job.job_id
    await terminal.terminate(c.manager, c.main, timer)
    proof = await c.manager.finalize_supervision(c.main, timer, generation=0)
    replay = await effects.apply(c.manager, c.main, h.event_id)
    assert replay["status"] == "owner_terminated_without_effect" and not replay["execution_authority"]
    assert h.model.calls == before
    inspected = await decisions.inspect(c.manager, c.main, h.event_id)
    assert inspected['effect_status'] == 'not_applied' and inspected['disposition'] == replay
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_effect'")
    assert await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action IN ('nudge','stop') ORDER BY event_id") == before_ops
    reopened = JobManager(await c.connect(), now_ms=lambda: c.policy.deadline_at_ms + 1)
    assert await effects.apply(reopened, c.main, h.event_id) == replay
    assert await reopened.finalize_supervision(c.main, timer, generation=0) == proof
    await consume_owner(h)
    assert all(r["supervision_status"] == "closed" for r in await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}"))


async def test_active_owner_cannot_be_disposed_to_make_supervision_green(handoff):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    with pytest.raises(JobError):
        await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_disposition'")


async def test_applied_but_unprocessed_continuation_retains_effect_and_negative_input_proof(handoff):
    h, c = handoff, handoff.c
    await decide(h, "continue")
    original_effect = await effects.apply(c.manager, c.main, h.event_id)
    [effect] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_effect'")
    await exited(h)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    [queue] = await c.store.inbox_items(c.sid)
    assert queue['status'] == 'void'
    [current] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_effect'")
    assert current['payload_json'] == effect['payload_json'] and current['delivery_status'] == 'handled'
    replay = await effects.apply(c.manager, c.main, h.event_id)
    assert replay['status'] == 'owner_terminated_before_input_processed'
    assert replay['operation_id'] == original_effect['operation_id']
    assert not replay['execution_authority'] and not replay['billing_settled']


async def test_expired_legacy_delivered_defer_is_retired_without_a_new_review(handoff):
    h, c = handoff, handoff.c
    await defer(h)
    [effect] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_effect'")
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='wait_review'")
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET delivery_status='delivered' WHERE event_type IN ('wait_decision','wait_effect')")
    original = await waits.read_wait(c.manager, c.main, h.registration.job.job_id)
    await exited(h)
    c.now[0] = c.policy.deadline_at_ms + 1
    results = await asyncio.gather(*(reviews.drive(h.loop, c.main, h.registration.job.job_id) for _ in range(3)))
    assert results == [results[0]] * 3 and results[0]["status"] == "owner_terminated_without_effect"
    assert not results[0]["billing_settled"]
    [current] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_effect'")
    assert current["payload_json"] == effect["payload_json"] and current["delivery_status"] == "handled"
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_review'")
    assert await waits.read_wait(c.manager, c.main, h.registration.job.job_id) == original


async def test_multiple_reviews_then_unapplied_intent_retains_all_prior_cycles(handoff):
    h, c = handoff, handoff.c
    previous = []
    for _ in range(2):
        previous.append(await defer(h))
        await advance(h)
    h.model.calls = 0
    await decide(h, "continue")
    await exited(h)
    result = await reviews.drive(h.loop, c.main, h.registration.job.job_id)
    assert result["decision"] == "continue" and result["status"] == "owner_terminated_without_effect"
    for event_id, reply in previous:
        assert await effects.apply(c.manager, c.main, event_id) == reply
    await consume_owner(h)
    assert all(r["supervision_status"] == "closed" for r in await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}"))


@pytest.mark.parametrize("fault", ["call_missing", "arguments", "receipt_missing", "receipt_body", "observation", "model_response", "owner_tracking"])
async def test_corrupt_original_evidence_cannot_become_a_negative_ack(handoff, fault):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    await exited(h)
    [op] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='wait_decision'")
    call = json.loads(op["call_ref_json"])
    receipt = json.loads(op["receipt_ref_json"])
    if fault == "call_missing":
        await c.manager.db.execute(f"DELETE FROM {c.store.t.messages} WHERE session_id=? AND seq=?", (c.main.session_id, int(call["seq"])))
    elif fault == "arguments":
        row = await c.manager.db.fetchone(f"SELECT tool_calls_json FROM {c.store.t.messages} WHERE session_id=? AND seq=?", (c.main.session_id, int(call["seq"])))
        calls = json.loads(row["tool_calls_json"])
        calls[0]["function"]["arguments"] = _json({"action": "decide", "event_id": h.event_id, "decision": "cancel"})
        await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET tool_calls_json=? WHERE session_id=? AND seq=?", (_json(calls), c.main.session_id, int(call["seq"])))
    elif fault == "receipt_missing":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=NULL WHERE event_id=?", (op["event_id"],))
    elif fault == "receipt_body":
        await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET content='{{}}' WHERE session_id=? AND seq=?", (c.main.session_id, int(receipt["seq"])))
    elif fault == "owner_tracking":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='model_attempt_tracking'", (c.caller.job_id,))
    else:
        kind = "wait_observed" if fault == "observation" else "model_attempt_finished"
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type=?", (kind,))
    with pytest.raises(JobError):
        await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('wait_disposition','wait_terminated')")
    assert (await c.manager.get_job(c.main, h.registration.job.job_id)).outcome is None


@pytest.mark.parametrize("point", ["disposition", "handled", "supervision"])
async def test_disposition_and_complete_supervision_transaction_roll_back_together(handoff, monkeypatch, point):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    await exited(h)
    timer_id = h.registration.job.job_id
    before = await c.manager.get_job(c.main, timer_id)
    async with c.manager._write() as tx:
        cls = type(tx)
    execute = cls.execute
    async def fail(tx, sql, params=()):
        result = await execute(tx, sql, params)
        if ((point == "disposition" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "wait_disposition")
                or (point == "handled" and sql.startswith("UPDATE " + c.manager.events + " SET delivery_status='handled' WHERE event_id=?") and params[0].startswith("wait_decision_"))
                or (point == "supervision" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "supervision_closed")):
            raise OSError("negative disposition write failed")
        return result
    monkeypatch.setattr(cls, "execute", fail)
    with pytest.raises(OSError, match="negative disposition"):
        await c.manager.finalize_supervision(c.main, timer_id, generation=0)
    assert await c.manager.get_job(c.main, timer_id) == before
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type IN ('wait_disposition','wait_terminated','result','supervision_closed')", (timer_id,))
    [decision] = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE event_type='wait_decision'")
    assert decision["delivery_status"] == "pending_delivery"


async def test_committed_termination_ack_loss_replays_without_a_second_disposition(handoff, monkeypatch):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    await exited(h)
    write, lost = c.manager._write, []
    @asynccontextmanager
    async def lose_ack():
        async with write() as tx:
            before = await tx.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_disposition'")
            yield tx
            after = await tx.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_disposition'")
        if len(after) > len(before) and not lost:
            lost.append(True)
            raise OSError("negative responsibility committed; ACK lost")
    monkeypatch.setattr(c.manager, "_write", lose_ack)
    with pytest.raises(OSError, match="ACK lost"):
        await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    reopened = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    replies = await asyncio.gather(*(terminal.terminate(reopened, c.main, h.registration.job.job_id) for _ in range(3)))
    assert replies == [replies[0]] * 3 and lost == [True]
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_disposition'")) == 1


@pytest.mark.parametrize("fault", ["rehashed", "deleted", "fork", "raw_receipt", "role"])
async def test_closed_disposition_is_revalidated_and_cannot_be_repaired_by_rehashing(handoff, fault):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    await exited(h)
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_disposition'")
    payload = json.loads(event["payload_json"])
    if fault == "rehashed":
        payload["decision_kind"] = "cancel"
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (_json(payload), _hash(payload), event["event_id"]))
    elif fault == "deleted":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (event["event_id"],))
    elif fault == "role":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET caller_session_id=? WHERE event_id=?", (c.sid, event["event_id"]))
    elif fault == "raw_receipt":
        await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET name='forged' WHERE session_id=? AND seq=?", (c.main.session_id, int(payload["receipt_ref"]["seq"])))
    else:
        cols = list(event)
        await c.manager.db.execute(f"INSERT INTO {c.manager.events} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
            tuple('orphan-disposition' if key == 'event_id' else event[key] for key in cols))
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)


async def test_history_does_not_control_or_modify_a_later_owner_generation(handoff):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    await exited(h)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2,executor_id='later-executor',boot_id='later-boot',resource_state='running',cancel_requested_at=NULL WHERE job_id=?", (c.owner.job_id,))
    before = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.owner.job_id,))
    reply = await effects.apply(c.manager, c.main, h.event_id)
    assert reply['original_owner']['generation'] == '1'
    assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.owner.job_id,)) == before
    with pytest.raises(JobError):
        await effects.apply(c.manager, replace(c.main, principal_id='foreign'), h.event_id)


@pytest.mark.parametrize("state", ["dispatched", "unknown"])
async def test_unknown_or_outstanding_provider_never_becomes_a_zero_cost_negative_ack(handoff, state):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    binding = Execution(h.loop.task_runtime, c.actor, c.caller)
    await attempts.enable(binding)
    attempt = await attempts.prepare(binding, nonce='unconfirmed-original-provider', round_index=0, retry_index=1,
        request_hash='a'*64, model='test-model', max_tokens=10)
    await attempts.dispatch(attempt)
    if state == "unknown":
        await attempts.finish(attempt, status='unknown', usage={'total_tokens': 11}, estimated=True)
    await exited(h, snapshot=False)
    with pytest.raises(JobError):
        await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_disposition'")
