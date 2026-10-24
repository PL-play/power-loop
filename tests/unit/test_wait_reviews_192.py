"""A defer transfers responsibility to a durable next cycle on the same timer."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import timer_waits as waits
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_notifications as notices
from power_loop.runtime import wait_reviews as reviews
from power_loop.runtime import wait_terminal as terminal
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.task_runtime import Execution

from .test_wait_effects_192 import child_loop, decide, handoff, registry_store, runtime, waiting
from .test_wait_terminal_192 import consume_owner, stop_owner

__all__ = ["handoff", "registry_store", "runtime", "waiting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def defer(h):
    original = h.event_id
    h.model.calls = 0
    await decide(h, "defer")
    replies = await asyncio.gather(*(effects.apply(h.c.manager, h.c.main, original) for _ in range(3)))
    assert replies == [replies[0]] * 3
    return original, replies[0]


async def advance(h):
    h.c.now[0] += 1000
    due = await waits.prepare_due(h.c.manager, h.c.main, h.registration.job.job_id)
    assert due["status"] == "decision_pending" and due["event_id"] != h.event_id
    h.event_id = due["event_id"]
    return due


async def test_defer_twice_then_continue_consumes_original_input_and_preserves_all_cycles(handoff):
    h, c = handoff, handoff.c
    saved = await waits.read_wait(c.manager, c.main, h.registration.job.job_id)
    first_due = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (h.event_id,))
    previous = []
    for number in (1, 2):
        event_id, reply = await defer(h)
        previous.append((event_id, reply))
        assert reply["effect_status"] == "handled"
        armed = await waits.prepare_due(c.manager, c.main, h.registration.job.job_id)
        assert armed["status"] == "armed" and armed["due_at_ms"] == reply["next_review_at_ms"]
        reopened = JobManager(await c.connect(), now_ms=lambda: c.now[0])
        assert await waits.prepare_due(reopened, c.main, h.registration.job.job_id) == armed
        due = await advance(h)
        assert due["payload"]["version"] == 2 and due["payload"]["review"]["cycle"] == number
        assert await waits.read_wait(c.manager, c.main, h.registration.job.job_id) == saved
        for old_id, old_reply in previous:
            assert await effects.apply(c.manager, c.main, old_id) == old_reply
    h.model.calls = 0
    await decide(h, "continue")
    await effects.apply(c.manager, c.main, h.event_id)
    loop, _, execution = await child_loop(h)
    async with h.loop.task_runtime.execution(execution, owner=True):
        result = await loop.flush_follow_ups(c.sid)
        await h.loop.task_runtime.record_scope(execution, {"status": "completed", "final_text": result.final_text})
    timer = await c.manager.get_job(c.main, h.registration.job.job_id)
    assert timer.outcome == "succeeded" and timer.generation == 0
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_review'")) == 2
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_due'")) == 3
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")) == 1
    assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (first_due["event_id"],)) != first_due  # Delivery status advances, content does not.
    last = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (first_due["event_id"],))
    assert last["request_hash"] == first_due["request_hash"] and last["payload_json"] == first_due["payload_json"]


async def test_cancel_after_two_reviews_closes_original_tree_and_freezes_every_cycle(handoff):
    h, c = handoff, handoff.c
    for _ in range(2):
        await defer(h)
        await advance(h)
    h.model.calls = 0
    await stop_owner(h)
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    proof = await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    [closure] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (proof["event_id"],))
    assert all(row["event_id"] in closure["payload_json"] for row in await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_review'"))
    await consume_owner(h)
    assert all(r["supervision_status"] == "closed" for r in await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}"))


@pytest.mark.parametrize("fault", ["missing", "wrong_type", "cycle", "hash", "deadline", "fork", "due_fork", "decision_receipt", "source_receipt", "observation"])
async def test_corrupt_cycle_or_original_provenance_cannot_issue_a_new_notification(handoff, fault):
    h, c = handoff, handoff.c
    await defer(h)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_review'")
    data = json.loads(row["payload_json"])
    if fault == "missing":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (row["event_id"],))
    elif fault == "wrong_type":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET event_type='operation' WHERE event_id=?", (row["event_id"],))
    elif fault in ("cycle", "hash", "deadline"):
        if fault == "cycle":
            data["cycle"] = 2
        elif fault == "hash":
            data["previous_due_hash"] = "forged"
        else:
            data["review_at_ms"] = c.policy.deadline_at_ms + 1
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (_json(data), _hash(data), row["event_id"]))
    elif fault in ("fork", "due_fork"):
        original = row if fault == "fork" else await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (h.event_id,))
        cols = list(original)
        await c.manager.db.execute(f"INSERT INTO {c.manager.events} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
            tuple("forged-branch" if key == "event_id" else original[key] for key in cols))
    elif fault.endswith("receipt"):
        action = "wait_decision" if fault == "decision_receipt" else "create"
        target = "" if action == "wait_decision" else " AND job_id='" + h.registration.job.job_id + "'"
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=NULL WHERE action=?" + target, (action,))
    else:
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='wait_observed'")
    c.now[0] += 1000
    with pytest.raises(JobError):
        await waits.prepare_due(c.manager, c.main, h.registration.job.job_id)
    assert len(await c.manager.db.fetchall(f"SELECT item_id FROM {c.store.t.follow_up_queue} WHERE kind='wait_handoff'")) == 1


@pytest.mark.parametrize("fault", ["cancel", "generation", "lease", "deadline"])
async def test_next_cycle_rechecks_original_authority_but_old_effect_remains_history(handoff, fault):
    h, c = handoff, handoff.c
    old_id, old_reply = await defer(h)
    c.now[0] += 1000
    if fault == "cancel":
        await c.manager.request_cancel(c.main, c.owner.job_id, reason="stop", system_key="cancel-review")
    elif fault == "generation":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (c.owner.job_id,))
    elif fault == "lease":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=1 WHERE job_id=?", (c.owner.job_id,))
    else:
        c.now[0] = c.policy.deadline_at_ms
    with pytest.raises(JobError):
        await waits.prepare_due(c.manager, c.main, h.registration.job.job_id)
    assert await effects.apply(c.manager, c.main, old_id) == old_reply
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_due'")) == 1


async def test_actual_review_commit_lost_ack_is_recalled_without_a_fork_or_new_work(handoff, monkeypatch):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    write, lost = c.manager._write, []
    @asynccontextmanager
    async def lose_ack():
        async with write() as tx:
            before = await tx.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_review'")
            yield tx
            after = await tx.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_review'")
        if len(after) > len(before) and not lost:
            lost.append(True)
            raise OSError("review transaction committed; ACK lost")
    monkeypatch.setattr(c.manager, "_write", lose_ack)
    with pytest.raises(OSError, match="ACK lost"):
        await effects.apply(c.manager, c.main, h.event_id)
    reopened = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    reply = await effects.apply(reopened, c.main, h.event_id)
    assert reply["effect_status"] == "handled" and lost == [True]
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_review'")) == 1
    await advance(h)


@pytest.mark.parametrize("expired", [False, True])
async def test_old_delivered_defer_is_upgraded_only_with_current_original_authority(handoff, expired):
    h, c = handoff, handoff.c
    old_id, reply = await defer(h)
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='wait_review'")
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET delivery_status='delivered' WHERE event_type IN ('wait_effect','wait_decision')")
    if expired:
        c.now[0] = c.policy.deadline_at_ms
        with pytest.raises(JobError, match="wait_deadline_exceeded"):
            await effects.apply(c.manager, c.main, old_id)
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_review'")
    else:
        assert await effects.apply(c.manager, c.main, old_id) == reply
        await advance(h)


async def test_repeatable_driver_delivers_only_to_main_and_never_runs_a_model_itself(handoff):
    h, c = handoff, handoff.c
    before = h.model.calls
    first = await reviews.drive(h.loop, c.main, h.registration.job.job_id)
    assert await reviews.drive(h.loop, c.main, h.registration.job.job_id) == first
    assert h.model.calls == before and first["status"] == "enqueued"
    h.model.options = {"event_id": h.event_id, "decision": "defer", "next_review_at_ms": c.now[0] + 1000}
    await h.loop.flush_follow_ups(c.main.session_id)
    before = h.model.calls
    reply = await reviews.drive(h.loop, c.main, h.registration.job.job_id)
    assert reply["status"] == "review_scheduled"
    assert (await reviews.drive(h.loop, c.main, h.registration.job.job_id))["status"] == "armed"
    c.now[0] += 1000
    second = await reviews.drive(h.loop, c.main, h.registration.job.job_id)
    assert second["event_id"] != h.event_id and second["main_session_id"] == c.main.session_id
    assert h.model.calls == before
    with pytest.raises(JobError, match="access_denied"):
        await reviews.drive(h.loop, c.actor, h.registration.job.job_id)
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_due'")) == 2


@pytest.mark.parametrize("point", ["review", "handled"])
async def test_review_and_responsibility_transfer_rollback_together(handoff, monkeypatch, point):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    before = await c.manager.get_job(c.main, h.registration.job.job_id)
    async with c.manager._write() as tx:
        cls = type(tx)
    original = cls.execute
    async def fail(tx, sql, params=()):
        value = await original(tx, sql, params)
        if ((point == "review" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "wait_review")
                or (point == "handled" and sql.startswith("UPDATE " + c.manager.events + " SET delivery_status='handled'"))):
            raise OSError("review transfer storage fault")
        return value
    monkeypatch.setattr(cls, "execute", fail)
    with pytest.raises(OSError, match="review transfer"):
        await effects.apply(c.manager, c.main, h.event_id)
    assert await c.manager.get_job(c.main, h.registration.job.job_id) == before
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('wait_review','wait_effect')")
    [decision] = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE event_type='wait_decision'")
    assert decision["delivery_status"] == "pending_delivery"


async def test_explicit_cycle_limit_rejects_new_defer_before_acceptance_and_keeps_old_history(handoff, monkeypatch):
    h, c = handoff, handoff.c
    monkeypatch.setattr(reviews, "MAX_REVIEWS", 1)
    old_id, old_reply = await defer(h)
    await advance(h)
    h.model.calls = 0
    await decide(h, "defer")
    [decision] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")
    assert decision["event_id"] == notices.decision_id(c.manager, old_id)
    with pytest.raises(JobError, match="wait_decision_required"):
        await effects.apply(c.manager, c.main, h.event_id)
    assert await effects.apply(c.manager, c.main, old_id) == old_reply


async def test_stopped_owner_does_not_silently_ack_an_unapplied_defer(handoff):
    h, c = handoff, handoff.c
    await decide(h, "defer")
    await attempts.enable(Execution(h.loop.task_runtime, c.actor, c.caller))
    await c.manager.request_cancel(c.main, c.owner.job_id, system_key="stop-unapplied-review", reason="stop")
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "cancelled"})
    await c.manager.record_execution_result(c.actor, c.owner, outcome="cancelled", result={"status": "cancelled"}, resource_state="stopped")
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    [disposed] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='wait_disposition'")
    payload = json.loads(disposed["payload_json"])
    assert payload["disposition"] == "owner_terminated_without_effect" and payload["effect"] is None
    assert payload["decision_kind"] == "defer" and payload["termination"]["event_type"] == "wait_terminated"
    [decision] = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE event_type='wait_decision'")
    assert decision["delivery_status"] == "handled"
