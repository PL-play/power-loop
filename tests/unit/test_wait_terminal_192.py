"""Generation-zero admission closure, stopped owners and frozen model exits."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_terminal as terminal
from power_loop.runtime.jobs import JobError, JobLease, JobManager, _json
from power_loop.runtime.task_runtime import Execution

from .test_model_attempts_192 import attempt_case, completed, prepare
from .test_wait_effects_192 import (
    decide,
    handoff,
    managed_workflow,
    registry_store,
    runtime,
    waiting,
)
from .test_wait_effects_192 import (
    test_actual_workflow_node_consumes_original_system_input_before_review_can_start as exercise_node,
)

__all__ = ["attempt_case", "handoff", "registry_store", "runtime", "waiting", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def stop_owner(h, *, outcome="cancelled", resource="stopped", snapshot=True):
    c = h.c
    if snapshot:
        await attempts.enable(Execution(h.loop.task_runtime, c.actor, c.caller))
    await decide(h, "cancel")
    await effects.apply(c.manager, c.main, h.event_id)
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "cancelled"}, resource_state=resource)
    await c.manager.record_execution_result(c.actor, c.owner, outcome=outcome, result={"status": outcome}, resource_state=resource)


async def consume_owner(h):
    c = h.c
    root = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.root.job.job_id,))
    await c.manager.close_model_execution(c.main, JobLease(root["job_id"], root["executor_id"], root["boot_id"], root["generation"]), evidence={"exit": "completed"})
    [created] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND action='create'", (c.owner.job_id,))
    await c.store.inbox_put(c.main.session_id, [{"item_id": "subagent:" + created["event_id"][3:], "kind": "task_done", "content": "Original child result is ready."}])
    h.model.calls = 10
    await h.loop.flush_follow_ups(c.main.session_id)


async def test_managed_wait_cannot_claim_a_fictional_executor(handoff):
    h, c = handoff, handoff.c
    before = await c.manager.get_job(c.main, h.registration.job.job_id)
    with pytest.raises(JobError, match="managed_wait_has_no_executor"):
        await c.manager.claim_job(c.main, before.job_id, executor_id="timer", boot_id="boot", expected_version=before.version)
    assert (await c.manager.get_job(c.main, before.job_id)).generation == 0
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='execution_scope'", (before.job_id,))


async def test_wait_termination_and_closure_are_recoverable_and_business_ack_does_not_change_proof(handoff):
    h, c = handoff, handoff.c
    await stop_owner(h)
    replies = await asyncio.gather(*(terminal.terminate(c.manager, c.main, h.registration.job.job_id) for _ in range(3)))
    assert all(r == replies[0] for r in replies)
    first = await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    [closure] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (first["event_id"],))
    data = json.loads(closure["payload_json"])
    assert data["version"] == 2 and data["scope"]["generation"] == 0 and data["scope"]["executor_id"] is None
    assert (await effects.apply(c.manager, c.main, h.event_id))["effect_status"] == "handled"
    restored = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    assert await terminal.terminate(restored, c.main, h.registration.job.job_id) == replies[0]
    # The owner result's eventual business ACK refines delivery, not physical exit.
    await consume_owner(h)
    assert await restored.finalize_supervision(c.main, h.registration.job.job_id, generation=0) == first
    rows = await c.manager.db.fetchall(f"SELECT kind,status,supervision_status FROM {c.manager.jobs}")
    assert all(row["supervision_status"] == "closed" for row in rows), rows


async def test_original_tree_finalization_can_terminate_wait_in_its_own_atomic_transaction(handoff):
    h, c = handoff, handoff.c
    await stop_owner(h)
    assert (await c.manager.get_job(c.main, h.registration.job.job_id)).outcome is None
    await consume_owner(h)
    timer = await c.manager.get_job(c.main, h.registration.job.job_id)
    assert timer.outcome == "cancelled" and timer.generation == 0 and timer.supervision_status == "closed"
    assert len(await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_terminated'")) == 1


async def test_successful_workflow_tree_closes_its_unexecuted_wait_without_any_extra_model(managed_workflow, monkeypatch):
    managed_workflow[6]["on_complete"] = None
    await exercise_node(managed_workflow, monkeypatch)
    _, manager, task_runtime, loop, model, parent, state, _ = managed_workflow
    before = model.counts.copy()
    await loop.flush_follow_ups(parent)
    assert model.counts["BUILD"] == before["BUILD"] and model.counts["REVIEW"] == before["REVIEW"]
    rows = await manager.db.fetchall(f"SELECT kind,generation,supervision_status FROM {manager.jobs}")
    assert all(r["supervision_status"] == "closed" for r in rows), rows
    assert (await manager.get_job(task_runtime.actor(parent), state["handles"][0].job_id)).supervision_status == "closed"


async def test_wait_closure_does_not_turn_a_callback_return_into_a_business_delivery_receipt(managed_workflow, monkeypatch):
    await exercise_node(managed_workflow, monkeypatch)
    _, manager, task_runtime, loop, _, parent, state, _ = managed_workflow
    await loop.flush_follow_ups(parent)
    [timer] = await manager.db.fetchall(f"SELECT supervision_status FROM {manager.jobs} WHERE kind='timer'")
    assert timer["supervision_status"] == "closed"
    with pytest.raises(JobError, match="finalization_notification_receipt_required"):
        await manager.finalize_supervision(task_runtime.actor(parent), state["handles"][0].job_id, generation=1)


@pytest.mark.parametrize("fault", ["running", "unknown", "legacy_exit", "cache", "missing_result", "missing_seal"])
async def test_owner_stop_must_be_proven_before_wait_can_be_terminated(handoff, fault):
    h, c = handoff, handoff.c
    if fault == "running":
        await decide(h, "cancel")
        await effects.apply(c.manager, c.main, h.event_id)
    else:
        await stop_owner(h, resource="unknown" if fault == "unknown" else "stopped", snapshot=fault != "legacy_exit")
        if fault == "cache":
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET resource_state='unknown' WHERE job_id=?", (c.owner.job_id,))
        elif fault == "missing_result":
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (c.owner.job_id,))
        elif fault == "missing_seal":
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='execution_membership_seal'", (c.owner.job_id,))
    with pytest.raises(JobError):
        await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    assert (await c.manager.get_job(c.main, h.registration.job.job_id)).outcome is None


@pytest.mark.parametrize("point", ["result", "termination", "close_view", "closure"])
async def test_negative_wait_terminal_result_and_closure_roll_back_together(handoff, monkeypatch, point):
    h, c = handoff, handoff.c
    await stop_owner(h)
    async with c.manager._write() as tx:
        transaction_type = type(tx)
    execute = transaction_type.execute

    async def fail(tx, sql, params=()):
        result = await execute(tx, sql, params)
        if ((point == "result" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "result")
                or (point == "termination" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "wait_terminated")
                or (point == "close_view" and sql.startswith("UPDATE " + c.manager.jobs + " SET status=?,supervision_status='closed'"))
                or (point == "closure" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "supervision_closed")):
            raise RuntimeError("terminal transaction failed")
        return result

    monkeypatch.setattr(transaction_type, "execute", fail)
    with pytest.raises(RuntimeError, match="terminal transaction failed"):
        await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    timer = await c.manager.get_job(c.main, h.registration.job.job_id)
    assert timer.outcome is None and timer.resource_state == "not_started" and timer.supervision_status == "open"
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type IN ('wait_terminated','supervision_closed') AND job_id=?", (timer.job_id,))


@pytest.mark.parametrize("fault", ["creation", "terminal", "result", "snapshot", "scope", "proof_shape", "version"])
async def test_closed_zero_generation_revalidates_all_original_terminal_evidence(handoff, fault):
    h, c = handoff, handoff.c
    await stop_owner(h)
    closed = await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    if fault in ("creation", "terminal", "result"):
        kind = {"creation": "operation", "terminal": "wait_terminated", "result": "result"}[fault]
        events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type=?", (h.registration.job.job_id, kind))
        [event] = [e for e in events if fault != "creation" or e["action"] == "create"]
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json='{{}}' WHERE event_id=?", (event["event_id"],))
    elif fault == "snapshot":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='model_attempt_tracking'", (c.caller.job_id,))
    else:
        [event] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (closed["event_id"],))
        data = json.loads(event["payload_json"])
        if fault == "scope":
            data["scope"]["generation"] = True
        elif fault == "proof_shape":
            data["proof"] = []
        else:
            data["version"] = True
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (_json(data), closed["event_id"]))
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)


@pytest.mark.parametrize("ending", ["cancelled", "failed", "interrupted", "hit_round_limit"])
async def test_instrumented_negative_model_exit_closes_only_as_a_model_exit_not_a_successful_task(attempt_case, ending):
    c = attempt_case
    await completed(c)
    await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={"exit": ending})
    proof = await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)
    job = await c.manager.get_job(c.actor, c.binding.lease.job_id)
    assert job.status == "model_closed" and job.outcome is None and job.supervision_status == "closed"
    assert await c.manager.finalize_supervision(c.actor, job.job_id, generation=1) == proof


async def test_frozen_model_exit_catches_total_attempt_deletion_before_supervision(attempt_case):
    c = attempt_case
    await completed(c)
    await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={"exit": "cancelled"})
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type IN ('model_attempt_prepared','model_attempt_dispatched','model_attempt_finished')")
    with pytest.raises(JobError, match="model_exit_snapshot_mismatch"):
        await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)


async def test_snapshot_is_runtime_owned_not_caller_asserted(attempt_case):
    c = attempt_case
    with pytest.raises(JobError, match="reserved_model_exit_evidence"):
        await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={"exit": "cancelled", "model_attempt_snapshot": {}})


@pytest.mark.parametrize("state", ["dispatched", "unknown"])
async def test_pending_or_unknown_provider_never_gets_negative_supervision_closure(attempt_case, state):
    c = attempt_case
    attempt = await prepare(c)
    await attempts.dispatch(attempt)
    if state == "unknown":
        await attempts.finish(attempt, status="unknown", usage={}, estimated=True)
    await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={"exit": "cancelled"})
    with pytest.raises(JobError, match="model_attempt_unsettled"):
        await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)


@pytest.mark.parametrize("submitted", ["succeeded", "failed", "timed_out", "interrupted"])
async def test_cancellation_preserves_original_submission_and_publishes_effective_canonical_result(handoff, submitted):
    h, c = handoff, handoff.c
    await attempts.enable(Execution(h.loop.task_runtime, c.actor, c.caller))
    await decide(h, "cancel")
    await effects.apply(c.manager, c.main, h.event_id)
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "cancelled"})
    status = "completed" if submitted == "succeeded" else submitted
    request = {"outcome": submitted, "result": {"status": status, "final_text": "Original execution response."}, "resource_state": "stopped"}
    identity = await c.manager.record_execution_result(c.actor, c.owner, **request)
    assert identity == await c.manager.record_execution_result(c.actor, c.owner, **request)
    [event] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (identity,))
    data = json.loads(event["payload_json"])
    assert data["outcome"] == "cancelled" and data["result"]["status"] == "cancelled"
    assert data["requested"] == request and data["result"]["execution_status"] == status
    assert (await c.manager.get_job(c.main, c.owner.job_id)).result == data["result"]
    with pytest.raises(JobError, match="result_conflict"):
        await c.manager.record_execution_result(c.actor, c.owner, **{**request, "result": {**request["result"], "changed": True}})
    await terminal.terminate(c.manager, c.main, h.registration.job.job_id)
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)


async def test_unstarted_callback_result_is_consistent_replayable_and_can_be_adopted_after_owner_stop(handoff):
    h, c = handoff, handoff.c
    await stop_owner(h)
    first = await c.manager.record_unstarted_result(c.actor, h.registration.job.job_id, outcome="failed", result={"reason": "Original callback reported failure."})
    assert first["status"] == "cancelled"
    assert first == await c.manager.record_unstarted_result(c.actor, h.registration.job.job_id, outcome="failed", result={"reason": "Original callback reported failure."})
    [row] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", ("result_" + h.registration.job.job_id,))
    body = json.loads(row["payload_json"])
    assert body["outcome"] == "cancelled" and body["result"] == first and body["requested"]["outcome"] == "failed"
    await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    assert (await c.manager.get_job(c.main, h.registration.job.job_id)).supervision_status == "closed"


async def test_legacy_unstarted_payload_is_adopted_without_rewriting_its_content(handoff):
    h, c = handoff, handoff.c
    await stop_owner(h)
    await c.manager.record_unstarted_result(c.actor, h.registration.job.job_id, outcome="cancelled", result={"reason": "legacy callback"})
    identity = "result_" + h.registration.job.job_id
    legacy = {"outcome": "cancelled", "result": {"reason": "legacy callback"}, "resource_state": "stopped", "unstarted": True}
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (_json(legacy), identity))
    first = await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0)
    assert await c.manager.finalize_supervision(c.main, h.registration.job.job_id, generation=0) == first
    [unchanged] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (identity,))
    assert json.loads(unchanged["payload_json"]) == legacy


@pytest.mark.parametrize("option", [0, 1, "false", None])
async def test_quiescence_budget_option_is_not_interpreted_by_truthiness(attempt_case, option):
    c = attempt_case
    row = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.binding.lease.job_id,))
    async with c.manager._write() as tx:
        with pytest.raises(JobError, match="invalid_model_quiescence_options"):
            await attempts.require_quiescent(c.manager, tx, row, include_budget=option)
