"""Storage-level restore admission; these tests do not claim engine restoration."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import workflow_resume as resume
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobError, JobManager, _hash
from power_loop.runtime.task_runtime import _SEND
from power_loop.workflow import journal

from .test_job_registry_192 import registry_store
from .test_node_nudge_runtime_192 import client
from .test_workflow_restore_plan_192 import finished, preview
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def admitting(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, parent, state, connect = managed_workflow
    c = await client(managed_workflow, monkeypatch)
    handle = await finished(managed_workflow)
    plan = await preview(managed_workflow)
    actor = runtime.actor(parent)
    c.plan, c.handle, c.actor = plan, handle, actor
    c.result, c.args, c.claimed = None, None, None
    c.fault_hits = []
    async def issue(*, fault=None, mutation=None, claim=False, claim_fault=None):
        async def handler():
            source = _SEND.get()
            seq, call_id = current_invocation(parent)
            ref = await store.message_ref(parent, seq)
            c.args = dict(caller=source.lease, call_ref=ref, tool_call_id=call_id, generation=1,
                node_id="BUILD", message="compute 20+23", executor_id=runtime.executor_id, boot_id=runtime.boot_id,
                lease_ms=runtime.lease_ms, plan_hash=_hash(plan))
            if mutation:
                await mutation(c)
            original = manager._write
            if fault:
                @asynccontextmanager
                async def broken():
                    async with original() as tx:
                        async def execute(sql, params=()):
                            silent = ((fault == "silent_jobs" and sql.startswith(f"UPDATE {manager.jobs} SET generation=")) or
                                (fault == "silent_operation" and sql.startswith(f"UPDATE {manager.events} SET payload_json=")) or
                                (fault == "silent_execution" and "execution" in params) or
                                (fault == "silent_scope" and "execution_scope" in params) or
                                (fault == "silent_reservation" and "workflow_node_reservation" in params) or
                                (fault == "silent_journal" and sql.startswith(f"UPDATE {store.t.session_runtime_state} SET value_json=")))
                            if silent:
                                c.fault_hits.append(fault)
                                return 1  # A lying rowcount must not manufacture receipt evidence.
                            value = await tx.execute(sql, params)
                            if ((fault == "reservation" and "workflow_node_reservation" in params) or
                                    (fault == "journal" and sql.startswith(f"UPDATE {store.t.session_runtime_state} SET value_json="))):
                                raise RuntimeError("injected statement failure")
                            return value
                        yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
                        if fault == "rollback":
                            raise RuntimeError("precommit failure")
                        if fault == "cancel":
                            raise asyncio.CancelledError()
                    if fault == "commit_ack":
                        raise RuntimeError("lost commit ACK")
                with monkeypatch.context() as patch:
                    patch.setattr(manager, "_write", broken)
                    try:
                        await resume.admit(manager, actor, handle.job_id, **c.args)
                    except (RuntimeError, asyncio.CancelledError):
                        return {"fault": fault}
                raise AssertionError("injected failure not reached")
            c.result = await resume.admit(manager, actor, handle.job_id, **c.args)
            if claim:
                if claim_fault:
                    @asynccontextmanager
                    async def broken_claim():
                        async with original() as tx:
                            async def execute(sql, params=()):
                                if claim_fault == "silent" and "workflow_dispatch" in params:
                                    return 1
                                return await tx.execute(sql, params)
                            yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
                            if claim_fault == "rollback":
                                raise RuntimeError("claim rollback")
                            if claim_fault == "cancel":
                                raise asyncio.CancelledError()
                        raise RuntimeError("claim committed, ACK lost")
                    with monkeypatch.context() as patch:
                        patch.setattr(manager, "_write", broken_claim)
                        try:
                            await resume.claim_dispatch(manager, actor, c.result[0].operation_id,
                                caller=source.lease, token="owner-token")
                        except (RuntimeError, asyncio.CancelledError, JobError):
                            pass
                else:
                    c.claimed = await resume.claim_dispatch(manager, actor, c.result[0].operation_id,
                        caller=source.lease, token="owner-token")
            return {"operation_id": c.result[0].operation_id, "generation": c.result[0].result["generation"], "status": "accepted"}
        return await c.issue(handler=handler)
    c.admit = issue
    return c


async def test_atomic_admission_preserves_history_reserves_nodes_and_resets_only_current_journal(admitting):
    c = admitting
    old = await c.manager.get_result_revision(c.actor, c.handle.job_id, 1)
    nodes = {key: await c.manager.get_job(c.actor, value["lease"]["job_id"]) for key, value in c.plan["nodes"].items()}
    before = c.model.counts.copy()
    await c.admit()
    operation, repeated = c.result
    assert not repeated and operation.result["mode"] == "workflow_continue"
    assert operation.result["run_id"] == c.handle.run_id and operation.result["execution_id"] != c.handle.run_id
    assert operation.result["generation"] == 2 and operation.result["result_event_id"] == "result_"+c.handle.job_id+"_g2"
    current = await c.manager.get_job(c.actor, c.handle.job_id)
    assert current.generation == 2 and current.session_id is None and current.outcome is None and current.result is None
    assert await c.manager.get_result_revision(c.actor, c.handle.job_id, 1) == old
    for nid, node in nodes.items():
        assert await c.manager.get_job(c.actor, node.job_id) == node
    j = await journal.read(c.store, c.parent, c.handle.run_id)
    assert j["task_generation"] == 2 and j["status"] == "running" and j["result"] is None
    assert all(s["status"] == "pending" for s in j["steps"])
    records = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='workflow_node_reservation'")
    assert {json.loads(r["payload_json"])["node_id"] for r in records} == {"BUILD", "REVIEW"}
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'")
    assert c.model.counts["BUILD"] == before["BUILD"] and c.model.counts["REVIEW"] == before["REVIEW"]
    await c.manager.check_execution(c.actor, resume.JobLease(**operation.result["request"]["target"]))
    with pytest.raises(JobError, match="workflow_resume"):
        await c.manager.bind_workflow_driver(c.actor, resume.JobLease(**operation.result["request"]["target"]), "not-a-driver")


@pytest.mark.parametrize("fault", ["rollback", "cancel", "reservation", "journal", "commit_ack"])
async def test_every_admission_write_rolls_back_or_lost_ack_recalls_exactly_one_intent(admitting, fault):
    c = admitting
    before = await journal.read(c.store, c.parent, c.handle.run_id)
    old = await c.manager.get_result_revision(c.actor, c.handle.job_id, 1)
    await c.admit(fault=fault)
    other = JobManager(await c.connect())
    committed = fault == "commit_ack"
    current = await other.get_job(c.actor, c.handle.job_id)
    assert current.generation == (2 if committed else 1)
    restored = await journal.read(other.store, c.parent, c.handle.run_id)
    assert restored["task_generation"] == 2 if committed else restored == before
    events = await other.db.fetchall(f"SELECT event_id FROM {other.events} WHERE event_type='workflow_node_reservation'")
    assert len(events) == (2 if committed else 0)
    assert await other.get_result_revision(c.actor, c.handle.job_id, 1) == old
    if committed:
        replay, seen = await resume.admit(other, c.actor, c.handle.job_id, **c.args)
        assert seen and replay.result["generation"] == 2
        assert not await other.db.fetchall(f"SELECT event_id FROM {other.events} WHERE event_type='workflow_dispatch'")


async def test_replay_is_read_only_even_after_original_send_closed_but_changed_arguments_conflict(admitting):
    c = admitting
    await c.admit()
    assert (await c.manager.get_job(c.actor, c.args["caller"].job_id)).status == "model_closed"
    original = c.result[0]
    other = JobManager(await c.connect())
    replay, repeated = await resume.admit(other, c.actor, c.handle.job_id, **c.args)
    assert repeated and replay.operation_id == original.operation_id
    for altered in ({"message": "different"}, {"executor_id": "other"}, {"node_id": "REVIEW"}, {"plan_hash": "0"*64}):
        with pytest.raises(JobError, match="operation_conflict"):
            await resume.admit(other, c.actor, c.handle.job_id, **{**c.args, **altered})
    assert not await other.db.fetchall(f"SELECT event_id FROM {other.events} WHERE event_type='workflow_dispatch'")


async def test_dispatch_claim_requires_active_original_send_and_is_exclusive(admitting):
    c = admitting
    await c.admit(claim=True)
    assert c.claimed
    operation = c.result[0]
    assert await resume.claim_dispatch(c.manager, c.actor, operation.operation_id, caller=c.args["caller"], token="owner-token")
    assert not await resume.claim_dispatch(c.manager, c.actor, operation.operation_id, caller=c.args["caller"], token="other-token")
    with pytest.raises(JobError, match="dispatch_mismatch"):
        await resume.claim_dispatch(c.manager, c.actor, operation.operation_id, caller=replace(c.args["caller"], generation=99), token="owner-token")
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='workflow_dispatch'")
    assert event["event_id"] == resume.dispatch_id(operation.operation_id) and event["delivery_status"] == "claimed"


async def test_unclaimed_dispatch_is_not_restarted_after_the_originating_model_send_ends(admitting):
    c = admitting
    await c.admit()
    with pytest.raises(JobError, match="intake_closed|lease_expired"):
        await resume.claim_dispatch(c.manager, c.actor, c.result[0].operation_id, caller=c.args["caller"], token="owner-token")
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'")


@pytest.mark.parametrize("mutation", ["plan_hash", "result_changed", "cancelled", "pending_input"])
async def test_transaction_rechecks_live_evidence_and_never_trusts_preview_json(admitting, mutation):
    c = admitting
    async def change(_):
        if mutation == "plan_hash":
            c.args["plan_hash"] = "0"*64
        elif mutation == "result_changed":
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET result_json=? WHERE job_id=?", ('{"status":"completed"}', c.handle.job_id))
        elif mutation == "cancelled":
            await c.manager.request_cancel(c.actor, c.handle.job_id, system_key="test-cancel", reason="test")
        else:
            sid = c.plan["nodes"]["BUILD"]["binding"]["session_id"]
            await c.store.inbox_put(sid, [{"item_id": "late-input", "content": "late", "mode": "steer"}])
    await c.admit(mutation=change)
    assert c.result is None
    assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_node_reservation'")


async def test_two_independent_managers_admit_same_call_only_once(admitting):
    c, outcomes = admitting, []
    async def compete(_):
        other = JobManager(await c.connect())
        outcomes.extend(await asyncio.gather(resume.admit(c.manager, c.actor, c.handle.job_id, **c.args),
            resume.admit(other, c.actor, c.handle.job_id, **c.args)))
    await c.admit(mutation=compete)
    assert sorted(repeated for _, repeated in outcomes) == [False, True]
    assert len({op.operation_id for op, _ in outcomes}) == 1
    assert c.result[1] and (await c.manager.get_job(c.actor, c.handle.job_id)).generation == 2
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_node_reservation'")) == 2


@pytest.mark.parametrize("fault", ["rollback", "cancel", "commit_ack", "silent"])
async def test_dispatch_transaction_rollback_cancel_and_lost_ack_keep_single_responsibility(admitting, fault):
    c = admitting
    await c.admit(claim=True, claim_fault=fault)
    other = JobManager(await c.connect())
    rows = await other.db.fetchall(f"SELECT * FROM {other.events} WHERE event_type='workflow_dispatch'")
    assert len(rows) == int(fault == "commit_ack")
    if fault == "commit_ack":
        assert await resume.claim_dispatch(other, c.actor, c.result[0].operation_id, caller=c.args["caller"], token="owner-token")
        assert not await resume.claim_dispatch(other, c.actor, c.result[0].operation_id, caller=c.args["caller"], token="stranger")
    else:
        with pytest.raises(JobError):
            await resume.claim_dispatch(other, c.actor, c.result[0].operation_id, caller=c.args["caller"], token="owner-token")


@pytest.mark.parametrize("change", [
    {"generation": True}, {"generation": 0}, {"generation": 2**63-1}, {"node_id": []}, {"message": ""},
    {"message": "x\x00y"}, {"executor_id": ""}, {"lease_ms": True}, {"lease_ms": 0}, {"lease_ms": 300001},
    {"plan_hash": "not-a-hash"}, {"caller": None},
])
async def test_invalid_request_fields_fail_without_any_extra_execution(admitting, change):
    c = admitting
    await c.admit()
    with pytest.raises(JobError):
        await resume.admit(c.manager, c.actor, c.handle.job_id, **{**c.args, **change})
    assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation == 2


@pytest.mark.parametrize("field,value", [("delivery_status", "void"), ("event_type", "other"), ("request_hash", "0"*64)])
async def test_corrupt_claim_receipt_does_not_grant_dispatch_again(admitting, field, value):
    c = admitting
    await c.admit(claim=True)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET {field}=? WHERE event_id=?", (value, resume.dispatch_id(c.result[0].operation_id)))
    with pytest.raises(JobError, match="dispatch_mismatch"):
        await resume.claim_dispatch(c.manager, c.actor, c.result[0].operation_id, caller=c.args["caller"], token="owner-token")


@pytest.mark.parametrize("fault", ["silent_jobs", "silent_operation", "silent_execution", "silent_scope", "silent_reservation", "silent_journal"])
async def test_silent_writes_even_with_success_rowcount_cannot_commit_partial_admission(admitting, fault):
    c = admitting
    j = await journal.read(c.store, c.parent, c.handle.run_id)
    await c.admit(fault=fault)
    assert c.fault_hits
    assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation == 1
    assert await journal.read(c.store, c.parent, c.handle.run_id) == j
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_node_reservation'")


@pytest.mark.parametrize("fault", ["wrong_role", "stale_caller", "wrong_ref", "driver_collision"])
async def test_invalid_controller_or_call_identity_cannot_admit(admitting, fault):
    c = admitting
    async def damage(_):
        if fault == "wrong_role":
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET execution_role='subagent' WHERE job_id=?", (c.args["caller"].job_id,))
        elif fault == "stale_caller":
            c.args["caller"] = replace(c.args["caller"], generation=99)
        elif fault == "wrong_ref":
            c.args["call_ref"] = replace(c.args["call_ref"], send_index=99)
        else:
            key = c.manager._origin_key(c.actor, "nudge", c.args["call_ref"], c.args["tool_call_id"], None)
            target = dict(c.plan["workflow"], generation=2)
            sid = "wf_driver_"+_hash([c.store.store_id, key, target])[:48]
            await c.store.create_session(session_id=sid)
    await c.admit(mutation=damage)
    assert c.result is None
    assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation == 1


async def test_replayed_journal_step_and_spec_come_from_canonical_not_mutable_cache(admitting):
    c = admitting
    async def change(_):
        c.args["node_id"] = "REVIEW"
        plan = await c.manager.get_workflow_restore_plan(c.actor, c.handle.job_id, generation=1,
            node_id="REVIEW", message=c.args["message"])
        c.args["plan_hash"] = _hash(plan)
        j = await journal.read(c.store, c.parent, c.handle.run_id)
        j["spec"] = {"name": "edited-untrusted-view"}
        j["steps"][0]["text"] = "FORGED-OUTPUT"
        await c.store.set_runtime_state(c.parent, journal.run_key(c.handle.run_id), j)
    await c.admit(mutation=change)
    assert c.result and not c.result[1]
    j = await journal.read(c.store, c.parent, c.handle.run_id)
    assert j["spec"] == c.plan["spec"] and j["live"] == {}
    steps = {s["node_id"]: s for s in j["steps"]}
    assert steps["BUILD"]["text"] == "PROOF-42" and steps["BUILD"]["status"] == "completed"
    assert steps["REVIEW"]["status"] == "pending" and not steps["REVIEW"]["text"] and steps["REVIEW"]["payload"] is None
