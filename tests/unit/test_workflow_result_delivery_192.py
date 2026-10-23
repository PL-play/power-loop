"""Canonical workflow delivery: generations, crash repair and no billing replay."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import execution_membership, workflow_results
from power_loop.runtime.jobs import JobError, JobLease, JobManager
from power_loop.runtime.task_runtime import Execution
from power_loop.workflow import journal

from .test_job_registry_192 import registry_store
from .test_processing_decisions_192 import processing_case, runtime
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "runtime", "managed_workflow", "processing_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_delivery_identity_fits_all_schema_dialects_and_partitions_generations():
    job_id = "job_" + "a"*32
    ids = [workflow_results.delivery_event_id(job_id, generation) for generation in (0, 1, 2, 2**63-1)]
    assert len(set(ids)) == 4 and all(len(value) <= 64 for value in ids)
    assert workflow_results.delivery_event_id("job_"+"b"*32, 1) not in ids
    assert all(len(workflow_results._id(kind, job_id, 2**63-1)) <= 64 for kind in ("projected", "cb_start", "cb_exit"))


@pytest.fixture
async def delivery_case(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, sid, state, connect = managed_workflow
    model.hold.clear()
    state["on_complete"] = None
    async def unavailable(*args, **kwargs):
        raise RuntimeError("test: projection unavailable after canonical result commit")
    with monkeypatch.context() as patch:
        patch.setattr(workflow_results, "project", unavailable)
        await loop.send("run", sid)
        await settle(managed_workflow)
    handle = state["handles"][0]
    row = await manager._row(manager.db, handle.job_id)
    from power_loop.runtime.workflow_controller import proof
    return SimpleNamespace(store=store, manager=manager, runtime=runtime, loop=loop, model=model,
        actor=runtime.actor(sid), job_id=handle.job_id, run_id=handle.run_id, lease=JobLease(**proof(row)),
        connect=connect, state=state)


async def test_projection_reopens_without_driver_and_deduplicates_after_main_consumption(delivery_case):
    c = delivery_case
    assert await c.store.get_session((await c.manager.get_job(c.actor, c.job_id)).session_id) is None
    counts = c.model.counts.copy()
    data = await workflow_results.prepare(c.manager, c.actor, c.job_id, 1, lease=c.lease)
    reopened = JobManager(await c.connect())
    assert await workflow_results.prepare(reopened, c.actor, c.job_id, 1) == data
    for _ in range(2):
        assert await workflow_results.project(c.loop, c.actor, c.job_id, 1) == data["envelope"]
    envelope = data["envelope"]
    assert envelope["event_id"] == envelope["result_event_id"] == "result_" + c.job_id
    assert envelope["generation"] == 1 and envelope["result_revision"] > 0
    assert c.model.counts == counts and (await c.store.inbox_counts(c.actor.session_id))["pending"] == 1
    await c.loop.flush_follow_ups(c.actor.session_id)
    assert (await c.manager.get_job(c.actor, c.job_id)).result_status == "handled"
    before = c.model.counts.copy()
    await workflow_results.project(c.loop, c.actor, c.job_id, 1)
    assert c.model.counts == before and (await c.store.inbox_counts(c.actor.session_id))["pending"] == 0
    assert not c.state["notifications"]


@pytest.mark.parametrize("fault", ["before_commit", "lost_ack", "journal", "inbox", "inbox_ack"])
async def test_failed_projection_repairs_views_only_and_preserves_one_revision(delivery_case, monkeypatch, fault):
    c = delivery_case
    original_write = c.manager._write
    @asynccontextmanager
    async def broken_write():
        async with original_write() as tx:
            async def execute(sql, params=()):
                if fault == "journal" and sql.startswith(f"UPDATE {c.store.t.session_runtime_state} SET value_json="):
                    raise RuntimeError("journal unavailable")
                inbox = sql.startswith("INSERT") and c.store.t.follow_up_queue in sql
                if fault == "inbox" and inbox:
                    raise RuntimeError("inbox unavailable")
                value = await tx.execute(sql, params)
                if fault == "inbox_ack" and inbox:
                    raise RuntimeError("inbox statement ACK lost before transaction commit")
                return value
            yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
            if fault == "before_commit":
                raise RuntimeError("rollback")
        if fault == "lost_ack":
            raise RuntimeError("commit ACK lost")
    counts = c.model.counts.copy()
    canonical = await c.manager.get_result_revision(c.actor, c.job_id, 1)
    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", broken_write)
        with pytest.raises(RuntimeError):
            await workflow_results.project(c.loop, c.actor, c.job_id, 1)
    rows = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_delivery'")
    assert len(rows) == (1 if fault == "lost_ack" else 0)
    await workflow_results.project(c.loop, c.actor, c.job_id, 1)
    await workflow_results.project(c.loop, c.actor, c.job_id, 1)
    assert c.model.counts == counts and not c.state["notifications"]
    assert await c.manager.get_result_revision(c.actor, c.job_id, 1) == canonical
    assert (await c.store.inbox_counts(c.actor.session_id))["pending"] == 1


async def test_competing_managers_freeze_one_identical_delivery(delivery_case):
    c = delivery_case
    other = JobManager(await c.connect())
    one, two = await asyncio.gather(*(workflow_results.prepare(m, c.actor, c.job_id, 1) for m in (c.manager, other)))
    assert one == two
    rows = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_delivery'")
    assert len(rows) == 1


@pytest.mark.parametrize("fault", ["result_missing", "result_json", "live_result", "scope", "unsealed", "lease", "generation",
                                  "journal_missing", "journal_generation", "journal_result", "projection_json", "projection_hash", "projection_type",
                                  "projection_version", "controller", "creation", "result_generation", "role",
                                  "unstarted_corrupt", "driver_missing", "journal_json"])
async def test_missing_or_corrupt_evidence_never_projects_a_guess(delivery_case, fault):
    c = delivery_case
    if fault.startswith("projection_"):
        await workflow_results.prepare(c.manager, c.actor, c.job_id, 1)
        field, value = {"projection_json": ("payload_json", "{}"), "projection_hash": ("request_hash", "wrong"),
                        "projection_type": ("event_type", "other"), "projection_version": ("job_version", 999)}[fault]
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET {field}=? WHERE event_id=?",
            (value, workflow_results.delivery_event_id(c.job_id, 1)))
    elif fault in ("result_missing", "scope", "unsealed"):
        key = {"result_missing": "result_"+c.job_id,
               "scope": execution_membership.scope_id(c.manager, c.job_id, 1),
               "unsealed": execution_membership.seal_id(c.manager, c.job_id, 1)}[fault]
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
    elif fault == "result_json":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json='[]' WHERE event_id=?", ("result_"+c.job_id,))
    elif fault == "live_result":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET result_json=? WHERE job_id=?", ("{}", c.job_id))
    elif fault == "role":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET execution_role='main' WHERE job_id=?", (c.job_id,))
    elif fault == "unstarted_corrupt":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=0 WHERE job_id=?", (c.job_id,))
    elif fault == "driver_missing":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET session_id=NULL WHERE job_id=?", (c.job_id,))
    elif fault == "result_generation":
        row = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", ("result_"+c.job_id,))
        data = json.loads(row["payload_json"])
        data["generation"] = True
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), "result_"+c.job_id))
    elif fault in ("controller", "creation"):
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND " +
            ("event_type='workflow_controller'" if fault == "controller" else "action='create'"), (c.job_id,))
    elif fault.startswith("journal_"):
        if fault == "journal_missing":
            await c.store.delete_runtime_state(c.actor.session_id, journal.run_key(c.run_id))
        elif fault == "journal_json":
            await c.manager.db.execute(f"UPDATE {c.store.t.session_runtime_state} SET value_json=? WHERE session_id=? AND state_key=?",
                ("[", c.actor.session_id, journal.run_key(c.run_id)))
        else:
            await journal.update(c.store, c.actor.session_id, c.run_id,
                **({"task_generation": 2} if fault == "journal_generation" else {"status": "completed", "result": {}}))
    counts = c.model.counts.copy()
    with pytest.raises(JobError):
        await workflow_results.project(c.loop, c.actor, c.job_id, 2 if fault == "generation" else 0 if fault == "unstarted_corrupt" else 1,
            lease=replace(c.lease, executor_id="wrong") if fault == "lease" else None if fault == "unstarted_corrupt" else c.lease)
    assert c.model.counts == counts and (await c.store.inbox_counts(c.actor.session_id))["pending"] == 0


@pytest.mark.parametrize("generation", [True, False, -1, 1.0, "1", None, 2**63])
async def test_delivery_rejects_noninteger_generations(delivery_case, generation):
    c = delivery_case
    with pytest.raises(JobError, match="invalid_result_generation"):
        await workflow_results.prepare(c.manager, c.actor, c.job_id, generation)


async def test_empty_new_execution_ledger_replaces_old_rollup_and_preserves_history(delivery_case):
    c = delivery_case
    original = await c.manager.get_result_revision(c.actor, c.job_id, 1)
    old_result = (await c.manager.get_job(c.actor, c.job_id)).result
    assert old_result["usage"]["total_tokens"] == 20
    # History-only fixture, no controller/admission/dispatch permission. A
    # future empty generation must not charge the reused nodes' old rollup.
    async with c.manager._write() as tx:
        await tx.execute(f"UPDATE {c.manager.jobs} SET generation=2,executor_id='future',status='running',resource_state='running',"
                         "result_status='none',result_json=NULL,outcome=NULL WHERE job_id=?", (c.job_id,))
        await execution_membership.start(c.manager, tx, await c.manager._row(tx, c.job_id))
    binding = Execution(c.runtime, c.actor, replace(c.lease, generation=2, executor_id="future"))
    # Storage-only synthetic history is no longer permission to report a
    # successful restored workflow. Preserve that guard and exercise the empty
    # ledger through explicit negative evidence instead.
    with pytest.raises(JobError, match="workflow_restore_execution_mismatch"):
        await c.runtime.record_scope(binding, old_result)
    saved = await c.runtime.record_scope(binding, {**old_result, "status": "failed"})
    assert saved["usage"] == {} and old_result["usage"]["total_tokens"] == 20
    assert await c.manager.execution_usage(c.actor, c.job_id, generation=1) == old_result["usage"]
    assert await c.manager.execution_usage(c.actor, c.job_id, generation=2) == {}
    assert await c.manager.get_result_revision(c.actor, c.job_id, 1) == original
    await journal.update(c.store, c.actor.session_id, c.run_id, task_generation=2)
    assert (await c.manager.get_result_revision(c.actor, c.job_id, 2))["payload"]["result"]["usage"] == {}
    with pytest.raises(JobError, match="workflow_restore_execution_mismatch"):
        await workflow_results.prepare(c.manager, c.actor, c.job_id, 2)
    with pytest.raises(JobError, match="generation_changed"):
        await workflow_results.prepare(c.manager, c.actor, c.job_id, 1)
    with pytest.raises(JobError):
        await c.manager.check_execution(c.actor, binding.lease)


@pytest.mark.parametrize("stage", ["prepare", "commit"])
async def test_old_workflow_result_does_not_enter_new_generation_request(processing_case, stage):
    c = processing_case
    job_id, row = await c.result(kind="workflow")
    if stage == "commit":
        snapshot = await c.prepare([row])
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2,result_status='pending_delivery' WHERE job_id=?", (job_id,))
    assert await c.prepare([row]) is None
    if stage == "commit":
        with pytest.raises(JobError, match="processing_revision_changed"):
            await c.commit(snapshot, await c.response())
    assert (await c.manager.get_job(c.actor, job_id)).result_status == "pending_delivery"


@pytest.mark.parametrize("field", ["tenant_id", "principal_id", "session_id", "driver"])
async def test_projection_is_restricted_to_original_main_subject(delivery_case, field):
    c = delivery_case
    actor = (replace(c.actor, session_id=(await c.manager.get_job(c.actor, c.job_id)).session_id)
             if field == "driver" else replace(c.actor, **{field: "foreign"}))
    with pytest.raises(JobError):
        await workflow_results.project(c.loop, actor, c.job_id, 1)
    assert (await c.store.inbox_counts(c.actor.session_id))["pending"] == 0


@pytest.mark.parametrize("fault", ["content", "kind", "mode", "void", "missing", "journal_noop"])
async def test_silent_view_conflict_cannot_be_reported_as_delivery(delivery_case, monkeypatch, fault):
    c = delivery_case
    key = "result_"+c.job_id
    if fault in ("journal_noop", "missing"):
        original = c.manager._write
        @asynccontextmanager
        async def no_op():
            async with original() as tx:
                async def execute(sql, params=()):
                    if ((fault == "missing" and sql.startswith("INSERT") and c.store.t.follow_up_queue in sql) or
                            (fault == "journal_noop" and sql.startswith(f"UPDATE {c.store.t.session_runtime_state} SET value_json="))):
                        return 0
                    return await tx.execute(sql, params)
                yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
        monkeypatch.setattr(c.manager, "_write", no_op)
    else:
        data = await workflow_results.prepare(c.manager, c.actor, c.job_id, 1)
        content = json.dumps(data["envelope"], ensure_ascii=False)
        await c.store.inbox_put(c.actor.session_id, [{"item_id": key, "kind": "task_done", "content": content}])
        field, value = {"content": ("content", "wrong"), "kind": ("kind", "wrong"), "mode": ("mode", "steer"),
                        "void": ("status", "void")}[fault]
        await c.manager.db.execute(f"UPDATE {c.store.t.follow_up_queue} SET {field}=? WHERE item_id=?", (value, key))
    with pytest.raises(JobError, match="workflow_delivery_(journal|inbox)_mismatch"):
        await workflow_results.project(c.loop, c.actor, c.job_id, 1)
    assert (await c.manager.get_job(c.actor, c.job_id)).result_status == "pending_delivery"


@pytest.mark.parametrize("ending", ["cancel", "claim_failure"])
async def test_unstarted_failure_uses_generation_zero_without_fabricated_scope(managed_workflow, monkeypatch, ending):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    state["on_complete"] = None
    original = runtime.claim
    release = asyncio.Event()
    async def unstarted(actor, registration):
        if registration.job.kind != "workflow":
            return await original(actor, registration)
        await release.wait()
        if ending == "cancel":
            await manager.request_cancel(actor, registration.job.job_id, system_key="unstarted", reason="stop")
        raise RuntimeError("test: execution never claimed")
    monkeypatch.setattr(runtime, "claim", unstarted)
    await loop.send("run", sid)
    release.set()
    await settle(managed_workflow)
    job_id = state["handles"][0].job_id
    envelope = await workflow_results.project(loop, runtime.actor(sid), job_id, 0)
    assert envelope["generation"] == 0 and envelope["status"] == ("cancelled" if ending == "cancel" else "failed")
    assert not model.counts["BUILD"] and (await store.inbox_counts(sid))["pending"] == 1
    assert not await manager.db.fetchone(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='execution_scope'", (job_id,))


async def test_cancel_racing_result_commit_keeps_effective_cancel_and_actual_exit_usage(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    original = manager.record_execution_result
    async def cancel_before_commit(actor, lease, **kwargs):
        row = await manager.get_job(actor, lease.job_id)
        if row.kind == "workflow":
            await manager.request_cancel(actor, lease.job_id, system_key="last-moment-stop", reason="stop")
        return await original(actor, lease, **kwargs)
    monkeypatch.setattr(manager, "record_execution_result", cancel_before_commit)
    await loop.send("run", sid)
    await settle(managed_workflow)
    job_id = state["handles"][0].job_id
    saved = await manager.get_job(runtime.actor(sid), job_id)
    canonical = await manager.get_result_revision(runtime.actor(sid), job_id, 1)
    assert saved.result["execution_status"] == canonical["payload"]["result"]["status"] == "completed"
    assert saved.result["status"] == saved.outcome == "cancelled" and saved.result["usage"]["total_tokens"] == 20
    envelope = await workflow_results.project(loop, runtime.actor(sid), job_id, 1)
    assert envelope["status"] == "cancelled" and "cancelled" in envelope["summary"]
    assert not state["notifications"] and (await store.inbox_counts(sid))["pending"] == 1
