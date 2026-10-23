"""Mechanical close/reopen uses durable original execution evidence only."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop.runtime import finalization, task_query
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_query import active_reasons
from power_loop.workflow.resume import resume_detached

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_node_resume_192 import ready_node
from .test_nudge_runtime_192 import issue_nudge
from .test_subagent_resume_192 import ready
from .test_task_runtime_192 import finish, managed
from .test_workflow_restore_plan_192 import finished
from .test_workflow_restore_runtime_192 import native_restore
from .test_workflow_task_runtime_192 import managed_workflow, records, settle

__all__ = ["registry_store", "registry_runtime", "managed_workflow", "managed", "ready", "ready_node", "native_restore"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def close(manager, actor, job):
    return await manager.finalize_supervision(actor, job.job_id, generation=job.generation)


async def test_inbox_workflow_closes_only_after_original_main_consumes(managed_workflow):
    store, manager, runtime, loop, model, sid, state, connect = managed_workflow
    handle = await finished(managed_workflow, consume=False)
    actor = runtime.actor(sid)
    before = await manager.get_job(actor, handle.job_id)
    assert before.supervision_status == "open"
    nodes = [r for r in await records(managed_workflow) if r.kind == "workflow_node"]
    assert nodes and all(r.supervision_status == "closed" and r.result_status == "handled" for r in nodes)
    with pytest.raises(JobError, match="required_result_unhandled"):
        await close(manager, actor, before)
    calls = model.counts.copy()
    await loop.flush_follow_ups(sid)
    assert model.counts["MAIN"] == calls["MAIN"] + 1
    rows = await records(managed_workflow)
    assert all(r.supervision_status == "closed" for r in rows), rows
    raw = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert all(not active_reasons(r) for r in raw)
    old = await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")
    wf = await manager.get_job(actor, handle.job_id)
    assert wf.status == wf.outcome == "succeeded"
    active = await task_query.list_jobs(manager, actor, root_job_id=wf.job_id, scope="self", status="active")
    terminal = await task_query.list_jobs(manager, actor, root_job_id=wf.job_id, scope="self", status="terminal")
    assert active["matched_count"] == 0 and not active["nodes"]
    assert terminal["matched_count"] == 1 and terminal["nodes"][0]["active_reason"] == []
    receipt = await close(manager, actor, wf)
    assert await close(JobManager(await connect()), actor, wf) == receipt
    assert await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id") == old
    assert await store.get_session(wf.session_id) is None


async def test_callback_return_is_not_business_receipt(managed_workflow):
    _, manager, runtime, loop, _, sid, state, _ = managed_workflow
    await loop.send("launch", sid)
    await settle(managed_workflow)
    await loop.flush_follow_ups(sid)
    wf = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert wf.result_status == "handled" and wf.supervision_status == "open"
    with pytest.raises(JobError, match="finalization_notification_receipt_required"):
        await close(manager, runtime.actor(sid), wf)


@pytest.mark.parametrize("fault", ["scope", "seal", "model", "node_ack", "decision", "delivery", "closure"])
async def test_closed_tree_revalidates_immutable_evidence(managed_workflow, fault):
    _, manager, runtime, _, _, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow)
    actor = runtime.actor(sid)
    wf = await manager.get_job(actor, handle.job_id)
    assert wf.supervision_status == "closed"
    kind = {"scope": "execution_scope", "seal": "execution_membership_seal", "model": "model_closed",
        "node_ack": "node_result_consumed", "decision": "processing_decision", "delivery": "workflow_delivery",
        "closure": "supervision_closed"}[fault]
    closure = await close(manager, actor, wf)
    async def refs(event_id):
        event = await manager.db.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
        found = [event] if event["event_type"] == kind else []
        if event["event_type"] == "supervision_closed":
            for ref in json.loads(event["payload_json"])["proof"]["references"]:
                found.extend(await refs(ref["event_id"]))
        return found
    events = await refs(closure["event_id"])
    assert events, kind
    event = events[0]
    data = json.loads(event["payload_json"])
    data["tampered"] = True
    await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), event["event_id"]))
    with pytest.raises(JobError, match="finalization_evidence_(changed|corrupt)"):
        await close(manager, actor, wf)


@pytest.mark.parametrize("field,value", [("resource_state", "running"), ("resource_state", "unknown"),
    ("intake", "open"), ("outcome", "failed"), ("result_json", '{}')])
async def test_closed_current_cache_must_match_proof(managed_workflow, field, value):
    _, manager, runtime, _, _, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow)
    actor = runtime.actor(sid)
    wf = await manager.get_job(actor, handle.job_id)
    await manager.db.execute(f"UPDATE {manager.jobs} SET {field}=? WHERE job_id=?", (value, wf.job_id))
    with pytest.raises(JobError, match="finalization_execution_changed"):
        await close(manager, actor, wf)


async def test_failed_finalization_does_not_replay_completed_business_send(managed_workflow, monkeypatch, caplog):
    _, manager, runtime, loop, model, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow, consume=False)
    original = manager.finalize_supervision
    async def unavailable(*args, **kwargs):
        raise RuntimeError("test: closure database unavailable")
    monkeypatch.setattr(manager, "finalize_supervision", unavailable)
    before = model.counts.copy()
    result = await loop.flush_follow_ups(sid)
    assert result.status == "completed" and model.counts["MAIN"] == before["MAIN"] + 1
    assert model.counts["BUILD"] == before["BUILD"] and model.counts["REVIEW"] == before["REVIEW"]
    assert "supervision remains open" in caplog.text
    wf = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert wf.supervision_status == "open" and wf.result_status == "handled"
    monkeypatch.setattr(manager, "finalize_supervision", original)
    await close(manager, runtime.actor(sid), wf)
    assert (await manager.get_job(runtime.actor(sid), wf.job_id)).supervision_status == "closed"


async def test_public_close_is_authorized_and_generation_fenced(managed_workflow):
    _, manager, runtime, _, _, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow)
    actor = runtime.actor(sid)
    with pytest.raises(JobError, match="access_denied"):
        await manager.finalize_supervision(replace(actor, session_id="other"), handle.job_id, generation=1)
    for generation in (0, True, "1", 2**63):
        with pytest.raises(JobError, match="invalid_result_generation"):
            await manager.finalize_supervision(actor, handle.job_id, generation=generation)
    with pytest.raises(JobError, match="finalization_generation_changed"):
        await manager.finalize_supervision(actor, handle.job_id, generation=2)


async def test_atomic_close_failure_rolls_back_view_and_receipt(managed_workflow, monkeypatch):
    _, manager, runtime, loop, _, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow, consume=False)
    original = manager.finalize_supervision
    async def postpone(*args, **kwargs):
        raise JobError("test_pending")
    monkeypatch.setattr(manager, "finalize_supervision", postpone)
    await loop.flush_follow_ups(sid)
    monkeypatch.setattr(manager, "finalize_supervision", original)
    actor = runtime.actor(sid)
    before = await manager.get_job(actor, handle.job_id)
    events = await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")
    write = manager._write
    @asynccontextmanager
    async def broken():
        async with write() as tx:
            execute = tx.execute
            async def fail(sql, params=()):
                if sql.startswith("INSERT") and "supervision_closed" in params:
                    raise RuntimeError("test: after view CAS before closure receipt")
                return await execute(sql, params)
            with monkeypatch.context() as patch:
                patch.setattr(tx, "execute", fail)
                yield tx
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", broken)
        with pytest.raises(RuntimeError, match="after view CAS"):
            await close(manager, actor, before)
    assert await manager.get_job(actor, handle.job_id) == before
    assert await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id") == events
    refs = await asyncio.gather(*(close(manager, actor, before) for _ in range(3)))
    assert refs[0] == refs[1] == refs[2]


async def test_closed_standalone_resume_reopens_creator_without_restarting_it(ready, monkeypatch):
    c = ready
    actor = c.runtime.actor(c.parent)
    old = await c.manager.get_job(actor, c.old.job_id)
    creator = await c.manager.get_job(actor, old.parent_job_id)
    assert old.supervision_status == creator.supervision_status == "closed"
    saved = await close(c.manager, actor, old)
    creator_version = creator.version
    await issue_nudge(c.fixture, monkeypatch)
    current = await c.manager.get_job(actor, old.job_id)
    parent = await c.manager.get_job(actor, creator.job_id)
    assert current.generation == 2 and current.supervision_status == "open"
    assert parent.generation == creator.generation and parent.status == "model_closed"
    assert parent.supervision_status == "open" and parent.version == creator_version + 1
    async with c.manager._write() as tx:
        parent_row = await c.manager._row(tx, parent.job_id)
        creator_event, creator_proof = await finalization.closed(c.manager, tx, parent_row)
        assert creator_proof["scope"]["generation"] == parent_row["generation"]
        assert creator_proof["scope"]["executor_id"] == parent_row["executor_id"]
        assert creator_proof["scope"]["boot_id"] == parent_row["boot_id"]
    with pytest.raises(JobError, match="finalization_generation_changed"):
        await close(c.manager, actor, old)
    async with c.manager._write() as tx:
        historical, _ = await finalization.closed(c.manager, tx, await c.manager._row(tx, old.job_id), 1)
    assert historical["event_id"] == saved["event_id"]
    await finish(c.fixture)
    await c.loop.flush_follow_ups(c.parent)
    assert (await c.manager.get_job(actor, old.job_id)).supervision_status == "closed"
    assert (await c.manager.get_job(actor, creator.job_id)).supervision_status == "closed"


async def test_closed_workflow_resume_reopens_ancestors_and_recloses_new_generation(native_restore):
    c = native_restore
    wf = await c.manager.get_job(c.actor, c.initial.job_id)
    creator = await c.manager.get_job(c.actor, wf.parent_job_id)
    assert wf.supervision_status == creator.supervision_status == "closed"
    original = await close(c.manager, c.actor, wf)
    handle = await c.restore()
    parent = await c.manager.get_job(c.actor, creator.job_id)
    assert parent.status == "model_closed" and parent.generation == creator.generation
    assert parent.supervision_status == "open"
    await asyncio.wait_for(handle.task, 8)
    await c.loop.flush_follow_ups(c.parent)
    current = await c.manager.get_job(c.actor, wf.job_id)
    assert current.generation == 2 and current.supervision_status == "closed"
    assert (await c.manager.get_job(c.actor, creator.job_id)).supervision_status == "closed"
    assert await close(c.manager, c.actor, current) != original
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='supervision_closed'", (creator.job_id,))
    assert len(events) == 2  # same original model generation, two supervision epochs


@pytest.mark.parametrize("exit,resource", [("completed", "unknown"), ("failed", "stopped"),
    ("cancelled", "stopped"), ("interrupted", "stopped")])
async def test_uncertain_model_exit_never_closes_from_python_exit_alone(registry_runtime, exit, resource):
    _, manager, actor, root, lease, _, _ = registry_runtime
    await manager.close_model_execution(actor, lease, evidence={"exit": exit}, resource_state=resource)
    job = await manager.get_job(actor, root.job.job_id)
    with pytest.raises(JobError, match="finalization_(model_exit_unconfirmed|execution_unsettled)"):
        await close(manager, actor, job)
    assert (await manager.get_job(actor, job.job_id)).supervision_status == "open"


async def test_live_independent_child_prevents_creator_supervision_close(registry_runtime):
    _, manager, actor, root, lease, _, _ = registry_runtime
    child = await manager.register_job(actor, kind="subagent", parent=lease, system_key="live-child", independent=True)
    await manager.claim_job(actor, child.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
    await manager.close_model_execution(actor, lease, evidence={"exit": "completed"})
    job = await manager.get_job(actor, root.job.job_id)
    with pytest.raises(JobError, match="finalization_execution_unsettled"):
        await close(manager, actor, job)
    assert (await manager.get_job(actor, job.job_id)).supervision_status == "open"
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='supervision_closed'")


async def test_completed_empty_send_closes_without_canonical_task_result(registry_runtime):
    _, manager, actor, root, lease, _, _ = registry_runtime
    await manager.close_model_execution(actor, lease, evidence={"exit": "completed"})
    job = await manager.get_job(actor, root.job.job_id)
    ref = await close(manager, actor, job)
    current = await manager._row(manager.db, job.job_id)
    assert ref["event_type"] == "supervision_closed" and not active_reasons(current)
    assert current["outcome"] is None and current["status"] == "model_closed"


@pytest.mark.parametrize("raw", ["[]", "null", "{", '{"version": NaN}', '{"version":true}', '{}'])
async def test_corrupt_closure_is_typed_error_and_cannot_dispatch(registry_runtime, raw):
    _, manager, actor, root, lease, _, _ = registry_runtime
    await manager.close_model_execution(actor, lease, evidence={"exit": "completed"})
    job = await manager.get_job(actor, root.job.job_id)
    ref = await close(manager, actor, job)
    await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (raw, ref["event_id"]))
    with pytest.raises(JobError, match="finalization_evidence_(corrupt|missing)"):
        await close(manager, actor, job)


async def test_input_processing_decision_is_part_of_new_generation_proof(ready, monkeypatch):
    c = ready
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    await c.loop.flush_follow_ups(c.parent)
    actor = c.runtime.actor(c.parent)
    job = await c.manager.get_job(actor, c.old.job_id)
    assert job.supervision_status == "closed" and job.generation == 2
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='processing_decision'", (job.job_id,))
    [event] = [e for e in events if json.loads(e["payload_json"])["event"]["event_id"].startswith("op_")]
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (event["event_id"],))
    with pytest.raises(JobError, match="finalization_evidence_changed"):
        await close(c.manager, actor, job)


async def test_host_stop_of_closed_scope_keeps_proof_and_future_admission_barrier(ready, monkeypatch):
    c = ready
    actor = c.runtime.actor(c.parent)
    before = await c.manager.get_job(actor, c.old.job_id)
    proof = await close(c.manager, actor, before)
    await c.manager.request_cancel(actor, before.job_id, system_key="host-stop-closed", reason="Stop this scope including future admissions")
    stopped = await c.manager.get_job(actor, before.job_id)
    assert stopped.cancel_requested_at is not None
    assert stopped.status == before.status and stopped.supervision_status == "closed"
    assert await close(c.manager, actor, stopped) == proof
    await issue_nudge(c.fixture, monkeypatch)
    assert (await c.manager.get_job(actor, before.job_id)).generation == 1
    assert c.model.child_calls == 1


async def test_original_creator_stop_after_reopen_reaches_new_child_generation(ready, monkeypatch):
    c = ready
    actor = c.runtime.actor(c.parent)
    await issue_nudge(c.fixture, monkeypatch)
    child = await c.manager.get_job(actor, c.old.job_id)
    creator = await c.manager.get_job(actor, child.parent_job_id)
    assert creator.supervision_status == "open" and child.generation == 2
    await c.manager.request_cancel(actor, creator.job_id, system_key="stop-old-creator", reason="Stop the complete original scope")
    await finish(c.fixture)
    result = await c.manager.get_job(actor, child.job_id)
    assert result.outcome != "succeeded" and result.generation == 2
    assert (await c.manager.get_job(actor, creator.job_id)).supervision_status == "open"


async def test_live_leaf_resume_requires_actual_engine_consumption(ready_node):
    c = ready_node
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET supervision_status='open',status='settling',result_status='none' WHERE job_id=?", (c.job_id,))
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='node_result_consumed'", (c.job_id,))
    await c.issue()
    assert not c.replies and c.model.counts["BUILD"] == 1
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1


@pytest.mark.parametrize("close_first", [True, False])
async def test_close_and_resume_race_cannot_overwrite_new_generation(ready, monkeypatch, close_first):
    c = ready
    actor = c.runtime.actor(c.parent)
    previous = await c.manager.get_job(actor, c.old.job_id)
    original = c.manager.resume_subagent
    observations = []
    async def raced(*args, **kwargs):
        actions = [close(c.manager, actor, previous), original(*args, **kwargs)] if close_first else [
            original(*args, **kwargs), close(c.manager, actor, previous)]
        results = await asyncio.gather(*actions, return_exceptions=True)
        observation, admitted = results if close_first else reversed(results)
        observations.append(observation)
        if isinstance(admitted, BaseException):
            raise admitted
        return admitted
    monkeypatch.setattr(c.manager, "resume_subagent", raced)
    await issue_nudge(c.fixture, monkeypatch)
    assert len(observations) == 1
    if isinstance(observations[0], JobError):
        assert observations[0].code == "finalization_generation_changed"
    else:
        assert observations[0]["event_type"] == "supervision_closed"
    job = await c.manager.get_job(actor, previous.job_id)
    assert job.generation == 2 and job.supervision_status == "open" and job.result is None
    await finish(c.fixture)
    await c.loop.flush_follow_ups(c.parent)
    assert (await c.manager.get_job(actor, job.job_id)).supervision_status == "closed"


async def test_live_node_continuation_then_original_whole_workflow_restoration(managed_workflow, monkeypatch):
    managed_workflow[6]['on_complete'] = None
    c = await ready_node.__wrapped__(managed_workflow, monkeypatch)
    await c.issue()
    assert len(c.replies) == 1
    await asyncio.wait_for(asyncio.gather(*list(c.engine._repairs.values())), 5)
    await settle(managed_workflow)
    await c.loop.flush_follow_ups(c.parent)
    node = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
    assert node.generation == 2 and node.supervision_status == 'closed'
    original_wf = await c.manager.get_job(c.runtime.actor(c.parent), c.state['handles'][0].job_id)
    assert original_wf.result['usage']['total_tokens'] == 40
    old_node_results = {g: await c.manager.get_result_revision(c.runtime.actor(c.parent), c.job_id, g) for g in (1, 2)}
    handles = []
    async def handler():
        h = await resume_detached(c.loop, c.parent, c.run_id, continuation=('BUILD', 'NUDGED-43', 3))
        handles.append(h)
        c.state['handles'].append(h)
        return {'operation_id': h.operation_id}
    await c.issue(handler=handler)
    assert len(handles) == 1, [(m.role, m.content) for m in await c.store.load_all_messages(c.parent) if m.role == 'tool']
    await handles[0].task
    wf = await c.manager.get_job(c.runtime.actor(c.parent), handles[0].job_id)
    assert wf.generation == 2 and wf.outcome == 'succeeded'
    assert wf.result['usage']['total_tokens'] == 20
    assert c.model.counts['BUILD'] == 3 and c.model.counts['AUDIT'] == 1 and c.model.counts['REVIEW'] == 2
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 3
    for g in (1, 2):
        assert await c.manager.get_result_revision(c.runtime.actor(c.parent), c.job_id, g) == old_node_results[g]
    await c.loop.flush_follow_ups(c.parent)
    assert (await c.manager.get_job(c.runtime.actor(c.parent), wf.job_id)).supervision_status == 'closed'
