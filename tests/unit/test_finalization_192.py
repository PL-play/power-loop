"""Mechanical close/reopen uses durable original execution evidence only."""
from __future__ import annotations

import asyncio

import pytest

from power_loop.runtime import finalization
from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_query import active_reasons

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_nudge_runtime_192 import issue_nudge
from .test_subagent_resume_192 import ready
from .test_task_runtime_192 import finish, managed

__all__ = ['registry_store', 'registry_runtime', 'managed', 'ready']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def close(manager, actor, job):
    return await manager.finalize_supervision(actor, job.job_id, generation=job.generation)
















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




@pytest.mark.parametrize("exit,resource", [("completed", "unknown"), ("failed", "stopped"),
    ("cancelled", "stopped"), ("interrupted", "stopped")])
async def test_uncertain_model_exit_never_closes_from_python_exit_alone(registry_runtime, exit, resource):
    _, manager, actor, root, lease, _, _ = registry_runtime
    await manager.close_model_execution(actor, lease, evidence={"exit": exit}, resource_state=resource)
    job = await manager.get_job(actor, root.job.job_id)
    if resource == 'unknown':
        with pytest.raises(JobError, match="finalization_(model_exit_unconfirmed|execution_unsettled)"):
            await close(manager, actor, job)
        assert (await manager.get_job(actor, job.job_id)).supervision_status == "open"
    else:
        await close(manager, actor, job)
        assert (await manager.get_job(actor, job.job_id)).supervision_status == 'closed'


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


async def test_consumed_input_and_result_keep_their_original_generation(ready, monkeypatch):
    c = ready
    await issue_nudge(c.fixture, monkeypatch)
    await finish(c.fixture)
    await c.loop.flush_follow_ups(c.parent)
    actor = c.runtime.actor(c.parent)
    job = await c.manager.get_job(actor, c.old.job_id)
    assert job.supervision_status == "closed" and job.generation == 2
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND action='nudge'", (job.job_id,))
    assert events and all(event["delivery_status"] == "handled" for event in events)


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
