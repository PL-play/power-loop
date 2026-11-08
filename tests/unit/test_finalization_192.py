"""Mechanical finalization preserves unsettled resources and canonical results."""
import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_query import active_reasons

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime

__all__ = ['registry_store', 'registry_runtime']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def close(manager, actor, job):
    return await manager.finalize_supervision(actor, job.job_id, generation=job.generation)



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
    store, manager, actor, root, lease, _, _ = registry_runtime
    sid = await store.create_session(parent_session_id=actor.session_id, metadata={"execution_role":"subagent"})
    child = await manager.register_job(actor, kind="subagent", session_id=sid, parent=lease, system_key="live-child", independent=True)
    await manager.claim_job(actor, child.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
    await manager.close_model_execution(actor, lease, evidence={"exit": "completed"})
    job = await manager.get_job(actor, root.job.job_id)
    with pytest.raises(JobError, match="finalization_execution_unsettled"):
        await close(manager, actor, job)
    assert (await manager.get_job(actor, job.job_id)).supervision_status == "open"
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='supervision_closed'")



async def test_completed_empty_send_uses_one_canonical_result_for_close(registry_runtime):
    _, manager, actor, root, lease, _, _ = registry_runtime
    await manager.close_model_execution(actor, lease, evidence={"exit": "completed"})
    job = await manager.get_job(actor, root.job.job_id)
    ref = await close(manager, actor, job)
    current = await manager._row(manager.db, job.job_id)
    assert ref["event_type"] == "result" and active_reasons(current) == ["result_available"]
    assert current["outcome"] is None and current["status"] == "model_closed"
