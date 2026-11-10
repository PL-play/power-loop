"""Finalization removes transient execution data only with a settled closure."""
import pytest

from power_loop.runtime import finalization
from power_loop.runtime.jobs import JobManager

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_send_sql_profile import _measure

__all__ = ['registry_store', 'registry_runtime']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def transient_rows(manager):
    return await manager.db.fetchall(
        f'SELECT * FROM {manager.prefix}execution_checkpoints ORDER BY job_id')


async def test_ordinary_send_leaves_no_checkpoint_or_job_rows(registry_store):
    store, connect = registry_store
    await _measure(store, 2, False)
    cold = JobManager(await connect())
    assert not await transient_rows(cold)
    jobs = await cold.db.fetchall(f'SELECT * FROM {cold.jobs}')
    assert jobs == []



async def test_failed_finalization_rolls_back_deletions_with_the_closure(registry_runtime, monkeypatch):
    _, manager, actor, root, lease, _, _ = registry_runtime
    await manager.close_model_execution(actor, lease, evidence={'exit': 'completed'})
    job = await manager.get_job(actor, root.job.job_id)
    await manager.db.execute(
        f'INSERT INTO {manager.prefix}execution_checkpoints (job_id, generation, batch_no, summary_json) VALUES (?,?,?,?)',
        (job.job_id, job.generation, 0, '{"closed": false}'))
    before = await transient_rows(manager)
    assert before
    publish = finalization._finish

    async def fail_after_delete(*args, **kwargs):
        await publish(*args, **kwargs)
        raise OSError('commit failed after cleanup')

    with monkeypatch.context() as patch:
        patch.setattr(finalization, '_finish', fail_after_delete)
        with pytest.raises(OSError, match='commit failed'):
            await manager.finalize_supervision(actor, job.job_id, generation=1)
    assert await transient_rows(manager) == before
    assert (await manager.get_job(actor, job.job_id)).supervision_status == 'open'
    await manager.finalize_supervision(actor, job.job_id, generation=1)
    assert not await transient_rows(manager)
    assert (await manager.get_job(actor, job.job_id)).supervision_status == 'closed'
