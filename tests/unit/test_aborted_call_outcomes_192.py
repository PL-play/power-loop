"""Normal query behavior retained after retiring exact abort-proof recovery."""
import pytest

from power_loop.runtime.abort_outcomes import pending_reviews
from power_loop.runtime.jobs import JobManager

from .test_job_registry_192 import registry_store, runtime

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

async def test_unrelated_large_send_result_is_not_an_aborted_batch_review(registry_store):
    from power_loop.runtime.jobs import JobActor
    store, _ = registry_store
    sid = await store.create_session()
    manager = JobManager(store)
    actor = JobActor('example-tenant', 'host', sid)
    root = await manager.register_job(actor, kind='send', session_id=sid, system_key='ordinary-source')
    lease = await manager.claim_job(actor, root.job.job_id, executor_id='original', boot_id='boot', expected_version=1)
    await manager.record_result(actor, lease, outcome='failed', result={'ordinary_host_payload':'A'*300000}, resource_state='stopped')
    before = await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id')
    async with manager._write() as tx:
        assert await pending_reviews(manager, tx, actor) == []
    assert await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id') == before
