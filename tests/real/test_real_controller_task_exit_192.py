"""A real provider request ends; only then does the actual controller Task exit."""
import asyncio

import pytest

from power_loop.runtime import delegation
from power_loop.runtime.controller_task_exit import read
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_original_owner_review_192 import original_owner_case, wait_owned

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_real_controller_task_exit_keeps_original_provider_usage_and_unhandled_result(registry_store, kind, record_property):
    provider = BoundedProvider(limit=5)
    c = await original_owner_case(registry_store, kind, allowed_calls=1, provider=provider)
    initial = set(delegation._TASKS)
    try:
        async with asyncio.timeout(150):
            await c.loop.send('Use launch_owned exactly once, then reply QUEUED.', c.state.root)
            await wait_owned(c, kind, initial)
            owner = c.state.owner_binding
            usage = await c.manager.execution_usage(owner.actor, owner.lease.job_id, generation=1)
            requests = provider.calls
            await c.runtime.flush_control_checkpoints()
            async with c.manager._write() as tx:
                row = await c.manager._authorize(tx, owner.actor, owner.lease.job_id)
                data = await read(c.manager, tx, row, owner.lease)
            assert data['task_exit']['state'] in ('returned', 'raised', 'cancelled')
            assert not data['descendant_stop_confirmed'] and not data['billing_settled']
            assert provider.calls == requests and c.state.counts['BUILD'] == 1 and not c.state.effects
            assert await c.manager.execution_usage(owner.actor, owner.lease.job_id, generation=1) == usage
            job = await c.manager.get_job(owner.actor, owner.lease.job_id)
            assert job.result_status == 'none' and job.resource_state == 'unknown' and job.supervision_status == 'open'
    finally:
        c.state.release.set()
        provider.report(record_property)
        await c.runtime.shutdown(timeout_s=3)
        await c.runtime.flush_control_checkpoints()
        await c.loop.aclose(drain_timeout_s=.1)
        await provider.close()
