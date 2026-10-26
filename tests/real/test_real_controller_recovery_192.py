"""Actual provider consumes original send and controller results, without replay."""
import asyncio

import pytest

from power_loop.runtime.controller_recovery import reconcile
from power_loop.runtime.finalization import closed
from tests.unit.test_controller_recovery_192 import cleanup, ready, recover_in_process
from tests.unit.test_job_registry_192 import registry_store

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_real_main_closes_original_exited_controller_without_repeating_effect_or_changing_usage(registry_store, kind, record_property):
    provider = BoundedProvider(limit=8)
    c = None
    try:
        async with asyncio.timeout(150):
            c = await ready(registry_store, kind, provider=provider)
            rt, actor, owner = c.recovery_runtime, c.actor, c.state.owner_binding
            usage = await rt.manager.execution_usage(actor, owner.lease.job_id, generation=1)
            calls = provider.calls
            await recover_in_process(c, owner.lease.job_id, record_property)
            assert provider.calls == calls
            pending = await rt.manager.get_job(actor, owner.lease.job_id)
            await c.recovery_loop.send('Read and acknowledge the interrupted original controller result. Do not repeat work.', c.state.root, tools=[])
            final = await rt.manager.get_job(actor, owner.lease.job_id)
            assert final.result_status == 'handled' and final.supervision_status == 'closed'
            assert final.outcome == 'interrupted' and final.result == pending.result and final.result['usage'] == usage
            assert await rt.manager.execution_usage(actor, owner.lease.job_id, generation=1) == usage
            assert not c.state.effects and c.state.counts['BUILD'] == 1 and c.state.counts['AFTER'] == 0
            async with rt.manager._write() as tx:
                row = await rt.manager._authorize(tx, actor, owner.lease.job_id)
                assert await closed(rt.manager, tx, row)
            if kind == 'workflow_node':
                workflow = await rt.manager.get_job(actor, final.parent_job_id)
                original = await rt.manager.get_result_revision(actor, workflow.job_id, 1)
                receipt = await reconcile(rt, actor, workflow.job_id)
                if receipt['notification_id'] is not None:
                    await c.recovery_loop.send('Acknowledge the original failed workflow without restarting it.', c.state.root, tools=[])
                settled = await rt.manager.get_job(actor, workflow.job_id)
                assert settled.supervision_status == 'closed' and settled.result == workflow.result
                assert (await rt.manager.get_result_revision(actor, workflow.job_id, 1))['payload'] == original['payload']
                assert c.state.counts['AFTER'] == 0 and not c.state.effects
    finally:
        if c is not None:
            await cleanup(c)
        provider.report(record_property)
        await provider.close()
