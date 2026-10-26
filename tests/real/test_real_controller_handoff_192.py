"""Actual main-model review after another OS process transfers an original negative."""
import asyncio

import pytest

from power_loop import HookPoint
from power_loop.runtime import delegation
from tests.unit.test_controller_handoff_192 import new_main, transfer_in_process
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_original_owner_review_192 import original_owner_case, wait_owned

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_real_original_main_handles_transferred_result_in_new_boot_without_replaying_child(
    registry_store, kind, record_property
):
    provider = BoundedProvider(limit=7)
    c = await original_owner_case(registry_store, kind, allowed_calls=1, provider=provider)
    initial = set(delegation._TASKS)
    loop = runtime = None
    seen = []
    try:
        async with asyncio.timeout(180):
            await c.loop.send('Use launch_owned exactly once, then reply QUEUED.', c.state.root)
            await wait_owned(c, kind, initial)
            actor = c.runtime.actor(c.state.root)
            original = await c.manager.get_job(actor, c.state.source)
            usage = await c.manager.execution_usage(actor, c.state.owner_binding.lease.job_id, generation=1)
            assert original.result_status == 'pending_delivery' and c.state.counts['BUILD'] == 1
            await c.runtime.shutdown(timeout_s=3)
            loop, runtime = await new_main(c, registry_store)
            requests = provider.calls
            receipt = await transfer_in_process(c, record_property)
            assert provider.calls == requests and not receipt['execution_authority']

            async def main_review(ctx):
                ctx.tools = []
                ctx.system_prompt += ('\nYou are the original main agent reviewing an original negative child outcome. '
                    'Read it and acknowledge that the original action was not dispatched. Give a short final text. '
                    'Do not redo the action or start any task. This review is not a successful child result.')
                seen.extend(m.get('content', '') for m in ctx.messages if isinstance(m.get('content'), str))

            loop.hooks.register(HookPoint.LLM_BEFORE, main_review, name='test.handoff.real-original-main')
            result = await loop.send('Review and acknowledge the queued original negative child result.', c.state.root, tools=[])
            assert result.status == 'completed' and result.final_text
            assert provider.calls == requests + 1
            assert any(c.state.source in text and 'Original tool outcome:' in text for text in seen)
            after = await runtime.manager.get_job(runtime.actor(c.state.root), c.state.source)
            assert after.result_status == 'handled' and after.result == original.result
            assert after.supervision_status == 'closed'
            from power_loop.runtime.abort_outcomes import handled_references
            async with runtime.manager._write() as tx:
                source = await runtime.manager._authorize(tx, runtime.actor(c.state.root), c.state.source)
                assert await handled_references(runtime.manager, tx, source)
            assert await runtime.manager.execution_usage(runtime.actor(c.state.root), c.state.owner_binding.lease.job_id, generation=1) == usage
            owner = await runtime.manager.get_job(runtime.actor(c.state.root), c.state.owner_binding.lease.job_id)
            assert owner.result_status == 'none' and owner.supervision_status == 'open'
            assert c.state.counts['BUILD'] == 1 and c.state.counts['AFTER'] == 0 and not c.state.effects
    finally:
        c.state.release.set()
        provider.report(record_property)
        if runtime is not None:
            await runtime.shutdown(timeout_s=3)
            await loop.aclose(drain_timeout_s=.1)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)
        await provider.close()
