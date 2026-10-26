"""Real original callback and main review; recovery runs in another process."""
import asyncio

import pytest

from power_loop.runtime.controller_recovery import reconcile as recover_owner
from tests.unit._native_return_faults_192 import fail_return
from tests.unit._owned_pending_case_192 import owned_pending_case
from tests.unit.test_controller_handoff_192 import new_main
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_owned_pending_review_192 import cleanup

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_owned_pending_192 import recover_in_process

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('kind,case', [('subagent', 'known'), ('workflow_node', 'known'),
                                     ('subagent', 'no_budget'), ('subagent', 'missing_checkpoint'),
                                     ('subagent', 'checkpoint_failure')])
async def test_real_native_return_recovery_uses_original_evidence_and_budget(registry_store, record_property, kind, case):
    provider = BoundedProvider(limit=8)
    c = loop = runtime = None
    def configure(c):
        if case == 'checkpoint_failure':
            c.return_fault = fail_return(c, 'write')
    try:
        async with asyncio.timeout(180):
            c = await owned_pending_case(registry_store, kind, provider=provider,
                native_tail='inspect_original_note', native_first=True, shared_calls=3 if case == 'no_budget' else 12,
                configure=configure, flush_checkpoints=False, raw_fail_at=999 if case == 'checkpoint_failure' else 1)
            assert c.state.native_calls == 1 and c.state.executed == c.state.prepared == 0
            sid = c.state.source_binding.actor.session_id
            assert (await c.store.get_state(sid)).pending['tool_calls'][0]['function']['name'] == 'inspect_original_note'
            original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
            usage = await c.manager.execution_usage(c.state.owner_binding.actor, c.state.owner_binding.lease.job_id, generation=1)
            if case == 'checkpoint_failure':
                assert c.return_fault.fired == 1 and c.runtime._native_returns.pending_count() == 1
                calls = provider.calls
                await c.runtime.flush_control_checkpoints()
                assert c.runtime._native_returns.pending_count() == 0 and provider.calls == calls
            if case == 'missing_checkpoint':
                await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='native_callback_returned'", (c.state.source,))
            loop, runtime = await new_main(c, registry_store)
            actor = runtime.actor(c.state.root)
            calls = provider.calls
            recovered = await recover_in_process(runtime, actor, c.state.source, record_property)
            assert provider.calls == calls
            if case == 'missing_checkpoint':
                assert recovered['state'] == 'unconfirmed'
                assert (await c.store.get_state(sid)).pending is not None
                assert (await runtime.manager.get_job(actor, c.state.source)).result_status == 'none'
            else:
                assert recovered['state'] == 'queued_for_review' and not recovered['acknowledgement_created']
                source = await runtime.manager.get_job(actor, c.state.source)
                value = source.result['aborted_call_batch']['calls'][0]
                assert value['resolution'] == 'callback_returned' and value['return_preview'] == 'Original native completion'
                response = await loop.send('Acknowledge the supplied original callback outcome. Do not repeat any action.', c.state.root, tools=[])
                source = await runtime.manager.get_job(actor, c.state.source)
                if case == 'no_budget':
                    assert response.status == 'degraded' and 'budget_max_calls_exhausted' in response.final_text
                    assert provider.calls == calls and source.result_status == 'pending_delivery'
                    assert source.supervision_status == 'open'
                else:
                    assert response.status == 'completed' and source.supervision_status == 'closed'
                    await recover_owner(runtime, actor, c.state.owner_binding.lease.job_id)
                    await loop.send('Acknowledge the original interrupted controller without restarting it.', c.state.root, tools=[])
                    owner = await runtime.manager.get_job(actor, c.state.owner_binding.lease.job_id)
                    assert owner.supervision_status == 'closed' and owner.outcome == 'interrupted'
            assert await runtime.manager.execution_usage(actor, c.state.owner_binding.lease.job_id, generation=1) == usage
            for row in original:
                assert await runtime.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
            assert not await runtime.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (sid,))
            assert c.state.native_calls == 1 and c.state.prepared == c.state.executed == c.state.lookups == c.state.counts['AFTER'] == 0
    finally:
        if c is not None:
            await cleanup(c, loop, runtime)
        provider.report(record_property)
        await provider.close()
