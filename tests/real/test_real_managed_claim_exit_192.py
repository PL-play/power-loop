"""Real child call, original claim lost ACK, separate-process repair and main review."""
import asyncio

import pytest

from power_loop.runtime.controller_recovery import reconcile as recover_owner
from tests.unit._managed_claim_faults_192 import fail_claim_transaction
from tests.unit._owned_pending_case_192 import owned_pending_case
from tests.unit.test_controller_handoff_192 import new_main
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_owned_pending_review_192 import cleanup

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_owned_pending_192 import recover_in_process

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('kind,case', [('subagent', 'commit_ack'), ('workflow_node', 'commit_ack'),
                                     ('subagent', 'cancel_ack'), ('subagent', 'no_budget')])
async def test_real_original_claim_exit_preserves_budget_and_never_replays(registry_store, record_property, kind, case):
    provider = BoundedProvider(limit=8)
    c = loop = runtime = None
    def configure(c):
        c.claim_fault = fail_claim_transaction(c.manager, 'cancel_ack' if case == 'cancel_ack' else 'commit_ack')
    try:
        async with asyncio.timeout(180):
            c = await owned_pending_case(registry_store, kind, configure=configure, provider=provider,
                                         shared_calls=3 if case == 'no_budget' else 12)
            assert c.claim_fault.fired and c.effect['generation'] == 1
            loop, runtime = await new_main(c, registry_store)
            actor = runtime.actor(c.state.root)
            original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
            usage = await c.manager.execution_usage(c.state.owner_binding.actor, c.state.owner_binding.lease.job_id, generation=1)
            canonical = await runtime.manager.get_result_revision(actor, c.effect['job_id'], 1)
            calls = provider.calls
            recovered = await recover_in_process(runtime, actor, c.state.source, record_property)
            assert recovered['state'] == 'queued_for_review' and provider.calls == calls
            assert not recovered['acknowledgement_created'] and not recovered['billing_settled']
            response = await loop.send('Acknowledge the original not-dispatched outcome. Do not repeat or start work.', c.state.root, tools=[])
            effect = await runtime.manager.get_job(actor, c.effect['job_id'])
            assert effect.result['managed_effect_result']['resolution'] == 'not_dispatched'
            if case == 'no_budget':
                assert response.status == 'degraded' and 'budget_max_calls_exhausted' in response.final_text
                assert provider.calls == calls and effect.result_status == 'pending_delivery'
                assert effect.supervision_status == 'open'
            else:
                assert response.status == 'completed' and effect.supervision_status == 'closed'
                assert (await runtime.manager.get_job(actor, c.state.source)).supervision_status == 'closed'
                await recover_owner(runtime, actor, c.state.owner_binding.lease.job_id)
                await loop.send('Acknowledge the original interrupted controller without restarting it.', c.state.root, tools=[])
                owner = await runtime.manager.get_job(actor, c.state.owner_binding.lease.job_id)
                assert owner.supervision_status == 'closed' and owner.outcome == 'interrupted'
            assert await runtime.manager.execution_usage(actor, c.state.owner_binding.lease.job_id, generation=1) == usage
            assert (await runtime.manager.get_result_revision(actor, effect.job_id, 1))['payload'] == canonical['payload']
            for row in original:
                assert await runtime.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
            assert not await runtime.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (c.state.source_binding.actor.session_id,))
            assert c.state.prepared == c.state.executed == c.state.lookups == len(c.state.effects) == c.state.counts['AFTER'] == 0
    finally:
        if c is not None:
            await cleanup(c, loop, runtime)
        provider.report(record_property)
        await provider.close()
