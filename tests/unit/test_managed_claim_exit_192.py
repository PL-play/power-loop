"""Claim lost ACK must not replay prepare, invent old exits or close storage early."""
import asyncio

import pytest

from power_loop.runtime.jobs import JobActor, JobError
from power_loop.runtime.managed_completion import flush_local, restore_result
from power_loop.runtime.managed_effects import _id, reconcile
from power_loop.runtime.pending_calls import recover_runtime

from ._managed_claim_faults_192 import fail_claim_transaction
from ._managed_completion_faults_192 import fail_completion_once
from ._owned_pending_case_192 import owned_pending_case
from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
@pytest.mark.parametrize('stage', ['rollback', 'commit_ack', 'cancel_ack'])
async def test_original_claim_transaction_exit_can_be_reviewed_without_executing(registry_store, kind, stage):
    def configure(c):
        c.claim_fault = fail_claim_transaction(c.manager, stage)
    c = await owned_pending_case(registry_store, kind, configure=configure)
    loop = runtime = None
    try:
        assert c.claim_fault.fired
        assert c.effect['generation'] == (0 if stage == 'rollback' else 1)
        assert c.runtime._managed_completions.pending_count() == 0
        original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        result = await recover_runtime(runtime, actor, c.state.source)
        assert result['state'] == 'queued_for_review'
        assert not result['acknowledgement_created'] and not result['billing_settled']
        if stage != 'rollback':
            repaired = await restore_result(runtime.manager, actor, c.effect['job_id'])
            assert repaired['owned_callback_exited'] and not repaired['tool_receipt_written']
            value = await runtime.manager.get_job(actor, c.effect['job_id'])
            assert value.resource_state == 'stopped' and value.supervision_status == 'open'
            assert value.result['managed_effect_result']['resolution'] == 'not_dispatched'
        await loop.send('Acknowledge the original not-dispatched outcome without starting work.', c.state.root, tools=[])
        assert (await runtime.manager.get_job(actor, c.effect['job_id'])).supervision_status == 'closed'
        assert (await runtime.manager.get_job(actor, c.state.source)).supervision_status == 'closed'
        for row in original:
            assert await runtime.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
        assert c.state.executed == c.state.prepared == c.state.lookups == c.state.counts['AFTER'] == 0
    finally:
        await cleanup(c, loop, runtime)


@pytest.mark.parametrize('stage', ['exit', 'anchor', 'exit_ack', 'result', 'result_ack'])
async def test_failed_claim_checkpoint_is_retained_and_retry_is_exact(registry_store, stage):
    def configure(c):
        c.claim_fault = fail_claim_transaction(c.manager, 'commit_ack')
        c.exit_fault = fail_completion_once(c.manager, stage)
    c = await owned_pending_case(registry_store, 'subagent', configure=configure, flush_checkpoints=False)
    try:
        assert c.claim_fault.fired and c.exit_fault.fired
        assert c.runtime._managed_completions.pending_count() == c.runtime._managed_completions.claim_exit_count() == 1
        assert not (await c.runtime.executor_status())['store_safe_to_close']
        with pytest.raises(JobError, match='managed_claim_exit_requires_reconciliation'):
            c.runtime.check_local_admission()
        actor = c.runtime.actor(c.state.root)
        original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
        # Concurrent retries share the retained original; no new Execution.
        results = await asyncio.gather(*(flush_local(c.runtime, actor, c.effect['job_id']) for _ in range(3)))
        assert results[0] == results[1] == results[2]
        assert c.runtime._managed_completions.pending_count() == 0
        assert await restore_result(c.manager, actor, c.effect['job_id']) == results[0]
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id') == original
        assert c.state.executed == c.state.prepared == c.state.lookups == 0
    finally:
        await cleanup(c)


async def test_missing_old_exit_cannot_be_inferred_from_not_dispatched_observation(registry_store):
    c = await owned_pending_case(registry_store, 'subagent', claim_failure='after')
    loop = runtime = None
    try:
        # A historical task with no original durable exit is still unknown.
        job = c.effect['job_id']
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (_id(c.manager, 'resource_exit', job, 1),))
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        with pytest.raises(JobError):
            await restore_result(runtime.manager, actor, job)
        assert (await recover_runtime(runtime, actor, c.state.source))['state'] == 'unconfirmed'
        await reconcile(runtime, actor, job, c.adapter)
        assert (await recover_runtime(runtime, actor, c.state.source))['state'] == 'unconfirmed'
        assert (await c.store.get_state(c.state.source_binding.actor.session_id)).pending is not None
        assert c.state.executed == c.state.prepared == 0
    finally:
        await cleanup(c, loop, runtime)


@pytest.mark.parametrize('cancelled', [False, True])
async def test_failure_reading_claim_commit_retains_witness_before_any_document(registry_store, cancelled):
    def configure(c):
        c.claim_fault = fail_claim_transaction(c.manager, 'commit_ack', read_failure='cancel' if cancelled else 'error')
    c = await owned_pending_case(registry_store, 'subagent', configure=configure, flush_checkpoints=False)
    try:
        cache = c.runtime._managed_completions
        slot = cache.find(c.effect['job_id'])
        assert c.claim_fault.resolve_failed and slot.document is None and slot.claim.exit == 'raised'
        assert cache.pending_count() == cache.claim_exit_count() == 1
        assert not (await c.runtime.executor_status())['store_safe_to_close']
        with pytest.raises(JobError, match='managed_claim_exit_requires_reconciliation'):
            c.runtime.check_local_admission()
        await c.runtime.flush_control_checkpoints()
        assert cache.pending_count() == 0
        result = await restore_result(c.manager, c.runtime.actor(c.state.root), c.effect['job_id'])
        assert result['owned_callback_exited'] and not result['acknowledgement_created']
        assert c.state.prepared == c.state.executed == c.state.lookups == 0
    finally:
        await cleanup(c)


async def test_foreign_runtime_and_actor_cannot_flush_original_claim_witness(registry_store):
    def configure(c):
        fail_claim_transaction(c.manager, 'commit_ack')
        fail_completion_once(c.manager, 'exit')
    c = await owned_pending_case(registry_store, 'subagent', configure=configure, flush_checkpoints=False)
    loop = runtime = None
    try:
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        with pytest.raises(JobError, match='handle_missing'):
            await flush_local(runtime, actor, c.effect['job_id'])
        with pytest.raises(JobError):
            await flush_local(c.runtime, JobActor('foreign', actor.principal_id, actor.session_id), c.effect['job_id'])
        assert c.runtime._managed_completions.pending_count() == 1
        await flush_local(c.runtime, actor, c.effect['job_id'])
        assert c.state.executed == c.state.prepared == c.state.lookups == 0
    finally:
        await cleanup(c, loop, runtime)
