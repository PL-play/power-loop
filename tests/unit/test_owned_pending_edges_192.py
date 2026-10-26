"""V2 review cannot infer exit, lose unknown calls, replay, or grant new work."""
import json
from contextlib import asynccontextmanager

import pytest

from power_loop import LLMResponse
from power_loop.runtime.jobs import JobActor, JobError
from power_loop.runtime.managed_delivery import enqueue
from power_loop.runtime.managed_effects import reconcile, terminate_unstarted
from power_loop.runtime.pending_calls import recover_runtime, retire_pending

from ._owned_pending_case_192 import owned_pending_case
from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_actual_main_can_inspect_before_disposition_but_cannot_ack_a_partial_batch(registry_store):
    c = await owned_pending_case(registry_store, 'subagent')
    loop = runtime = None
    try:
        loop, runtime = await new_main(c, registry_store)
        actor, sid = runtime.actor(c.state.root), c.state.source_binding.actor.session_id
        original = (await c.store.get_state(sid)).pending
        await enqueue(loop, actor, c.effect['job_id'])
        result = await loop.send('Inspect the original outcome. Do not repeat work.', c.state.root)
        assert result.status == 'completed'
        assert (await c.store.get_state(sid)).pending == original
        assert (await runtime.manager.get_job(actor, c.effect['job_id'])).result_status == 'pending_delivery'
        assert not await runtime.manager.db.fetchall(f"SELECT event_id FROM {runtime.manager.events} WHERE event_type='processing_decision' AND job_id=?", (c.effect['job_id'],))
        await retire_pending(loop, actor, c.state.source)
        await loop.send('Now handle the original known outcome without repeating work.', c.state.root)
        assert (await runtime.manager.get_job(actor, c.effect['job_id'])).supervision_status == 'closed'
        assert c.state.executed == len(c.state.effects) == 1
    finally:
        await cleanup(c, loop, runtime)


@pytest.mark.parametrize('fault', ['task_exit', 'retirement', 'owner_generation', 'foreign_main', 'child_role'])
async def test_missing_or_foreign_original_exit_cannot_transfer_child_pending(registry_store, fault):
    c = await owned_pending_case(registry_store, 'subagent')
    loop = runtime = None
    try:
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        await enqueue(loop, actor, c.effect['job_id'])
        if fault in ('task_exit', 'retirement'):
            kind = 'controller_task_exit' if fault == 'task_exit' else 'controller_retired'
            await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE job_id=? AND event_type=?', (c.state.owner_binding.lease.job_id, kind))
        elif fault == 'owner_generation':
            await c.manager.db.execute(f'UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?', (c.state.owner_binding.lease.job_id,))
        elif fault == 'foreign_main':
            actor = JobActor(actor.tenant_id, actor.principal_id, await c.store.create_session())
        else:
            actor = c.state.source_binding.actor
        sid = c.state.source_binding.actor.session_id
        before = (await c.store.get_state(sid)).pending
        with pytest.raises(JobError):
            await retire_pending(loop, actor, c.state.source)
        assert (await c.store.get_state(sid)).pending == before
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('pending_owner_transfer','pending_call_disposition')")
        assert c.state.executed == 1
    finally:
        await cleanup(c, loop, runtime)


@pytest.mark.parametrize('fault', ['rollback', 'lost_ack', 'cancelled'])
async def test_transfer_and_pending_retirement_are_atomic_and_idempotent(registry_store, monkeypatch, fault):
    import asyncio
    c = await owned_pending_case(registry_store, 'subagent')
    loop, runtime = await new_main(c, registry_store)
    manager, actor = runtime.manager, runtime.actor(c.state.root)
    original, tripped = manager._write, False
    @asynccontextmanager
    async def fail_once():
        nonlocal tripped
        async with original() as tx:
            yield tx
            if not tripped and fault != 'lost_ack':
                tripped = True
                if fault == 'cancelled':
                    raise asyncio.CancelledError('retirement cancelled before commit')
                raise OSError('retirement transaction rollback')
        if not tripped:
            tripped = True
            raise OSError('retirement commit acknowledgement lost')
    try:
        await enqueue(loop, actor, c.effect['job_id'])
        monkeypatch.setattr(manager, '_write', fail_once)
        with pytest.raises(asyncio.CancelledError if fault == 'cancelled' else OSError):
            await retire_pending(loop, actor, c.state.source)
        monkeypatch.setattr(manager, '_write', original)
        sid = c.state.source_binding.actor.session_id
        assert ((await c.store.get_state(sid)).pending is None) == (fault == 'lost_ack')
        results = await asyncio.gather(*(retire_pending(loop, actor, c.state.source) for _ in range(3)))
        assert all(r == results[0] for r in results)
        for kind in ('pending_owner_transfer', 'pending_call_disposition'):
            assert len(await manager.db.fetchall(f'SELECT event_id FROM {manager.events} WHERE event_type=?', (kind,))) == 1
        assert (await manager.get_job(actor, c.effect['job_id'])).result_status == 'pending_delivery'
        assert c.state.executed == 1 and c.state.lookups == 0
    finally:
        monkeypatch.setattr(manager, '_write', original)
        await cleanup(c, loop, runtime)


async def test_unknown_original_effect_needs_its_observer_and_never_reexecutes(registry_store):
    c = await owned_pending_case(registry_store, 'workflow_node', unknown=True)
    loop = runtime = None
    try:
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        original = (await c.store.get_state(c.state.source_binding.actor.session_id)).pending
        result = await recover_runtime(runtime, actor, c.state.source)
        assert result['state'] == 'unconfirmed' and not result['acknowledgement_created']
        assert (await c.store.get_state(c.state.source_binding.actor.session_id)).pending == original
        assert c.state.lookups == 0
        c.state.resolved = True
        await reconcile(runtime, actor, c.effect['job_id'], c.adapter)
        result = await recover_runtime(runtime, actor, c.state.source)
        assert result['state'] == 'queued_for_review'
        await loop.send('Handle the resolved original effect without repeating it.', c.state.root)
        assert (await runtime.manager.get_job(actor, c.effect['job_id'])).supervision_status == 'closed'
        assert c.state.executed == c.state.prepared == c.state.lookups == len(c.state.effects) == 1
        assert c.state.counts['AFTER'] == 0
    finally:
        await cleanup(c, loop, runtime)


@pytest.mark.parametrize('claim_failure', ['before', 'after'])
async def test_never_claimed_and_claimed_without_exit_are_different_facts(registry_store, claim_failure):
    c = await owned_pending_case(registry_store, 'subagent', claim_failure=claim_failure)
    loop = runtime = None
    try:
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        result = await recover_runtime(runtime, actor, c.state.source)
        if claim_failure == 'before':
            assert result['state'] == 'queued_for_review'
            await loop.send('Handle the original not-dispatched outcome; do not start work.', c.state.root)
            assert (await runtime.manager.get_job(actor, c.state.source)).supervision_status == 'closed'
        else:
            assert result['state'] == 'unconfirmed'
            c.state.resolved = True
            await reconcile(runtime, actor, c.effect['job_id'], c.adapter)
            assert (await recover_runtime(runtime, actor, c.state.source))['state'] == 'unconfirmed'
            with pytest.raises(JobError):
                await terminate_unstarted(runtime.manager, actor, c.effect['job_id'])
            assert (await c.store.get_state(c.state.source_binding.actor.session_id)).pending is not None
        assert c.state.executed == c.state.prepared == len(c.state.effects) == 0
    finally:
        await cleanup(c, loop, runtime)


async def test_outcome_review_cannot_start_a_new_business_task_through_an_actual_registered_handler(registry_store, monkeypatch):
    c = await owned_pending_case(registry_store, 'subagent')
    loop, runtime = await new_main(c, registry_store)
    try:
        actor = runtime.actor(c.state.root)
        await recover_runtime(runtime, actor, c.state.source)
        before = await runtime.manager.db.fetchall(f"SELECT job_id FROM {runtime.manager.jobs} WHERE kind='subagent'")
        async def unsafe(request, **kwargs):
            assert not request.tools
            return LLMResponse(raw_text='', tool_calls=[{'id': 'unwanted-followup', 'type': 'function',
                'function': {'name': 'launch_owned', 'arguments': json.dumps({})}}])
        monkeypatch.setattr(c.model, 'complete', unsafe)
        with pytest.raises(JobError, match='outcome_review_requires_control_only'):
            await loop.send('Review the original result.', c.state.root)
        assert await runtime.manager.db.fetchall(f"SELECT job_id FROM {runtime.manager.jobs} WHERE kind='subagent'") == before
        assert (await runtime.manager.get_job(actor, c.effect['job_id'])).result_status == 'pending_delivery'
        assert c.state.executed == len(c.state.effects) == 1
    finally:
        await cleanup(c, loop, runtime)
