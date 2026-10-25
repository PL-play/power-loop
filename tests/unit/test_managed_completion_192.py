"""Original callback exit/result recovery cannot invent delivery or execution."""
import asyncio
import json

import pytest

from power_loop.runtime.jobs import JobActor, JobError, JobManager, _hash, _json
from power_loop.runtime.managed_completion import flush_local, restore_result
from power_loop.runtime.managed_effects import _id, read_raw_receipt, require_source_resolved
from power_loop.runtime.task_runtime import TaskRuntime

from ._managed_completion_faults_192 import fail_completion_once
from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def failure_case(registry_store, stage='result'):
    c = await scenario(registry_store, budgeted=True, completion_failure=stage)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert c.state.prepared == c.state.dispatched == c.model.calls == c.host.reserve_calls == 1
    assert c.state.lookups == 0
    assert c.runtime._managed_completions.pending_count() == 1
    return c, row


@pytest.mark.parametrize('stage', ['exit', 'anchor', 'result', 'publication', 'exit_ack', 'result_ack'])
async def test_real_callback_exit_retained_across_storage_failures_and_exact_retry(registry_store, stage):
    c, row = await failure_case(registry_store, stage)
    original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    raw = await read_raw_receipt(c.manager, c.actor, row['job_id'])
    assert not raw['acknowledges_result']
    if stage in ('exit', 'anchor'):
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='managed_effect_resource_exit'")
        with pytest.raises(JobError, match='evidence_missing'):
            await restore_result(c.manager, c.actor, row['job_id'])
    repaired = await flush_local(c.runtime, c.actor, row['job_id'])
    assert repaired['owned_callback_exited'] and not repaired['acknowledgement_created']
    assert not repaired['tool_receipt_written'] and not repaired['execution_authority'] and not repaired['billing_settled']
    assert c.runtime._managed_completions.pending_count() == 0
    assert await restore_result(c.manager, c.actor, row['job_id']) == repaired
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq') == original
    current = await c.manager.get_job(c.actor, row['job_id'])
    assert current.resource_state == 'stopped' and current.result_status == 'pending_delivery'
    assert current.result['display'] == 'Original effect completed'
    assert c.state.prepared == c.state.dispatched == c.model.calls == c.host.reserve_calls == 1
    assert c.state.lookups == 0
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
    async with c.manager._write() as tx:
        with pytest.raises(JobError, match='managed_effect_pending'):
            await require_source_resolved(c.manager, tx, await c.manager._row(tx, row['parent_job_id']))


async def test_new_connection_restores_durable_checkpoint_without_runtime_or_business_query(registry_store):
    c, row = await failure_case(registry_store)
    before = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    reader = JobManager(await registry_store[1](schema='compatible'))
    result = await restore_result(reader, c.actor, row['job_id'])
    assert await restore_result(reader, c.actor, row['job_id']) == result
    assert c.state.lookups == 0 and c.model.calls == c.host.reserve_calls == c.state.dispatched == 1
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq') == before
    # The original process must confirm its retained copy has been durably saved.
    assert c.runtime._managed_completions.pending_count() == 1
    assert await flush_local(c.runtime, c.actor, row['job_id']) == result
    assert c.runtime._managed_completions.pending_count() == 0


async def test_new_runtime_cannot_invent_lost_exit_from_remote_outcome_or_expired_lease(registry_store):
    c, row = await failure_case(registry_store, 'exit')
    other = TaskRuntime(c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id,
                        executor_id='new-executor', boot_id='new-boot')
    with pytest.raises(JobError, match='handle_missing'):
        await flush_local(other, c.actor, row['job_id'])
    with pytest.raises(JobError, match='evidence_missing'):
        await restore_result(other.manager, c.actor, row['job_id'])
    assert c.state.dispatched == 1 and c.state.lookups == 0 and c.model.calls == 1
    assert (await c.manager.get_job(c.actor, row['job_id'])).resource_state in ('running', 'unknown')


@pytest.mark.parametrize('part', ['exit_missing', 'anchor_missing', 'anchor_hash', 'exit_repaired_hash',
                                  'scope', 'dispatch', 'prepared', 'generation', 'raw_call'])
async def test_corrupt_or_wrong_generation_cannot_restore_original_result(registry_store, part):
    c, row = await failure_case(registry_store)
    if part in ('exit_missing', 'anchor_missing'):
        kind = 'resource_exit' if part == 'exit_missing' else 'completion'
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (_id(c.manager, kind, row['job_id'], 1),))
    elif part in ('anchor_hash', 'exit_repaired_hash', 'dispatch', 'prepared'):
        kind = {'anchor_hash': 'completion', 'exit_repaired_hash': 'resource_exit', 'dispatch': 'dispatch', 'prepared': 'prepared'}[part]
        identity = _id(c.manager, kind, row['job_id'], 1)
        saved = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (identity,))
        value = json.loads(saved['payload_json'])
        if part == 'exit_repaired_hash':
            value['completion']['result']['managed_effect_result']['reference']['receipt_id'] = 'invented'
        elif part == 'anchor_hash':
            value['exit_hash'] = 'f' * 64
        elif part == 'prepared':
            value['preparation']['reference']['text'] = 'changed'
        else:
            value['preparation_hash'] = 'f' * 64
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(value), _hash(value), identity))
    elif part == 'scope':
        from power_loop.runtime.execution_membership import scope_id
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (scope_id(c.manager, row['job_id'], 1),))
    elif part == 'raw_call':
        observed = await read_raw_receipt(c.manager, c.actor, row['job_id'])
        await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET tool_calls_json='[]' WHERE id=?", (int(observed['call_ref']['id']),))
    else:
        await c.manager.db.execute(f'UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?', (row['job_id'],))
    with pytest.raises(JobError):
        await restore_result(c.manager, c.actor, row['job_id'])
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (row['job_id'],))
    assert c.model.calls == c.host.reserve_calls == c.state.dispatched == 1 and c.state.lookups == 0


@pytest.mark.parametrize('fault', ['result', 'publication', 'result_ack'])
async def test_mechanical_result_repair_rolls_back_or_recalls_same_commit(registry_store, fault):
    c, row = await failure_case(registry_store)
    fail_completion_once(c.manager, fault)
    with pytest.raises(OSError, match='test_completion_'):
        await restore_result(c.manager, c.actor, row['job_id'])
    results = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (row['job_id'],))
    assert len(results) == (1 if fault == 'result_ack' else 0)
    first = await restore_result(c.manager, c.actor, row['job_id'])
    assert await restore_result(c.manager, c.actor, row['job_id']) == first
    assert c.model.calls == c.host.reserve_calls == c.state.dispatched == 1 and c.state.lookups == 0


async def test_parallel_readers_publish_only_one_original_result(registry_store):
    c, row = await failure_case(registry_store)
    readers = [JobManager(await registry_store[1](schema='compatible')) for _ in range(2)]
    values = await asyncio.gather(*(restore_result(m, c.actor, row['job_id']) for m in readers))
    assert values[0] == values[1]
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (row['job_id'],))) == 1
    assert c.model.calls == c.host.reserve_calls == c.state.dispatched == 1


async def test_foreign_actor_and_runtime_rebinding_cannot_use_retained_completion(registry_store):
    c, row = await failure_case(registry_store, 'exit')
    foreign = JobActor('other-tenant', c.actor.principal_id, c.sid)
    for fn, first in [(flush_local, c.runtime), (restore_result, c.manager)]:
        with pytest.raises(JobError):
            await fn(first, foreign, row['job_id'])
    c.runtime.boot_id = 'changed-boot'
    with pytest.raises(JobError, match='runtime_changed'):
        await flush_local(c.runtime, c.actor, row['job_id'])
    assert c.state.dispatched == 1 and c.state.lookups == 0


async def test_bound_before_dispatch_capacity_never_discards_existing_obligations(registry_store):
    c = await scenario(registry_store)
    cache = c.runtime._managed_completions
    slots = [cache.reserve(c.actor, 1000+i, 'call') for i in range(1000)]
    with pytest.raises(JobError, match='capacity_reached'):
        cache.reserve(c.actor, 3000, 'call')
    assert len(cache._slots) == 1000
    for slot in slots:
        cache.release_uncompleted(slot)
    assert not cache._slots


async def test_shutdown_does_not_declare_store_safe_while_completion_unpersisted(registry_store):
    c, row = await failure_case(registry_store)
    result = await c.runtime.shutdown(timeout_s=1, grace_s=0.01)
    assert result['managed_completion_pending'] == 1 and not result['store_safe_to_close']
    status = await c.runtime.executor_status()
    assert status['managed_completion_pending'] == 1 and not status['store_safe_to_close']
    await flush_local(c.runtime, c.actor, row['job_id'])
    assert c.runtime._managed_completions.pending_count() == 0
    assert (await c.runtime.executor_status())['managed_completion_pending'] == 0


async def test_cancellation_before_repair_retains_original_business_result_and_cancelled_outcome(registry_store):
    c, row = await failure_case(registry_store)
    await c.manager.request_cancel(c.actor, row['job_id'], system_key='original-stop', reason='cancel original')
    await restore_result(c.manager, c.actor, row['job_id'])
    result = await c.manager.get_job(c.actor, row['job_id'])
    assert result.outcome == 'cancelled' and result.result['display'] == 'Original effect completed'
    assert result.result['execution_status'] == 'completed' and result.result['status'] == 'cancelled'
    assert not (await restore_result(c.manager, c.actor, row['job_id']))['acknowledgement_created']


@pytest.mark.parametrize('context', ['_SEND', '_OWNER'])
async def test_public_repair_ports_cannot_be_called_from_a_model_or_child_scope(registry_store, context):
    from power_loop.runtime import task_runtime

    c, row = await failure_case(registry_store)
    variable = getattr(task_runtime, context)
    token = variable.set(object())
    try:
        for function, host in [(flush_local, c.runtime), (restore_result, c.manager)]:
            with pytest.raises(JobError, match='requires_host'):
                await function(host, c.actor, row['job_id'])
    finally:
        variable.reset(token)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (row['job_id'],))
    assert c.model.calls == c.state.dispatched == 1


@pytest.mark.parametrize('invalid', ['mapping', 'oversize'])
async def test_invalid_callback_return_preserves_actual_exit_and_unknown_outcome(registry_store, invalid):
    c = await scenario(registry_store, budgeted=True, invalid_result=invalid, max_result_chars=32)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    result = json.loads(row['result_json'])
    assert result['managed_effect_pending'] and result['managed_effect_result'] is None
    assert len(result['display']) <= 32 and row['resource_state'] == 'stopped'
    assert c.state.dispatched == c.model.calls == c.host.reserve_calls == 1 and c.state.lookups == 0
    assert c.runtime._managed_completions.pending_count() == 0


async def test_preparation_failure_respects_a_small_contract_display_limit(registry_store):
    c = await scenario(registry_store, prepare_failure=True, max_result_chars=1)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    result = json.loads(row['result_json'])
    assert len(result['display']) == 1 and result['managed_effect_result']['resolution'] == 'not_dispatched'
    assert row['supervision_status'] == 'closed' and c.state.dispatched == 0


async def test_durable_exit_can_publish_after_original_boot_and_job_leases_expire(registry_store):
    c, row = await failure_case(registry_store)
    c.manager.now = lambda: 9_000_000_000_000
    result = await restore_result(c.manager, c.actor, row['job_id'])
    assert result['owned_callback_exited'] and not result['execution_authority']
    assert (await c.manager.get_job(c.actor, row['job_id'])).result['display'] == 'Original effect completed'
    assert c.model.calls == c.host.reserve_calls == c.state.dispatched == 1 and c.state.lookups == 0


@pytest.mark.parametrize('proof_kind', ['lookalike', 'false', 'different_result'])
async def test_publication_proof_is_a_real_interface_bound_to_the_original_result(registry_store, proof_kind):
    from power_loop.runtime.jobs import JobLease
    from power_loop.runtime.managed_completion import _CheckpointProof, validate_exit
    from power_loop.runtime.managed_effects import descriptor
    from power_loop.runtime.result_proof import ResultPublicationProof

    c, row = await failure_case(registry_store)
    calls = []

    class Lookalike:
        async def validate(self, *args):
            calls.append('lookalike')
            return True

    class FalseProof(ResultPublicationProof):
        async def validate(self, *args):
            calls.append('false')
            return False

    async with c.manager._write() as tx:
        checkpoint = await validate_exit(c.manager, tx, row, await descriptor(c.manager, tx, row))
        payload = json.loads(_json(checkpoint['completion']))
        if proof_kind == 'different_result':
            proof = _CheckpointProof(_hash(checkpoint))
            payload['result']['display'] = 'replacement result'
        else:
            proof = Lookalike() if proof_kind == 'lookalike' else FalseProof()
        with pytest.raises(JobError, match='result_publication_evidence_invalid|managed_completion_result_mismatch'):
            await c.manager._record_result_tx(tx, c.actor, JobLease(**checkpoint['lease']), evidence_only=True,
                outcome=payload['outcome'], result=payload['result'], resource_state='stopped', publication_proof=proof)
    assert calls == (['false'] if proof_kind == 'false' else [])
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (row['job_id'],))
