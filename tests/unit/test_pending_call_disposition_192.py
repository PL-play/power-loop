"""Actual raw-receipt failure leaves a captured pending obligation, never a replay."""
import json
from contextlib import asynccontextmanager

import pytest

from power_loop.runtime.jobs import JobActor, JobError, _hash, _json
from power_loop.runtime.managed_delivery import enqueue
from power_loop.runtime.pending_calls import SNAPSHOT, retire_pending
from power_loop.runtime.representation import ProjectedRepresentation

from .test_job_registry_192 import registry_store
from .test_managed_delivery_192 import controller
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def original_pending(registry_store):
    c = await scenario(registry_store, budgeted=True, raw_receipt_failure=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    source = await c.manager.get_job(c.actor, row['parent_job_id'])
    assert source.status == 'model_closed' and source.resource_state == 'stopped'
    pending = (await c.store.get_state(c.sid)).pending
    assert pending['tool_call_ids'] == ['original-call']
    closed = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', ('model_closed_' + source.job_id,))
    assert json.loads(closed['payload_json'])['evidence'][SNAPSHOT]['pending'] == pending
    assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    assert c.state.prepared == c.state.dispatched == c.model.calls == 1
    loop, reviewer = await controller(c)
    loop.config.representation = ProjectedRepresentation()
    await enqueue(loop, c.actor, row['job_id'])
    try:
        yield c, row, source, pending, loop, reviewer
    finally:
        await loop.aclose(drain_timeout_s=0.1)


async def test_pending_retires_only_for_actual_review_and_raw_messages_remain_original(original_pending):
    c, row, source, pending, loop, reviewer = original_pending
    raw = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    disposition = await retire_pending(loop, c.actor, source.job_id)
    assert await retire_pending(loop, c.actor, source.job_id) == disposition
    assert not disposition['execution_authority'] and not disposition['acknowledgement_created']
    assert not disposition['billing_settled'] and reviewer.calls == 0
    assert (await c.store.get_state(c.sid)).pending is None
    assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'pending_delivery'
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.actor, source.job_id, generation=1)
    result = await loop.send('Read the original result and acknowledge without repeating it.', c.sid)
    assert result.status == 'completed' and reviewer.calls == 1
    current = await c.manager.get_job(c.actor, row['job_id'])
    assert current.result_status == 'handled' and current.supervision_status == 'closed'
    assert (await c.manager.get_job(c.actor, source.job_id)).supervision_status == 'closed'
    stored = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (disposition['disposition']['event_id'],))
    assert json.loads(stored['payload_json'])['snapshot']['pending'] == pending
    for message in raw:
        assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (message['id'],)) == message
    assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    assert c.state.dispatched == c.state.prepared == c.model.calls == reviewer.calls == 1
    assert c.state.lookups == 0 and c.host.reserve_calls == 2


@pytest.mark.parametrize('part', ['model_missing', 'resource_unknown', 'source_generation', 'snapshot_call',
                                  'model_attempt', 'notice_missing', 'pending_changed', 'pending_missing'])
async def test_incomplete_or_rewritten_original_evidence_cannot_dispose_pending(original_pending, part):
    c, row, source, pending, loop, reviewer = original_pending
    key = 'model_closed_' + source.job_id
    if part == 'model_missing':
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (key,))
    elif part == 'resource_unknown':
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET resource_state='unknown' WHERE job_id=?", (source.job_id,))
    elif part == 'source_generation':
        await c.manager.db.execute(f'UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?', (source.job_id,))
    elif part == 'snapshot_call':
        event = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (key,))
        value = json.loads(event['payload_json'])
        value['evidence'][SNAPSHOT]['pending']['tool_calls'][0]['function']['arguments'] = '{"text":"rewritten"}'
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?', (_json(value), key))
    elif part == 'model_attempt':
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='model_attempt_finished'", (source.job_id,))
    elif part == 'notice_missing':
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='managed_effect_delivery'", (row['job_id'],))
    else:
        await c.store.set_pending(c.sid, {**pending, 'round_index': 20} if part == 'pending_changed' else None)
    before = (await c.store.get_state(c.sid)).pending
    with pytest.raises(JobError):
        await retire_pending(loop, c.actor, source.job_id)
    assert (await c.store.get_state(c.sid)).pending == before
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='pending_call_disposition'")
    assert reviewer.calls == 0 and c.state.dispatched == c.model.calls == 1


@pytest.mark.parametrize('stage', ['record', 'state', 'commit', 'commit_ack'])
async def test_pending_state_and_disposition_are_atomic_and_lost_ack_is_recalled(original_pending, stage):
    c, row, source, pending, loop, reviewer = original_pending
    original, fired = c.manager._write, False

    @asynccontextmanager
    async def inject():
        nonlocal fired
        async with original() as tx:
            execute = tx.execute
            async def fault(sql, args=()):
                nonlocal fired
                target = (stage == 'record' and sql.startswith('INSERT INTO ' + c.manager.events)
                          or stage == 'state' and sql.startswith('UPDATE ' + c.store.t.session_state))
                if target and not fired:
                    fired = True
                    raise OSError('test_pending_write_failed')
                return await execute(sql, args)
            tx.execute = fault
            yield tx
            if stage == 'commit' and not fired:
                fired = True
                raise OSError('test_pending_commit_failed')
        if stage == 'commit_ack' and not fired:
            fired = True
            raise OSError('test_pending_ack_lost')
    c.manager._write = inject
    with pytest.raises(OSError):
        await retire_pending(loop, c.actor, source.job_id)
    assert fired
    c.manager._write = original
    assert ((await c.store.get_state(c.sid)).pending is None) == (stage == 'commit_ack')
    result = await retire_pending(loop, c.actor, source.job_id)
    assert await retire_pending(loop, c.actor, source.job_id) == result
    assert reviewer.calls == 0 and c.state.dispatched == 1
    records = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='pending_call_disposition'")
    assert len(records) == 1 and (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'pending_delivery'


async def test_foreign_main_cannot_clear_original_pending_and_duplicate_cannot_clear_new_pending(original_pending):
    c, row, source, pending, loop, reviewer = original_pending
    other = await c.store.create_session()
    with pytest.raises(JobError):
        await retire_pending(loop, JobActor(c.actor.tenant_id, c.actor.principal_id, other), source.job_id)
    assert (await c.store.get_state(c.sid)).pending == pending
    first = await retire_pending(loop, c.actor, source.job_id)
    newer = {'assistant_seq': 999, 'round_index': 1, 'tool_calls': [], 'tool_call_ids': []}
    await c.store.set_pending(c.sid, newer)
    assert await retire_pending(loop, c.actor, source.job_id) == first
    assert (await c.store.get_state(c.sid)).pending == newer and reviewer.calls == 0


@pytest.mark.parametrize('part', ['delete', 'repair_hash'])
async def test_closed_parent_rechecks_retained_pending_responsibility(original_pending, part):
    c, row, source, pending, loop, reviewer = original_pending
    result = await retire_pending(loop, c.actor, source.job_id)
    await loop.send('Read and handle the known result.', c.sid)
    assert (await c.manager.get_job(c.actor, source.job_id)).supervision_status == 'closed'
    key = result['disposition']['event_id']
    if part == 'delete':
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (key,))
    else:
        record = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (key,))
        value = json.loads(record['payload_json'])
        value['calls'][0]['notice_hash'] = 'f' * 64
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(value), _hash(value), key))
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.actor, source.job_id, generation=1)
    assert reviewer.calls == 1 and c.state.dispatched == 1


async def test_cancel_after_model_exit_preserves_phase_and_pending_responsibility(original_pending):
    c, row, source, pending, loop, reviewer = original_pending
    await c.manager.request_cancel(c.actor, source.job_id, system_key='original-stop', reason='Stop original work')
    cancelled = await c.manager.get_job(c.actor, source.job_id)
    assert cancelled.cancel_requested_at is not None and cancelled.status == 'model_closed'
    disposition = await retire_pending(loop, c.actor, source.job_id)
    assert not disposition['acknowledgement_created'] and reviewer.calls == 0
    await loop.send('Read the original outcome. Do not repeat the cancelled work.', c.sid)
    assert reviewer.calls == 1 and c.state.dispatched == 1
    assert (await c.manager.get_job(c.actor, source.job_id)).cancel_requested_at is not None
    assert (await c.manager.get_job(c.actor, source.job_id)).supervision_status == 'closed'


async def test_execution_context_cannot_use_host_pending_recovery_port(original_pending):
    from power_loop.runtime.task_runtime import _SEND

    c, row, source, pending, loop, reviewer = original_pending
    token = _SEND.set(object())
    try:
        with pytest.raises(JobError):
            await retire_pending(loop, c.actor, source.job_id)
    finally:
        _SEND.reset(token)
    assert (await c.store.get_state(c.sid)).pending == pending and reviewer.calls == 0


async def test_concurrent_readers_recall_one_disposition_without_new_execution(original_pending, registry_store):
    import asyncio

    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.task_runtime import TaskRuntime
    from power_loop.tools.registry import ToolRegistry

    c, row, source, pending, loop, reviewer = original_pending
    reader = await registry_store[1](schema='compatible')
    runtime = TaskRuntime(JobManager(reader), tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id,
                          executor_id='query-only-reader', boot_id='query-boot')
    other = StatefulAgentLoop(llm=reviewer, store=reader, task_runtime=runtime, tool_registry=ToolRegistry(), config=AgentLoopConfig(delegation_policy="single_level"))
    try:
        first, second = await asyncio.gather(retire_pending(loop, c.actor, source.job_id),
                                             retire_pending(other, c.actor, source.job_id))
        assert first == second and reviewer.calls == 0
        assert (await c.store.get_state(c.sid)).pending is None
        assert c.state.dispatched == c.model.calls == 1
        [record] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='pending_call_disposition'")
        assert record['event_id'] == first['disposition']['event_id']
    finally:
        await other.aclose(drain_timeout_s=0.1)


@pytest.mark.parametrize('failure', ['hidden', 'model_error'])
async def test_retired_call_is_still_unhandled_without_actual_main_processing(original_pending, failure):
    from power_loop import HookPoint

    c, row, source, pending, loop, reviewer = original_pending
    await retire_pending(loop, c.actor, source.job_id)
    if failure == 'model_error':
        reviewer.fail = True
    else:
        reviewer.sees = False
        async def hide(ctx):
            ctx.messages = [m for m in ctx.messages if 'Original tool outcome:' not in str(m.get('content'))]
        loop.hooks.register(HookPoint.LLM_BEFORE, hide, name='test.pending.hide-result')
    await loop.send('Review the original outcome without repeating its action.', c.sid)
    assert reviewer.calls == 1 and c.state.dispatched == c.model.calls == 1
    assert (await c.manager.get_job(c.actor, row['job_id'])).result_status != 'handled'
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.actor, source.job_id, generation=1)
