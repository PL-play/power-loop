"""A positive SDK pre-admission checkpoint, never absence-based non-execution."""
import json

import pytest

from power_loop.runtime.jobs import JobError

from ._tool_plan_faults_192 import fail_plan_once
from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('stage', ['first_registration', 'seal', 'commit', 'commit_ack'])
async def test_original_prepare_failure_records_abort_and_retains_unprocessed_pending(registry_store, stage):
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, stage)
    c = await scenario(registry_store, call_count=2, budgeted=True, configure=configure,
                       expected_error='test_plan_' + stage)
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type IN ('tool_call_abort','tool_call_abort_anchor')")
    assert len(events) == 2
    assert c.model.calls == c.host.reserve_calls == 1
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    assert source['status'] == 'model_closed' and source['resource_state'] == 'stopped'
    pending = (await c.store.get_state(c.sid)).pending
    assert pending['tool_call_ids'] == ['original-call', 'original-call-2']
    from power_loop.runtime.tool_call_abort import read_abort
    raw = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    value = await read_abort(c.manager, c.actor, source['job_id'], assistant_seq=pending['assistant_seq'])
    assert value['decision'] == 'admission_not_granted'
    assert len(value['entries']) == 2 and not value['execution_authority']
    assert not value['acknowledgement_created'] and not value['billing_settled']
    assert value['plan'] is None if stage != 'commit_ack' else value['plan'] is not None
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq') == raw
    assert (await c.store.get_state(c.sid)).pending == pending
    closed = json.loads((await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", ('model_closed_' + source['job_id'],)))['payload_json'])
    assert closed['evidence']['pending_call_snapshot']['version'] == 3
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='pending_call_disposition'")
    assert not any(row['role'] == 'tool' for row in raw)


@pytest.mark.parametrize('part', ['abort_missing', 'anchor_missing', 'payload_hash', 'scope_boolean', 'version_boolean', 'raw_name', 'dispatch', 'plan_deleted', 'source_protocol'])
async def test_abort_read_and_parent_close_refuse_missing_or_conflicting_originals(registry_store, part):
    from power_loop.runtime.jobs import _hash, _json
    from power_loop.runtime.tool_call_abort import read_abort
    stage = 'commit_ack' if part == 'plan_deleted' else 'seal'
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, stage)
    c = await scenario(registry_store, call_count=2, configure=configure, expected_error='test_plan_' + stage)
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='tool_call_abort'")
    value = json.loads(event['payload_json'])
    if part in ('abort_missing', 'anchor_missing'):
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (event['event_id'] + ('_anchor' if part == 'anchor_missing' else ''),))
    elif part in ('payload_hash', 'scope_boolean', 'version_boolean'):
        if part == 'scope_boolean':
            value['scope']['generation'] = True
        elif part == 'version_boolean':
            value['version'] = True
        else:
            value['decision'] = 'guessed_from_absence'
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(value), _hash(value), event['event_id']))
    elif part == 'raw_name':
        row = await c.store.get_message(c.sid, int(value['assistant_ref']['seq']))
        calls = row.tool_calls
        calls[0]['function']['name'] = 'changed'
        await c.manager.db.execute(f'UPDATE {c.store.t.messages} SET tool_calls_json=? WHERE id=?', (_json(calls), int(value['assistant_ref']['id'])))
    elif part == 'dispatch':
        plan_id = event['event_id'].removesuffix('_abort')
        await c.manager.db.execute(f'INSERT INTO {c.manager.events} (event_id,job_id,event_type,job_version,payload_json,request_hash,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)',
                                  (plan_id+'_start_'+_hash(value['entries'][0]['call_id'])[:24], source['job_id'], 'tool_call_dispatch_started', source['version'], '{}', _hash({}), 'not_applicable', c.manager.now()))
    elif part == 'source_protocol':
        creation = await c.manager.db.fetchone(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (source['job_id'],))
        payload = json.loads(creation['payload_json'])
        del payload['request']['tool_call_plan_protocol']
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?', (_json(payload), creation['event_id']))
    else:
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='tool_call_plan'")
    before = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    with pytest.raises(JobError):
        await read_abort(c.manager, c.actor, source['job_id'], assistant_seq=int(value['assistant_ref']['seq']))
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.actor, source['job_id'], generation=1)
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == before
    assert (await c.store.get_state(c.sid)).pending is not None
    assert c.model.calls == 1 and c.state.dispatched == 0


async def test_abort_checkpoint_cannot_be_reused_from_host_or_dispatch_or_claim(registry_store, monkeypatch):
    from power_loop.runtime import tool_call_abort, tool_call_plans
    from power_loop.runtime.invocation import invocation_context
    actual = tool_call_abort._record
    captured, rejected = [], []
    async def observe(attempt):
        value = await actual(attempt)
        captured.append(attempt)
        runtime = attempt.execution.runtime
        manager, actor = runtime.manager, attempt.execution.actor
        jobs = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
        assert len(jobs) == 2
        for job in jobs:
            with pytest.raises(JobError, match='tool_call_batch_aborted'):
                await manager.claim_job(actor, job['job_id'], executor_id='forbidden', boot_id='boot', expected_version=1)
        entry = value['entries'][0]
        with invocation_context(actor.session_id, attempt.ref.seq, entry['call_id']):
            with pytest.raises(JobError, match='tool_call_batch_aborted'):
                await tool_call_plans.dispatch(attempt.pipeline, entry['tool_name'], {'text': 'hello'}, attempt.pipeline.tool_registry.get(entry['tool_name']))
        rejected.append(True)
        return value
    monkeypatch.setattr(tool_call_abort, '_record', observe)
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, 'commit_ack')
    c = await scenario(registry_store, call_count=2, configure=configure, expected_error='test_plan_commit_ack')
    assert rejected == [True]
    with pytest.raises(JobError, match='original_stack_required'):
        await actual(captured[0])
    assert c.state.prepared == c.state.dispatched == 0


async def test_exception_after_actual_admission_cannot_be_reclassified_as_not_admitted(registry_store, monkeypatch):
    from power_loop.runtime import tool_call_plans
    actual = tool_call_plans._prepare
    async def after_grant(pipeline, attempt):
        await actual(pipeline, attempt)
        raise OSError('test_after_actual_admission')
    monkeypatch.setattr(tool_call_plans, '_prepare', after_grant)
    c = await scenario(registry_store, expected_error='test_after_actual_admission')
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='tool_call_abort'")
    assert c.state.dispatched == 0 and c.model.calls == 1


@pytest.mark.parametrize('stage', ['checkpoint', 'anchor', 'commit', 'commit_ack'])
async def test_checkpoint_storage_failure_preserves_original_failure_and_unknown_responsibility(registry_store, stage):
    from ._tool_abort_faults_192 import fail_abort_once
    faults = []
    async def configure(loop, state):
        manager = loop.task_runtime.manager
        fail_plan_once(manager, 'seal')
        faults.append(fail_abort_once(manager, stage))
    c = await scenario(registry_store, configure=configure, expected_error='test_plan_seal')
    assert faults[0].fired and c.result.error == 'test_plan_seal'
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type IN ('tool_call_abort','tool_call_abort_anchor')")
    assert len(events) == (2 if stage == 'commit_ack' else 0)
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    assert source['status'] == 'model_closed'
    assert source['resource_state'] == ('stopped' if stage == 'commit_ack' else 'unknown')
    assert (await c.store.get_state(c.sid)).pending is not None
    assert c.model.calls == 1 and c.state.prepared == c.state.dispatched == 0


@pytest.mark.parametrize('when', ['plan', 'abort'])
async def test_actual_cancellation_before_checkpoint_keeps_unknown_pending_and_no_grant(registry_store, when):
    import asyncio
    from contextlib import asynccontextmanager
    captured = {}
    async def configure(loop, state):
        manager = loop.task_runtime.manager
        captured.update(manager=manager, model=loop.llm, state=state)
        if when == 'abort':
            fail_plan_once(manager, 'seal')
        original = manager._write
        class Transaction:
            def __init__(self, target):
                self.target = target
            def __getattr__(self, name):
                return getattr(self.target, name)
            async def execute(self, statement, params=()):
                if statement.startswith(f'INSERT INTO {manager.events} ') and len(params) > 2 and params[2] == ('tool_call_plan' if when == 'plan' else 'tool_call_abort'):
                    raise asyncio.CancelledError()
                return await self.target.execute(statement, params)
        @asynccontextmanager
        async def cancel_write():
            async with original() as tx:
                yield Transaction(tx)
        manager._write = cancel_write
    with pytest.raises(asyncio.CancelledError):
        await scenario(registry_store, configure=configure)
    manager = captured['manager']
    [source] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='send'")
    assert source['status'] == 'model_closed' and source['resource_state'] == 'unknown'
    assert captured['model'].calls == 1 and captured['state'].prepared == captured['state'].dispatched == 0
    assert (await manager.store.get_state(source['session_id'])).pending is not None
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type IN ('tool_call_abort','tool_call_abort_anchor','tool_call_dispatch_started')")


async def test_positive_checkpoint_reader_does_not_grant_another_principal_access(registry_store):
    from dataclasses import replace

    from power_loop.runtime.tool_call_abort import read_abort
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, 'seal')
    c = await scenario(registry_store, configure=configure, expected_error='test_plan_seal')
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    pending = (await c.store.get_state(c.sid)).pending
    with pytest.raises(JobError):
        await read_abort(c.manager, replace(c.actor, principal_id='another'), source['job_id'], assistant_seq=pending['assistant_seq'])
    assert c.state.dispatched == 0 and c.model.calls == 1
