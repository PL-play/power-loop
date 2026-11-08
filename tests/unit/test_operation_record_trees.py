"""Metadata recording keeps source permission and target publication separate."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, ToolDefinition
from power_loop.runtime import control_requests, operation_records, session_obligations
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.runtime_state import get_tool_runtime_context
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

from .test_executor_activation_trees import observe
from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import child, registry_store, runtime

__all__ = ['registry_store', 'runtime']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def operation_case(runtime):
    store, manager, actor, old, first, _, _ = runtime
    registered = await child(runtime, independent=True)
    target = await manager.claim_job(actor, registered.job.job_id,
        executor_id='independent-worker', boot_id='worker-boot', expected_version=1)
    await manager.close_model_execution(actor, first, evidence={'exit': 'completed'}, resource_state='stopped')
    current = await manager.register_job(actor, kind='send', session_id=actor.session_id, system_key='later-source',
        request={'send_index': 2})
    caller = await manager.claim_job(actor, current.job.job_id,
        executor_id='later-source-worker', boot_id='source-boot', expected_version=1)
    seq = await store.append_message(actor.session_id, role='assistant', send_index=2, round_index=0,
        tool_calls=[{'id': 'actual-record', 'type': 'function', 'function': {'name': 'work', 'arguments': '{}'}}])
    ref = await store.message_ref(actor.session_id, seq)
    return SimpleNamespace(store=store, manager=manager, actor=actor, old=old, registered=registered,
        target=target, current=current, caller=caller, ref=ref, connect=runtime[5], now=runtime[6])


async def record(c, action='steer', *, manager=None, request=None, result=None):
    return await (manager or c.manager).register_operation(c.actor, c.target, caller=c.caller, action=action,
        call_ref=c.ref, tool_call_id='actual-record', request={'message': 'record metadata'} if request is None else request,
        result={'status': 'accepted', 'saved': {'value': 'original'}} if result is None else result)


@pytest.mark.parametrize('action', ['steer', 'nudge'])
async def test_recording_an_operation_on_an_old_independent_tree_has_one_root_per_writer(operation_case, monkeypatch, action):
    c = operation_case
    store, manager, actor, old = c.store, c.manager, c.actor, c.old
    registered, current, target, ref = c.registered, c.current, c.target, c.ref
    assert registered.job.root_job_id == old.job.job_id != current.job.root_job_id
    inbox = await store.inbox_counts(actor.session_id)
    payload = {'status': 'accepted', 'saved': {'value': 'original'}}
    async with observe(manager, monkeypatch) as footprints:
        operation = await record(c, action, result=payload)
    assert operation.job_id == target.job_id and operation.action == action and operation.result == payload
    assert operation.call_ref == ref.to_dict()
    assert await store.inbox_counts(actor.session_id) == inbox
    assert (await manager.get_job(actor, target.job_id)).generation == 1
    assert (await manager.get_job(actor, old.job.job_id)).status == 'model_closed'
    assert footprints and all(len(trees) <= 1 for trees in footprints), footprints


@pytest.mark.parametrize('action', ['steer', 'nudge'])
@pytest.mark.parametrize('stage', ['accepted', 'target-uncommitted', 'target-committed', 'source-uncommitted', 'source-committed'])
async def test_real_process_death_replays_metadata_without_a_second_operation_or_input(operation_case, tmp_path, action, stage):
    c = operation_case
    manager, store = c.manager, c.store
    if manager.db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if manager.db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'actor': asdict(c.actor), 'caller': asdict(c.caller), 'target': asdict(c.target),
        'ref': c.ref.to_dict(), 'now': c.now[0], 'action': action}
    marker = tmp_path/'operation-death.json'
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._operation_record_worker',
        dsn, manager.prefix, json.dumps(config), stage, str(marker),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    inbox = await store.inbox_counts(c.actor.session_id)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 94, stdout.decode()
        assert json.loads(marker.read_text()) == {'stage': stage, 'pid': process.pid}
        originals = await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id')
        target_before = [row for row in originals if row['event_type'] == 'operation' and row['action'] == action]
        assert len(target_before) == int(stage in ('target-committed', 'source-uncommitted', 'source-committed'))
        pending = await manager.db.fetchall(f"SELECT * FROM {manager.prefix}session_obligations WHERE kind='control_request'")
        assert len(pending) == int(stage != 'source-committed')
        cold = JobManager(await c.connect(), now_ms=manager.now)
        receipt = await record(c, action, manager=cold)
        assert receipt.result == {'status': 'accepted', 'saved': {'value': 'original'}}
        after = await cold.db.fetchall(f'SELECT * FROM {cold.events} ORDER BY event_id')
        assert all(row in after for row in originals)
        targets = [row for row in after if row['event_type'] == 'operation' and row['action'] == action]
        assert len(targets) == 1 and targets[0]['event_id'] == receipt.operation_id
        assert not await cold.db.fetchall(f"SELECT * FROM {cold.prefix}session_obligations WHERE kind='control_request'")
        assert await record(c, action, manager=cold) == receipt
        assert await cold.db.fetchall(f'SELECT * FROM {cold.events} ORDER BY event_id') == after
        assert await store.inbox_counts(c.actor.session_id) == inbox
        assert (await cold.get_job(c.actor, c.target.job_id)).generation == 1
        assert (await cold.get_job(c.actor, c.caller.job_id)).generation == 1
    finally:
        await stop_worker(process)


async def test_accepted_metadata_permission_can_finish_after_its_caller_exits(operation_case, monkeypatch):
    c = operation_case
    async def before_target(*args, **kwargs):
        raise OSError('source committed before target')
    with monkeypatch.context() as patch:
        patch.setattr(operation_records, '_apply', before_target)
        with pytest.raises(OSError, match='source committed'):
            await record(c)
    pending = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    assert len(pending) == 1
    await c.manager.close_model_execution(c.actor, c.caller, evidence={'exit': 'completed'}, resource_state='stopped')
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    outcome = await operation_records.reconcile_request(cold, c.actor, pending[0]['source_id'])
    assert outcome.job_id == c.target.job_id and outcome.action == 'steer'
    assert (await cold.get_job(c.actor, c.caller.job_id)).status == 'model_closed'
    assert not await cold.db.fetchall(f"SELECT * FROM {cold.prefix}session_obligations WHERE kind='control_request'")


async def test_expired_target_gets_one_refusal_and_closes_the_source_obligation(operation_case, monkeypatch):
    c = operation_case
    apply = operation_records._apply
    async def expired(*args, **kwargs):
        c.now[0] += 30001
        return await apply(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(operation_records, '_apply', expired)
        with pytest.raises(JobError, match='execution_lease_expired'):
            await record(c)
    before = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    [request] = [row for row in before if row['event_type'] == 'task_control_request' and json.loads(row['payload_json'])['phase'] == 'accepted']
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    with pytest.raises(JobError, match='execution_lease_expired'):
        await operation_records.reconcile_request(cold, c.actor, request['event_id'])
    assert await cold.db.fetchall(f'SELECT * FROM {cold.events} ORDER BY event_id') == before
    assert not await cold.db.fetchall(f"SELECT * FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert not [row for row in before if row['event_type'] == 'operation' and row['action'] == 'steer']


async def test_duplicate_connections_keep_the_first_result_for_one_recorded_call(operation_case):
    c = operation_case
    other = JobManager(await c.connect(), now_ms=c.manager.now)
    replies = await asyncio.gather(record(c, result={'first': 1}), record(c, manager=other, result={'first': 2}))
    assert replies[0] == replies[1]
    assert replies[0].result in ({'first': 1}, {'first': 2})
    assert await record(c, result={'ignored replay': True}) == replies[0]


@pytest.mark.parametrize('action', ['steer', 'nudge'])
@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_each_record_transaction_rollback_or_unknown_ack_recovers_its_original_result(
        operation_case, monkeypatch, action, phase, committed):
    c = operation_case
    transaction, hits = c.manager.db.transaction, []
    inbox = await c.store.inbox_counts(c.actor.session_id)

    @asynccontextmanager
    async def fault():
        async with transaction() as tx:
            yield tx
            events = await tx.fetchall(f'SELECT * FROM {c.manager.events}')
            grants = [event for event in events if event['event_type'] == 'task_control_request'
                and json.loads(event['payload_json'])['phase'] == 'accepted']
            current = None
            if grants:
                if any(event['event_id'] == control_requests.completion_id(grants[0]['event_id']) for event in events):
                    current = 'completed'
                elif any(event['event_type'] == 'operation' and event['action'] == action for event in events):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = not hits and current == phase
            if hit:
                hits.append((phase, committed))
                if not committed:
                    raise OSError('record transaction rolled back')
        if hit:
            raise OSError('record committed; ACK lost')

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, 'transaction', fault)
        with pytest.raises(OSError, match='rolled back|ACK lost'):
            await record(c, action)
    assert hits == [(phase, committed)]
    originals = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    receipt = await record(c, action, manager=cold)
    assert receipt.result == {'status': 'accepted', 'saved': {'value': 'original'}}
    after = await cold.db.fetchall(f'SELECT * FROM {cold.events} ORDER BY event_id')
    assert all(event in after for event in originals)
    assert len([event for event in after if event['event_type'] == 'operation' and event['action'] == action]) == 1
    assert not await cold.db.fetchall(f"SELECT * FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert await record(c, action, manager=cold, result={'ignored': True}) == receipt
    assert await cold.db.fetchall(f'SELECT * FROM {cold.events} ORDER BY event_id') == after
    assert await c.store.inbox_counts(c.actor.session_id) == inbox
    assert (await cold.get_job(c.actor, c.target.job_id)).generation == 1








@pytest.mark.parametrize('action', ['steer', 'nudge'])
async def test_actual_model_tool_records_an_old_tree_and_binds_its_real_receipt_once(registry_store, monkeypatch, action):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='record-loop', principal_id='host', executor_id='record-loop', boot_id='boot')
    calls = []
    tools = ToolRegistry()

    async def callback(text):
        context, binding = get_tool_runtime_context(), _SEND.get()
        assert binding is not None
        ref = await store.message_ref(sid, context.assistant_seq)
        calls.append(await manager.register_operation(actor, lease, caller=binding.lease, action=action,
            call_ref=ref, tool_call_id=context.tool_call_id, request={'message': text}, result={'status': 'accepted'}))
        return 'metadata recorded'

    tools.register(ToolDefinition('record_metadata', 'Record acceptance metadata',
        input_schema={'type': 'object', 'properties': {'text': {'type': 'string'}}}, pricing=Free('fixture')), callback)

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            return LLMResponse(raw_text='done' if self.calls > 1 else '', tool_calls=None if self.calls > 1 else [
                {'id': 'actual-loop-record', 'type': 'function',
                 'function': {'name': 'record_metadata', 'arguments': '{"text":"metadata"}'}}])
        def stream(self, request):
            raise AssertionError('nonstream fixture')
        async def close(self):
            pass

    model = Model()
    loop = await RunningAgent(store, model, tools, task_runtime=runtime,
        config=AgentLoopConfig(model='test', max_tokens=50, delegation_policy='single_level',
                               compactor=None, max_rounds=3)).start()
    sid = loop.target.session_id
    actor = runtime.actor(sid)
    old = await manager.register_job(actor, kind='send', session_id=sid, system_key='old-parent')
    first = await manager.claim_job(actor, old.job.job_id, executor_id='old', boot_id='old', expected_version=1)
    child_sid = await store.create_session(parent_session_id=sid, metadata={'execution_role': 'subagent'})
    target = await manager.register_job(actor, kind='subagent', session_id=child_sid, parent=first, independent=True, system_key='old-independent')
    lease = await manager.claim_job(actor, target.job.job_id, executor_id='target', boot_id='target', expected_version=1)
    await manager.close_model_execution(actor, first, evidence={'exit': 'completed'}, resource_state='stopped')
    try:
        result = await loop.send('record old work')
        assert result.status == 'completed' and result.final_text == 'done'
        assert model.calls == 2 and len(calls) == 1
        operation = await manager.get_operation(actor, calls[0].operation_id)
        assert operation.job_id == lease.job_id and operation.receipt_ref is not None
        receipt = await store.message_by_id(sid, operation.receipt_ref['id'])
        assert receipt.role == 'tool' and receipt.tool_call_id == 'actual-loop-record'
        assert not (await store.get_state(sid)).pending
        assert (await manager.get_job(actor, lease.job_id)).generation == 1
        assert (await store.inbox_counts(sid))['pending'] == 0
        assert not await manager.db.fetchall(f"SELECT * FROM {manager.prefix}session_obligations WHERE kind='control_request'")
    finally:
        await loop.aclose()


@pytest.mark.parametrize('action', ['steer', 'nudge'])
@pytest.mark.parametrize('entry', ['verify', 'rebuild'])
async def test_cold_entry_refuses_a_target_record_without_its_original_source_permission(operation_case, action, entry):
    c = operation_case
    await record(c, action)
    originals = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    grants = [event for event in originals if event['event_type'] == 'task_control_request']
    assert len(grants) == 2
    try:
        for event in grants:
            assert await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (event['event_id'],)) == 1
        before = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
        jobs = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id')
        pending = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}session_obligations ORDER BY source_id')
        with pytest.raises(StoreSchemaError, match='original caller permission'):
            if entry == 'verify':
                await session_obligations.verify(c.manager, c.actor)
            elif entry == 'rebuild':
                await session_obligations.rebuild(c.manager)
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == before
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id') == jobs
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}session_obligations ORDER BY source_id') == pending
        assert (await c.manager.db.fetchone(f'SELECT protocol_version FROM {c.manager.prefix}task_store WHERE id=1'))['protocol_version'] == TASK_PROTOCOL_VERSION
    finally:
        for event in grants:
            columns = tuple(event)
            await c.manager.db.execute(f"INSERT INTO {c.manager.events} ({','.join(columns)}) VALUES ("
                + ','.join('?' for _ in columns) + ')', tuple(event.values()))
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == originals
