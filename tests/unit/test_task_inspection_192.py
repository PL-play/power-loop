"""Deterministic persistent alerts, without model polling or invented task outcomes."""
import asyncio
import json

import pytest

from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.task_inspection import InspectionPolicy, inspect_tasks, read_alert

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime

__all__ = ['registry_store', 'registry_runtime']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
POLICY = InspectionPolicy(grace_ms=1000, runtime_ms=10000, result_ms=1000, escalation_ms=5000)


@pytest.fixture
async def runtime(registry_runtime):
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import Command, RuntimeDescriptor, TargetRef, TrustedActor, canonical, plain
    store, manager, actor, root, *_ = registry_runtime
    descriptor = RuntimeDescriptor(1, 'agent', 'inspection@1', None, 'tools@1', 'hooks@1', 'resources@1',
                                   {'max_rounds': 3}, {'run_id': 101})
    await manager.db.execute(f'UPDATE {store.t.sessions} SET metadata_json=? WHERE session_id=?',
        (canonical({'tenant_id': actor.tenant_id, 'principal_id': actor.principal_id,
                    'runtime_descriptor': plain(descriptor)}), actor.session_id))
    port = RuntimeInputStore(store)
    async with manager._write() as tx:
        await port.accept_input(tx, Command('original', TargetRef(store.store_id, actor.session_id),
            'send', {'body': 'original task'}, 'new_turn'), TrustedActor(actor.tenant_id, actor.principal_id, None))
        await tx.execute(f"UPDATE {port.q} SET status='delivered',delivered_job_id=?,delivered_job_generation=1 "
                         "WHERE session_id=? AND item_id='original'", (root.job.job_id, actor.session_id))
    # A later definition edit must not change the original alert's profile or attribution.
    await manager.db.execute(f'UPDATE {store.t.sessions} SET metadata_json=? WHERE session_id=?',
        (canonical({'tenant_id': actor.tenant_id, 'principal_id': actor.principal_id,
                    'runtime_descriptor': plain(RuntimeDescriptor(1, 'agent', 'inspection@2', None,
                        'tools@2', 'hooks@2', 'resources@2', {'max_rounds': 9}, {'run_id': 202}))}), actor.session_id))
    return registry_runtime


async def child(runtime, *, key='child'):
    store, manager, actor, _, lease, *_ = runtime
    sid = await store.create_session(parent_session_id=actor.session_id, metadata={'execution_role': 'subagent'})
    return await manager.register_job(actor, kind='subagent', parent=lease, session_id=sid, system_key=key)


async def troubled(runtime, *, state='unknown'):
    store, manager, actor, root, lease, connect, now = runtime
    reg = await child(runtime)
    proof = await manager.claim_job(actor, reg.job.job_id, executor_id='child', boot_id='child-boot',
                                    expected_version=1)
    if state == 'unknown':
        await manager.db.execute(f"UPDATE {manager.jobs} SET resource_state='unknown' WHERE job_id=?", (proof.job_id,))
    elif state == 'cancel':
        await manager.request_cancel(actor, proof.job_id, system_key='cancel-once', reason='stop original')
    elif state == 'runtime':
        now[0] += POLICY.runtime_ms
    elif state == 'result':
        await manager.record_execution_result(actor, proof, outcome='failed', result={'text': 'failed'}, resource_state='stopped')
    await manager.close_model_execution(actor, lease, evidence={'exit': 'completed'})
    await manager.db.execute(f"UPDATE {manager.jobs} SET result_status='handled' WHERE job_id=?", (root.job.job_id,))
    now[0] += 2000
    return proof


@pytest.mark.parametrize('state,reason', [('unknown', 'resource_unknown'), ('cancel', 'stop_unconfirmed'),
                                         ('runtime', 'long_running'), ('result', 'result_unhandled')])
async def test_current_job_alert_reason_without_job_lease(runtime, state, reason):
    from power_loop.runtime.task_inspection import _reason
    _, manager, _, _, _, _, now = runtime
    proof = await troubled(runtime, state=state)
    row = await manager.db.fetchone(f'SELECT * FROM {manager.jobs} WHERE job_id=?', (proof.job_id,))
    assert _reason(row, now[0], POLICY) == reason


@pytest.mark.parametrize('state,reason', [('unknown', 'resource_unknown'), ('cancel', 'stop_unconfirmed'),
                                         ('runtime', 'long_running'), ('result', 'result_unhandled')])
async def test_original_problem_creates_one_main_alert_without_finishing_or_stopping_work(runtime, state, reason):
    store, manager, actor, root, lease, connect, now = runtime
    proof = await troubled(runtime, state=state)
    before = await manager.get_job(actor, proof.job_id)
    first = await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY)
    assert len(first['notification_ids']) == 1
    key = first['notification_ids'][0]
    _, event, data = await read_alert(manager, actor, key)
    assert data['reason'] == reason and data['affected']['job_id'] == proof.job_id
    assert data['level'] == 1 and not data['execution_authority'] and not data['resource_stop_confirmed']
    assert event['job_id'] == root.job.job_id and data['main_session_id'] == actor.session_id
    assert await manager.get_job(actor, proof.job_id) == before
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='processing_decision'")
    inbox = await manager.db.fetchone(f'SELECT * FROM {store.t.follow_up_queue} WHERE item_id=?', (key,))
    assert inbox['session_id'] == actor.session_id and 'not a task result' in inbox['content']
    public = json.loads(json.loads(inbox['content']).split('Task alert data:\n', 1)[1])
    meta = json.loads(inbox['meta_json'])
    assert meta['operation'] == 'append' and meta['descriptor']['definition_ref'] == 'inspection@1'
    assert meta['descriptor']['host_context']['run_id'] == 101 and meta['options']['max_rounds'] == 3
    assert 'root' not in public and public['root_job_id'] == root.job.job_id
    assert 'executor_id' not in inbox['content'] and 'boot_id' not in inbox['content']
    assert (await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY))['notification_ids'] == []


async def test_parallel_inspectors_and_new_process_connection_do_not_duplicate_or_flood_unread_alerts(runtime):
    store, manager, actor, root, lease, connect, now = runtime
    await troubled(runtime)
    second = JobManager(await connect(), now_ms=manager.now)
    pages = await asyncio.gather(*(inspect_tasks(m, actor, root.job.job_id, policy=POLICY)
                                  for m in [manager, second, manager, second]))
    keys = [key for page in pages for key in page['notification_ids']]
    assert len(keys) == len(set(keys)) == 1
    now[0] += 6000
    assert (await inspect_tasks(second, actor, root.job.job_id, policy=POLICY))['notification_ids'] == []
    await manager.db.execute(f"UPDATE {store.t.follow_up_queue} SET status='delivered' WHERE item_id=?", (keys[0],))
    next_page = await inspect_tasks(second, actor, root.job.job_id, policy=POLICY)
    assert len(next_page['notification_ids']) == 1
    assert (await read_alert(second, actor, next_page['notification_ids'][0]))[2]['level'] == 2
    now[0] += 60000
    assert (await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY))['notification_ids'] == []


@pytest.mark.parametrize('fault', ['resolved', 'superseded', 'claimed', 'content', 'event', 'authority_flag', 'foreign'])
async def test_stale_changed_hidden_or_foreign_alert_is_not_fresh_wake_authority(runtime, fault):
    store, manager, actor, root, lease, connect, now = runtime
    proof = await troubled(runtime)
    [key] = (await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY))['notification_ids']
    original = await read_alert(manager, actor, key)
    if fault == 'resolved':
        await manager.db.execute(f"UPDATE {manager.jobs} SET resource_state='stopped' WHERE job_id=?", (proof.job_id,))
    elif fault == 'superseded':
        await manager.db.execute(f'UPDATE {manager.jobs} SET generation=generation+1 WHERE job_id=?', (proof.job_id,))
    elif fault == 'claimed':
        await manager.db.execute(f"UPDATE {store.t.follow_up_queue} SET status='claimed',claim_token='owner',"
                                "claimed_at=?,claim_owner_epoch=1 WHERE item_id=?", (now[0], key))
    elif fault == 'content':
        await manager.db.execute(f"UPDATE {store.t.follow_up_queue} SET content='hidden' WHERE item_id=?", (key,))
    elif fault == 'authority_flag':
        from power_loop.runtime.jobs import _hash, _json
        data = dict(original[2], execution_authority=True)
        await manager.db.execute(f'UPDATE {manager.events} SET payload_json=?,request_hash=? WHERE event_id=?',
                                  (_json(data), _hash(data), key))
    elif fault == 'event':
        data = dict(original[2], level=2)
        await manager.db.execute(f'UPDATE {manager.events} SET payload_json=? WHERE event_id=?', (json.dumps(data), key))
    else:
        actor = JobActor(actor.tenant_id, 'different-host', actor.session_id)
    with pytest.raises(JobError):
        await read_alert(manager, actor, key)
    if fault in ('resolved', 'superseded', 'claimed'):
        assert (await read_alert(manager, actor, key, pending=False))[2] == original[2]


async def test_explicit_wait_does_not_become_runaway_and_pages_do_not_drop_descendants(runtime):
    store, manager, actor, root, lease, connect, now = runtime
    reg = await child(runtime, key='waiting-child')
    await manager.db.execute(f"UPDATE {manager.jobs} SET status='waiting' WHERE job_id=?", (reg.job.job_id,))
    now[0] += 100000
    after, found, visited = '', [], set()
    while True:
        page = await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY, limit=1, after_job_id=after)
        found.extend(page['notification_ids'])
        if page['next_cursor'] is None:
            break
        assert page['next_cursor'] not in visited
        visited.add(page['next_cursor'])
        after = page['next_cursor']
    # The old running root can be long-running; the explicit wait is never one.
    for key in found:
        assert (await read_alert(manager, actor, key))[2]['affected']['job_id'] != reg.job.job_id


@pytest.mark.parametrize('values', [{'grace_ms': True}, {'runtime_ms': 0}, {'escalation_ms': 1000}])
async def test_policy_bounds_are_strict(values):
    with pytest.raises(JobError, match='invalid_inspection_policy'):
        InspectionPolicy(**values)


@pytest.mark.parametrize('stage', ['rollback', 'lost_ack'])
async def test_event_and_inbox_are_atomic_and_ack_retry_is_idempotent(runtime, monkeypatch, stage):
    from contextlib import asynccontextmanager

    store, manager, actor, root, lease, connect, now = runtime
    await troubled(runtime)
    original = manager._write
    failed = False

    @asynccontextmanager
    async def fault():
        nonlocal failed
        if failed:
            async with original() as tx:
                yield tx
            return
        failed = True
        async with original() as tx:
            yield tx
            if stage == 'rollback':
                raise OSError('original alert transaction rolled back')
        raise OSError('original alert committed; acknowledgement lost')

    monkeypatch.setattr(manager, '_write', fault)
    with pytest.raises(OSError):
        await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY)
    events = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='task_alert'")
    assert len(events) == (0 if stage == 'rollback' else 1)
    retry = await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY)
    assert len(retry['notification_ids']) == (1 if stage == 'rollback' else 0)
    [event] = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='task_alert'")
    assert await read_alert(manager, actor, event['event_id'])
    inbox = await manager.db.fetchall(f"SELECT item_id FROM {store.t.follow_up_queue} WHERE kind='task_alert'")
    assert len(inbox) == 1


async def test_observation_time_can_precede_event_commit_without_invalidating_alert(runtime):
    store, manager, actor, root, lease, connect, now = runtime
    await troubled(runtime)
    clock = now[0]

    def ticking():
        nonlocal clock
        clock += 1
        return clock

    manager.now = ticking
    [key] = (await inspect_tasks(manager, actor, root.job.job_id, policy=POLICY))['notification_ids']
    _, event, value = await read_alert(manager, actor, key)
    assert event['created_at'] > value['observed_at_ms']
