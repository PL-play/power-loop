"""A new boot proves old roots without retaining a control writer."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from power_loop.runtime import execution_membership
from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import spawn_worker, stop_worker, wait_ready
from .test_executor_registry_192 import boot, root, start
from .test_job_registry_192 import registry_store

__all__ = ['boot', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def old_roots(c):
    await start(c)
    leases = [(await root(c, key=key))[1] for key in ('old-first', 'old-second')]
    for lease in leases:
        await c.manager.close_model_execution(c.actor, lease,
            evidence={'exit': 'completed'}, resource_state='stopped')
    c.now[0] += 1001
    identity = replace(c.identity, boot_id='new-boot')
    await registry.register(c.manager, identity, contract_hash='a'*64, startup_lease_ms=300000)
    return identity, leases


@asynccontextmanager
async def observe(manager, monkeypatch):
    rows = await manager.db.fetchall(f'SELECT job_id,root_job_id FROM {manager.jobs}')
    roots = {row['job_id']: row['root_job_id'] for row in rows}
    write, one, many = manager._write, ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault('_activation_reads', set()).update(
            row['job_id'] for row in rows if row is not None and 'job_id' in row)

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    @asynccontextmanager
    async def scoped():
        async with write() as tx:
            yield tx
        jobs = tx.touched_jobs | tx.__dict__.get('_activation_reads', set())
        footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(manager, '_write', scoped)
        patch.setattr(ControlTransaction, 'fetchone', fetchone)
        patch.setattr(ControlTransaction, 'fetchall', fetchall)
        yield footprints


async def test_activation_of_two_old_roots_has_only_zero_tree_metadata_writers(boot, monkeypatch):
    c = boot
    identity, leases = await old_roots(c)
    events = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    jobs = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id')
    async with observe(c.manager, monkeypatch) as footprints:
        receipt = await registry.activate(c.manager, identity, expected_version=1, system_key='startup')
    assert receipt['executor']['state'] == 'accepting' and not receipt['physical_stop_confirmed']
    assert not receipt['execution_authority'] and not receipt['replayed']
    assert {lease.job_id for lease in leases} == {job['job_id'] for job in jobs}
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == events
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id') == jobs
    assert footprints and all(not scope for scope in footprints), footprints


async def test_suspended_original_activation_proof_allows_another_root_to_stop(boot, monkeypatch):
    c = boot
    identity, _ = await old_roots(c)
    other = JobManager(await c.connect(), now_ms=c.manager.now)
    third = await other.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='third-root')
    original = execution_membership.read
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(manager, reader, *args, **kwargs):
        assert manager._control_intents.get() is None
        result = await original(manager, reader, *args, **kwargs)
        if manager is c.manager:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(execution_membership, 'read', held)
    action = asyncio.create_task(registry.activate(c.manager, identity, expected_version=1, system_key='startup'))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(other.request_cancel(c.actor, third.job.job_id,
            system_key='stop-third', reason='test'), 3)
    finally:
        release.set()
        await asyncio.wait_for(action, 10)
    assert (await other.get_job(c.actor, third.job.job_id)).cancel_requested_at is not None
    assert (await registry.get(c.manager, identity))['state'] == 'accepting'


async def test_begin_closes_old_admission_before_the_readonly_proof(boot, monkeypatch):
    c = boot
    identity, _ = await old_roots(c)
    inspect, observed = registry._inspect_prior_executions, []

    async def after_barrier(manager, actual):
        assert manager._control_intents.get() is None
        assert (await registry.get(manager, c.identity))['state'] == 'draining'
        queued = await manager.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='old-queued')
        with pytest.raises(JobError, match='executor_not_accepting'):
            await manager.claim_job(c.actor, queued.job.job_id, executor_id=c.identity.executor_id,
                boot_id=c.identity.boot_id, expected_version=1)
        with pytest.raises(JobError, match='executor_registration_required'):
            await manager.claim_job(c.actor, queued.job.job_id, executor_id=c.identity.executor_id,
                boot_id='unregistered-old-boot', expected_version=1)
        assert (await manager.get_job(c.actor, queued.job.job_id)).generation == 0
        observed.append(True)
        return await inspect(manager, actual)

    monkeypatch.setattr(registry, '_inspect_prior_executions', after_barrier)
    await registry.activate(c.manager, identity, expected_version=1, system_key='startup')
    assert observed == [True]


@pytest.mark.parametrize('stage', ['begin-rollback', 'begin-ack', 'publish-rollback', 'publish-ack'])
async def test_native_activation_failure_retains_only_committed_admission_and_receipt(boot, monkeypatch, stage):
    c = boot
    identity, _ = await old_roots(c)
    save, begin, publish = registry._save, registry._begin_activation, registry._publish_activation
    hits = []

    async def saved(manager, tx, actual, *args, **kwargs):
        value = await save(manager, tx, actual, *args, **kwargs)
        if ((stage == 'begin-rollback' and actual == c.identity)
                or (stage == 'publish-rollback' and actual == identity)):
            hits.append(stage)
            raise OSError('activation native fault')
        return value

    async def begun(*args, **kwargs):
        value = await begin(*args, **kwargs)
        if stage == 'begin-ack':
            hits.append(stage)
            raise OSError('activation native fault')
        return value

    async def published(*args, **kwargs):
        value = await publish(*args, **kwargs)
        if stage == 'publish-ack':
            hits.append(stage)
            raise OSError('activation native fault')
        return value

    jobs = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id')
    events = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    with monkeypatch.context() as patch:
        patch.setattr(registry, '_save', saved)
        patch.setattr(registry, '_begin_activation', begun)
        patch.setattr(registry, '_publish_activation', published)
        with pytest.raises(OSError, match='activation native fault'):
            await registry.activate(c.manager, identity, expected_version=1, system_key='startup')
    assert hits == [stage]
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    assert (await registry.get(cold, c.identity))['state'] == ('accepting' if stage == 'begin-rollback' else 'draining')
    assert (await registry.get(cold, identity))['state'] == ('accepting' if stage == 'publish-ack' else 'starting')
    before = await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id')
    receipt = await registry.activate(cold, identity, expected_version=1, system_key='startup')
    assert receipt['replayed'] == (stage == 'publish-ack')
    assert receipt['executor']['state'] == 'accepting'
    after = await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id')
    assert all(row in after for row in before)
    final = [row for row in after if row['executor_key'] == receipt['executor']['executor_key'] and row['action'] == 'activate']
    assert len(final) == 1 and final[0]['event_id'] == receipt['event_id']
    assert await cold.db.fetchall(f'SELECT * FROM {cold.jobs} ORDER BY job_id') == jobs
    assert await cold.db.fetchall(f'SELECT * FROM {cold.events} ORDER BY event_id') == events


@pytest.mark.parametrize('stage', ['old-closure-uncommitted', 'barrier', 'prepared', 'activation-uncommitted', 'activated'])
async def test_real_process_activation_death_reuses_committed_barriers_and_exact_final_receipt(boot, tmp_path, stage):
    c = boot
    identity, _ = await old_roots(c)
    jobs = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id')
    events = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    if c.manager.db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await c.manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if c.manager.db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    marker = tmp_path/'activation-death.json'
    config = {'identity': asdict(identity), 'now': c.now[0]}
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._executor_activation_worker',
        dsn, c.manager.prefix, json.dumps(config), stage, str(marker),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 93, stdout.decode()
        assert json.loads(marker.read_text()) == {'stage': stage, 'pid': process.pid}
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        assert (await registry.get(cold, c.identity))['state'] == ('accepting' if stage == 'old-closure-uncommitted' else 'draining')
        assert (await registry.get(cold, identity))['state'] == ('accepting' if stage == 'activated' else 'starting')
        originals = await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id')
        receipt = await registry.activate(cold, identity, expected_version=1, system_key='startup')
        assert receipt['replayed'] == (stage == 'activated')
        assert receipt['executor']['state'] == 'accepting' and not receipt['physical_stop_confirmed']
        after = await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id')
        assert all(row in after for row in originals)
        assert len(after) == 5
        replay = await registry.activate(cold, identity, expected_version=1, system_key='startup')
        assert replay['replayed'] and replay['event_id'] == receipt['event_id']
        assert replay['executor'] == receipt['executor']
        assert await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id') == after
        assert await cold.db.fetchall(f'SELECT * FROM {cold.jobs} ORDER BY job_id') == jobs
        assert await cold.db.fetchall(f'SELECT * FROM {cold.events} ORDER BY event_id') == events
    finally:
        await stop_worker(process)


async def test_boot_metadata_change_requires_a_fresh_original_proof_before_activation(boot, monkeypatch):
    c = boot
    identity, _ = await old_roots(c)
    publish, inspect = registry._publish_activation, registry._inspect_prior_executions
    proofs, publications = [], []
    late = replace(identity, boot_id='late-expired-boot')

    async def inspected(*args, **kwargs):
        proofs.append(True)
        return await inspect(*args, **kwargs)

    async def changed(*args, **kwargs):
        if not publications:
            await registry.register(c.manager, late, contract_hash='a'*64, startup_lease_ms=1)
            c.now[0] += 2
        publications.append(True)
        return await publish(*args, **kwargs)

    monkeypatch.setattr(registry, '_inspect_prior_executions', inspected)
    monkeypatch.setattr(registry, '_publish_activation', changed)
    receipt = await registry.activate(c.manager, identity, expected_version=1, system_key='startup')
    assert proofs == publications == [True, True]
    assert receipt['executor']['state'] == 'accepting'
    assert (await registry.get(c.manager, late))['state'] == 'draining'


async def test_real_send_process_cannot_dispatch_old_boot_while_new_activation_proof_is_prepared(boot, tmp_path):
    c = boot
    identity, _ = await old_roots(c)
    jobs = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id')
    if c.manager.db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await c.manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if c.manager.db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    marker = tmp_path/'activation-ready.json'
    config = {'identity': asdict(identity), 'now': c.now[0]}
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._executor_activation_worker',
        dsn, c.manager.prefix, json.dumps(config), 'hold-prepared', str(marker),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    send_process = None
    try:
        await wait_ready(process, marker)
        assert json.loads(marker.read_text()) == {'stage': 'hold-prepared', 'pid': process.pid}
        assert (await registry.get(c.manager, c.identity))['state'] == 'draining'
        assert (await registry.get(c.manager, identity))['state'] == 'starting'
        send_process, _, release, output = await spawn_worker(c, tmp_path, 'new-send')
        stdout, _ = await asyncio.wait_for(send_process.communicate(), 30)
        assert send_process.returncode == 0, stdout.decode()
        assert json.loads(output.read_text()) == {'error': 'executor_not_accepting', 'calls': 0}
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id') == jobs
        marker.with_suffix('.release').touch()
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 0, stdout.decode()
        receipt = json.loads(marker.with_suffix('.done').read_text())
        assert receipt['executor']['state'] == 'accepting' and not receipt['replayed']
        release.touch()
    finally:
        marker.with_suffix('.release').touch()
        if send_process is not None:
            await stop_worker(send_process)
        await stop_worker(process)


async def test_drain_during_activation_proof_cannot_be_revived(boot, monkeypatch):
    c = boot
    identity, _ = await old_roots(c)
    publish = registry._publish_activation
    stops = []

    async def drained(*args, **kwargs):
        stops.append(await registry.drain(c.manager, identity, system_key='drain-new-boot', reason='shutdown'))
        return await publish(*args, **kwargs)

    monkeypatch.setattr(registry, '_publish_activation', drained)
    with pytest.raises(JobError, match='executor_activation_conflict'):
        await registry.activate(c.manager, identity, expected_version=1, system_key='startup')
    assert len(stops) == 1
    assert (await registry.get(c.manager, identity))['state'] == 'draining'
    operation = registry._operation_key(c.manager, identity, 'activate', 'startup')
    assert not await c.manager.db.fetchone(f'SELECT event_id FROM {c.manager.prefix}task_executor_events WHERE operation_key=?', (operation,))


async def test_duplicate_activation_has_one_original_receipt_across_connections(boot):
    c = boot
    identity, _ = await old_roots(c)
    other = JobManager(await c.connect(), now_ms=c.manager.now)
    results = await asyncio.gather(*(registry.activate(manager, identity, expected_version=1, system_key='startup')
                                    for manager in (c.manager, other)))
    assert sorted(result['replayed'] for result in results) == [False, True]
    assert results[0]['event_id'] == results[1]['event_id']
    assert results[0]['executor'] == results[1]['executor']
