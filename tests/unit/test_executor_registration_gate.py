"""First registration and unregistered claims must share an absence fence.

These component process races use actual public claim/registration ports in
native writers,so the retained global mutex cannot hide a missing group gate.
Final root-lock acceptance must also exercise full unmodified public send.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from power_loop.runtime import executor_admissions
from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.store.factory import _make_database

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_executor_registry_192 import boot, start
from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled, _counts

__all__ = ['boot', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('warm', [False, True])
@pytest.mark.parametrize('mode', ['hold-claim', 'hold-registration'])
async def test_first_registration_cannot_cross_an_uncommitted_unregistered_claim(boot, monkeypatch, tmp_path, mode, warm):
    c = boot
    if warm:
        prior = await c.manager.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='prior-boot')
        lease = await c.manager.claim_job(c.actor, prior.job.job_id, executor_id=c.identity.executor_id,
            boot_id='prior-boot', expected_version=1)
        await c.manager.close_model_execution(c.actor, lease, evidence={'exit': 'completed'}, resource_state='stopped')
    queued = await c.manager.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='unregistered-claim')
    engine = c.manager.db.dialect.name
    dsn = next(row['file'] for row in await c.manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main') if engine == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    observer = await _make_database(dsn)
    ready, release, output = (tmp_path/name for name in ('registration.ready', 'registration.release', 'registration.done'))
    config = {'actor': asdict(c.actor), 'identity': asdict(c.identity), 'job_id': queued.job.job_id, 'now': c.now[0]}
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._executor_registration_gate_worker',
        dsn, c.manager.prefix, json.dumps(config), mode, str(ready), str(release), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    action = None

    @asynccontextmanager
    async def native():
        async with c.manager.db.transaction() as tx:
            yield tx

    async def compete():
        try:
            if mode == 'hold-claim':
                return await registry.register(c.manager, c.identity, contract_hash='a'*64)
            return await c.manager.claim_job(c.actor, queued.job.job_id,
                executor_id=c.identity.executor_id, boot_id=c.identity.boot_id, expected_version=1)
        except JobError as exc:
            return exc.code

    try:
        await wait_ready(process, ready)
        assert json.loads(ready.read_text()) == {'mode': mode, 'pid': process.pid}
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, '_control_transaction', native)
            action = asyncio.create_task(compete())
            async with asyncio.timeout(5):
                while not action.done():
                    if engine == 'sqlite':
                        waiting = c.manager.db._lock.locked() and not c.manager.db._conn.in_transaction
                    elif engine == 'postgres':
                        waiting = await observer.fetchone('SELECT pid FROM pg_stat_activity WHERE datname=current_database() '
                            "AND wait_event_type='Lock' AND query LIKE ?", ('%'+c.manager.prefix+'%',))
                    else:
                        waiting = await observer.fetchone('SELECT w.REQUESTING_ENGINE_TRANSACTION_ID '
                            'FROM performance_schema.data_lock_waits w JOIN performance_schema.data_locks l '
                            'ON l.ENGINE_LOCK_ID=w.REQUESTING_ENGINE_LOCK_ID '
                            'WHERE l.OBJECT_SCHEMA=DATABASE() AND l.OBJECT_NAME LIKE ?', (c.manager.prefix+'%',))
                    if waiting:
                        break
                    await asyncio.sleep(.01)
            assert not action.done(), 'first registration crossed an uncommitted claim admission'
            release.touch()
            value = await asyncio.wait_for(action, 10)
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stdout.decode()
        if mode == 'hold-claim':
            assert value == 'executor_history_requires_reconciliation'
            assert json.loads(output.read_text()) == {'job_id': queued.job.job_id, 'generation': 1}
            assert not await c.manager.db.fetchone(f'SELECT * FROM {c.manager.prefix}task_executors WHERE executor_key=?',
                (registry._key(c.manager, c.identity),))
            saved = await c.manager.get_job(c.actor, queued.job.job_id)
            assert saved.generation == 1
            binding = await c.manager.db.fetchone(f'SELECT executor_id,boot_id,executor_key FROM {c.manager.jobs} WHERE job_id=?',
                (queued.job.job_id,))
            assert binding == {'executor_id': c.identity.executor_id, 'boot_id': c.identity.boot_id, 'executor_key': None}
        else:
            assert value == 'executor_not_accepting'
            assert json.loads(output.read_text())['executor']['state'] == 'starting'
            assert (await c.manager.get_job(c.actor, queued.job.job_id)).generation == 0
    finally:
        release.touch()
        await stop_worker(process)
        if action is not None:
            await asyncio.gather(action, return_exceptions=True)
        await observer.close()


@pytest.mark.parametrize('committed', [False, True])
async def test_registration_gate_and_original_share_rollback_and_unknown_ack(boot, monkeypatch, committed):
    c = boot
    transaction, hits = c.manager.db.transaction, []

    @asynccontextmanager
    async def fault():
        async with transaction() as tx:
            yield tx
            found = await tx.fetchone(f"SELECT event_id FROM {c.manager.prefix}task_executor_events WHERE action='register'")
            hit = bool(found) and not hits
            if hit:
                hits.append(True)
                if not committed:
                    raise OSError('registration rolled back')
        if hit:
            raise OSError('registration committed; ACK lost')

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, 'transaction', fault)
        with pytest.raises(OSError, match='rolled back|ACK lost'):
            await registry.register(c.manager, c.identity, contract_hash='a'*64)
    assert hits == [True]
    groups = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}task_executor_admissions')
    assert groups == ([{'group_key':executor_admissions.key(c.manager, c.identity), 'registered':1}] if committed else [])
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    reply = await registry.register(cold, c.identity, contract_hash='a'*64)
    assert reply['replayed'] == committed and reply['executor']['state'] == 'starting'
    assert len(await cold.db.fetchall(f"SELECT * FROM {cold.prefix}task_executor_events WHERE action='register'")) == 1


@pytest.mark.parametrize('committed', [False, True])
async def test_real_registration_process_death_keeps_gate_and_original_atomic(boot, tmp_path, committed):
    c = boot
    engine = c.manager.db.dialect.name
    dsn = next(row['file'] for row in await c.manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main') if engine == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    ready, release, output = (tmp_path/name for name in ('registration.death', 'unused.release', 'unused.output'))
    mode = 'registration-committed' if committed else 'registration-uncommitted'
    config = {'actor':asdict(c.actor), 'identity':asdict(c.identity), 'now':c.now[0]}
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._executor_registration_gate_worker',
        dsn, c.manager.prefix, json.dumps(config), mode, str(ready), str(release), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 96, stdout.decode()
        assert json.loads(ready.read_text()) == {'mode':mode, 'pid':process.pid}
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        groups = await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_admissions')
        assert groups == ([{'group_key':executor_admissions.key(cold, c.identity), 'registered':1}] if committed else [])
        originals = await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id')
        assert len(originals) == int(committed)
        reply = await registry.register(cold, c.identity, contract_hash='a'*64)
        assert reply['replayed'] == committed and reply['executor']['state'] == 'starting'
        after = await cold.db.fetchall(f'SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id')
        assert len(after) == 1 and all(row in after for row in originals)
    finally:
        await stop_worker(process)






async def test_a_later_registered_boot_revokes_unregistered_admission_without_a_cached_absence(boot):
    c = boot
    first = await c.manager.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='unregistered')
    lease = await c.manager.claim_job(c.actor, first.job.job_id,
        executor_id=c.identity.executor_id, boot_id=c.identity.boot_id, expected_version=1)
    fresh = replace(c.identity, boot_id='registered-later')
    await registry.register(c.manager, fresh, contract_hash='a'*64)
    with pytest.raises(JobError, match='executor_prior_execution_unsettled'):
        await registry.activate(c.manager, fresh, expected_version=1, system_key='startup')
    with pytest.raises(JobError, match='executor_registration_required'):
        await c.manager.renew_execution(c.actor, lease)
    await c.manager.close_model_execution(c.actor, lease, evidence={'exit':'cancelled'}, resource_state='stopped')
    accepted = await registry.activate(c.manager, fresh, expected_version=1, system_key='startup')
    assert accepted['executor']['state'] == 'accepting'
    other = await c.manager.register_job(c.actor, kind='send', session_id=c.actor.session_id, system_key='after-registration')
    with pytest.raises(JobError, match='executor_registration_required'):
        await c.manager.claim_job(c.actor, other.job.job_id,
            executor_id=c.identity.executor_id, boot_id=c.identity.boot_id, expected_version=1)
    assert (await c.manager.get_job(c.actor, other.job.job_id)).generation == 0


@pytest.mark.parametrize('stage', ['first-absence', 'warm-absence', 'registered'])
async def test_registered_boot_keeps_one_admission_query_and_absence_cost_is_explicit(boot, stage):
    c = boot
    if stage == 'registered':
        await start(c)
    elif stage == 'warm-absence':
        async with c.manager._write() as tx:
            assert await registry.require_accepting(c.manager, tx, c.identity) is None
    originals = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}task_executor_events ORDER BY event_id')
    raw = c.manager.db
    profiled = Profiled(raw)
    c.manager.db = c.store._db = profiled
    trust = c.manager._trust_own_writes.set(True)
    try:
        async with c.manager._write() as tx:
            row = await registry.require_accepting(c.manager, tx, c.identity)
            assert (row is not None) == (stage == 'registered')
    finally:
        c.manager._trust_own_writes.reset(trust)
        c.manager.db = c.store._db = raw
    queries = [r for r in profiled.records if any(site.startswith((
        'runtime.executor_registry:', 'runtime.executor_admissions:')) for site in r['chain'])]
    assert len(queries) == {'first-absence':4, 'warm-absence':2, 'registered':1}[stage]
    assert _counts(profiled.records, profiled.tx_sites)['control_tx'] == 1
    assert await raw.fetchall(f'SELECT * FROM {c.manager.prefix}task_executor_events ORDER BY event_id') == originals
