"""First mandatory policy and unbudgeted entry share an actual DB fence."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime import budget_admissions
from power_loop.runtime.jobs import JobError
from power_loop.runtime.store.factory import _make_database
from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION, migrate_task_schema

from ._budget_policy_admission_worker import make_loop, wait_file
from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case

__all__ = ['budget_case', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def wait_for_admission_lock(manager, observer, action):
    engine = manager.db.dialect.name
    async with asyncio.timeout(5):
        while not action.done():
            if engine == 'sqlite':
                waiting = manager.db._lock.locked() and not manager.db._conn.in_transaction
            elif engine == 'postgres':
                waiting = await observer.fetchone('SELECT pid FROM pg_stat_activity '
                    "WHERE datname=current_database() AND wait_event_type='Lock' AND query LIKE ?",
                    ('%'+manager.prefix+'%',))
            else:
                waiting = await observer.fetchone('SELECT w.REQUESTING_ENGINE_TRANSACTION_ID '
                    'FROM performance_schema.data_lock_waits w JOIN performance_schema.data_locks l '
                    'ON l.ENGINE_LOCK_ID=w.REQUESTING_ENGINE_LOCK_ID '
                    'WHERE l.OBJECT_SCHEMA=DATABASE() AND l.OBJECT_NAME LIKE ?', (manager.prefix+'%',))
            if waiting:
                break
            await asyncio.sleep(.01)
    assert not action.done(), 'entry passed an uncommitted conflicting policy admission'


@pytest.mark.parametrize('worker_paid', [False, True])
@pytest.mark.parametrize('unbudget_registered', [False, True])
async def test_first_policy_cannot_cross_an_uncommitted_unbudgeted_send(
        registry_store, monkeypatch, tmp_path, worker_paid, unbudget_registered):
    store, _ = registry_store
    loop, runtime, model, host = make_loop(store, executor='parent', budgeted=not worker_paid,
        registered=not worker_paid or unbudget_registered)
    sid, worker_sid = await loop.new_session(), await loop.new_session()
    await runtime.ensure_started()
    engine = runtime.manager.db.dialect.name
    dsn = next(row['file'] for row in await runtime.manager.db.fetchall('PRAGMA database_list')
        if row['name'] == 'main') if engine == 'sqlite' else os.environ[
            'POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    observer = await _make_database(dsn)
    process = await asyncio.create_subprocess_exec(sys.executable, '-m',
        'tests.unit._budget_policy_admission_worker', dsn, store.table_prefix, worker_sid,
        str(int(worker_paid)), str(tmp_path), 'send' if worker_paid or unbudget_registered else 'send-unregistered',
        cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    action = None

    @asynccontextmanager
    async def native():
        async with runtime.manager.db.transaction() as tx:
            yield tx

    async def send():
        try:
            return await loop.send('parent once', sid)
        except JobError as exc:
            return exc.code

    try:
        await wait_ready(process, tmp_path/'entry.ready')
        assert json.loads((tmp_path/'entry.ready').read_text()) == {
            'pid': process.pid, 'budgeted': worker_paid}
        with monkeypatch.context() as patch:
            patch.setattr(runtime.manager, '_control_transaction', native)
            action = asyncio.create_task(send())
            await wait_for_admission_lock(runtime.manager, observer, action)
            assert model.calls == host.reserve_calls == host.apply_calls == 0
            (tmp_path/'entry.release').touch()
            value = await asyncio.wait_for(action, 10)
        assert value == ('budget_provider_required' if worker_paid else 'budget_legacy_execution_unsettled')
        assert model.calls == host.reserve_calls == host.apply_calls == 0
        await wait_file(tmp_path/'model.ready')
        assert json.loads((tmp_path/'model.ready').read_text()) == {'pid': process.pid, 'calls': 1}
        manager = runtime.manager
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.jobs} WHERE kind='send'"))['n'] == 1
        assert (await manager.db.fetchone(f'SELECT COUNT(*) AS n FROM {manager.prefix}task_budget_policies'))['n'] == int(worker_paid)
        (tmp_path/'model.release').touch()
        stdout, _ = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, stdout.decode()
        assert json.loads((tmp_path/'worker.done').read_text()) == {'status': 'completed', 'text': 'done',
            'calls': 1, 'reserve': int(worker_paid), 'apply': int(worker_paid)}
        assert [row.role for row in await store.load_all_messages(worker_sid)] == ['user', 'assistant']
    finally:
        (tmp_path/'entry.release').touch()
        (tmp_path/'model.release').touch()
        if action is not None and not action.done():
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
        await stop_worker(process)
        if not worker_paid or unbudget_registered:
            await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
        await observer.close()


@pytest.mark.parametrize('mode', ['crash-before-commit', 'crash-after-commit'])
async def test_policy_original_and_admission_survive_actual_entry_process_death(
        registry_store, monkeypatch, tmp_path, mode):
    store, _ = registry_store
    loop, runtime, model, host = make_loop(store, executor='parent', budgeted=False)
    sid, worker_sid = await loop.new_session(), await loop.new_session()
    await runtime.ensure_started()
    manager, engine = runtime.manager, runtime.manager.db.dialect.name
    dsn = next(row['file'] for row in await manager.db.fetchall('PRAGMA database_list')
        if row['name'] == 'main') if engine == 'sqlite' else os.environ[
            'POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    observer = await _make_database(dsn)
    process = await asyncio.create_subprocess_exec(sys.executable, '-m',
        'tests.unit._budget_policy_admission_worker', dsn, store.table_prefix, worker_sid,
        '1', str(tmp_path), mode, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    action = None

    @asynccontextmanager
    async def native():
        async with manager.db.transaction() as tx:
            yield tx

    async def send():
        try:
            return (await loop.send('parent once', sid)).status
        except JobError as exc:
            return exc.code

    try:
        await wait_ready(process, tmp_path/'entry.ready')
        with monkeypatch.context() as patch:
            patch.setattr(manager, '_control_transaction', native)
            action = asyncio.create_task(send())
            await wait_for_admission_lock(manager, observer, action)
            assert model.calls == 0 and not (tmp_path/'model.ready').exists()
            if mode == 'crash-after-commit':
                (tmp_path/'entry.release').touch()
                await wait_ready(process, tmp_path/'entry.committed')
                assert json.loads((tmp_path/'entry.committed').read_text()) == {'pid': process.pid}
            assert not (tmp_path/'model.ready').exists()
            process.kill()
            stdout, _ = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == -9, stdout.decode()
            value = await asyncio.wait_for(action, 10)
        committed = mode == 'crash-after-commit'
        assert value == ('budget_provider_required' if committed else 'completed')
        assert model.calls == int(not committed)
        assert host.reserve_calls == host.apply_calls == 0
        assert not (tmp_path/'model.ready').exists() and not (tmp_path/'worker.done').exists()
        policies = await manager.db.fetchall(f'SELECT * FROM {manager.prefix}task_budget_policies')
        originals = await manager.db.fetchall(f'SELECT * FROM {manager.prefix}task_budget_policies')
        assert len(policies) == len(originals) == int(committed)
        admission = await manager.db.fetchone(f'SELECT policy_hash FROM {manager.prefix}task_principal_admissions')
        assert admission['policy_hash'] == (originals[0]['request_hash'] if committed else None)
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.jobs} WHERE kind='send'"))['n'] == 1
    finally:
        for name in ('entry.release', 'model.release', 'commit.release'):
            (tmp_path/name).touch()
        if action is not None and not action.done():
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
        await stop_worker(process)
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
        await observer.close()


@pytest.mark.parametrize('registered', [False, True])
async def test_a_warm_unbudgeted_runtime_observes_another_boots_mandatory_policy(registry_store, registered):
    store, connect = registry_store
    free, free_runtime, free_model, free_host = make_loop(store, executor='free', budgeted=False, registered=registered)
    paid_store = await connect()
    paid, paid_runtime, paid_model, paid_host = make_loop(paid_store, executor='paid', budgeted=True)
    sid, paid_sid = await free.new_session(), await paid.new_session()
    try:
        assert (await free.send('before policy', sid)).status == 'completed'
        assert free_model.calls == 1
        assert (await paid.send('first policy', paid_sid)).status == 'completed'
        with pytest.raises(JobError, match='^budget_provider_required$'):
            await free.send('after policy', sid)
        assert free_model.calls == paid_model.calls == 1
        assert free_host.reserve_calls == free_host.apply_calls == 0
        assert paid_host.reserve_calls == paid_host.apply_calls == 1
        assert (await free_runtime.manager.db.fetchone(
            f"SELECT COUNT(*) AS n FROM {free_runtime.manager.jobs} WHERE kind='send'"))['n'] == 2
    finally:
        for runtime, loop in ((free_runtime, free), (paid_runtime, paid)):
            if runtime is paid_runtime or registered:
                await runtime.shutdown(timeout_s=1)
            await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('outcome', ['rollback', 'lost-ack'])
async def test_first_policy_write_failure_does_not_separate_the_admission_from_its_original(
        registry_store, monkeypatch, outcome):
    store, connect = registry_store
    loop, runtime, model, host = make_loop(store, executor='paid', budgeted=True)
    sid = await loop.new_session()
    await runtime.ensure_started()
    manager = runtime.manager
    activate, transaction = budget_admissions.activated, manager._control_transaction
    changed = []

    async def activated(*args):
        await activate(*args)
        changed.append(True)
        if outcome == 'rollback':
            raise OSError('first policy transaction rollback')

    @asynccontextmanager
    async def lose_ack():
        async with transaction() as tx:
            yield tx
        if changed:
            raise OSError('first policy transaction lost ACK')

    cold_loop, cold_runtime, cold_model, cold_host = make_loop(await connect(), executor='cold', budgeted=False)
    cold_sid = await cold_loop.new_session()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(budget_admissions, 'activated', activated)
            if outcome == 'lost-ack':
                patch.setattr(manager, '_control_transaction', lose_ack)
            with pytest.raises(OSError, match='^first policy transaction '):
                await loop.send('first policy', sid)
        assert changed == [True]
        assert model.calls == host.reserve_calls == host.apply_calls == 0
        committed = outcome == 'lost-ack'
        originals = await manager.db.fetchall(f'SELECT * FROM {manager.prefix}task_budget_policies')
        assert len(originals) == int(committed)
        assert (await manager.db.fetchone(f'SELECT COUNT(*) AS n FROM {manager.prefix}task_budget_policies'))['n'] == int(committed)
        assert (await manager.db.fetchone(f'SELECT policy_hash FROM {manager.prefix}task_principal_admissions'))[
            'policy_hash'] == (originals[0]['request_hash'] if committed else None)
        if committed:
            with pytest.raises(JobError, match='^budget_provider_required$'):
                await cold_loop.send('unbudgeted', cold_sid)
        else:
            assert (await cold_loop.send('unbudgeted', cold_sid)).status == 'completed'
        assert cold_model.calls == int(not committed)
        assert cold_host.reserve_calls == cold_host.apply_calls == 0
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.jobs} WHERE kind='send'"))['n'] == 1
    finally:
        for active_runtime, active_loop in ((runtime, loop), (cold_runtime, cold_loop)):
            await active_runtime.shutdown(timeout_s=1)
            await active_loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('damage', ['missing-row', 'null-hash', 'wrong-hash', 'missing-table'])
async def test_explicit_maintenance_rebuilds_principal_admission_from_policy_originals(budget_case, damage):
    c = budget_case
    await c.runtime.shutdown(timeout_s=1)
    table = c.manager.prefix+'task_principal_admissions'
    before = await c.manager.db.fetchall(f'SELECT * FROM {table}')
    originals = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    assert len(before) == 1 and before[0]['policy_hash'] is not None
    if damage == 'missing-row':
        await c.manager.db.execute(f'DELETE FROM {table}')
    elif damage in ('null-hash', 'wrong-hash'):
        await c.manager.db.execute(f'UPDATE {table} SET policy_hash=?', (None if damage == 'null-hash' else '0'*64,))
    else:
        await c.manager.db.execute(f'DROP TABLE {table}')
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    assert await c.manager.db.fetchall(f'SELECT * FROM {table}') == before
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == originals
    assert (await c.manager.db.fetchone(f'SELECT protocol_version FROM {c.manager.prefix}task_store WHERE id=1'))[
        'protocol_version'] == TASK_PROTOCOL_VERSION
