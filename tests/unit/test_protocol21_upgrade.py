"""Offline 21 -> final rehearsals on the deployed DDL in three private databases."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from urllib.parse import urlparse

import pytest

from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION, TASK_SCHEMA_VERSION
from power_loop.workflow.journal import new_journal, run_key
from scripts.upgrade_protocol21 import upgrade

from ._protocol21_fixture import retain_protocol21_format
from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def cli(dsn, prefix, *args):
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'scripts.upgrade_protocol21',
        '--dsn', dsn, '--prefix', prefix, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    output, errors = await asyncio.wait_for(process.communicate(), 30)
    assert process.returncode == 0, errors.decode()+output.decode()
    return json.loads(output)


async def test_deployed_ddl_upgrade_keeps_original_ids_rekeys_and_rebuilds_once(registry_store, tmp_path):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path/'registry.db') if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    if backend != 'sqlite':
        assert urlparse(dsn).path.endswith('_test')
    manager = JobManager(store)
    sid = await store.create_session()
    actor = JobActor('upgrade', 'host', sid)
    root = await manager.register_job(actor, kind='send', session_id=sid, system_key='root')
    lease = await manager.claim_job(actor, root.job.job_id, executor_id='e', boot_id='b', expected_version=1)
    seq = await store.append_message(sid, role='assistant', tool_calls=[{
        'id':'call', 'type':'function', 'function':{'name':'delegate', 'arguments':'{}'}}])
    ref = await store.message_ref(sid, seq)
    child = await manager.register_job(actor, kind='subagent', parent=lease, independent=True,
        call_ref=ref, tool_call_id='call')
    operation = await store._db.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',
        (child.operation.operation_id,))
    run_id = 'operator supplied workflow id'
    journal = new_journal(run_id, 'fixture')
    await store.set_runtime_state(sid, 'workflow:run:'+run_id, journal)
    await store.record_usage(sid, send_index=1, round_index=0, prompt_tokens=7, completion_tokens=5, total_tokens=12)
    originals = await store._db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY id')
    await retain_protocol21_format(store)
    # The deployed lookup columns permit this former spelling. Only lookup
    # keys change; event IDs, raw messages, references and payloads stay intact.
    await store._db.execute(f'UPDATE {manager.events} SET operation_key=? WHERE event_id=?',
        ('legacy-call-key', operation['event_id']))
    before = await cli(dsn, store.table_prefix)
    assert before['source'] == [10, 21] and not before['applied']
    with pytest.raises(StoreSchemaError, match='maintenance_required'):
        await upgrade(dsn, store.table_prefix, apply=True)
    with pytest.raises(StoreSchemaError):
        await connect()
    applied = await cli(dsn, store.table_prefix, '--apply', '--writers-stopped')
    assert applied['applied'] and applied['after']['version'] == TASK_SCHEMA_VERSION
    cold = await connect()
    assert cold.store_id == store.store_id and await cold.message_ref(sid, seq) == ref
    assert await cold._db.fetchall(f'SELECT * FROM {cold.t.messages} ORDER BY id') == originals
    assert (await cold.list_usage_rounds(sid))[0]['total_tokens'] == 12
    assert await cold.get_runtime_state(sid, run_key(run_id)) == journal
    assert await cold.get_runtime_state(sid, 'workflow:run:'+run_id) is None
    restored = await cold._db.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (operation['event_id'],))
    assert restored == operation
    rows = await cold._db.fetchall(f'SELECT * FROM {store.table_prefix}session_obligations ORDER BY source_id')
    again = await cli(dsn, store.table_prefix, '--apply', '--writers-stopped')
    assert again['source'] == [TASK_SCHEMA_VERSION, TASK_PROTOCOL_VERSION]
    assert await cold._db.fetchall(f'SELECT * FROM {cold.t.messages} ORDER BY id') == originals
    assert await cold._db.fetchall(f'SELECT * FROM {store.table_prefix}session_obligations ORDER BY source_id') == rows


async def test_unreleased_intermediate_protocol_is_refused_without_changes(registry_store):
    store, _ = registry_store
    await store._db.execute(f'UPDATE {store.table_prefix}task_store SET protocol_version=22 WHERE id=1')
    from power_loop.runtime.store.task_schema import migrate_task_schema
    try:
        with pytest.raises(StoreSchemaError, match='unsupported_migration_source'):
            await migrate_task_schema(store._db, store.table_prefix, maintenance=True)
        assert (await store._db.fetchone(f'SELECT protocol_version FROM {store.table_prefix}task_store'))['protocol_version'] == 22
    finally:
        await store._db.execute(f'UPDATE {store.table_prefix}task_store SET protocol_version=? WHERE id=1',
            (TASK_PROTOCOL_VERSION,))
