"""An accepted host context cancellation survives removal of derived state."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop.runtime import root_context
from power_loop.runtime.jobs import JobError, JobManager

from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store
from .test_root_cancel_trees import spawn_worker
from .test_root_context_192 import NAMESPACE, root_case
from .test_root_control_192 import PAYLOAD, send, stop

__all__ = ['registry_store', 'root_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('started', [False, True])
async def test_cold_admission_remains_cancelled_after_derived_state_is_removed(root_case, started):
    c = root_case
    if started:
        await send(c)
    receipt = await stop(c)
    assert len(receipt['roots']) == int(started)
    jobs = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id')
    calls = c.model.calls.copy()
    await c.manager.db.execute(f'DELETE FROM {c.store.t.session_runtime_state} WHERE session_id=?', (c.sid,))
    cold = JobManager(await c.connect())
    document = root_context._document(NAMESPACE, PAYLOAD)
    with pytest.raises(JobError, match='root_context_cancelled'):
        await cold.register_job(c.actor, kind='send', session_id=c.sid, system_key='send:999',
            request={'host_context': document, 'send_index': 999, 'model_attempt_protocol': 1})
    assert await cold.db.fetchall(f'SELECT * FROM {cold.jobs} ORDER BY job_id') == jobs
    assert await stop(c, manager=cold) == receipt
    assert c.model.calls == calls
    another = root_context._document(NAMESPACE, {'origin_run_id': 202})
    accepted = await cold.register_job(c.actor, kind='send', session_id=c.sid, system_key='send:1000',
        request={'host_context': another, 'send_index': 1000, 'model_attempt_protocol': 1})
    assert not accepted.replayed and accepted.job.cancel_requested_at is None


@pytest.mark.parametrize('started', [False, True])
async def test_real_new_process_cannot_reopen_cancelled_admission_after_cache_removal(root_case, tmp_path, started):
    c = root_case
    if started:
        await send(c)
    await stop(c)
    originals = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}task_context_events ORDER BY event_id')
    messages = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
    jobs = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id')
    await c.manager.db.execute(f'DELETE FROM {c.store.t.session_runtime_state} WHERE session_id=?', (c.sid,))
    process, _ready, release, output = await spawn_worker(c, tmp_path, 'new-send')
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 0, stdout.decode()
        assert json.loads(output.read_text()) == {'error': 'root_context_cancelled', 'calls': 0}
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}task_context_events ORDER BY event_id') == originals
        after = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
        assert after[:len(messages)] == messages
        [incoming] = after[len(messages):]
        assert incoming['role'] == 'user' and incoming['session_id'] == c.sid
        assert incoming['content'] == 'new send during context stop'
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.jobs} ORDER BY job_id') == jobs
    finally:
        release.touch()
        await stop_worker(process)


async def previous_schema(c, started):
    if started:
        await send(c)
    receipt = await stop(c)
    await c.loop.aclose(drain_timeout_s=.1)
    prefix, db = c.manager.prefix, c.manager.db
    messages = await db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
    events = await db.fetchall(f'SELECT * FROM {prefix}task_context_events ORDER BY event_id')
    assert len(events) == 4
    # Reconstruct the actual immediately previous storage: four former
    # originals in runtime_state,no context event table,schema10/protocol21.
    for event in events:
        await db.execute(f'INSERT INTO {c.store.t.session_runtime_state} '
            '(session_id,state_key,value_json,updated_at) VALUES (?,?,?,?)',
            (event['session_id'], event['state_key'], event['payload_json'], event['created_at']))
    await db.execute(f'DROP TABLE {prefix}task_context_events')
    await db.execute(f'UPDATE {prefix}schema_migrations SET version=10 WHERE id=1')
    await db.execute(f'UPDATE {prefix}task_store SET protocol_version=21 WHERE id=1')
    return receipt, prefix, db, events, messages
