import json

import pytest

from power_loop.runtime.jobs import JobError, _hash, _json
from power_loop.runtime.managed_completion import restore_result
from power_loop.runtime.managed_effects import _id, reconcile
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.store.task_schema import migrate_task_schema, verify_task_schema
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_managed_effects_192 import scenario

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
__all__ = ['registry_store']


async def test_legacy_protocol_eight_exit_is_readable_but_cannot_invent_a_completion_checkpoint(registry_store):
    c = await scenario(registry_store, budgeted=True, pending=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    identity = _id(c.manager, 'resource_exit', row['job_id'], 1)
    event = await c.manager.db.fetchone(f'SELECT * FROM {c.manager.events} WHERE event_id=?', (identity,))
    value = json.loads(event['payload_json'])
    value['version'] = 1
    value.pop('completion')
    await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(value), _hash(value), identity))
    for kind in ['completion', 'result_recovery']:
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (_id(c.manager, kind, row['job_id'], 1),))
    original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    await c.manager.db.execute(f'UPDATE {c.manager.prefix}task_store SET protocol_version=8 WHERE id=1')
    with pytest.raises(StoreSchemaError):
        await verify_task_schema(c.manager.db, c.manager.prefix)
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    with pytest.raises(JobError, match='checkpoint_unavailable'):
        await restore_result(c.manager, c.actor, row['job_id'])
    c.state.resolved = True
    assert await reconcile(c.runtime, c.actor, row['job_id'], c.adapter)
    await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
    assert (await c.manager.get_job(c.actor, row['job_id'])).supervision_status == 'closed'
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq') == original
    assert c.state.dispatched == c.state.lookups == c.model.calls == c.host.reserve_calls == 1
    legacy = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (identity,))
    assert json.loads(legacy['payload_json']) == value
