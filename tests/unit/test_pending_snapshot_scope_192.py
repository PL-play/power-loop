"""Independent proof: JSON booleans cannot alias an original generation."""
import json

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.pending_calls import SNAPSHOT, _verify
from tests.unit._tool_plan_faults_192 import fail_plan_once
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['legacy', 'sealed', 'aborted'])
async def test_pending_snapshot_generation_boolean_is_never_an_original_scope(registry_store, kind):
    async def configure(loop, state):
        manager = loop.task_runtime.manager
        if kind == 'aborted':
            fail_plan_once(manager, 'seal')
        elif kind == 'legacy':
            original = manager.register_job
            async def old_source(actor, **kwargs):
                if kwargs.get('kind') == 'send':
                    kwargs['request'] = dict(kwargs['request'])
                    kwargs['request'].pop('tool_call_plan_protocol', None)
                return await original(actor, **kwargs)
            manager.register_job = old_source
    c = await scenario(registry_store, configure=configure, raw_receipt_failure=kind != 'aborted',
                       expected_error='test_plan_seal' if kind == 'aborted' else None)
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    closed = json.loads((await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', ('model_closed_'+source['job_id'],)))['payload_json'])
    snapshot = closed['evidence'][SNAPSHOT]
    assert snapshot['version'] == {'legacy': 1, 'sealed': 2, 'aborted': 3}[kind]
    async with c.manager._write() as tx:
        await _verify(c.manager, tx, source, snapshot)
        snapshot['scope']['generation'] = True
        with pytest.raises(JobError, match='snapshot_mismatch'):
            await _verify(c.manager, tx, source, snapshot)
