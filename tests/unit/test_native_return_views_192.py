"""Original callback data is inspectable, never an acknowledgement or a lease."""
import json

import pytest

from power_loop.runtime import task_events
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.jobs import JobActor, JobError, _hash, _json
from power_loop.runtime.native_returns import anchor_id
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

from ._owned_pending_case_192 import owned_pending_case
from .test_job_registry_192 import registry_store
from .test_jobs_tool_192 import invoke
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def returned(registry_store):
    text = 'Original untrusted data: 云"\\\n' * 120
    def configure(c):
        original = c.loop.tool_registry.get('native_data')
        async def native():
            c.state.native_calls += 1
            return text
        c.loop.tool_registry.register(original.definition, FunctionRuntimeTool(native,
            ToolRuntimeContract('example.return-view'), policy=lambda args: ToolDispatchPolicy(True)), overwrite=True)
    c = await owned_pending_case(registry_store, 'subagent', native_tail='native_data', native_first=True, configure=configure)
    try:
        [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='native_callback_returned'", (c.state.source,))
        c.return_event, c.return_text = event, text
        yield c
    finally:
        await cleanup(c, None, None)


async def read(c, **options):
    return await invoke(c.loop, c.state.root, action='get', job_id=c.state.source, view='returns', **options)


async def test_chunks_and_export_preserve_original_data_without_ack_or_execution(returned, tmp_path):
    c = returned
    before = [await c.manager.db.fetchall(f'SELECT * FROM {table} ORDER BY 1')
              for table in (c.manager.events, c.manager.jobs, c.store.t.messages)]
    listing = await read(c)
    assert listing['read_only'] and not listing['execution_authority']
    assert [e['event_id'] for e in listing['events']] == [c.return_event['event_id']]
    offset, parts = 0, []
    while offset is not None:
        response = await read(c, event_id=c.return_event['event_id'], char_offset=offset, max_chars=2048)
        assert len(json.dumps(response, ensure_ascii=False, separators=(',', ':'))) <= 2048
        parts.append(response['text'])
        offset = response['next_char_offset']
    assert len(parts) > 1
    value = json.loads(''.join(parts))
    assert value['return_text'] == c.return_text
    assert value['call_ref'] == json.loads(c.return_event['payload_json'])['call_ref']
    assert value['callback_returned'] and not value['resource_stop_proven'] and not value['billing_settled']
    assert not {'scope', 'boot_id', 'executor_id', 'plan_hash', 'entry_hash'} & value.keys()
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
        exported = await read(c, event_id=c.return_event['event_id'], output_file='original.jsonl')
    assert exported['status'] == 'exported' and exported['records'] == 1
    saved = json.loads((tmp_path/'original.jsonl').read_text())
    assert saved == {**value, 'source': 'task_events'}
    after = [await c.manager.db.fetchall(f'SELECT * FROM {table} ORDER BY 1')
             for table in (c.manager.events, c.manager.jobs, c.store.t.messages)]
    assert after == before
    assert c.state.native_calls == 1 and c.state.executed == 0


@pytest.mark.parametrize('fault', ['foreign_actor', 'wrong_view', 'missing_anchor', 'rehashed_return'])
async def test_return_view_rejects_wrong_scope_or_corrupted_original(returned, fault):
    c = returned
    actor = c.runtime.actor(c.state.root)
    options = dict(view='returns', event_id=c.return_event['event_id'])
    if fault == 'foreign_actor':
        actor = JobActor('foreign', actor.principal_id, actor.session_id)
    elif fault == 'wrong_view':
        options['view'] = 'results'
    elif fault == 'missing_anchor':
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (anchor_id(c.return_event['event_id']),))
    else:
        value = json.loads(c.return_event['payload_json'])
        value['return_text'] = 'FORGED'
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?',
                                   (_json(value), _hash(value), c.return_event['event_id']))
    if fault == 'wrong_view':
        assert (await task_events.read_events(c.manager, actor, c.state.source, **options))['status'] == 'unavailable'
    else:
        with pytest.raises(JobError):
            await task_events.read_events(c.manager, actor, c.state.source, **options)
    source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
    assert source.result_status == 'none' and source.supervision_status == 'open'
    assert c.state.native_calls == 1 and c.state.executed == 0
