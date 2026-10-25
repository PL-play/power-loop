"""Recovery refuses oversized or incomplete evidence without dropping responsibility."""
import json

import pytest

from power_loop.runtime.jobs import JobError, _json
from power_loop.runtime.pending_calls import SNAPSHOT, retire_pending

from .test_job_registry_192 import registry_store
from .test_pending_call_disposition_192 import original_pending

__all__ = ['registry_store', 'original_pending']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('part', ['pending_bytes', 'model_bytes', 'disposition_bytes',
                                  'boolean_round', 'boolean_version', 'omitted_unreturned_call'])
async def test_bounds_and_complete_call_set_cannot_be_bypassed_by_known_first_result(original_pending, part):
    c, row, source, pending, loop, reviewer = original_pending
    if part == 'pending_bytes':
        await c.store.set_pending(c.sid, {**pending, 'padding': 'x' * 32769})
    elif part == 'model_bytes':
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?',
            (_json({'padding': 'x' * (4 * 1024 * 1024)}), 'model_closed_' + source.job_id))
    elif part == 'disposition_bytes':
        result = await retire_pending(loop, c.actor, source.job_id)
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?',
            (_json({'padding': 'x' * 262145}), result['disposition']['event_id']))
    elif part in ('boolean_round', 'boolean_version'):
        event = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', ('model_closed_' + source.job_id,))
        data = json.loads(event['payload_json'])
        if part == 'boolean_round':
            data['evidence'][SNAPSHOT]['pending']['round_index'] = False
            await c.store.set_pending(c.sid, {**pending, 'round_index': False})
        else:
            data['evidence'][SNAPSHOT]['version'] = True
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?', (_json(data), 'model_closed_' + source.job_id))
    else:
        message = await c.manager.db.fetchone(f'SELECT tool_calls_json FROM {c.store.t.messages} WHERE session_id=? AND seq=?', (c.sid, pending['assistant_seq']))
        calls = json.loads(message['tool_calls_json'])
        calls.append({'id': 'other-unreturned-call', 'type': 'function',
                      'function': {'name': 'post_once', 'arguments': '{"text":"another"}'}})
        await c.manager.db.execute(f'UPDATE {c.store.t.messages} SET tool_calls_json=? WHERE session_id=? AND seq=?', (_json(calls), c.sid, pending['assistant_seq']))
    before = (await c.store.get_state(c.sid)).pending
    with pytest.raises(JobError):
        await retire_pending(loop, c.actor, source.job_id)
    assert (await c.store.get_state(c.sid)).pending == before
    assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'pending_delivery'
    assert reviewer.calls == c.state.lookups == 0 and c.state.dispatched == c.model.calls == 1
