"""Native tool intentions cannot outlive their atomic durable pending marker."""
from contextlib import asynccontextmanager

import pytest

from power_loop.agent.sink import SQLiteSink

from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
CALLS = [{'id': 'first', 'type': 'function', 'function': {'name': 'example', 'arguments': '{ "n": 1 }'}}]


@pytest.mark.parametrize('stage', ['message', 'pending', 'commit', 'commit_ack'])
async def test_raw_assistant_and_pending_are_one_transaction_even_with_lost_ack(registry_store, stage):
    store, connect = registry_store
    sid = await store.create_session()
    original, fired = store._db.transaction, False

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags
        def __getattr__(self, name):
            return getattr(self.target, name)
        async def execute(self, statement, args=()):
            nonlocal fired
            if statement.startswith(f'INSERT INTO {store.t.messages} '):
                self.flags['target'] = True
                if stage == 'message' and not fired:
                    fired = True
                    raise OSError('test_atomic_' + stage)
            if statement.startswith(f'UPDATE {store.t.session_state} SET pending_json=') and stage == 'pending' and not fired:
                fired = True
                raise OSError('test_atomic_' + stage)
            return await self.target.execute(statement, args)

    @asynccontextmanager
    async def intercepted():
        nonlocal fired
        flags = {'target': False}
        async with original() as tx:
            yield Transaction(tx, flags)
            if flags['target'] and stage == 'commit' and not fired:
                fired = True
                raise OSError('test_atomic_' + stage)
        if flags['target'] and stage == 'commit_ack' and not fired:
            fired = True
            raise OSError('test_atomic_' + stage)
    store._db.transaction = intercepted
    with pytest.raises(OSError, match='test_atomic_'):
        await store.append_message(sid, role='assistant', tool_calls=CALLS, round_index=4, send_index=3, pending_tool_calls=True)
    assert fired
    store._db.transaction = original
    reader = await connect(schema='compatible')
    rows = await reader.load_all_messages(sid)
    state = await reader.get_state(sid)
    if stage != 'commit_ack':
        assert not rows and state.pending is None and state.next_seq == 1
        return
    assert len(rows) == 1 and rows[0].tool_calls == CALLS
    assert state.pending == {'assistant_seq': rows[0].seq, 'round_index': 4, 'tool_call_ids': ['first'], 'tool_calls': CALLS}
    reference = await reader.message_ref(sid, state.pending['assistant_seq'])
    assert reference.role == 'assistant' and reference.id is not None
    before = await reader.get_state(sid)
    with pytest.raises(ValueError, match='pending_tool_calls_exist'):
        await reader.append_message(sid, role='assistant', tool_calls=CALLS, pending_tool_calls=True)
    assert await reader.get_state(sid) == before and len(await reader.load_all_messages(sid)) == 1


async def test_native_pending_survives_failure_before_the_later_pending_callback(registry_store, monkeypatch):
    async def fail_before_memory(*args, **kwargs):
        raise OSError('test_before_later_pending_callback')
    with monkeypatch.context() as patch:
        patch.setattr(SQLiteSink, 'on_assistant_tool_calls', fail_before_memory)
        c = await scenario(registry_store, call_count=2, expected_error='test_before_later_pending_callback')
    pending = (await c.store.get_state(c.sid)).pending
    assert pending['tool_call_ids'] == ['original-call', 'original-call-2']
    original = await c.store.get_message(c.sid, pending['assistant_seq'])
    assert original.tool_calls == pending['tool_calls'] and original.role == 'assistant'
    assert c.state.prepared == c.state.dispatched == 0 and c.model.calls == 1
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")


async def test_plain_raw_append_keeps_legacy_behavior_and_cannot_remove_existing_pending(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    await store.append_message(sid, role='assistant', tool_calls=CALLS)
    assert (await store.get_state(sid)).pending is None
    seq = await store.append_message(sid, role='assistant', tool_calls=CALLS, pending_tool_calls=True)
    pending = (await store.get_state(sid)).pending
    assert pending['assistant_seq'] == seq
    await store.append_message(sid, role='user', content='audit input')
    assert (await store.get_state(sid)).pending == pending


@pytest.mark.parametrize('value', [1, 'true', None])
async def test_atomic_pending_flag_is_strict_and_never_mutates_on_invalid_input(registry_store, value):
    store, _ = registry_store
    sid = await store.create_session()
    with pytest.raises(ValueError, match='invalid_pending_tool_message'):
        await store.append_message(sid, role='assistant', tool_calls=CALLS, pending_tool_calls=value)
    assert not await store.load_all_messages(sid) and (await store.get_state(sid)).pending is None


async def test_native_assistant_append_lost_ack_keeps_raw_pending_without_memory_seq(registry_store):
    async def configure(loop, state):
        store = loop.task_runtime.manager.store
        original = store.append_message
        async def lose_ack(*args, **kwargs):
            seq = await original(*args, **kwargs)
            if kwargs.get('role') == 'assistant' and kwargs.get('tool_calls'):
                assert kwargs.get('pending_tool_calls') is True
                raise OSError('test_native_assistant_ack_lost')
            return seq
        store.append_message = lose_ack
    c = await scenario(registry_store, budgeted=True, call_count=2, configure=configure,
                       expected_error='test_native_assistant_ack_lost')
    pending = (await c.store.get_state(c.sid)).pending
    assert pending['tool_call_ids'] == ['original-call', 'original-call-2']
    row = await c.store.get_message(c.sid, pending['assistant_seq'])
    assert row.role == 'assistant' and row.tool_calls == pending['tool_calls']
    assert c.model.calls == c.host.reserve_calls == 1
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='background_tool'")
