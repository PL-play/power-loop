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
    await _atomic_pending_case(registry_store, stage)


@pytest.mark.parametrize('stage', ['message', 'pending', 'commit', 'commit_ack'])
async def test_cached_allocation_preserves_atomic_pending_and_deferred_round_on_failure(registry_store, stage):
    await _atomic_pending_case(registry_store, stage, warm=True)


async def _atomic_pending_case(registry_store, stage, *, warm=False):
    store, connect = registry_store
    sid = await store.create_session()
    if warm:
        await store.get_state(sid)
        store.defer_round_index(sid, 4)
    original, fired = store._db.transaction, False

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags
        def __getattr__(self, name):
            return getattr(self.target, name)
        def inject(self, statement):
            nonlocal fired
            if statement.startswith(f'INSERT INTO {store.t.messages} '):
                self.flags['target'] = True
                if stage == 'message' and not fired:
                    fired = True
                    raise OSError('test_atomic_' + stage)
            if (statement.startswith(f'UPDATE {store.t.session_state} SET ')
                    and 'pending_json=?' in statement.split(' WHERE ', 1)[0]
                    and stage == 'pending' and not fired):
                fired = True
                raise OSError('test_atomic_' + stage)

        async def execute(self, statement, args=()):
            self.inject(statement)
            return await self.target.execute(statement, args)

        async def fetchone(self, statement, args=()):
            self.inject(statement)
            return await self.target.fetchone(statement, args)

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
    reader = await connect(schema="verify")
    rows = await reader.load_all_messages(sid)
    state = await reader.get_state(sid)
    if stage != 'commit_ack':
        assert not rows and state.pending is None and state.next_seq == 1
        if warm:
            # Retrying the first append must still persist the deferred round,
            # including when the failed transaction reached its commit boundary.
            await store.append_message(sid, role='user', content='retry', round_index=4)
            retried = await reader.get_state(sid)
            assert retried.round_index == 4 and retried.next_seq == 2
        return
    assert len(rows) == 1 and rows[0].tool_calls == CALLS
    assert state.pending == {'assistant_seq': rows[0].seq, 'round_index': 4, 'tool_call_ids': ['first'], 'tool_calls': CALLS}
    if warm:
        assert state.round_index == 4
    reference = await reader.message_ref(sid, state.pending['assistant_seq'])
    assert reference.role == 'assistant' and reference.id is not None
    before = await reader.get_state(sid)
    with pytest.raises(ValueError, match='pending_tool_calls_exist'):
        await reader.append_message(sid, role='assistant', tool_calls=CALLS, pending_tool_calls=True)
    assert await reader.get_state(sid) == before and len(await reader.load_all_messages(sid)) == 1


@pytest.mark.parametrize('stage', ['commit', 'commit_ack'])
@pytest.mark.parametrize('keep_second', [False, True])
async def test_tool_result_and_remaining_interactions_commit_together(registry_store, stage, keep_second):
    store, connect = registry_store
    sid = await store.create_session()
    calls = CALLS + ([{**CALLS[0], 'id': 'second'}] if keep_second else [])
    sink = SQLiteSink(store, sid)
    await sink.on_message_appended({'role': 'assistant', 'tool_calls': calls}, round_index=4)
    await sink.on_assistant_tool_calls(assistant_seq=1, tool_calls=calls, round_index=4)
    pending = (await store.get_state(sid)).pending
    interactions = [{'tool_call_id': call['id'], 'interaction_id': call['id']} for call in calls]
    pending['pending_interactions'] = interactions
    await store.set_pending(sid, pending)
    original = store._db.transaction

    @asynccontextmanager
    async def intercepted():
        async with original() as tx:
            yield tx
            if stage == 'commit':
                raise OSError('test_tool_result_commit')
        raise OSError('test_tool_result_commit_ack')

    store._db.transaction = intercepted
    message = {'role': 'tool', 'tool_call_id': 'first', 'content': 'actual result'}
    try:
        with pytest.raises(OSError, match='test_tool_result_commit'):
            await sink.on_message_appended(message, round_index=4)
    finally:
        store._db.transaction = original
    reader = await connect(schema="verify")
    state = await reader.get_state(sid)
    if stage == 'commit':
        assert state.pending == pending and state.next_seq == 2
        assert len(await reader.load_all_messages(sid)) == 1
        await sink.on_message_appended(message, round_index=4)
        state = await reader.get_state(sid)
    rows = await reader.load_all_messages(sid)
    assert state.next_seq == 3 and len(rows) == 2 and rows[-1].content == 'actual result'
    assert state.pending == ({**pending, 'tool_call_ids': ['second'], 'tool_calls': calls[1:],
                              'pending_interactions': interactions[1:]} if keep_second else None)


async def test_failed_receipt_binding_preserves_pending_and_raw_result(registry_store, monkeypatch):
    from power_loop.runtime import task_runtime

    store, _ = registry_store
    sid = await store.create_session()
    sink = SQLiteSink(store, sid)
    await sink.on_message_appended({'role': 'assistant', 'tool_calls': CALLS}, round_index=4)
    await sink.on_assistant_tool_calls(assistant_seq=1, tool_calls=CALLS, round_index=4)
    pending = (await store.get_state(sid)).pending
    pending['pending_interactions'] = [{'tool_call_id': 'first', 'interaction_id': 'original'}]
    await store.set_pending(sid, pending)

    async def unavailable(*args, **kwargs):
        raise OSError('receipt_binding_unavailable')

    async def no_compensation(*args, **kwargs):
        pytest.fail('pending must already be durable; no compensating write is allowed')

    monkeypatch.setattr(task_runtime, 'tool_receipt_required', lambda *args: True)
    monkeypatch.setattr(task_runtime, 'bind_tool_receipt', unavailable)
    monkeypatch.setattr(store, 'set_pending', no_compensation)
    with pytest.raises(OSError, match='receipt_binding_unavailable'):
        await sink.on_message_appended({'role': 'tool', 'tool_call_id': 'first', 'content': 'actual result'}, round_index=4)
    assert (await store.get_state(sid)).pending == pending
    rows = await store.load_all_messages(sid)
    assert len(rows) == 2 and rows[-1].content == 'actual result'
    assert sink.persisted_assistant_seq == 1


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
