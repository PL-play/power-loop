"""Response usage shares one original and survives uncertain append commits."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest

from power_loop.agent.sink import SQLiteSink

from .session_native_fixture import new_session
from .test_send_sql_profile import Profiled

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
USAGE = dict(prompt_tokens=7, completion_tokens=5, total_tokens=12, cached_tokens=3, model='fixture')


async def test_response_original_has_usage_without_an_additional_statement(registry_store):
    store, connect = registry_store
    sid = (await new_session(store)).session_id
    await store.append_message(sid, role='user', content='input', send_index=1)
    original = store._db
    store._db = counted = Profiled(original)
    try:
        seq = await store.append_message(sid, role='assistant', content='response', send_index=1,
            round_index=0, meta={'visible':'kept'}, round_usage=USAGE)
        assert counted.transactions == 1
        assert len(counted.records) == (3 if original.dialect.name == 'mysql' else 2)
        assert not any("usage:" in row["sql"] for row in counted.records)
        raw = await original.fetchone(f'SELECT meta_json FROM {store.t.messages} '
            'WHERE session_id=? AND seq=?', (sid, seq))
        assert json.loads(raw['meta_json']) == {'visible':'kept', '_round_usage':USAGE}
        assert (await store.load_all_messages(sid))[-1].meta == {'visible':'kept'}
        cold = await connect()
        [detail] = await cold.list_usage_rounds(sid)
        assert {key:detail[key] for key in USAGE} == USAGE
        assert (detail['session_id'], detail['send_index'], detail['round_index']) == (sid, 1, 0)
        exported = await store.export_session(sid)
        message = next(row for row in exported['tables']['messages'] if row['seq'] == seq)
        assert json.loads(message['meta_json'])['_round_usage'] == USAGE
    finally:
        store._db = original


async def test_lost_append_ack_and_round_end_observation_count_one_round(registry_store, monkeypatch):
    store, connect = registry_store
    sid = (await new_session(store)).session_id
    sink = SQLiteSink(store, sid)
    sink.send_index = 1
    transaction = store._db.transaction

    @asynccontextmanager
    async def unknown():
        async with transaction() as tx:
            yield tx
        raise OSError('append committed,ACK lost')

    with monkeypatch.context() as patch:
        patch.setattr(store._db, 'transaction', unknown)
        with pytest.raises(OSError, match='ACK lost'):
            await sink.on_message_appended({'role':'assistant','content':'original','send_index':1,
                '_round_usage':USAGE}, round_index=0)
    await sink.on_round_ended(0, usage=USAGE)
    await sink.on_round_ended(0, usage=USAGE)
    cold = await connect()
    assert len(await cold.load_all_messages(sid)) == 1
    [detail] = await cold.list_usage_rounds(sid)
    assert detail['total_tokens'] == 12 and detail['cached_tokens'] == 3
    assert (await cold._db.fetchone(f'SELECT COUNT(*) AS n FROM {cold.t.session_runtime_state} '
        "WHERE session_id=? AND state_key LIKE 'usage:%'", (sid,)))['n'] == 1


async def test_pruning_details_counts_distinct_rounds_and_preserves_message_bodies(registry_store):
    store, _ = registry_store
    sid = (await new_session(store)).session_id
    await store.record_usage(sid, send_index=1, round_index=0, **USAGE)
    await store.append_message(sid, role='assistant', content='one', send_index=1,
        round_index=0, round_usage=USAGE)
    await store.record_usage(sid, send_index=1, round_index=1, **USAGE)
    await store.append_message(sid, role='assistant', content='two', send_index=2,
        round_index=0, round_usage=USAGE)
    assert len(await store.list_usage_rounds(sid)) == 3
    assert await store.prune_usage_rounds(sid, keep_last=1) == 2
    [detail] = await store.list_usage_rounds(sid)
    assert (detail['send_index'], detail['round_index'], detail['total_tokens']) == (2, 0, 12)
    assert [row.content for row in await store.load_all_messages(sid)] == ['one', 'two']
    assert await store.prune_usage_rounds(sid) == 1
    assert await store.list_usage_rounds(sid) == []


async def test_export_keeps_response_and_empty_round_usage(registry_store):
    store, _ = registry_store
    sid = (await new_session(store)).session_id
    await store.append_message(sid, role='assistant', content='original', send_index=7,
        round_index=0, round_usage=USAGE)
    await store.record_usage(sid, send_index=7, round_index=1, **USAGE)
    exported = await store.export_session(sid)
    observation = next(r for r in exported['tables']['session_runtime_state'] if r['state_key']=='usage:7:1')
    assert json.loads(observation['value_json'])['total_tokens'] == 12
    details = await store.list_usage_rounds(sid)
    assert [(r['send_index'],r['round_index'],r['total_tokens']) for r in details] == [(7,0,12),(7,1,12)]
    [message] = await store.load_all_messages(sid)
    assert message.content == 'original' and '_round_usage' not in message.meta


async def test_hook_audit_and_corrected_auxiliary_usage_share_schema17(registry_store):
    store, connect = registry_store
    sid = (await new_session(store)).session_id
    seq = await store.append_message(sid, role='assistant', content='public', hook_injected={
        'hook_point':'llm_before','hook':'memory','kind':'context','payload':{'text':'private'}})
    peer = await connect()
    [audit] = await peer.list_hook_events(sid, message_seq=seq)
    assert audit.payload == {'text':'private'}
    assert (await peer.load_all_messages(sid))[0].content == 'public'
    await store.record_auxiliary_usage(sid, 'fold', USAGE)
    await peer.record_auxiliary_usage(sid, 'fold', USAGE)
    await peer.record_auxiliary_usage(sid, 'fold', {**USAGE, 'total_tokens':15}, revision=1)
    stats = await store.get_session_stats(sid)
    assert stats.llm_calls == 1 and stats.total_tokens == 15
    assert (await store.list_session_stats())[0] == stats
    assert not any(r['state_key'].startswith('hook:') for r in (await store.export_session(sid))['tables']['session_runtime_state'])
