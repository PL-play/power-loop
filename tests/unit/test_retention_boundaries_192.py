"""Indexed retention boundaries preserve sends, age filters and other sessions."""
import pytest

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('keep,age,remaining', [(2,None,[(2,0),(3,0)]), (10,None,[(1,0),(1,1),(2,0),(3,0)]),
    (2,3600000,[(1,0),(1,1),(2,0),(3,0)]), (2,-1000,[(2,0),(3,0)]), (0,None,[])])
async def test_usage_retention_uses_global_send_round_order_and_intersects_age(registry_store, keep, age, remaining):
    store, _ = registry_store
    sid, other = await store.create_session(), await store.create_session()
    for send, round_ in [(1,0),(1,1),(2,0),(3,0)]:
        await store.record_usage(sid, send_index=send, round_index=round_, prompt_tokens=2, completion_tokens=3, total_tokens=5)
    await store.record_usage(other, send_index=1, round_index=0, prompt_tokens=7, completion_tokens=0, total_tokens=7)
    foreign = await store._db.fetchall(f'SELECT * FROM {store.t.usage_rounds} WHERE session_id=?', (other,))
    deleted = await store.prune_usage_rounds(sid, keep_last=keep, older_than_ms=age)
    actual = await store._db.fetchall(f'SELECT send_index,round_index FROM {store.t.usage_rounds} WHERE session_id=? ORDER BY send_index,round_index', (sid,))
    assert [(row['send_index'],row['round_index']) for row in actual] == remaining
    assert deleted == 4-len(remaining)
    assert await store._db.fetchall(f'SELECT * FROM {store.t.usage_rounds} WHERE session_id=?', (other,)) == foreign


@pytest.mark.parametrize('keep,age,deleted', [(2,None,2),(10,None,0),(2,3600000,0),(2,-1000,2),(0,None,4)])
async def test_message_retention_preserves_active_window_and_insufficient_keep_is_noop(registry_store, keep, age, deleted):
    store, _ = registry_store
    sid = await store.create_session()
    for i in range(6):
        await store.append_message(sid, role='user', content=f'original-{i}')
    await store.record_compaction(sid, from_seq=1, to_seq=4, note_content='original summary', before_tokens=100,
        after_tokens=10, round_index=1, fold_seqs=[1,2,3,4], order_key=1)
    active = await store._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? AND state=? ORDER BY seq', (sid,'active'))
    assert await store.prune_compacted_messages(sid, keep_recent=keep, older_than_ms=age) == deleted
    assert await store._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? AND state=? ORDER BY seq', (sid,'active')) == active
    rows = await store._db.fetchall(f'SELECT seq FROM {store.t.messages} WHERE session_id=? AND state=? ORDER BY seq', (sid,'compacted_out'))
    assert [row['seq'] for row in rows] == ([3,4] if deleted==2 else [] if deleted==4 else [1,2,3,4])
