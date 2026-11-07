"""Global admission closure and cumulative maintenance bounds on all three stores."""
import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.contracts.errors import SessionBusy

from .test_job_registry_192 import registry_store
from .test_retention import NOW, OLD, insert, root
from .test_runtime_messaging_203 import accept, command, fixture

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_close_and_remote_admission_serialize(registry_store):
    store, connect = registry_store
    peer = await connect()
    port, target, actor = await fixture(peer)
    async def send():
        try:
            await accept(port, command(target, 'race'), actor)
            return 'accepted'
        except RuntimeRequestError as error:
            assert error.code == 'session_closed'
            return 'closed'
    status, accepted = await asyncio.gather(store.close_if_idle(target.session_id), send())
    assert (status, accepted) in {('busy', 'accepted'), ('closed', 'closed')}
    state = await store._db.fetchone(f'SELECT lifecycle FROM {store.t.session_state} WHERE session_id=?', (target.session_id,))
    assert state['lifecycle'] == ('open' if status == 'busy' else 'closing')
    if status == 'busy':
        with pytest.raises(SessionBusy):
            await store.close_session(target.session_id)


async def test_close_keeps_remote_child_and_unknown_resources(registry_store):
    store, connect = registry_store
    parent = await store.create_session()
    peer = await connect()
    child = await peer.create_session(parent_session_id=parent)
    job, _, _ = await root(peer, 'remote', ended=False)
    await peer._db.execute(f'UPDATE {peer.table_prefix}jobs SET session_id=? WHERE job_id=?', (child, job))
    assert await store.close_if_idle(parent) == 'busy'
    with pytest.raises(SessionBusy):
        await store.close_session(parent)
    await peer._db.execute(f"UPDATE {peer.table_prefix}jobs SET supervision_status='closed',resource_state='unknown' WHERE job_id=?", (job,))
    assert await store.close_if_idle(parent) == 'busy'
    assert (await store._db.fetchone(f'SELECT lifecycle FROM {store.t.session_state} WHERE session_id=?', (parent,)))['lifecycle'] == 'open'


@asynccontextmanager
async def measure_writes(store, monkeypatch):
    """Count actual affected rows, including UPDATEs, by transaction and call."""
    factory = store._db.transaction
    batches = []
    @asynccontextmanager
    async def transaction():
        async with factory() as tx:
            batch = []
            original = tx.execute
            async def execute(sql, params=()):
                count = await original(sql, params)
                if sql.lstrip().split()[0] in ('INSERT','UPDATE','DELETE'):
                    batch.append(max(0, count))
                return count
            tx.execute = execute
            batches.append(batch)
            yield tx
    with monkeypatch.context() as patch:
        patch.setattr(store._db, 'transaction', transaction)
        yield batches


async def test_large_tree_mixed_categories_progress_with_one_allowance(registry_store, monkeypatch):
    store, _ = registry_store
    db, p = store._db, store.table_prefix
    large, sid, _ = await root(store, 'large')
    small, _, _ = await root(store, 'small')
    active, _, _ = await root(store, 'active', ended=False)
    async with db.transaction() as tx:
        for i in range(1100):
            await insert(tx, p+'job_events', event_id=f'bulk-{i}', job_id=large,
                         event_type='trace', job_version=1, payload_json='{}', delivery_status='handled', created_at=OLD)
        for i in range(20):
            await insert(tx, store.t.follow_up_queue, session_id=sid, item_id=f'old-{i}', content='',
                         kind='user', mode='queue', status='void', created_at=OLD, meta_json='{}')
    for _ in range(8):
        async with measure_writes(store, monkeypatch) as batches:
            await store.purge_expired(NOW, limit=250)
        assert sum(map(sum, batches)) <= 250
        assert all(sum(batch) <= 250 for batch in batches)
        events = await db.fetchone(f'SELECT event_id FROM {p}job_events WHERE job_id=? LIMIT 1', (large,))
        if events:
            assert await db.fetchone(f'SELECT job_id FROM {p}jobs WHERE job_id=?', (large,))
    assert not await db.fetchone(f'SELECT job_id FROM {p}jobs WHERE job_id IN (?,?)', (large, small))
    assert await db.fetchone(f'SELECT job_id FROM {p}jobs WHERE job_id=?', (active,))


async def test_explicit_delete_is_bounded_and_blocks_new_inputs(registry_store, monkeypatch):
    store, _ = registry_store
    port, target, actor = await fixture(store)
    sid = target.session_id
    async with store._db.transaction() as tx:
        for seq in range(1, 1101):
            await insert(tx, store.t.messages, session_id=sid, seq=seq, role='user', content='old', created_at=OLD)
    for i in range(4):
        async with measure_writes(store, monkeypatch) as batches:
            deleted = await store.close_session_tree(sid, limit=400)
        assert sum(map(sum, batches)) <= 400
        if i == 0:
            assert not deleted and await store.get_session(sid)
            with pytest.raises(RuntimeRequestError, match='session_closed'):
                await accept(port, command(target, 'late'), actor)
    assert await store.get_session(sid) is None


async def test_resource_guard_and_expired_child_descriptor(registry_store):
    import json
    store, _ = registry_store
    parent = await store.create_session()
    child = await store.create_session(parent_session_id=parent, kind='subagent',
        metadata={'runtime_descriptor': {'test': 'temporary'}, 'keep':'business'})
    assert await store.close_if_idle(parent, resource_guard=lambda ids: child not in ids) == 'busy'
    assert (await store._db.fetchone(f'SELECT lifecycle FROM {store.t.session_state} WHERE session_id=?', (parent,)))['lifecycle']=='open'
    await store._db.execute(f'UPDATE {store.t.sessions} SET updated_at=? WHERE session_id=?', (OLD, child))
    await store.purge_expired(NOW, limit=2)
    row = await store._db.fetchone(f'SELECT metadata_json FROM {store.t.sessions} WHERE session_id=?', (child,))
    assert json.loads(row['metadata_json']) == {'keep':'business'}
    assert (await store._db.fetchone(f'SELECT lifecycle FROM {store.t.session_state} WHERE session_id=?', (child,)))['lifecycle']=='closed'
