"""Schema 17 close/delete and retention, including shared-store resource guards."""
import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.contracts.errors import SessionBusy

from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_active_inputs_pending_tools_and_remote_owner_are_protected(registry_store):
    store, connect = registry_store
    peer = await connect(schema='verify')
    port = RuntimeInputStore(peer)
    root = await target(port)
    receipt = await port.accept(root, 'work', message_id='start', actor=ACTOR)
    assert await store.close_if_idle(root.session_id) == 'busy'
    batch = await port.claim(root, node_id='remote', boot_id='remote')
    with pytest.raises(SessionBusy):
        await store.close_session(root.session_id)
    assert not await store.purge_expired(2**62, retention_days={'inbox': 0})
    await port.final(batch, final_text='done')
    # Pending tool evidence protects even an otherwise idle session.
    await peer._db.execute(f'UPDATE {peer.t.sessions} SET pending_tool_seq=? WHERE session_id=?',
                           (2, root.session_id))
    assert await store.close_if_idle(root.session_id) == 'busy'
    await peer._db.execute(f'UPDATE {peer.t.sessions} SET pending_tool_seq=NULL WHERE session_id=?', (root.session_id,))
    assert (await port.get_result(receipt.input_ref, actor=ACTOR)).final_text == 'done'
    assert await store.close_if_idle(root.session_id) == 'closed'
    assert (await store.get_session(root.session_id)).lifecycle == 'closing'


async def test_linked_children_independent_survivors_and_resource_guard(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(await connect())
    parent = await target(port, 'parent')
    child = await target(port, 'linked', parent=parent)
    independent = await target(port, 'independent', parent=parent, cancel_propagation='independent')
    receipt = await port.accept(child, 'remote child', message_id='start', actor=ACTOR)
    await port.accept(independent, 'independent', message_id='start', actor=ACTOR)
    with pytest.raises(SessionBusy):
        await store.close_session(parent.session_id)
    await port.cancel(receipt.input_ref, actor=ACTOR)
    with pytest.raises(SessionBusy):
        await store.close_session_tree(parent.session_id, resource_guard=lambda ids: child.session_id not in ids)
    assert set(await store.close_session_tree(parent.session_id)) == {'parent', 'linked'}
    saved = await store.get_session('independent')
    assert saved.parent_session_id is None and saved.descriptor == DESCRIPTOR


async def test_waiting_continuation_prevents_cleanup(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    receipt = await port.accept(root, 'start', message_id='start', actor=ACTOR)
    batch = await port.claim(root, node_id='a', boot_id='a')
    await port.final(batch, final_text='done')
    await store._db.execute(f"UPDATE {store.t.follow_up_queue} SET status='waiting_continuation', "
        'terminal_status=NULL,result_send_index=NULL,result_seq=NULL,completed_at=NULL WHERE item_id=?',
        (receipt.message_id,))
    assert await store.close_if_idle(root.session_id) == 'busy'
    assert not await store.purge_expired(2**62)


async def test_terminal_inputs_graph_projection_cleanup_and_row_limit(registry_store, monkeypatch):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, 'start', message_id='start', actor=ACTOR)
    batch = await port.claim(root, node_id='a', boot_id='a')
    await port.final(batch, final_text='done')
    await store.set_runtime_state(root.session_id, 'workflow:run:start', {'status': 'completed'})
    await store.upsert_project_message(root.session_id, send_index=1, kind='project', content='done')
    assert await store.close_if_idle(root.session_id) == 'closed'
    original = store._db.transaction
    writes = []

    @asynccontextmanager
    async def transaction():
        async with original() as tx:
            execute = tx.execute
            async def measured(sql, params=()):
                count = await execute(sql, params)
                if sql.lstrip().split()[0] in {'UPDATE', 'DELETE', 'INSERT'}:
                    writes.append(max(0, count))
                return count
            tx.execute = measured
            yield tx
    monkeypatch.setattr(store._db, 'transaction', transaction)
    totals = {}
    for _ in range(10):
        writes.clear()
        counts = await store.purge_expired(2**62, limit=3)
        assert sum(writes) <= 3 and sum(counts.values()) == sum(writes)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        if await store.get_session(root.session_id) is None:
            break
    assert totals == {'inbox': 1, 'projections': 1, 'runtime': 2, 'messages': 2, 'sessions': 1}
    assert await store.get_session(root.session_id) is None


async def test_bounded_delete_blocks_new_admission_and_uses_only_current_schema(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    for i in range(10):
        await store.append_message(root.session_id, role='user', content=str(i))
    assert not await store.close_session_tree(root.session_id, limit=4)
    assert (await store.get_session(root.session_id)).lifecycle == 'closing'
    with pytest.raises(RuntimeRequestError, match='session_closed'):
        await port.accept(root, 'late', message_id='late', actor=ACTOR)
    for _ in range(5):
        await store.close_session_tree(root.session_id, limit=4)
    assert await store.get_session(root.session_id) is None


async def test_close_serializes_with_remote_accept(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(await connect())
    root = await target(port)
    async def accept():
        try:
            await port.accept(root, 'race', message_id='race', actor=ACTOR)
            return 'accepted'
        except RuntimeRequestError as error:
            assert error.code == 'session_closed'
            return 'closed'
    status, accepted = await asyncio.gather(store.close_if_idle(root.session_id), accept())
    assert (status, accepted) in {('busy', 'accepted'), ('closed', 'closed')}
