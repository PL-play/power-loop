"""The bundled single-process implementation passes the application contract."""
import unittest

import pytest

from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from power_loop.testing.session_ownership import SessionOwnershipContractTests

pytestmark = pytest.mark.unit


class TestMemorySessionOwnership(SessionOwnershipContractTests, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.shared = MemorySessionOwnershipCoordinator()

    async def coordinator(self):
        return self.shared

    async def asyncTearDown(self):
        self.assertFalse(self.shared._leases)


@pytest.mark.asyncio
async def test_store_session_lock_passes_the_shared_contract(registry_store):
    """The default database lock passes the shared-coordinator contract on this backend."""
    store, _ = registry_store

    class Checks(SessionOwnershipContractTests, unittest.IsolatedAsyncioTestCase):
        async def coordinator(self):
            from power_loop.runtime.store_session_ownership import StoreSessionOwnershipCoordinator
            return StoreSessionOwnershipCoordinator(store)

    case = Checks()
    for name in ("test_exclusive_acquisition", "test_renew_and_release_then_reacquire",
                 "test_expiry_takeover_notifies_old_holder_and_rejects_renewal",
                 "test_expired_local_handle_rejects_writes_without_waiting_for_renewal"):
        await getattr(case, name)()
    assert not await store._db.fetchall(f"SELECT session_id FROM {store.t.session_leases}")


class TestDatabaseSessionOwnership(SessionOwnershipContractTests, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import tempfile
        self.directory = tempfile.TemporaryDirectory()

    async def coordinator(self):
        from power_loop.testing.db_session_ownership import DatabaseSessionOwnershipCoordinator
        return DatabaseSessionOwnershipCoordinator(self.directory.name + "/coordinator.db")

    async def asyncTearDown(self):
        self.directory.cleanup()


@pytest.mark.asyncio
async def test_delayed_acquire_releases_conflicting_grant_and_restarts(registry_store):
    import asyncio

    from power_loop.runtime.session_ownership import hold_session_ownership

    from .test_session_ownership_tasks import holder

    store, _ = registry_store
    sid = await store.create_session()
    entered, resume = asyncio.Event(), asyncio.Event()
    class Delayed(MemorySessionOwnershipCoordinator):
        calls = 0
        grants = []
        async def acquire(self, key, **kwargs):
            if kwargs['owner_id'] == 'old':
                self.calls += 1
                if self.calls == 1:
                    entered.set()
                    await resume.wait()
            lease = await super().acquire(key, **kwargs)
            if kwargs['owner_id'] == 'old':
                self.grants.append(lease)
            return lease
    coordinator = Delayed()
    async def old_request():
        async with hold_session_ownership(holder(store, coordinator, 'old'), sid) as lease:
            assert lease.epoch == 2
    old = asyncio.create_task(old_request())
    await entered.wait()
    async with hold_session_ownership(holder(store, coordinator, 'new'), sid) as lease:
        assert lease.epoch == 1
    resume.set()
    await old
    assert coordinator.calls == 2
    assert coordinator.grants[0].released
    assert coordinator.grants[0].token != coordinator.grants[1].token


@pytest.mark.asyncio
async def test_old_release_and_late_writes_cannot_change_new_owner(registry_store):
    import time

    from power_loop import SessionLeaseLost

    store, _ = registry_store
    sid = await store.create_session()
    coordinator = MemorySessionOwnershipCoordinator()
    key = (store.store_id, sid)
    old = await coordinator.acquire(key, owner_id='old', ttl_s=90, on_lost=lambda: None)
    old.node_id, old.boot_id = 'old', 'old'
    assert await store.install_session_owner(sid, old, 0)
    # A missed local notification cannot grant writes after a newer epoch.
    new = type(old)(key, 'new', time.monotonic()+90, lambda: None, node_id='new', boot_id='new')
    assert await store.install_session_owner(sid, new, 1)
    await store.release_session_owner(sid, old)
    row = await store._db.fetchone(f'SELECT owner_epoch,owner_token FROM {store.t.session_state} WHERE session_id=?', (sid,))
    assert row == {'owner_epoch': 2, 'owner_token': new.token}
    async with store.send_read_scope(sid, ownership=old):
        with pytest.raises(SessionLeaseLost):
            await store.append_message(sid, role='assistant', content='late')
        with pytest.raises(SessionLeaseLost):
            await store.set_pending(sid, {'late': True})
        with pytest.raises(SessionLeaseLost):
            await store.set_runtime_state(sid, 'late', {'value': True})
    assert not await store.load_all_messages(sid)
    assert await store.get_runtime_state(sid, 'late') is None
    # Result publication uses its existing state lock in the job transaction.
    from types import SimpleNamespace

    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import JobRef, Result
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.task_runtime import TaskRuntime
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='tenant', principal_id='host', executor_id='old', boot_id='old')
    actor = runtime.actor(sid)
    registration = await manager.register_job(actor, kind='send', session_id=sid, system_key='late-result')
    proof = await runtime.claim(actor, registration)
    ref = JobRef(store.store_id, proof.lease.job_id, proof.lease.generation)
    port = RuntimeInputStore(store)
    with pytest.raises(SessionLeaseLost):
        async with manager._write() as tx:
            await port.commit_result_and_parent_ref(tx, ref, Result(ref, None, 'completed'),
                SimpleNamespace(epoch=old.epoch, lease=old))
    assert (await manager.get_job(actor, ref.job_id)).result_status == 'none'
    await store.release_session_owner(sid, new)
    await coordinator.release(old)


@pytest.mark.asyncio
async def test_lease_expiring_while_waiting_for_real_state_lock_rolls_back(registry_store):
    import asyncio

    from power_loop import SessionLeaseLost

    store, connect = registry_store
    writer = await connect()
    sid = await store.create_session()
    coordinator = MemorySessionOwnershipCoordinator()
    lease = await coordinator.acquire((store.store_id, sid), owner_id='old', ttl_s=90, on_lost=lambda: None)
    lease.node_id, lease.boot_id = 'old', 'old'
    assert await store.install_session_owner(sid, lease, 0)
    started = asyncio.Event()
    async def write():
        async with writer.send_read_scope(sid, ownership=lease):
            started.set()
            with pytest.raises(SessionLeaseLost):
                await writer.set_pending(sid, {'late': True})
    async with store._db.transaction() as tx:
        await store._db.dialect.lock_state(tx, store.t.session_state, sid)
        task = asyncio.create_task(write())
        await started.wait()
        await asyncio.sleep(.05)
        lease.expires_at = 0
    await task
    assert (await store.get_state(sid)).pending is None
    await store.release_session_owner(sid, lease)
    await coordinator.release(lease)


from .test_job_registry_192 import registry_store  # noqa: E402

__all__ = ['registry_store']


@pytest.mark.asyncio
async def test_closed_session_rejects_acquisition_without_retrying(registry_store):
    from power_loop.agent.runtime_types import RuntimeRequestError
    from power_loop.runtime.session_ownership import hold_session_ownership

    from .test_session_ownership_tasks import holder
    store, _ = registry_store
    sid = await store.create_session()
    await store._db.execute(f"UPDATE {store.t.session_state} SET lifecycle='closed' WHERE session_id=?", (sid,))
    coordinator = MemorySessionOwnershipCoordinator()
    with pytest.raises(RuntimeRequestError, match='session_closed'):
        async with hold_session_ownership(holder(store, coordinator), sid):
            pytest.fail('closed session acquired')
    assert not coordinator._leases
