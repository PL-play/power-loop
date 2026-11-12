"""Memory lease contract and schema-17 database owner fences."""
import time
import unittest

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator, SessionOwnership
from power_loop.runtime.store_session_ownership import StoreSessionOwnershipCoordinator
from power_loop.testing.session_ownership import SessionOwnershipContractTests

from .session_native_fixture import ACTOR, target

pytestmark = pytest.mark.unit


class TestMemorySessionOwnership(SessionOwnershipContractTests, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.shared = MemorySessionOwnershipCoordinator()

    async def coordinator(self):
        return self.shared

    async def asyncTearDown(self):
        self.assertFalse(self.shared._leases)


@pytest.mark.asyncio
async def test_owner_fence_renew_final_release_and_stale_handle(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "one", message_id="one", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    coordinator = StoreSessionOwnershipCoordinator(store)
    lease = SessionOwnership(
        (store.store_id, root.session_id), "a", time.monotonic() + 90, lambda: None,
        token=batch.owner_token, epoch=batch.owner_epoch, node_id="a", boot_id="a",
    )
    assert await coordinator.acquire(
        lease.key, owner_id="b", ttl_s=30, on_lost=lambda: None
    ) is None
    assert await coordinator.renew(lease, ttl_s=30)
    with pytest.raises(RuntimeRequestError, match="final or recover"):
        await coordinator.release(lease)
    await port.final(batch, final_text="done")
    await coordinator.release(lease)
    assert lease.released

    await port.accept(root, "two", message_id="two", actor=ACTOR)
    new_batch = await port.claim(root, node_id="b", boot_id="b")
    assert new_batch.owner_epoch == batch.owner_epoch + 1
    await coordinator.release(lease)
    state = await store._db.fetchone(
        f"SELECT owner_epoch,owner_token FROM {store.t.session_state} WHERE session_id=?",
        (root.session_id,),
    )
    assert state == {"owner_epoch": new_batch.owner_epoch, "owner_token": new_batch.owner_token}
    await port.final(new_batch, final_text="new")


@pytest.mark.asyncio
async def test_shared_coordinator_takeover_enters_recovery_before_release(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "one", message_id="one", actor=ACTOR)
    await port.claim(root, node_id="old", boot_id="old")
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
        (root.session_id,),
    )
    coordinator = StoreSessionOwnershipCoordinator(store)
    lease = await coordinator.acquire(
        (store.store_id, root.session_id), owner_id="new", ttl_s=30, on_lost=lambda: None,
        node_id="new", boot_id="new",
    )
    assert lease is not None
    recovered = await port.recover(
        root, node_id="new", boot_id="new", ownership=lease,
    )
    assert recovered.execution_state == "idle"
    await coordinator.release(lease)


@pytest.mark.asyncio
async def test_session_owner_fence_blocks_stale_active_writes(registry_store):
    from power_loop.contracts.errors import SessionLeaseLost

    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "one", message_id="one", actor=ACTOR)
    old = await port.claim(root, node_id="old", boot_id="old")
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
        (root.session_id,),
    )
    await port.accept(root, "two", message_id="two", actor=ACTOR)
    with pytest.raises(SessionLeaseLost):
        await port.cancel_active(old)
    recovered = await port.recover(root, node_id="new", boot_id="new")
    assert recovered.execution_state == "queued"
