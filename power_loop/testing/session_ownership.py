"""Subclass with unittest.IsolatedAsyncioTestCase and implement coordinator()."""
from __future__ import annotations

import asyncio
import uuid

from power_loop.contracts.errors import SessionLeaseLost


class SessionOwnershipContractTests:
    """Each call to coordinator() must use the same isolated shared namespace.

Return separate clients when testing a shared implementation. Dispose of its
test namespace in asyncTearDown. The mixin uses unittest assertions and does
not introduce a pytest dependency into the SDK.
"""

    async def coordinator(self):
        raise NotImplementedError

    async def test_exclusive_acquisition(self):
        first, second = await self.coordinator(), await self.coordinator()
        key = (uuid.uuid4().hex, uuid.uuid4().hex)
        leases = await asyncio.gather(*(
            client.acquire(key, owner_id=str(i), ttl_s=5, on_lost=lambda: None)
            for i, client in enumerate((first, second))))
        try:
            self.assertEqual(sum(lease is not None for lease in leases), 1)
        finally:
            for client, lease in zip((first, second), leases, strict=True):
                if lease is not None:
                    await client.release(lease)

    async def test_renew_and_release_then_reacquire(self):
        client = await self.coordinator()
        key, lost = (uuid.uuid4().hex, uuid.uuid4().hex), []
        lease = await client.acquire(key, owner_id='same-owner', ttl_s=5,
                                     on_lost=lambda: lost.append(True))
        try:
            before = lease.expires_at
            self.assertTrue(await client.renew(lease, ttl_s=10))
            self.assertGreater(lease.expires_at, before)
            self.assertIsNone(await client.acquire(key, owner_id='same-owner', ttl_s=5,
                                                   on_lost=lambda: None))
        finally:
            await client.release(lease)
        self.assertFalse(lost)
        with self.assertRaises(SessionLeaseLost):
            lease.require_owned()
        next_lease = await client.acquire(key, owner_id='same-owner', ttl_s=5,
                                         on_lost=lambda: None)
        try:
            self.assertIsNotNone(next_lease)
            self.assertNotEqual(next_lease.token, lease.token)
            await client.release(lease)
            self.assertTrue(await client.renew(next_lease, ttl_s=5))
        finally:
            await client.release(next_lease)

    async def test_expiry_takeover_notifies_old_holder_and_rejects_renewal(self):
        first, second = await self.coordinator(), await self.coordinator()
        key, lost = (uuid.uuid4().hex, uuid.uuid4().hex), []
        old = await first.acquire(key, owner_id='old', ttl_s=.05,
                                  on_lost=lambda: lost.append(True))
        await asyncio.sleep(.08)
        new = await second.acquire(key, owner_id='new', ttl_s=5, on_lost=lambda: None)
        try:
            self.assertIsNotNone(new)
            self.assertFalse(await first.renew(old, ttl_s=5))
            self.assertEqual(lost, [True])
            with self.assertRaises(SessionLeaseLost):
                old.require_owned()
            await first.release(old)
            self.assertTrue(await second.renew(new, ttl_s=5))
        finally:
            await first.release(old)
            await second.release(new)

    async def test_expired_local_handle_rejects_writes_without_waiting_for_renewal(self):
        client = await self.coordinator()
        lost = []
        lease = await client.acquire((uuid.uuid4().hex, uuid.uuid4().hex), owner_id='old', ttl_s=.05,
                                     on_lost=lambda: lost.append(True))
        try:
            await asyncio.sleep(.08)
            with self.assertRaises(SessionLeaseLost):
                lease.require_owned()
            self.assertEqual(lost, [True])
            self.assertFalse(await client.renew(lease, ttl_s=5))
            self.assertEqual(lost, [True])
        finally:
            await client.release(lease)
