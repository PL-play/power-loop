"""Default session lock for multiple nodes, stored in the SDK's own database.

One row per session in ``session_leases``. Every acquire, renew, and release is a
single conditional write, so two nodes racing for the same session leave exactly
one owner. The row outlives a crashed node until its deadline; the next acquire
replaces an expired row.

The store's ``owner_epoch`` stays the write fence: this lock decides who may
install the next epoch, the epoch decides whose writes land.
"""
from __future__ import annotations

import time

from power_loop.runtime.session_ownership import SessionOwnership, SessionOwnershipCoordinator


class StoreSessionOwnershipCoordinator(SessionOwnershipCoordinator):
    scope = "shared"

    def __init__(self, store):
        self.store = store
        self.table = store.t.session_leases

    def _deadline(self, started, ttl_s):
        return int(started * 1000) + int(ttl_s * 1000)

    async def acquire(self, key, *, owner_id, ttl_s, on_lost):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        started = time.monotonic()
        lease = SessionOwnership(key, owner_id, started + ttl_s, on_lost)
        now, deadline = self._deadline(started, 0), self._deadline(started, ttl_s)
        async with self.store._db.transaction() as tx:
            # MySQL counts an unchanged duplicate as a match, so the token decides.
            if self.store._db.dialect.name == "mysql":
                await tx.execute(
                    f"INSERT INTO {self.table} (session_id, owner_id, token, acquired_at, expires_at) "
                    "VALUES (?,?,?,?,?) ON DUPLICATE KEY UPDATE "
                    "owner_id=IF(expires_at<=?, VALUES(owner_id), owner_id), "
                    "token=IF(expires_at<=?, VALUES(token), token), "
                    "acquired_at=IF(expires_at<=?, VALUES(acquired_at), acquired_at), "
                    "expires_at=IF(expires_at<=?, VALUES(expires_at), expires_at), "
                    "fence=IF(expires_at<=?, fence+1, fence)",
                    (key[1], owner_id, lease.token, now, deadline, now, now, now, now, now))
            else:
                await tx.execute(
                    f"INSERT INTO {self.table} (session_id, owner_id, token, acquired_at, expires_at) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                    "owner_id=excluded.owner_id, token=excluded.token, acquired_at=excluded.acquired_at, "
                    f"expires_at=excluded.expires_at, fence={self.table}.fence+1 "
                    f"WHERE {self.table}.expires_at<=?",
                    (key[1], owner_id, lease.token, now, deadline, now))
            row = await tx.fetchone(f"SELECT token FROM {self.table} WHERE session_id=?", (key[1],))
        return lease if row is not None and row["token"] == lease.token else None

    async def renew(self, lease, *, ttl_s):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        started = time.monotonic()
        if lease.lost or lease.released or lease.expires_at <= started:
            lease.lose()
            return False
        changed = await self.store._db.execute(
            f"UPDATE {self.table} SET expires_at=? WHERE session_id=? AND token=? AND expires_at>?",
            (self._deadline(started, ttl_s), lease.key[1], lease.token, self._deadline(started, 0)))
        if not changed or lease.lost or lease.released:
            lease.lose()
            return False
        lease.expires_at = started + ttl_s
        return True

    async def release(self, lease):
        await self.store._db.execute(
            f"DELETE FROM {self.table} WHERE session_id=? AND token=?", (lease.key[1], lease.token))
        lease.released = True
