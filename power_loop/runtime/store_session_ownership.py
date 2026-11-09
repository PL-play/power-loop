"""Default session lock for multiple nodes, stored on ``session_state``.

One conditional write takes the session: it sets the owner token and deadline
and advances ``owner_epoch``. A crashed node keeps the row until the deadline;
the next acquire replaces an expired owner. Renewal and release check the token,
so an older holder cannot extend or clear a newer one.

The deadline is wall-clock milliseconds because other processes compare it.
The local handle still expires on this process's monotonic clock.
"""
from __future__ import annotations

import time

from power_loop.runtime.session_ownership import SessionOwnership, SessionOwnershipCoordinator


class StoreSessionOwnershipCoordinator(SessionOwnershipCoordinator):
    scope = "shared"
    installs_session_fence = True

    def __init__(self, store):
        self.store = store
        self.table = store.t.session_state

    def _deadline(self, ttl_s):
        return int(time.time() * 1000) + int(ttl_s * 1000)

    async def acquire(self, key, *, owner_id, ttl_s, on_lost, node_id=None, boot_id=None):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        node_id, boot_id = node_id or owner_id, boot_id or owner_id
        started = time.monotonic()
        lease = SessionOwnership(key, owner_id, started + ttl_s, on_lost)
        lease.installs_fence = True
        lease.node_id, lease.boot_id = node_id, boot_id
        now, deadline = int(time.time() * 1000), self._deadline(ttl_s)
        table, session_id = self.table, key[1]
        async with self.store._db.transaction() as tx:
            if self.store._db.dialect.name == "mysql":
                # MySQL reads a column assigned earlier in this UPDATE as its new
                # value, so the takeover decision is captured once.
                await tx.execute(
                    f"INSERT INTO {table} (session_id,owner_token,owner_node_id,owner_boot_id,"
                    "owner_lease_until_ms,owner_epoch) VALUES (?,?,?,?,?,1) AS incoming "
                    "ON DUPLICATE KEY UPDATE "
                    f"owner_epoch=IF(@takeover:=({table}.lifecycle='open' AND "
                    f"({table}.owner_token IS NULL OR {table}.owner_lease_until_ms<=?)), "
                    f"{table}.owner_epoch+1, {table}.owner_epoch), "
                    f"owner_node_id=IF(@takeover, incoming.owner_node_id, {table}.owner_node_id), "
                    f"owner_boot_id=IF(@takeover, incoming.owner_boot_id, {table}.owner_boot_id), "
                    f"owner_lease_until_ms=IF(@takeover, incoming.owner_lease_until_ms, "
                    f"{table}.owner_lease_until_ms), "
                    f"owner_token=IF(@takeover, incoming.owner_token, {table}.owner_token)",
                    (session_id, lease.token, node_id, boot_id, deadline, now))
                row = await tx.fetchone(
                    f"SELECT owner_token,owner_epoch,lifecycle FROM {table} WHERE session_id=?",
                    (session_id,))
            else:
                row = await tx.fetchone(
                    f"INSERT INTO {table} (session_id,owner_token,owner_node_id,owner_boot_id,"
                    "owner_lease_until_ms,owner_epoch) VALUES (?,?,?,?,?,1) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "owner_token=excluded.owner_token, owner_node_id=excluded.owner_node_id, "
                    "owner_boot_id=excluded.owner_boot_id, owner_lease_until_ms=excluded.owner_lease_until_ms, "
                    f"owner_epoch={table}.owner_epoch+1 "
                    f"WHERE {table}.lifecycle='open' AND ({table}.owner_token IS NULL OR {table}.owner_lease_until_ms<=?) "
                    "RETURNING owner_token,owner_epoch,lifecycle",
                    (session_id, lease.token, node_id, boot_id, deadline, now))
                if row is None:
                    row = await tx.fetchone(
                        f"SELECT owner_token,owner_epoch,lifecycle FROM {table} WHERE session_id=?", (session_id,))
        if row is not None and row["lifecycle"] != "open":
            from power_loop.agent.runtime_types import RuntimeRequestError
            raise RuntimeRequestError("session_closed")
        if row is None or row["owner_token"] != lease.token:
            return None
        lease.epoch = int(row["owner_epoch"])
        return lease

    async def renew(self, lease, *, ttl_s):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        started = time.monotonic()
        if lease.lost or lease.released or lease.expires_at <= started:
            lease.lose()
            return False
        now = int(time.time() * 1000)
        changed = await self.store._db.execute(
            f"UPDATE {self.table} SET owner_lease_until_ms=? "
            "WHERE session_id=? AND owner_token=? AND lifecycle='open' AND owner_lease_until_ms>?",
            (self._deadline(ttl_s), lease.key[1], lease.token, now))
        if not changed or lease.lost or lease.released:
            lease.lose()
            return False
        lease.expires_at = started + ttl_s
        return True

    async def release(self, lease):
        await self.store._db.execute(
            f"UPDATE {self.table} SET owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,"
            "owner_lease_until_ms=NULL WHERE session_id=? AND owner_token=?",
            (lease.key[1], lease.token))
        lease.released = True
