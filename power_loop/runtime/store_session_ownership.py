"""Shared owner fence stored on session_state.

Claim/recover install ownership in their own transactions. This coordinator only
renews or takes over an already active send; it never creates a send or a claim token.
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

    async def claim(self, inputs, target, **options):
        """Install the fence in the same transaction as the input batch."""
        return await inputs.claim(target, **options)

    async def recover(self, inputs, target, **options):
        return await inputs.recover(target, **options)

    async def acquire(self, key, *, owner_id, ttl_s, on_lost, node_id=None, boot_id=None):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        node_id, boot_id = node_id or owner_id, boot_id or owner_id
        started = time.monotonic()
        lease = SessionOwnership(key, owner_id, started + ttl_s, on_lost)
        lease.installs_fence = True
        lease.node_id, lease.boot_id = node_id, boot_id
        async with self.store._db.transaction() as tx:
            from power_loop.agent.runtime_store import RuntimeInputStore
            now = await RuntimeInputStore(self.store)._db_now(tx)
            deadline = now + int(ttl_s * 1000)
            changed = await tx.execute(
                f"UPDATE {self.table} SET execution_state=CASE "
                "WHEN execution_state='running' THEN 'recovering' ELSE execution_state END,"
                "owner_epoch=owner_epoch+1,owner_token=?,"
                "owner_node_id=?,owner_boot_id=?,owner_lease_until_ms=? "
                "WHERE session_id=? AND execution_state IN ('running','recovering') "
                "AND (owner_token IS NULL OR owner_lease_until_ms<=?) "
                "AND EXISTS (SELECT 1 FROM " + self.store.t.sessions + " s "
                "WHERE s.session_id=" + self.table + ".session_id AND s.lifecycle='open')",
                (lease.token, node_id, boot_id, deadline, key[1], now),
            )
            if not changed:
                return None
            row = await tx.fetchone(
                f"SELECT owner_epoch,pending_json FROM {self.table} WHERE session_id=?", (key[1],)
            )
        lease.epoch = int(row["owner_epoch"])
        lease.pending_json = row["pending_json"]
        return lease

    async def renew(self, lease, *, ttl_s):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        started = time.monotonic()
        if lease.lost or lease.released or lease.expires_at <= started:
            lease.lose()
            return False
        from power_loop.agent.runtime_store import RuntimeInputStore
        now = await RuntimeInputStore(self.store)._db_now(self.store._db)
        changed = await self.store._db.execute(
            f"UPDATE {self.table} SET owner_lease_until_ms=? "
            "WHERE session_id=? AND owner_epoch=? AND owner_token=? "
            "AND execution_state IN ('running','recovering') AND owner_lease_until_ms>? "
            "AND EXISTS (SELECT 1 FROM " + self.store.t.sessions + " s "
            "WHERE s.session_id=" + self.table + ".session_id AND s.lifecycle='open')",
            (now + int(ttl_s * 1000), lease.key[1], lease.epoch, lease.token, now),
        )
        if not changed or lease.lost or lease.released:
            lease.lose()
            return False
        lease.expires_at = started + ttl_s
        return True

    async def release(self, lease):
        from power_loop.agent.runtime_types import RuntimeRequestError

        row = await self.store._db.fetchone(
            f"SELECT execution_state,owner_epoch,owner_token FROM {self.table} WHERE session_id=?",
            (lease.key[1],),
        )
        if row and row["owner_token"] == lease.token and row["owner_epoch"] == lease.epoch:
            if row["execution_state"] in {"running", "recovering"}:
                raise RuntimeRequestError("invalid_options", "final or recover before release")
        await self.store._db.execute(
            f"UPDATE {self.table} SET owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,"
            "owner_lease_until_ms=NULL WHERE session_id=? AND owner_epoch=? AND owner_token=? "
            "AND execution_state IN ('idle','queued')",
            (lease.key[1], lease.epoch, lease.token),
        )
        lease.released = True
