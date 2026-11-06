"""A SQLite coordinator for real, same-host multiprocess contract tests.

This test adapter uses its own database, never the application's SDK tables.
The namespace is bounded by the test fixture; release deletes its row and an
expired row is replaced on acquire. Hosts implement the same token contract
with their chosen shared service.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing, contextmanager

from power_loop.runtime.session_ownership import SessionOwnership, SessionOwnershipCoordinator


class DatabaseSessionOwnershipCoordinator(SessionOwnershipCoordinator):
    scope = "shared"

    def __init__(self, path):
        self.path = str(path)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS leases (key TEXT PRIMARY KEY, token TEXT NOT NULL, expires_at REAL NOT NULL)")

    @contextmanager
    def _transaction(self):
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    async def acquire(self, key, *, owner_id, ttl_s, on_lost):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        started = time.monotonic()
        lease = SessionOwnership(key, owner_id, started + ttl_s, on_lost)
        with self._transaction() as db:
            changed = db.execute(
                "INSERT INTO leases VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "token=excluded.token,expires_at=excluded.expires_at WHERE leases.expires_at<=?",
                (json.dumps(key), lease.token, lease.expires_at, started)).rowcount
        return lease if changed else None

    async def renew(self, lease, *, ttl_s):
        started = time.monotonic()
        if lease.lost or lease.released or lease.expires_at <= started:
            lease.lose()
            return False
        with self._transaction() as db:
            changed = db.execute("UPDATE leases SET expires_at=? WHERE key=? AND token=? AND expires_at>?",
                                 (started + ttl_s, json.dumps(lease.key), lease.token, started)).rowcount
        if not changed or lease.lost or lease.released:
            lease.lose()
            return False
        lease.expires_at = started + ttl_s
        return True

    async def release(self, lease):
        with self._transaction() as db:
            db.execute("DELETE FROM leases WHERE key=? AND token=?", (json.dumps(lease.key), lease.token))
        lease.released = True
