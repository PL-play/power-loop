"""Entry reads owned by one send, refreshed by its committed writes."""
from __future__ import annotations

from contextvars import ContextVar

from power_loop.contracts.errors import SessionLeaseLost

ACTIVE = ContextVar("power_loop_send_reads", default=None)


class SessionHolding:
    """One holding period's mutable state and immutable lease identity."""

    def __init__(self, store, sid, *, lease_owner_id=None):
        self.store, self.sid = store, sid
        self.active = True
        self.rows, self.runtime = {}, {}
        self.assistant_send_index = None
        self.assistant_ref = None
        self.lease_owner_id = lease_owner_id
        self.next_seq = None
        self.pending_known, self.pending_json = False, None
        self.deferred_round = None
        self.touched = 0
        self.change_snapshot = None
        self.appended_call = None

    def retire(self):
        self.active = False
        self.rows.clear()
        self.runtime.clear()
        self.assistant_send_index = self.assistant_ref = None
        self.next_seq = self.deferred_round = None
        self.pending_known, self.pending_json = False, None
        self.touched, self.change_snapshot, self.appended_call = 0, None, None


def current(store, sid):
    value = ACTIVE.get()
    return value if value is not None and value.active and value.store is store and value.sid == sid else None


def trusted(store, sid):
    from ..jobs import TRUST_OWN_WRITES
    return current(store, sid) if TRUST_OWN_WRITES.get() else None


def lease_writer(store, sid):
    """An inherited old turn never borrows a new period with the same owner ID."""
    value = ACTIVE.get()
    if value is None or value.store is not store or value.sid != sid or value.lease_owner_id is None:
        return None
    if not value.active:
        raise SessionLeaseLost(sid)
    return value


def forget(store, sid, *keys):
    value = current(store, sid)
    if value is not None:
        for key in keys:
            value.rows.pop(key, None)


def take_state(store, sid):
    value = current(store, sid)
    return value.rows.pop("state", None) if value is not None else None


def remember_state(store, sid, row):
    value = current(store, sid)
    if value is not None and row is not None:
        value.rows["state"] = row
