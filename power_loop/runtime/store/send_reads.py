"""Entry reads owned by one send, refreshed by its committed writes."""
from __future__ import annotations

from contextvars import ContextVar

ACTIVE = ContextVar("power_loop_send_reads", default=None)


class SendReads:
    def __init__(self, store, sid):
        self.store, self.sid = store, sid
        self.active = True
        self.rows, self.runtime = {}, {}
        self.assistant_send_index = None
        self.assistant_ref = None
        self.lease_owner_id = None


def current(store, sid):
    value = ACTIVE.get()
    return value if value is not None and value.active and value.store is store and value.sid == sid else None


def trusted(store, sid):
    from ..jobs import TRUST_OWN_WRITES
    return current(store, sid) if TRUST_OWN_WRITES.get() else None


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
