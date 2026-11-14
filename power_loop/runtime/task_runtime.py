"""The in-memory execution context of one session send."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class SessionSend:
    """The session-native execution identity of one running send (design 215).

    No job, generation, lease upgrade or supervision: the claim transaction already
    installed the owner fence, and ``final`` writes the result message plus the input
    terminal states. The loop only appends messages and advances this in-memory record.
    """

    store: object
    target: object
    batch: object
    ownership: object
    stop_renewal: asyncio.Event = field(default_factory=asyncio.Event, compare=False, repr=False)
    consumed_inputs: list = field(default_factory=list, compare=False, repr=False)
    usage_totals: dict = field(default_factory=dict, compare=False, repr=False)
    error_code: str | None = field(default=None, compare=False, repr=False)
    resource_evidence: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def session_id(self):
        return self.target.session_id

    @property
    def send_index(self):
        return self.batch.send_index


_SESSION_SEND: ContextVar[SessionSend | None] = ContextVar("power_loop_session_send", default=None)


def current_session_send(store=None, session_id=None) -> SessionSend | None:
    send = _SESSION_SEND.get()
    if send is None:
        return None
    if store is not None and send.store is not store:
        return None
    if session_id is not None and send.session_id != session_id:
        return None
    return send


def mark_unconfirmed_resource():
    send = current_session_send()
    if send is not None:
        send.resource_evidence["resource_state"] = "unknown"


def preserve_model_usage(session_id, usage):
    send = current_session_send(session_id=session_id)
    if send is not None:
        send.usage_totals = dict(usage)


def atomic_pending_required(store, sid):
    return current_session_send(store, sid) is not None
