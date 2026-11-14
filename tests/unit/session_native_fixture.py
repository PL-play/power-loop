"""Small schema-17 fixtures shared only by the P1 storage tests."""
from __future__ import annotations

from contextlib import asynccontextmanager

from power_loop.agent.runtime_types import RuntimeDescriptor, TrustedActor

ACTOR = TrustedActor("tenant", "host", None)
DESCRIPTOR = RuntimeDescriptor(
    2, "agent", "test.agent@215", None, "test.tools@215", "test.hooks@215",
    "test.resources@215", {"max_rounds": 4}, {"tenant_id": "tenant"},
)


async def target(port, session_id="session", **kwargs):
    return await port.create_session(
        DESCRIPTOR, actor=ACTOR, options={"session_id": session_id}, **kwargs
    )


async def new_session(store, session_id=None):
    from power_loop.agent.runtime_store import RuntimeInputStore

    options = {"session_id": session_id} if session_id is not None else {}
    return await RuntimeInputStore(store).create_session(DESCRIPTOR, actor=ACTOR, options=options)


class UnknownCommit(Exception):
    pass


def fail_next_commit(monkeypatch, store):
    """Commit the real transaction, then report an unknown outcome exactly once."""
    original = type(store._db).transaction
    enabled = True

    @asynccontextmanager
    async def transaction(db):
        nonlocal enabled
        if db is store._db and enabled:
            enabled = False
            async with original(db) as tx:
                yield tx
            raise UnknownCommit()
        else:
            async with original(db) as tx:
                yield tx

    monkeypatch.setattr(type(store._db), "transaction", transaction)


async def fixture(store):
    from power_loop import RuntimeDescriptor, TrustedActor
    from power_loop.agent.runtime_store import RuntimeInputStore
    descriptor = RuntimeDescriptor(2, "agent", "test.agent@215", None, "builtin.empty@215",
                                   "builtin.none@215", "builtin.none@215", {"max_rounds": 2}, {})
    actor = TrustedActor("tenant", "principal", None)
    port = RuntimeInputStore(store)
    return port, await port.create_session(descriptor, actor=actor), actor
