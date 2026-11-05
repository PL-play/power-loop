"""Ownership lifetime and next_seq reject stale message writers."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop import AgentLoopConfig, SessionLeaseLost, StatefulAgentLoop

from .test_job_registry_192 import registry_store
from .test_round_change_signals import Model

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

async def test_detached_child_cannot_append_after_its_lease_scope_finishes(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(compactor=None))
    release = asyncio.Event()

    async def late():
        await release.wait()
        await store.append_message(sid, role="assistant", content="late")

    try:
        async with loop._session_lease(sid):
            await store.get_state(sid)
            child = asyncio.create_task(late())
        release.set()
        with pytest.raises(SessionLeaseLost):
            await child
        assert not await store.load_all_messages(sid)
    finally:
        release.set()
        await loop.aclose(drain_timeout_s=.1)


async def test_old_child_cannot_reuse_the_same_loops_next_lease_or_consume_its_round(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(compactor=None))
    release = asyncio.Event()

    async def late():
        await release.wait()
        await store.append_message(sid, role="assistant", content="stale")

    try:
        async with loop._session_lease(sid):
            await store.get_state(sid)
            child = asyncio.create_task(late())
        async with loop._session_lease(sid):
            await store.get_state(sid)
            store.defer_round_index(sid, 7)
            release.set()
            with pytest.raises(SessionLeaseLost):
                await child
            await store.append_message(sid, role="assistant", content="current")
            assert (await store.get_state(sid)).round_index == 7
            assert [row.content for row in await store.load_all_messages(sid)] == ["current"]
    finally:
        release.set()
        await loop.aclose(drain_timeout_s=.1)


async def test_writer_waiting_for_a_transaction_cannot_restore_an_expired_turns_round(registry_store, monkeypatch):
    store, _ = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(compactor=None))
    entered, release = asyncio.Event(), asyncio.Event()
    transaction = store._db.transaction
    child = None
    delayed = False

    @asynccontextmanager
    async def blocked():
        nonlocal delayed
        if asyncio.current_task() is child and not delayed:
            delayed = True
            entered.set()
            await release.wait()
        async with transaction() as tx:
            yield tx

    async def late():
        await store.append_message(sid, role="assistant", content="stale")

    monkeypatch.setattr(store._db, "transaction", blocked)
    try:
        async with loop._session_lease(sid):
            await store.get_state(sid)
            store.defer_round_index(sid, 5)
            child = asyncio.create_task(late())
            await asyncio.wait_for(entered.wait(), 5)
        async with loop._session_lease(sid):
            await store.get_state(sid)
            store.defer_round_index(sid, 7)
            await store.append_message(sid, role="assistant", content="current")
            release.set()
            with pytest.raises(SessionLeaseLost):
                await child
            assert (await store.get_state(sid)).round_index == 7
            assert [row.content for row in await store.load_all_messages(sid)] == ["current"]
    finally:
        release.set()
        if child is not None:
            await asyncio.gather(child, return_exceptions=True)
        await loop.aclose(drain_timeout_s=.1)



@pytest.mark.parametrize("operation", ["append", "pending", "round"])
async def test_next_seq_rejects_a_writer_after_replacement_committed(registry_store, monkeypatch, operation):
    store, connect = registry_store
    sid = await store.create_session()
    other = await connect(schema="compatible")
    loop = StatefulAgentLoop(llm=Model(), store=store, config=AgentLoopConfig(compactor=None))
    transaction = store._db.transaction
    injected = False

    @asynccontextmanager
    async def replaced():
        nonlocal injected
        if not injected:
            injected = True
            await other.append_message(sid, role="user", content="replacement")
        async with transaction() as tx:
            yield tx

    try:
        async with loop._session_lease(sid):
            await store.get_state(sid)
            monkeypatch.setattr(store._db, "transaction", replaced)
            action = {"append": lambda: store.append_message(sid, role="assistant", content="stale"),
                      "pending": lambda: store.set_pending(sid, None),
                      "round": lambda: store.set_round_index(sid, 999)}[operation]
            with pytest.raises(SessionLeaseLost):
                await action()
        assert [row.content for row in await other.load_all_messages(sid)] == ["replacement"]
        assert (await other.get_state(sid)).round_index != 999
    finally:
        await loop.aclose(drain_timeout_s=.1)
