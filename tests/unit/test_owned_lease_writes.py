"""Ownership lifetime and next_seq reject stale message writers."""
from __future__ import annotations

import asyncio
import contextvars
import time
from contextlib import asynccontextmanager

import pytest

from power_loop import SessionLeaseLost
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.runtime.session_ownership import _CURRENT, SessionOwnership
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

class ClaimedTurns:
    """Real accept/claim/final; the separate worker tests verify actual binding."""

    def __init__(self, store, node="test-owner"):
        self.store, self.port, self.node = store, RuntimeInputStore(store), node
        self.counter = 0

    @asynccontextmanager
    async def turn(self, sid, *, body="input", finalize=True):
        self.counter += 1
        from power_loop import TargetRef
        root = TargetRef(self.store.store_id, sid)
        await self.port.accept(root, body, message_id=f"{self.node}-{self.counter}", actor=STORE_ACTOR)
        batch = await self.port.claim(root, node_id=self.node, boot_id=self.node)
        assert batch is not None
        lease = SessionOwnership(
            (self.store.store_id, sid), self.node, time.monotonic() + 90, lambda: None,
            epoch=batch.owner_epoch, token=batch.owner_token,
            node_id=self.node, boot_id=self.node, installs_fence=True,
        )
        token = _CURRENT.set(lease)
        try:
            async with self.store.send_read_scope(sid, ownership=lease):
                yield lease
                if finalize:
                    await self.port.final(batch, final_text="done")
        finally:
            lease.released = True
            _CURRENT.reset(token)


async def assistant_contents(store, sid):
    return [row.content for row in await store.load_all_messages(sid) if row.role == "assistant"]


async def test_detached_child_cannot_append_after_its_lease_scope_finishes(registry_store):
    store, _ = registry_store
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    turns = ClaimedTurns(store)
    release = asyncio.Event()

    async def late():
        await release.wait()
        await store.append_message(sid, role="assistant", content="late")

    try:
        async with turns.turn(sid):
            await store.get_state(sid)
            child = asyncio.create_task(late())
        release.set()
        with pytest.raises(SessionLeaseLost):
            await child
        assert await assistant_contents(store, sid) == ["done"]
        assert [row.role for row in await store.load_all_messages(sid)] == ["user", "assistant"]
    finally:
        release.set()
        assert _CURRENT.get() is None


async def test_old_child_cannot_reuse_the_same_loops_next_lease_or_consume_its_round(registry_store):
    store, _ = registry_store
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    turns = ClaimedTurns(store)
    release = asyncio.Event()

    async def late():
        await release.wait()
        await store.append_message(sid, role="assistant", content="stale")

    try:
        async with turns.turn(sid):
            await store.get_state(sid)
            child = asyncio.create_task(late())
        async with turns.turn(sid):
            await store.get_state(sid)
            store.defer_round_index(sid, 7)
            release.set()
            with pytest.raises(SessionLeaseLost):
                await child
            await store.append_message(sid, role="assistant", content="current")
            assert (await store.get_state(sid)).round_index == 7
            assert await assistant_contents(store, sid) == ["done", "current"]
            assert len([row for row in await store.load_all_messages(sid) if row.role == "user"]) == 2
    finally:
        release.set()
        assert _CURRENT.get() is None


async def test_writer_waiting_for_a_transaction_cannot_restore_an_expired_turns_round(registry_store, monkeypatch):
    store, _ = registry_store
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    turns = ClaimedTurns(store)
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
        async with turns.turn(sid):
            await store.get_state(sid)
            store.defer_round_index(sid, 5)
            child = asyncio.create_task(late())
            await asyncio.wait_for(entered.wait(), 5)
        async with turns.turn(sid):
            await store.get_state(sid)
            store.defer_round_index(sid, 7)
            await store.append_message(sid, role="assistant", content="current")
            release.set()
            with pytest.raises(SessionLeaseLost):
                await child
            assert (await store.get_state(sid)).round_index == 7
            assert await assistant_contents(store, sid) == ["done", "current"]
            assert len([row for row in await store.load_all_messages(sid) if row.role == "user"]) == 2
    finally:
        release.set()
        if child is not None:
            await asyncio.gather(child, return_exceptions=True)
        assert _CURRENT.get() is None



@pytest.mark.parametrize("operation", ["append", "pending", "round"])
async def test_owner_epoch_rejects_a_writer_after_replacement_committed(registry_store, monkeypatch, operation):
    store, connect = registry_store
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    other = await connect(schema="verify")
    from power_loop import TargetRef
    root = TargetRef(store.store_id, sid)
    turns = ClaimedTurns(store)
    transaction = store._db.transaction
    injected = False

    @asynccontextmanager
    async def replaced():
        nonlocal injected
        if not injected:
            injected = True
            async def replacement():
                # Only time is injected. Recovery and the next owner installation
                # use the real state transitions on another connection/context.
                await other._db.execute(
                    f"UPDATE {other.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
                    (sid,),
                )
                recovered = await RuntimeInputStore(other).recover(
                    root, node_id="new-owner", boot_id="new-owner")
                assert recovered.execution_state == "idle"
                async with ClaimedTurns(other, "new-owner").turn(sid, body="replacement"):
                    await other.set_round_index(sid, 7)
                    await other.append_message(sid, role="assistant", content="replacement")
            await asyncio.create_task(replacement(), context=contextvars.Context())
        async with transaction() as tx:
            yield tx

    try:
        async with turns.turn(sid, finalize=False) as old_lease:
            await store.get_state(sid)
            monkeypatch.setattr(store._db, "transaction", replaced)
            action = {"append": lambda: store.append_message(sid, role="assistant", content="stale"),
                      "pending": lambda: store.set_pending(sid, None),
                      "round": lambda: store.set_round_index(sid, 999)}[operation]
            with pytest.raises(SessionLeaseLost):
                await action()
            assert not old_lease.lost and not old_lease.released
            assert old_lease.expires_at > time.monotonic()
        assert await assistant_contents(other, sid) == ["replacement", "done"]
        assert len([row for row in await other.load_all_messages(sid) if row.role == "user"]) == 2
        assert (await other.get_state(sid)).round_index == 7
        assert (await other.get_state(sid)).pending is None
    finally:
        assert _CURRENT.get() is None
