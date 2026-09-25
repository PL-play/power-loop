"""Cross-process session leases + the DB-backed follow-up queue (schema v7).

The in-process lock in StatefulAgentLoop only serializes ONE interpreter. With several agent
processes on one database, the lease row is the shared arbiter that stops two of them from driving
the same session at once.
"""

from __future__ import annotations

import asyncio

import pytest

from power_loop import SessionStore


@pytest.fixture
async def store():
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_live_lease_is_never_stolen(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    assert await store.acquire_session_lease(sid, owner_id="A", ttl_ms=60_000)
    assert not await store.acquire_session_lease(sid, owner_id="B", ttl_ms=60_000)

    held = await store.session_lease_holder(sid)
    assert held is not None and held["owner_id"] == "A" and held["fence"] == 1


@pytest.mark.asyncio
async def test_expired_lease_is_stealable_and_bumps_the_fence(store: SessionStore) -> None:
    """A holder that dies stops renewing; the lease must become takeable, and the fence must
    advance so a future fencing check can reject the dead holder's late writes."""
    sid = await store.create_session(system_prompt="S")
    assert await store.acquire_session_lease(sid, owner_id="A", ttl_ms=1)
    await asyncio.sleep(0.02)

    assert await store.session_lease_holder(sid) is None, "expired lease still reported as live"
    assert await store.acquire_session_lease(sid, owner_id="B", ttl_ms=60_000)
    held = await store.session_lease_holder(sid)
    assert held is not None and held["owner_id"] == "B" and held["fence"] == 2


@pytest.mark.asyncio
async def test_renew_fails_once_dispossessed(store: SessionStore) -> None:
    """The signal a stalled holder needs: renew returning False means someone else owns the
    session now and this run must stop rather than interleave with it."""
    sid = await store.create_session(system_prompt="S")
    await store.acquire_session_lease(sid, owner_id="A", ttl_ms=1)
    await asyncio.sleep(0.02)
    await store.acquire_session_lease(sid, owner_id="B", ttl_ms=60_000)

    assert not await store.renew_session_lease(sid, owner_id="A", ttl_ms=60_000)
    assert await store.renew_session_lease(sid, owner_id="B", ttl_ms=60_000)


@pytest.mark.asyncio
async def test_release_is_scoped_to_the_owner(store: SessionStore) -> None:
    """A dispossessed holder releasing on its way out must not delete the NEW holder's lease."""
    sid = await store.create_session(system_prompt="S")
    await store.acquire_session_lease(sid, owner_id="A", ttl_ms=1)
    await asyncio.sleep(0.02)
    await store.acquire_session_lease(sid, owner_id="B", ttl_ms=60_000)

    await store.release_session_lease(sid, owner_id="A")
    held = await store.session_lease_holder(sid)
    assert held is not None and held["owner_id"] == "B"

    await store.release_session_lease(sid, owner_id="B")
    assert await store.session_lease_holder(sid) is None


@pytest.mark.asyncio
async def test_concurrent_acquire_yields_exactly_one_winner(store: SessionStore) -> None:
    """The whole point: N processes racing for a free session, one winner."""
    sid = await store.create_session(system_prompt="S")
    results = await asyncio.gather(
        *(store.acquire_session_lease(sid, owner_id=f"p{i}", ttl_ms=60_000) for i in range(8))
    )
    assert sum(results) == 1, f"expected exactly one winner, got {sum(results)}"


def _items(*texts: str, kind: str = "user", mode: str = "queue") -> list[dict]:
    return [{"item_id": t, "content": t, "kind": kind, "mode": mode} for t in texts]


@pytest.mark.asyncio
async def test_inbox_round_trip(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    assert await store.inbox_put(sid, _items("one", "two")) == ["one", "two"]
    assert (await store.inbox_counts(sid))["pending"] == 2

    claimed = await store.inbox_claim(sid)
    assert [r["content"] for r in claimed] == ["one", "two"]  # oldest first
    assert await store.inbox_claim(sid) == []                 # claimed ≠ claimable again
    # still undelivered until a transcript row marks it
    assert (await store.inbox_counts(sid))["pending"] == 2
    await store.append_message(sid, role="user", content="one\n\ntwo",
                               inbox={"claim_token": claimed[0]["claim_token"],
                                      "ids": [r["id"] for r in claimed]})
    assert (await store.inbox_counts(sid))["pending"] == 0
    rows = await store.inbox_items(sid)
    assert {r["status"] for r in rows} == {"delivered"}
    assert all(r["delivered_seq"] == rows[0]["delivered_seq"] for r in rows)


@pytest.mark.asyncio
async def test_inbox_same_item_id_is_accepted_once_ever(store: SessionStore) -> None:
    """The dedupe key survives delivery: re-sending an item that already reached the model is a
    no-op, not a second copy in the transcript (design/124 Z4)."""
    sid = await store.create_session(system_prompt="S")
    assert await store.inbox_put(sid, _items("m1")) == ["m1"]
    assert await store.inbox_put(sid, _items("m1", "m2")) == ["m2"]
    claimed = await store.inbox_claim(sid)
    await store.append_message(sid, role="user", content="x",
                               inbox={"claim_token": claimed[0]["claim_token"],
                                      "ids": [r["id"] for r in claimed]})
    assert await store.inbox_put(sid, _items("m1", "m2")) == []
    assert (await store.inbox_counts(sid))["pending"] == 0


@pytest.mark.asyncio
async def test_inbox_release_and_stale_claim_return_to_pending(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    await store.inbox_put(sid, _items("a", "b"))
    c1 = await store.inbox_claim(sid)
    assert await store.inbox_release(sid, c1[0]["claim_token"]) == 2
    c2 = await store.inbox_claim(sid)
    assert [r["item_id"] for r in c2] == ["a", "b"]
    # a claim whose claimer died (never delivered) comes back after the stale window
    assert await store.inbox_claim(sid, stale_after_ms=60_000) == []
    c3 = await store.inbox_claim(sid, stale_after_ms=-1)
    assert [r["item_id"] for r in c3] == ["a", "b"]


@pytest.mark.asyncio
async def test_inbox_void_withdraws_only_undelivered(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    await store.inbox_put(sid, _items("keep", "drop"))
    assert await store.inbox_void(sid, ["drop"]) == 1
    claimed = await store.inbox_claim(sid)
    assert [r["item_id"] for r in claimed] == ["keep"]


@pytest.mark.asyncio
async def test_inbox_counts_steer(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    await store.inbox_put(sid, _items("q1") + _items("s1", mode="steer"))
    assert await store.inbox_counts(sid) == {"pending": 2, "steer": 1}


@pytest.mark.asyncio
async def test_concurrent_drains_do_not_double_deliver(store: SessionStore) -> None:
    """Steering must be delivered once — a claim race must not hand the same item to two drains
    (which would replay a user's message into the model twice)."""
    sid = await store.create_session(system_prompt="S")
    await store.inbox_put(sid, _items(*(f"m{i}" for i in range(20))))

    drained = await asyncio.gather(*(store.inbox_claim(sid) for _ in range(4)))
    seen = [r["item_id"] for batch in drained for r in batch]
    assert sorted(seen) == sorted(f"m{i}" for i in range(20))
    assert len(seen) == len(set(seen)), "an item was delivered more than once"


@pytest.mark.asyncio
async def test_close_session_clears_lease_and_queue(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    await store.acquire_session_lease(sid, owner_id="A", ttl_ms=60_000)
    await store.inbox_put(sid, _items("steer"))

    await store.close_session(sid)
    assert await store.session_lease_holder(sid) is None
    assert (await store.inbox_counts(sid))["pending"] == 0
