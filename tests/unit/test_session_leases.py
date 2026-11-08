"""Durable follow-up inbox operations; ownership contracts are tested separately."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import SessionStore


@pytest.fixture
async def store():
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


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
async def test_inbox_release_returns_pending_but_age_never_steals_a_claim(store: SessionStore, monkeypatch) -> None:
    sid = await store.create_session(system_prompt="S")
    await store.inbox_put(sid, _items("a", "b"))
    c1 = await store.inbox_claim(sid)
    assert await store.inbox_release(sid, c1[0]["claim_token"]) == 2
    c2 = await store.inbox_claim(sid)
    assert [r["item_id"] for r in c2] == ["a", "b"]
    # Session ownership controls recovery; elapsed time alone is not authority.
    from power_loop.runtime.store import store as module
    monkeypatch.setattr(module, '_now_ms', lambda: 2**60)
    assert await store.inbox_claim(sid) == []
    assert await store.inbox_release(sid,c2[0]['claim_token']) == 2
    c3 = await store.inbox_claim(sid)
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
async def test_close_session_clears_queue(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    await store.inbox_put(sid, _items("steer"))

    from power_loop import SessionBusy
    with pytest.raises(SessionBusy):
        await store.close_session(sid)
    assert await store.inbox_void(sid,["steer"]) == 1
    await store.close_session(sid)
    assert (await store.inbox_counts(sid))["pending"] == 0
