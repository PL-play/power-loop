"""The lease under REAL server concurrency (PostgreSQL).

Why a separate PG file: SQLite serializes writers, so it silently satisfies these properties even
when the SQL is wrong. The first cut of ``drain_follow_up_queue`` did a plain SELECT-then-DELETE —
green on SQLite, but under PostgreSQL's READ COMMITTED two concurrent drains both read the same
rows and each delivered the same steering to its model. Only a server backend catches that.

Gated on a reachable server (skips otherwise), mirroring test_store_parity_pg.
"""

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse

import pytest

pytest.importorskip("asyncpg")

from power_loop.runtime.store.factory import open_store  # noqa: E402

PG_DSN = os.environ.get(
    "POWER_LOOP_TEST_PG_DSN",
    "postgresql://deeptalk:deeptalk@localhost:5433/power_loop_test",
)


def _reachable(dsn: str) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 5432), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _reachable(PG_DSN), reason=f"Postgres not reachable at {PG_DSN}"),
]

N = 8


@pytest.fixture
async def stores():
    """N INDEPENDENT stores = N connection pools, standing in for N agent processes. A single
    store would share one pool and mask exactly the races under test."""
    pools = [await open_store(PG_DSN, table_prefix="pltest_") for _ in range(N)]
    await pools[0]._db.execute(
        f"TRUNCATE {pools[0].t.session_leases}, {pools[0].t.follow_up_queue}"
    )
    try:
        yield pools
    finally:
        for s in pools:
            await s.close()


@pytest.mark.asyncio
async def test_only_one_process_wins_a_free_session(stores) -> None:
    sid = await stores[0].create_session(system_prompt="S")
    won = await asyncio.gather(
        *(s.acquire_session_lease(sid, owner_id=f"p{i}", ttl_ms=60_000)
          for i, s in enumerate(stores))
    )
    assert sum(won) == 1, f"{sum(won)} processes think they own the session"

    await stores[0].close_session(sid)


@pytest.mark.asyncio
async def test_only_the_holder_can_renew(stores) -> None:
    sid = await stores[0].create_session(system_prompt="S")
    await asyncio.gather(
        *(s.acquire_session_lease(sid, owner_id=f"p{i}", ttl_ms=60_000)
          for i, s in enumerate(stores))
    )
    renewed = await asyncio.gather(
        *(s.renew_session_lease(sid, owner_id=f"p{i}", ttl_ms=60_000)
          for i, s in enumerate(stores))
    )
    assert sum(renewed) == 1

    await stores[0].close_session(sid)


@pytest.mark.asyncio
async def test_concurrent_drains_deliver_each_item_exactly_once(stores) -> None:
    """The regression that SQLite could not see: steering must never be replayed."""
    sid = await stores[0].create_session(system_prompt="S")
    for i in range(50):
        await stores[0].enqueue_follow_up(sid, f"m{i}")

    batches = await asyncio.gather(*(s.drain_follow_up_queue(sid) for s in stores))
    seen = [item for batch in batches for item in batch]

    assert len(seen) == len(set(seen)), "an item was delivered to more than one drain"
    assert sorted(seen) == sorted(f"m{i}" for i in range(50)), "items lost or duplicated"

    await stores[0].close_session(sid)


# ── end-to-end: the loop honors the lease ───────────────────────────────────────────────
#
# NOTE on simulating "another process": the in-process session lock is now process-GLOBAL, so a
# second loop object in this interpreter would block on that lock and never reach the lease. To
# exercise the lease itself, the foreign holder is represented by a lease row written directly
# through the store under a different owner_id — which is exactly what another process looks like
# from here.


class _GateLLM:
    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.release = asyncio.Event()
        self.calls: list[list[dict]] = []

    async def complete(self, request, **kwargs):
        from power_loop._vendor.llm_client.interface import LLMResponse

        self.calls.append(list(request.messages))
        if len(self.calls) == 1:
            await self.release.wait()
        return LLMResponse(raw_text=self.text)

    def stream(self, request):
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _loop(store, llm):
    from power_loop import AgentLoopConfig, StatefulAgentLoop

    return StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=3, compactor=None,
            distributed_sessions=True, session_lease_ttl_s=30.0,
        ),
    )


@pytest.mark.asyncio
async def test_send_refuses_a_session_held_by_another_process(stores) -> None:
    """The conv-119 failure mode across processes: no second run over a held session."""
    from power_loop import SessionBusy

    llm = _GateLLM()
    loop = _loop(stores[0], llm)
    sid = await loop.new_session()
    await stores[1].acquire_session_lease(sid, owner_id="other-process", ttl_ms=30_000)

    with pytest.raises(SessionBusy):
        await loop.send("competing", sid)
    assert not llm.calls, "the loop ran despite the lease"

    await stores[1].release_session_lease(sid, owner_id="other-process")
    await loop.aclose()
    await stores[0].close_session(sid)


@pytest.mark.asyncio
async def test_follow_up_parks_steering_for_the_remote_holder(stores) -> None:
    """Losing the race must not lose the message, and must not start a competing run."""
    llm = _GateLLM()
    loop = _loop(stores[0], llm)
    sid = await loop.new_session()
    await stores[1].acquire_session_lease(sid, owner_id="other-process", ttl_ms=30_000)

    queued = await loop.follow_up("steer me", sid)
    assert queued.__class__.__name__ == "FollowUpQueued"
    assert not llm.calls, "follow_up started a run over a held session"
    assert await stores[0].pending_follow_up_depth(sid) == 1

    await stores[1].release_session_lease(sid, owner_id="other-process")
    await loop.aclose()
    await stores[0].close_session(sid)


@pytest.mark.asyncio
async def test_holder_drains_steering_parked_by_another_process(stores) -> None:
    """The other half of folding: what a remote process parked gets picked up by the running
    holder at its next round boundary, and reaches history exactly once."""
    llm = _GateLLM()
    loop = _loop(stores[0], llm)
    sid = await loop.new_session()

    run = asyncio.create_task(loop.send("go", sid))
    for _ in range(400):
        if llm.calls:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("the run never reached its LLM")

    # A different process parks steering while this one holds the lease.
    await stores[1].enqueue_follow_up(sid, "steer from elsewhere")
    llm.release.set()
    await run

    msgs = await loop.get_messages(sid)
    hits = [m for m in msgs if "steer from elsewhere" in str(m.get("content") or "")]
    assert len(hits) == 1, f"steering reached history {len(hits)} times"
    assert await stores[0].pending_follow_up_depth(sid) == 0

    await loop.aclose()
    await stores[0].close_session(sid)
