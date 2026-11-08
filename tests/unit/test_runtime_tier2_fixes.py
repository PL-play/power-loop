"""Regression tests for Tier-2/3 runtime fixes:

- C24: Independent blackboard objects atomically share and delete one board.
- C16: LLMRetryPolicy.total_timeout bounds an IN-FLIGHT call, not just the gap between
  attempts — a hung call can't blow past the wall-clock budget.
- C18: close_session(cascade) clears EVERY session-keyed table (guards against a future
  per-session table being forgotten in _delete_session_tree → orphaned rows).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from power_loop import CancellationToken, LLMRetryPolicy, SessionStore, StoreBlackboard
from power_loop.contracts.errors import LLMTimeout
from power_loop.runtime.retry import with_retry

pytestmark = pytest.mark.unit


# ── C24: blackboard locks are weakly bounded ─────────────────────────────────


async def test_blackboard_instances_share_atomic_updates_and_release_scope() -> None:
    store = await SessionStore.open(":memory:")
    try:
        first, second = StoreBlackboard(store), StoreBlackboard(store)
        entries = await asyncio.gather(*(board.post("shared", text=str(i))
            for i, board in enumerate([first, second] * 10)))
        assert len({e.id for e in entries}) == 20
        assert {e.text for e in await second.read("shared")} == {str(i) for i in range(20)}
        await first.delete("shared")
        assert await second.read("shared") == []
    finally:
        await store.close()


# ── C16: total_timeout bounds an in-flight call ──────────────────────────────


async def test_total_timeout_bounds_a_hung_call() -> None:
    policy = LLMRetryPolicy(total_timeout=0.2, max_attempts=3, retry_on=(ValueError,))

    async def hung():
        await asyncio.sleep(10)  # never returns within the budget

    t0 = time.monotonic()
    with pytest.raises(LLMTimeout):
        await with_retry(hung, policy=policy, token=CancellationToken.never())
    assert time.monotonic() - t0 < 2.0  # bounded by total_timeout, not the 10s sleep


async def test_total_timeout_allows_a_fast_call() -> None:
    policy = LLMRetryPolicy(total_timeout=5.0, max_attempts=3, retry_on=(ValueError,))

    async def quick():
        return "ok"

    assert await with_retry(quick, policy=policy, token=CancellationToken.never()) == "ok"


# ── C18: close_session clears every session-keyed table ──────────────────────


async def test_close_session_clears_normal_descendant_resources() -> None:
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session()
        child = await store.create_session(parent_session_id=sid)
        for target in (sid, child):
            await store.append_message(target, role="user", content="retained history")
            await store.add_note(target, "note")
            await store.set_runtime_state(target, "resource", {"value": 1})
            await store.create_timer(target, due_at=2**50, note="future")
        await store.close_session(sid, cascade=True)
        for table in (store.t.sessions, store.t.messages, store.t.notes,
                      store.t.session_runtime_state, store.t.timers):
            assert not await store._db.fetchall(f"SELECT * FROM {table} WHERE session_id IN (?,?)", (sid, child))
    finally:
        await store.close()
