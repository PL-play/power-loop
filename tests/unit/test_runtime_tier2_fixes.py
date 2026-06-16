"""Regression tests for Tier-2/3 runtime fixes:

- C24: SqliteBlackboard._locks is weak-valued, so per-board_id locks don't accumulate
  forever in a long-lived process driving many distinct board ids.
- C16: LLMRetryPolicy.total_timeout bounds an IN-FLIGHT call, not just the gap between
  attempts — a hung call can't blow past the wall-clock budget.
- C18: close_session(cascade) clears EVERY session-keyed table (guards against a future
  per-session table being forgotten in _delete_session_tree → orphaned rows).
"""

from __future__ import annotations

import asyncio
import gc
import time

import pytest

from power_loop import CancellationToken, LLMRetryPolicy, SessionStore
from power_loop.contracts.errors import LLMTimeout
from power_loop.runtime.blackboard import SqliteBlackboard
from power_loop.runtime.retry import with_retry

pytestmark = pytest.mark.unit


# ── C24: blackboard locks are weakly bounded ─────────────────────────────────


def test_blackboard_locks_do_not_accumulate() -> None:
    bb = SqliteBlackboard(store=None)  # _lock() does not touch the store

    # same id returns the same lock while it is referenced (mutual exclusion holds)
    held = bb._lock("x")
    assert bb._lock("x") is held

    # many distinct, unheld board ids must NOT pile up forever
    for i in range(500):
        bb._lock(f"board-{i}")  # result not retained → collectible
    del held
    gc.collect()
    assert len(bb._locks) < 50  # was ~500 (an unbounded plain dict)


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


def _dummy(coltype: str):
    t = (coltype or "").upper()
    if "INT" in t:
        return 0
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return 0.0
    return "x"


def test_close_session_clears_every_session_keyed_table() -> None:
    store = SessionStore.open(":memory:")
    try:
        conn = store._conn
        all_tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        session_tables = []
        for t in all_tables:
            cols = conn.execute(f"PRAGMA table_info({t})").fetchall()  # noqa: S608 (schema names)
            if any(c[1] == "session_id" for c in cols):
                session_tables.append((t, cols))
        assert {t for t, _ in session_tables}  # sanity: there ARE session-keyed tables

        sid = store.create_session()  # also creates a session_state row
        for t, cols in session_tables:
            have = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE session_id=?", (sid,)).fetchone()[0]  # noqa: S608
            if have:
                continue
            if t == "messages":
                store.append_message(sid, role="user", content="m", round_index=0)
                continue
            names, vals = ["session_id"], [sid]
            for _cid, name, ctype, notnull, dflt, _pk in cols:
                if name == "session_id":
                    continue
                if notnull and dflt is None:
                    names.append(name)
                    vals.append(_dummy(ctype))
            ph = ",".join("?" * len(names))
            with store._lock, conn:
                conn.execute(f"INSERT INTO {t} ({','.join(names)}) VALUES ({ph})", vals)  # noqa: S608

        # every session-keyed table is seeded for sid
        for t, _ in session_tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE session_id=?", (sid,)).fetchone()[0]  # noqa: S608
            assert n > 0, f"failed to seed {t}"

        store.close_session(sid, cascade=True)

        # close_session must have cleared EVERY session-keyed table
        for t, _ in session_tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE session_id=?", (sid,)).fetchone()[0]  # noqa: S608
            assert n == 0, f"close_session left rows in {t} — add it to _delete_session_tree"
    finally:
        store.close()
