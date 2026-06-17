"""G2 regression: the blocking sync API must drive every call on ONE persistent event
loop, so an owned PG/MySQL connection pool (bound to the loop it was created on) stays
valid across calls.

Pre-fix, ``send_sync`` / ``follow_up_sync`` / ``close`` each spun a fresh ``asyncio.run``;
the first call opened (and then closed) a loop the pool was bound to, so the SECOND sync
call found a dead-loop pool and raised. SQLite is loop-agnostic (so it never surfaced the
bug); the PG test below is the real regression and is gated on a reachable server.
"""

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse

import pytest

from power_loop.agent.stateful_loop import StatefulAgentLoop
from power_loop.runtime.stub_provider import EchoLLMService

pytestmark = pytest.mark.unit


def test_repeated_sync_calls_share_one_persistent_loop(tmp_path) -> None:
    loop = StatefulAgentLoop(llm=EchoLLMService(), db_path=str(tmp_path / "s.db"))
    try:
        s1 = loop.new_session_sync()
        s2 = loop.new_session_sync()  # 2nd sync entry — used to fail on PG/MySQL
        s3 = loop.new_session_sync()
        assert len({s1, s2, s3}) == 3
        # all three ran on the SAME dedicated loop
        runner = loop._sync_runner
        assert runner is not None and not runner._loop.is_closed()
    finally:
        loop.close()
    # close tears the dedicated loop down and releases the store
    assert loop._sync_runner is None
    assert loop.store is None


async def test_sync_api_rejected_inside_running_loop(tmp_path) -> None:
    loop = StatefulAgentLoop(llm=EchoLLMService(), db_path=str(tmp_path / "s.db"))
    try:
        with pytest.raises(RuntimeError, match="within a running event loop"):
            loop.send_sync("hi", "sess_x")
    finally:
        await loop.aclose()


# ── the real regression: a server backend whose pool is event-loop-bound ──────────────
PG_DSN = os.environ.get(
    "POWER_LOOP_TEST_PG_DSN",
    "postgresql://deeptalk:deeptalk@localhost:5433/power_loop_test",
)


def _reachable(dsn: str, default_port: int) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or default_port), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _reachable(PG_DSN, 5432), reason=f"Postgres not reachable at {PG_DSN}")
def test_repeated_sync_calls_on_postgres(tmp_path) -> None:
    pytest.importorskip("asyncpg")
    loop = StatefulAgentLoop(llm=EchoLLMService(), dsn=PG_DSN, table_prefix="plsync_")
    try:
        ids = [loop.new_session_sync() for _ in range(3)]  # pre-fix: 2nd call raised
        assert len(set(ids)) == 3
    finally:
        loop.close()  # closing a pool bound to the dedicated loop must also work (R02)


def test_old_style_asyncio_run_reuse_would_fail_on_postgres() -> None:
    """Document/verify the underlying failure mode the fix avoids: reusing one store across
    fresh asyncio.run loops (what the old sync API did) breaks on a pooled server backend."""
    if not _reachable(PG_DSN, 5432):
        pytest.skip(f"Postgres not reachable at {PG_DSN}")
    pytest.importorskip("asyncpg")
    from power_loop.runtime.store.factory import open_store

    store = asyncio.run(open_store(PG_DSN, table_prefix="plsync_"))  # pool bound to loop #1
    try:
        with pytest.raises(Exception):  # noqa: B017 - asyncpg InterfaceError on a dead-loop pool
            asyncio.run(store.create_session(system_prompt="x"))  # loop #2 reuses dead pool
    finally:
        try:
            asyncio.run(store.close())
        except Exception:
            pass
