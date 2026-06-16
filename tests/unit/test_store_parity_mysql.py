"""MySQL conformance: run the SAME parity scenarios (test_store_parity.SCENARIOS)
against a real MySQL-backed async store, with the legacy SQLite store as the oracle.

Gated on a reachable server (skips otherwise) so CI without MySQL stays green. Points at
``POWER_LOOP_TEST_MYSQL_DSN`` (default: the local docker server's isolated
``power_loop_test`` database) and TRUNCATEs the pl_ tables before each scenario for a
clean slate.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

pytest.importorskip("aiomysql")

from power_loop.runtime.store.factory import open_store  # noqa: E402
from tests.unit.test_store_parity import SCENARIOS, run_parity  # noqa: E402

MYSQL_DSN = os.environ.get(
    "POWER_LOOP_TEST_MYSQL_DSN",
    "mysql://deeptalk:deeptalk@localhost:3307/power_loop_test",
)


def _reachable(dsn: str) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 3306), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _reachable(MYSQL_DSN), reason=f"MySQL not reachable at {MYSQL_DSN}"),
]

_DATA_TABLES = (
    "sessions", "messages", "compactions", "usage_rounds", "session_state",
    "session_runtime_state", "shared_state", "background_tasks", "session_stats",
    "timers", "notes",
)


@pytest.fixture
async def mysql_store():
    store = await open_store(MYSQL_DSN)
    # TRUNCATE each pl_ table for a clean slate per scenario (FK-free schema → order-agnostic).
    for n in _DATA_TABLES:
        await store._db.execute(f"TRUNCATE TABLE {getattr(store.t, n)}")
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.parametrize("name,scn,snap", SCENARIOS, ids=[s[0] for s in SCENARIOS])
async def test_mysql_parity(mysql_store, name: str, scn, snap) -> None:
    await run_parity(scn, snap, new_store=mysql_store)
