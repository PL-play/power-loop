"""PostgreSQL conformance: run the SAME parity scenarios (test_store_parity.SCENARIOS)
against a real Postgres-backed async store, with the legacy SQLite store as the oracle.

Gated on a reachable server (skips otherwise) so CI without Postgres stays green. Points
at ``POWER_LOOP_TEST_PG_DSN`` (default: the local docker server's isolated
``power_loop_test`` database) and TRUNCATEs the pl_ tables before each scenario for a
clean slate — DeepTalk's own data is never touched (separate DB + ``pl_`` prefix).
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

pytest.importorskip("asyncpg")

from power_loop.runtime.store.factory import open_store  # noqa: E402
from tests.unit.test_store_parity import SCENARIOS, run_parity  # noqa: E402

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

_DATA_TABLES = (
    "sessions", "messages", "compactions", "usage_rounds", "session_state",
    "session_runtime_state", "shared_state", "background_tasks", "session_stats",
    "timers", "notes", "project_messages",
)


@pytest.fixture
async def pg_store():
    store = await open_store(PG_DSN)
    tables = ",".join(getattr(store.t, n) for n in _DATA_TABLES)
    await store._db.execute(f"TRUNCATE {tables}")  # clean slate per scenario
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.parametrize("name,scn,snap", SCENARIOS, ids=[s[0] for s in SCENARIOS])
async def test_pg_parity(pg_store, name: str, scn, snap) -> None:
    await run_parity(scn, snap, new_store=pg_store)


async def test_pg_project_messages(pg_store) -> None:
    # v2 send-context projection table: DDL/upsert/load conformance on real Postgres.
    from tests.unit.test_project_messages_store import exercise_project_messages_crud

    await exercise_project_messages_crud(pg_store)
