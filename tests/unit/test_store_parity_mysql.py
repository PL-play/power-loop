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
    "timers", "notes", "project_messages", "hook_events",
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


async def test_mysql_project_messages(mysql_store) -> None:
    # v2 send-context projection table: DDL/upsert/load conformance on real MySQL.
    from tests.unit.test_project_messages_store import exercise_project_messages_crud

    await exercise_project_messages_crud(mysql_store)


async def test_mysql_export_import_send_index(mysql_store) -> None:
    # send_index (BIGINT on MySQL) round-trips through export/import on real MySQL.
    from tests.unit.test_project_messages_store import exercise_export_import_send_index

    await exercise_export_import_send_index(mysql_store)


async def test_mysql_hook_events(mysql_store) -> None:
    # v3 hook-events audit table: DDL/insert/list/cleanup conformance on real MySQL.
    from tests.unit.test_project_messages_store import exercise_hook_events_crud

    await exercise_hook_events_crud(mysql_store)


async def test_mysql_large_payload(mysql_store) -> None:
    # H2: >64 KiB free-text/JSON round-trips on real MySQL. Pre-fix the TEXT (64 KiB) columns
    # raised DataError(1406) under strict sql_mode; LONGTEXT (DDL + v3→v4 migration) fixes it.
    from tests.unit.test_store_parity import exercise_large_payload

    await exercise_large_payload(mysql_store)


_MIG_PREFIX = "plmig_"  # isolated prefix so the migration test never collides with the real pl_ schema


async def _content_data_type(db, table: str) -> str:
    row = await db.fetchone(
        "SELECT data_type AS dt FROM information_schema.columns WHERE table_schema=DATABASE() "
        "AND table_name=? AND column_name='content'",
        (table,),
    )
    return str(row["dt"]).lower()


async def test_mysql_v3_to_v4_widens_text_columns() -> None:
    # Provision the full v4 schema, downgrade one column back to TEXT + stamp v3 (simulating a
    # pre-fix store), then REOPEN so ensure_schema runs the real v3→v4 migration. Assert the
    # column becomes LONGTEXT and a >64 KiB write that fails on TEXT now succeeds end-to-end.
    from power_loop.runtime.store.factory import open_store

    prefix, big = _MIG_PREFIX, "A" * 100_000
    msgs = f"{prefix}messages"
    store = await open_store(MYSQL_DSN, table_prefix=prefix)
    try:
        # Simulate a v3 store: revert content TEXT and roll the version back to 3.
        await store._db.execute(f"ALTER TABLE {msgs} MODIFY content TEXT")
        await store._db.execute(f"UPDATE {prefix}schema_migrations SET version=3 WHERE id=1")
        assert await _content_data_type(store._db, msgs) == "text"  # precondition
    finally:
        await store.close()

    store2 = await open_store(MYSQL_DSN, table_prefix=prefix)  # → runs the v3→v4 migration
    try:
        assert store2.schema_version == 4
        assert await _content_data_type(store2._db, msgs) == "longtext"  # widened by migration
        await store2.create_session(session_id="m")
        await store2.append_message("m", role="assistant", content=big, round_index=0)
        rows = await store2.load_active_messages("m")
        assert rows[0].content == big and len(rows[0].content) == 100_000
        # Idempotent: re-running the migration steps on an already-LONGTEXT column is a no-op.
        await store2._db.execute(f"UPDATE {prefix}schema_migrations SET version=3 WHERE id=1")
    finally:
        await store2.close()

    store3 = await open_store(MYSQL_DSN, table_prefix=prefix)  # migration re-runs cleanly
    try:
        assert store3.schema_version == 4
        for n in _DATA_TABLES + ("schema_migrations",):
            await store3._db.execute(f"DROP TABLE IF EXISTS {prefix}{n}")
    finally:
        await store3.close()
