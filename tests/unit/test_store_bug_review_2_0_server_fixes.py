"""Real-server regression tests for the BUG_REVIEW_2.0 fixes that CANNOT manifest on the
single-writer SQLite backend: the per-session id-allocation race (G9), concurrent first-boot
provisioning (G14) on PostgreSQL, and the MySQL parameterless-``%`` collapse (G18).

Gated on the same reachable test servers as the parity suites; skips cleanly otherwise.
"""

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse

import pytest

from power_loop.runtime.store.schema import _STORE_TABLES
from power_loop.runtime.store.task_schema import TASK_TABLES

pytestmark = pytest.mark.unit

PG_DSN = os.environ.get(
    "POWER_LOOP_TEST_PG_DSN",
    f"postgresql://deeptalk:{os.environ.get('POSTGRES_PASSWORD', 'deeptalk')}@localhost:5433/power_loop_test",
)
MYSQL_DSN = os.environ.get(
    "POWER_LOOP_TEST_MYSQL_DSN",
    "mysql://deeptalk:deeptalk@localhost:3307/power_loop_test",
)


_ALL_TABLES = tuple(dict.fromkeys((*_STORE_TABLES, *(n for n in TASK_TABLES if n != 'task_store'))))


def _reachable(dsn: str, default_port: int) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or default_port), timeout=2):
            return True
    except OSError:
        return False


_pg = pytest.mark.skipif(not _reachable(PG_DSN, 5432), reason=f"Postgres not reachable at {PG_DSN}")
_my = pytest.mark.skipif(not _reachable(MYSQL_DSN, 3306), reason=f"MySQL not reachable at {MYSQL_DSN}")


# ── G9: concurrent per-session id allocation must not collide on PG ──────────


@_pg
async def test_pg_concurrent_create_timer_no_pk_collision():
    pytest.importorskip("asyncpg")
    from power_loop.runtime.store.factory import open_store

    store = await open_store(PG_DSN)
    try:
        tables = ",".join(store.table_prefix + n for n in _ALL_TABLES if n != "schema_migrations")
        await store._db.execute(f"TRUNCATE {tables}")
        sid = await store.create_session(session_id="race_timers")
        # Without the session_state FOR UPDATE lock, concurrent allocators read the same
        # MAX(timer_id) and the 2nd INSERT raises UniqueViolation on the (session_id, timer_id) PK.
        n = 12
        timers = await asyncio.gather(
            *(store.create_timer(sid, due_at=i, note=f"t{i}") for i in range(n))
        )
        ids = sorted(t.timer_id for t in timers)
        assert ids == list(range(1, n + 1))  # all unique, gap-free
    finally:
        await store.close()


@_pg
async def test_pg_concurrent_add_note_no_pk_collision():
    pytest.importorskip("asyncpg")
    from power_loop.runtime.store.factory import open_store

    store = await open_store(PG_DSN)
    try:
        tables = ",".join(store.table_prefix + n for n in _ALL_TABLES if n != "schema_migrations")
        await store._db.execute(f"TRUNCATE {tables}")
        sid = await store.create_session(session_id="race_notes")
        n = 12
        notes = await asyncio.gather(*(store.add_note(sid, f"n{i}") for i in range(n)))
        ids = sorted(x.note_id for x in notes)
        assert ids == list(range(1, n + 1))
    finally:
        await store.close()


# ── G14: concurrent first-boot AUTO_CREATE converges (no raw duplicate-object) ──


@_pg
async def test_pg_concurrent_first_boot_provisioning_converges():
    pytest.importorskip("asyncpg")
    import secrets

    from power_loop.runtime.store.factory import open_store
    assert urlparse(PG_DSN).path.endswith('_test')
    prefix = 'firstboot_' + secrets.token_hex(6) + '_'
    stores = await asyncio.gather(*(open_store(PG_DSN, table_prefix=prefix) for _ in range(8)))
    try:
        assert all(s.schema_version == 16 for s in stores)
        assert len({s.store_id for s in stores}) == 1
        # The schema is usable and stamped exactly once (idempotent stamp).
        vtable = f"{stores[0].table_prefix}schema_migrations"
        row = await stores[0]._db.fetchone(f"SELECT COUNT(*) AS c FROM {vtable}")
        assert row["c"] == 1
    finally:
        await stores[0]._db.execute(f"DROP VIEW IF EXISTS {prefix}usage_rounds")
        for name in (*TASK_TABLES, *_STORE_TABLES, 'schema_migrations'):
            await stores[0]._db.execute(f'DROP TABLE IF EXISTS {prefix}{name}')
        await asyncio.gather(*(s.close() for s in stores))


# ── G18: a parameterless statement with a literal % must not leak %% on MySQL ──


@_my
async def test_mysql_parameterless_percent_literal_not_doubled():
    pytest.importorskip("aiomysql")
    from power_loop.runtime.store.factory import open_store

    store = await open_store(MYSQL_DSN)
    try:
        # No bound params + a literal '%'. dialect.translate doubles it to '%%'; the driver's
        # `query % args` pass (now always run, because _args returns a tuple) collapses it back.
        # Before the fix the pass was skipped for parameterless SQL and '%%' leaked verbatim.
        row = await store._db.fetchone("SELECT '100%' AS v")
        assert row["v"] == "100%"
        row2 = await store._db.fetchone("SELECT 'a%b%c' AS v")
        assert row2["v"] == "a%b%c"
    finally:
        await store.close()
