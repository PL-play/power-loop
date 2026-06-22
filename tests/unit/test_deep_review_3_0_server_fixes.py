"""Real-Postgres regression for deep-review B1: the v1→v2 ``send_index`` ALTER must be scoped to
the CURRENT schema. Pre-fix, ``_column_exists`` probed ``information_schema.columns`` across ALL
schemas, so a same-named ``pl_messages`` with ``send_index`` in another schema made the ALTER get
skipped (then v2 stamped) → every ``append_message`` in the un-migrated schema crashed.

Cannot manifest on SQLite (no schemas). Gated on the PG test server; skips cleanly otherwise.
"""

from __future__ import annotations

import os
import socket
import urllib.parse
from urllib.parse import urlparse

import pytest

from power_loop.runtime.store.factory import open_store

pytestmark = pytest.mark.unit

PG_DSN = os.environ.get(
    "POWER_LOOP_TEST_PG_DSN", "postgresql://deeptalk:deeptalk@localhost:5433/power_loop_test"
)


def _reachable(dsn: str, default_port: int) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or default_port), timeout=2):
            return True
    except OSError:
        return False


_pg = pytest.mark.skipif(not _reachable(PG_DSN, 5432), reason=f"Postgres not reachable at {PG_DSN}")


def _dsn(schema: str) -> str:
    return f"{PG_DSN}?options=" + urllib.parse.quote(f"-csearch_path={schema}")


@_pg
async def test_v1_to_v2_send_index_alter_is_schema_scoped() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    a_schema, b_schema = "pltest_b1_a", "pltest_b1_b"
    raw = await asyncpg.connect(PG_DSN)
    try:
        for s in (a_schema, b_schema):
            await raw.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
            await raw.execute(f"CREATE SCHEMA {s}")
    finally:
        await raw.close()
    try:
        # Two schemas each get a full v2 store (both pl_messages have send_index).
        for s in (a_schema, b_schema):
            st = await open_store(_dsn(s))
            await st.close()
        # Downgrade schema B to a v1-shaped DB: drop send_index + reset its recorded version to 1.
        rb = await asyncpg.connect(_dsn(b_schema))
        try:
            await rb.execute("ALTER TABLE pl_messages DROP COLUMN send_index")
            await rb.execute("UPDATE pl_schema_migrations SET version=1")
        finally:
            await rb.close()
        # Reopen B → the v1→v2 ladder must re-add send_index to B (NOT be fooled by A's column).
        b2 = await open_store(_dsn(b_schema))
        try:
            rb = await asyncpg.connect(_dsn(b_schema))
            try:
                has = await rb.fetchval(
                    "SELECT 1 FROM information_schema.columns WHERE table_schema=$1 "
                    "AND table_name='pl_messages' AND column_name='send_index'",
                    b_schema,
                )
            finally:
                await rb.close()
            assert has is not None, "v1→v2 migration skipped the send_index ALTER for schema B"
            # And an append that writes send_index must succeed (it crashed pre-fix).
            sid = await b2.create_session(system_prompt="B")
            await b2.append_message(sid, role="user", content="x", round_index=0, send_index=1)
        finally:
            await b2.close()
    finally:
        raw = await asyncpg.connect(PG_DSN)
        try:
            for s in (a_schema, b_schema):
                await raw.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        finally:
            await raw.close()
