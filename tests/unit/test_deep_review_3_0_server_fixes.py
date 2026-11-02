"""The explicit protocol-21 upgrade probes only the current PostgreSQL schema."""

from __future__ import annotations

import os
import socket
import urllib.parse
from urllib.parse import urlparse

import pytest

from power_loop.runtime.store.factory import open_store

pytestmark = pytest.mark.unit

PG_DSN = os.environ.get(
    "POWER_LOOP_TEST_PG_DSN", f"postgresql://deeptalk:{os.environ.get('POSTGRES_PASSWORD', 'deeptalk')}@localhost:5433/power_loop_test"
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
async def test_protocol21_upgrade_columns_are_schema_scoped() -> None:
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
        from power_loop.runtime.store.change_signals import COLUMNS
        from power_loop.runtime.store.schema import StoreSchemaError
        from power_loop.runtime.store.task_schema import migrate_task_schema

        from ._protocol21_fixture import retain_protocol21_format
        for schema_name in (a_schema, b_schema):
            st = await open_store(_dsn(schema_name), schema="tasks")
            await st.close()
        original = await open_store(_dsn(b_schema), schema="tasks")
        try:
            sid = await original.create_session(system_prompt="B")
            seq = await original.append_message(sid, role="user", content="original", send_index=1)
            ref = await original.message_ref(sid, seq)
            await retain_protocol21_format(original)
        finally:
            await original.close()
        with pytest.raises(StoreSchemaError):
            await open_store(_dsn(b_schema), schema="tasks")
        from power_loop.runtime.store.factory import _make_database
        db = await _make_database(_dsn(b_schema))
        try:
            before = await db.fetchall("SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name='pl_session_state'")
            assert not set(COLUMNS) & {row['column_name'] for row in before}
            await migrate_task_schema(db, 'pl_', maintenance=True)
            after = await db.fetchall("SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name='pl_session_state'")
            assert set(COLUMNS) <= {row['column_name'] for row in after}
        finally:
            await db.close()
        upgraded = await open_store(_dsn(b_schema), schema="tasks")
        try:
            assert await upgraded.message_ref(sid, seq) == ref
            assert (await upgraded.load_active_messages(sid))[0].content == "original"
            await upgraded.append_message(sid, role="user", content="after", send_index=2)
        finally:
            await upgraded.close()
    finally:
        raw = await asyncpg.connect(PG_DSN)
        try:
            for s in (a_schema, b_schema):
                await raw.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        finally:
            await raw.close()
