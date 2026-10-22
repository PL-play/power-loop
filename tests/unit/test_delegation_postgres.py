"""Fresh identity schema and recall on an isolated prefix in the dedicated test database."""
import json
import os
import secrets
from urllib.parse import urlparse

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.recall import recall
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import _STORE_TABLES

pytestmark = pytest.mark.unit


async def test_postgres_projection_identity_and_snapshot_jsonl(tmp_path):
    pytest.importorskip("asyncpg")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        pytest.skip("POSTGRES_PASSWORD required for the dedicated test database")
    dsn = os.environ.get("POWER_LOOP_TEST_PG_DSN",
                         f"postgresql://deeptalk:{password}@localhost:5433/power_loop_test")
    assert urlparse(dsn).path.endswith("_test"), "refuse non-test database"
    prefix = "dt191_" + secrets.token_hex(4) + "_"
    store = await open_store(dsn, table_prefix=prefix)
    try:
        parent = await store.create_session()
        child = await store.create_session(parent_session_id=parent)
        await store.upsert_project_message(child, send_index=1, kind="project", content={"proof": "yes"},
                                           source_seq_lo=1, source_seq_hi=2)
        row = (await store.load_project_messages(child))[0]
        assert row.id is not None
        await store.upsert_project_message(child, send_index=1, kind="project", content={"proof": "updated"})
        assert (await store.load_project_messages(child))[0].id == row.id
        await store.append_message(child, role="assistant", content="record", send_index=1)
        data = json.loads(await recall(store, parent, sub_session=child, source="projection", id=row.id))
        assert json.loads(data["text"])["content"]["proof"] == "updated"
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            saved = json.loads(await recall(store, parent, sub_session=child, ids=[1, 1], output_file="data.jsonl"))
            assert saved["status"] == "exported" and saved["records"] == 1
            assert json.loads((tmp_path / "data.jsonl").read_text())["content"] == "record"
    finally:
        for table in (*_STORE_TABLES, "schema_migrations"):
            await store._db.execute(f"DROP TABLE IF EXISTS {prefix}{table}")
        await store.close()
