"""SchemaPolicy provisioning, StoreSchemaError(.ddl), current schemas,
the next_seq monotonicity invariant the window cache relies on, and the delta load."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from power_loop.runtime.store.backends.sqlite import SqliteDatabase
from power_loop.runtime.store.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaPolicy,
    StoreSchemaError,
    ensure_schema,
    provisioning_ddl,
)
from power_loop.runtime.store.store import SessionStore

from .session_native_fixture import new_session

pytestmark = pytest.mark.unit


# ── SchemaPolicy ─────────────────────────────────────────────────────────────


async def test_auto_create_then_verify(tmp_path):
    db = str(tmp_path / "s.db")
    s = await SessionStore.open(db, schema=SchemaPolicy.AUTO_CREATE)
    assert s.schema_version == CURRENT_SCHEMA_VERSION
    await s.close()
    # schema now exists → VERIFY opens cleanly
    s2 = await SessionStore.open(db, schema=SchemaPolicy.VERIFY)
    assert s2.schema_version == CURRENT_SCHEMA_VERSION
    await s2.close()


async def test_verify_on_fresh_db_raises_with_complete_ddl(tmp_path):
    db = str(tmp_path / "fresh.db")
    with pytest.raises(StoreSchemaError) as ei:
        await SessionStore.open(db, schema=SchemaPolicy.VERIFY)
    err = ei.value
    assert "not initialized" in str(err)
    # .ddl is a COMPLETE provisioning script: migrations table + a store table + the stamp
    joined = ";".join(err.ddl)
    assert "pl_schema_migrations" in joined
    assert "pl_sessions" in joined
    assert any("INSERT INTO pl_schema_migrations" in d for d in err.ddl)
    # __str__ prints the runnable script
    assert "CREATE TABLE" in str(err) and "INSERT INTO pl_schema_migrations" in str(err)


async def test_auto_create_is_idempotent(tmp_path):
    db = str(tmp_path / "idem.db")
    for _ in range(3):
        s = await SessionStore.open(db, schema=SchemaPolicy.AUTO_CREATE)
        assert s.schema_version == CURRENT_SCHEMA_VERSION
        await s.close()




async def test_auto_create_ddl_failure_raises_storeschemaerror_with_ddl(monkeypatch):
    """A DDL failure during AUTO_CREATE (e.g. the role lacks CREATE rights) surfaces as a
    StoreSchemaError carrying the full provisioning script + an actionable hint, not a raw
    driver error."""
    db = SqliteDatabase.open(":memory:")
    monkeypatch.setattr(db.dialect, "ddl", lambda prefix: ["THIS IS NOT VALID SQL"])
    with pytest.raises(StoreSchemaError) as ei:
        await ensure_schema(db, "pl_", policy=SchemaPolicy.AUTO_CREATE)
    assert "permission" in str(ei.value).lower()
    assert ei.value.ddl  # full script attached
    await db.close()


async def test_provisioning_ddl_shape():
    db = SqliteDatabase.open(":memory:")
    ddl = provisioning_ddl(db, "pl_")
    assert ddl[0].startswith("CREATE TABLE IF NOT EXISTS pl_schema_migrations")
    assert ddl[-1].startswith("INSERT INTO pl_schema_migrations")
    assert any("pl_messages" in d for d in ddl) and any("pl_sessions" in d for d in ddl)
    await db.close()


# ── next_seq monotonicity (the window-cache validity invariant) ──────────────


async def test_append_message_bumps_next_seq():
    s = await SessionStore.open(":memory:")
    sid = (await new_session(s, "s")).session_id
    before = (await s.get_state(sid)).next_seq
    await s.append_message(sid, role="user", content="x")
    after = (await s.get_state(sid)).next_seq
    assert after == before + 1
    await s.close()


# ── delta load (load_active_messages after_seq) ──────────────────────────────


async def test_load_active_after_seq_returns_only_the_tail():
    s = await SessionStore.open(":memory:")
    sid = (await new_session(s, "s")).session_id
    seqs = [await s.append_message(sid, role="user", content=f"m{i}") for i in range(5)]
    full = [r.seq for r in await s.load_active_messages(sid)]
    assert full == seqs
    tail = [r.seq for r in await s.load_active_messages(sid, after_seq=seqs[2])]
    assert tail == seqs[2:]  # inclusive of after_seq
    none_after = await s.load_active_messages(sid, after_seq=seqs[-1] + 1)
    assert none_after == []
    await s.close()


def test_table_prefix_length_is_capped() -> None:
    # store-dialect-3: a long prefix would overflow MySQL's 64-char identifier limit for derived
    # index names; validate_table_prefix rejects it.
    from power_loop.runtime.store.schema import _MAX_TABLE_PREFIX_LEN, validate_table_prefix

    assert validate_table_prefix("a" * _MAX_TABLE_PREFIX_LEN) == "a" * _MAX_TABLE_PREFIX_LEN  # at limit
    with pytest.raises(ValueError, match="too long"):
        validate_table_prefix("a" * (_MAX_TABLE_PREFIX_LEN + 1))


async def test_schema17_has_no_legacy_tables_and_enforces_owner_state(registry_store):
    from power_loop.runtime.store.session_native_ddl import FORBIDDEN_TABLES

    store, _ = registry_store
    if store._db.dialect.name == "sqlite":
        rows = await store._db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    elif store._db.dialect.name == "postgres":
        rows = await store._db.fetchall(
            "SELECT tablename AS name FROM pg_tables WHERE schemaname=current_schema()"
        )
    else:
        rows = await store._db.fetchall(
            "SELECT table_name AS name FROM information_schema.tables WHERE table_schema=DATABASE()"
        )
    names = {row["name"] for row in rows}
    assert not {store.table_prefix + table for table in FORBIDDEN_TABLES} & names
    with pytest.raises(Exception):
        await store._db.execute(
            f"INSERT INTO {store.t.session_state}(session_id,execution_state,active_send_index) "
            "VALUES ('bad','running',1)"
        )


@pytest.mark.parametrize("column,value", (("version", 16), ("protocol_version", 42)))
async def test_old_schema_or_protocol_is_rejected(registry_store, column, value):
    store, connect = registry_store
    await store._db.execute(
        f"UPDATE {store.table_prefix}schema_migrations SET {column}=? WHERE id=1", (value,)
    )
    with pytest.raises(StoreSchemaError):
        await connect(schema="verify")


async def test_session_indexes(registry_store):
    store, connect = registry_store
    index = store.table_prefix + "idx_session_recovery"
    statement = f"DROP INDEX {index}"
    if store._db.dialect.name == "mysql":
        statement += f" ON {store.t.session_state}"
    await store._db.execute(statement)
    with pytest.raises(StoreSchemaError, match="idx_session_recovery"):
        await connect(schema="verify")


async def test_projection_session_id_offline_migration(registry_store, monkeypatch):
    from power_loop.runtime.store import task_schema
    from power_loop.runtime.store.migrate_projection_session_id import add_projection_session_id_index

    store, connect = registry_store
    sid = (await new_session(store, "projection-migration")).session_id
    await store.append_message(sid, role="user", content="original", send_index=1)
    await store.upsert_project_message(sid, send_index=1, kind="user", content={"original": True})
    await store.upsert_project_message(
        sid, send_index=1, kind="compact", content={"summary": "kept"},
        compact_from_send=1, compact_to_send=1,
    )
    await store.set_runtime_state(sid, "migration-probe", {"unchanged": True})
    tables = (store.t.messages, store.t.project_messages, store.t.session_runtime_state)
    before = {table: [dict(row) for row in await store._db.fetchall(f"SELECT * FROM {table}")]
              for table in tables}
    name = store.table_prefix + "idx_projection_session_id"
    statement = f"DROP INDEX {name}"
    if store._db.dialect.name == "mysql":
        statement += f" ON {store.t.project_messages}"
    await store._db.execute(statement)

    reduced = {key: value for key, value in task_schema.INDEXES.items()
               if key != "idx_projection_session_id"}
    with monkeypatch.context() as patch:
        patch.setattr(task_schema, "INDEXES", reduced)
        legacy = await connect(schema="verify")
        await legacy.close()
    with pytest.raises(StoreSchemaError, match="idx_projection_session_id"):
        await connect(schema="verify")

    assert await add_projection_session_id_index(store._db, store.table_prefix) == {
        "index": name, "changed": True,
    }
    assert await add_projection_session_id_index(store._db, store.table_prefix) == {
        "index": name, "changed": False,
    }
    reopened = await connect(schema="verify")
    for table in tables:
        assert [dict(row) for row in await reopened._db.fetchall(f"SELECT * FROM {table}")] == before[table]
    compact = await reopened.latest_project_compact(sid)
    assert compact is not None and compact.compact_to_send == 1
    migration = await reopened._db.fetchone(
        f"SELECT version,protocol_version FROM {store.table_prefix}schema_migrations WHERE id=1"
    )
    assert dict(migration) == {"version": 17, "protocol_version": 43}


async def test_projection_session_id_migration_rejects_incompatible_index(registry_store):
    from power_loop.runtime.store.migrate_projection_session_id import add_projection_session_id_index

    store, _ = registry_store
    name = store.table_prefix + "idx_projection_session_id"
    statement = f"DROP INDEX {name}"
    if store._db.dialect.name == "mysql":
        statement += f" ON {store.t.project_messages}"
    await store._db.execute(statement)
    await store._db.execute(
        f"CREATE INDEX {name} ON {store.t.project_messages}(session_id,send_index)"
    )
    with pytest.raises(StoreSchemaError, match="idx_projection_session_id"):
        await add_projection_session_id_index(store._db, store.table_prefix)


async def test_projection_session_id_migration_reports_ddl_failure(monkeypatch):
    from power_loop.runtime.store.migrate_projection_session_id import add_projection_session_id_index

    store = await SessionStore.open(":memory:")
    name = store.table_prefix + "idx_projection_session_id"
    await store._db.execute(f"DROP INDEX {name}")
    original = type(store._db).transaction

    @asynccontextmanager
    async def transaction(db):
        async with original(db) as tx:
            class FailDDL:
                def __getattr__(self, attr):
                    return getattr(tx, attr)

                async def execute(self, sql, params=()):
                    if sql.startswith("CREATE INDEX"):
                        raise RuntimeError("DDL denied by test")
                    return await tx.execute(sql, params)

            yield FailDDL()

    monkeypatch.setattr(type(store._db), "transaction", transaction)
    with pytest.raises(RuntimeError, match="DDL denied by test"):
        await add_projection_session_id_index(store._db, store.table_prefix)
    await store.close()
