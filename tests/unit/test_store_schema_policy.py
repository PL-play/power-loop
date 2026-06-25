"""SchemaPolicy provisioning, StoreSchemaError(.ddl), the create_schema deprecated alias,
the next_seq monotonicity invariant the window cache relies on, and the delta load."""

from __future__ import annotations

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


async def test_create_schema_bool_alias_maps_to_policy(tmp_path):
    db = str(tmp_path / "alias.db")
    s = await SessionStore.open(db, create_schema=True)  # → AUTO_CREATE
    await s.close()
    s2 = await SessionStore.open(db, create_schema=False)  # → VERIFY (schema exists)
    assert s2.schema_version == CURRENT_SCHEMA_VERSION
    await s2.close()
    # explicit policy wins over the bool alias
    db2 = str(tmp_path / "wins.db")
    with pytest.raises(StoreSchemaError):
        await SessionStore.open(db2, schema=SchemaPolicy.VERIFY, create_schema=True)


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
    sid = await s.create_session(session_id="s")
    before = (await s.get_state(sid)).next_seq
    await s.append_message(sid, role="user", content="x")
    after = (await s.get_state(sid)).next_seq
    assert after == before + 1
    await s.close()


async def test_record_compaction_bumps_next_seq():
    """A fold allocates the compact_note's seq from the per-session counter, so it bumps
    next_seq — which is what lets the loop's window cache detect a fold and invalidate."""
    s = await SessionStore.open(":memory:")
    sid = await s.create_session(session_id="s")
    seqs = [await s.append_message(sid, role="user", content=f"m{i}") for i in range(3)]
    before = (await s.get_state(sid)).next_seq
    await s.record_compaction(
        sid, from_seq=min(seqs), to_seq=max(seqs), note_content="n",
        before_tokens=1, after_tokens=1, round_index=0, fold_seqs=seqs, order_key=seqs[0],
    )
    after = (await s.get_state(sid)).next_seq
    assert after > before
    await s.close()


# ── delta load (load_active_messages after_seq) ──────────────────────────────


async def test_load_active_after_seq_returns_only_the_tail():
    s = await SessionStore.open(":memory:")
    sid = await s.create_session(session_id="s")
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
