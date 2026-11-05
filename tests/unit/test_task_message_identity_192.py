"""Message identity, explicit schema entry and refusal of unsupported migration sources."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import SessionStore
from power_loop.runtime.message_ref import integer_id
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION, SchemaPolicy, StoreSchemaError

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def task_store(tmp_path):
    store = await SessionStore.open(str(tmp_path / "tasks.db"), schema=SchemaPolicy.TASKS)
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.parametrize("mode", ["tasks", "compatible"])
async def test_old_experimental_task_protocol_is_not_implicitly_upgraded(tmp_path, mode):
    path = str(tmp_path / "old-experimental.db")
    s = await SessionStore.open(path, schema="tasks")
    await s._db.execute("UPDATE pl_task_store SET protocol_version=1 WHERE id=1")
    await s.close()
    with pytest.raises(StoreSchemaError, match="protocol"):
        await SessionStore.open(path, schema=mode)
    import sqlite3
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT protocol_version FROM pl_task_store").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM pl_jobs").fetchone()[0] == 0


async def test_partial_control_owner_schema_is_rejected_on_open(tmp_path):
    path = str(tmp_path / "partial-owner.db")
    s = await SessionStore.open(path, schema="tasks")
    await s._db.execute("ALTER TABLE pl_jobs DROP COLUMN control_owner_boot_id")
    await s.close()
    with pytest.raises(StoreSchemaError, match="control owner schema incomplete"):
        await SessionStore.open(path, schema="compatible")


async def test_message_ids_are_store_wide_while_seq_remains_session_local(task_store):
    s = task_store
    a, b = await s.create_session(), await s.create_session()
    assert await s.append_message(a, role="user", content="A") == 1
    assert await s.append_message(b, role="user", content="B") == 1
    ar, br = await s.message_ref(a, 1), await s.message_ref(b, 1)
    assert ar.id != br.id and ar.store_id == br.store_id == s.store_id
    assert (await s.message_by_id(a, str(ar.id))).content == "A"
    assert await s.message_by_id(b, ar.id) is None
    assert json.loads(json.dumps(ar.to_dict()))["id"] == str(ar.id)
    assert (await s.load_all_messages(a))[0].id == ar.id


async def test_ids_not_reused_after_deletion_or_import(task_store):
    s = task_store
    sid = await s.create_session()
    await s.append_message(sid, role="assistant", content="old")
    old = await s.message_ref(sid, 1)
    exported = await s.export_session(sid)
    assert "id" not in exported["tables"]["messages"][0], "cross-store IDs must not be copied"
    await s.close_session(sid)
    imported = await s.import_session(exported)
    new = await s.message_ref(imported, 1)
    assert new.id > old.id and new.session_id != old.session_id
    assert await s.message_by_id(sid, old.id) is None


async def test_identity_survives_reopen_and_both_bridge_modes(tmp_path):
    path = str(tmp_path / "reopen.db")
    s = await SessionStore.open(path, schema="tasks")
    sid = await s.create_session()
    await s.append_message(sid, role="assistant", tool_calls=[{"id": "call-1"}])
    ref = await s.message_ref(sid, 1)
    await s.close()
    for policy in ("tasks", "compatible"):
        s = await SessionStore.open(path, schema=policy)
        assert await s.message_ref(sid, 1) == ref
        await s.close()
    # The old default entry remains unable to start a legacy writer on v10.
    for policy in (None, "verify"):
        with pytest.raises(StoreSchemaError):
            await SessionStore.open(path, schema=policy)


async def test_legacy_mode_degrades_and_never_automatically_migrates(tmp_path):
    path = str(tmp_path / "legacy.db")
    s = await SessionStore.open(path)
    sid = await s.create_session()
    await s.append_message(sid, role="user", content="legacy")
    assert (await s.message_ref(sid, 1)).to_dict()["id_status"] == "unsupported"
    with pytest.raises(ValueError, match="unsupported"):
        await s.message_by_id(sid, "1")
    await s.close()
    with pytest.raises(StoreSchemaError, match="maintenance migration"):
        await SessionStore.open(path, schema="tasks")
    s = await SessionStore.open(path, schema="compatible")
    assert s.schema_version == CURRENT_SCHEMA_VERSION and (await s.load_all_messages(sid))[0].content == "legacy"
    await s.close()




async def test_two_connections_allocate_unique_ids_and_keep_same_identity(tmp_path):
    path = str(tmp_path / "concurrent.db")
    a = await SessionStore.open(path, schema="tasks")
    b = await SessionStore.open(path, schema="tasks")
    try:
        sids = [await a.create_session() for _ in range(2)]
        await asyncio.gather(*(store.append_message(sid, role="user", content=str(i))
            for i in range(20) for store, sid in zip((a, b), sids)))
        rows = [r for sid in sids for r in await a.load_all_messages(sid)]
        assert len(rows) == 40 and len({r.id for r in rows}) == 40
        assert a.store_id == b.store_id
        assert [r.seq for r in rows[:20]] == list(range(1, 21))
    finally:
        await a.close()
        await b.close()


@pytest.mark.parametrize("value", [True, False, 1.0, "1.0", "", "-1", " 1", "１２", 0, 2**63])
async def test_message_identity_rejects_imprecise_or_invalid_coordinates(value):
    with pytest.raises(ValueError):
        integer_id(value)


async def test_bigint_is_returned_as_decimal_string(task_store):
    s = task_store
    sid = await s.create_session()
    large = 2**53 + 9
    # Allocate a real bigint identity rather than mutating an immutable ID after
    # the writer has already cached it.
    await s._db.execute("INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)", (s.t.messages, large - 1))
    await s.append_message(sid, role="user", content="large")
    ref = await s.message_ref(sid, 1)
    assert ref.to_dict()["id"] == str(large)
    assert (await s.message_by_id(sid, str(large))).id == large


async def test_legacy_zero_send_provenance_is_preserved_without_inventing_a_new_send(task_store):
    sid = await task_store.create_session()
    await task_store.append_message(sid, role="user", content="legacy import", send_index=0, round_index=0)
    ref = await task_store.message_ref(sid, 1)
    assert ref.to_dict()["send_index"] == "0" and ref.round_index == 0
