"""Raw message identity and current-store reopen behavior under schema 17."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import SessionStore
from power_loop.runtime.message_ref import integer_id
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION

from .session_native_fixture import new_session

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_message_identity_reopen_old_reject(tmp_path):
    path = str(tmp_path / "identity.db")
    store = await SessionStore.open(path)
    first = (await new_session(store, "a")).session_id
    second = (await new_session(store, "b")).session_id
    assert await store.append_message(first, role="user", content="A") == 1
    assert await store.append_message(second, role="user", content="B") == 1
    first_ref, second_ref = await store.message_ref(first, 1), await store.message_ref(second, 1)
    assert first_ref.id != second_ref.id and first_ref.store_id == second_ref.store_id == store.store_id
    assert first_ref.to_dict()["id"] == str(first_ref.id)
    await store.close()
    reopened = await SessionStore.open(path, schema="verify")
    assert reopened.schema_version == CURRENT_SCHEMA_VERSION == 17
    assert await reopened.message_ref(first, 1) == first_ref
    await reopened.close()


@pytest.mark.asyncio
async def test_two_connections_keep_store_identity_and_unique_raw_ids(tmp_path):
    path = str(tmp_path / "concurrent.db")
    stores = [await SessionStore.open(path), await SessionStore.open(path)]
    try:
        sid = (await new_session(stores[0], "shared")).session_id
        await asyncio.gather(*(
            store.append_message(sid, role="user", content=str(index))
            for index in range(10) for store in stores
        ))
        rows = await stores[0].load_all_messages(sid)
        assert len(rows) == 20 and len({row.id for row in rows}) == 20
        assert stores[0].store_id == stores[1].store_id
        assert [row.seq for row in rows] == list(range(1, 21))
    finally:
        for store in stores:
            await store.close()


@pytest.mark.parametrize("value", [True, False, 1.0, "1.0", "", "-1", " 1", "１２", 0, 2**63])
def test_message_identity_rejects_imprecise_coordinates(value):
    with pytest.raises(ValueError):
        integer_id(value)
