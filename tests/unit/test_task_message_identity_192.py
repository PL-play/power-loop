"""Raw message identity and current-store reopen behavior under schema 17."""
from __future__ import annotations

import asyncio

import pytest

from power_loop.runtime.message_ref import integer_id
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION

from .session_native_fixture import new_session

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_message_identity_reopen_old_reject(registry_store):
    store, connect = registry_store
    first = (await new_session(store, "a")).session_id
    second = (await new_session(store, "b")).session_id
    assert await store.append_message(first, role="user", content="A") == 1
    assert await store.append_message(second, role="user", content="B") == 1
    first_ref, second_ref = await store.message_ref(first, 1), await store.message_ref(second, 1)
    assert first_ref.id != second_ref.id and first_ref.store_id == second_ref.store_id == store.store_id
    assert first_ref.to_dict()["id"] == str(first_ref.id)
    reopened = await connect(schema="verify")
    assert reopened.schema_version == CURRENT_SCHEMA_VERSION == 18
    assert await reopened.message_ref(first, 1) == first_ref


@pytest.mark.asyncio
async def test_two_connections_keep_store_identity_and_unique_raw_ids(registry_store):
    first, connect = registry_store
    second = await connect(schema="verify")
    sid = (await new_session(first, "shared")).session_id
    await asyncio.gather(*(
        store.append_message(sid, role="user", content=str(index))
        for index in range(10) for store in (first, second)
    ))
    rows = await first.load_all_messages(sid)
    assert len(rows) == 20 and len({row.id for row in rows}) == 20
    assert first.store_id == second.store_id
    assert [row.seq for row in rows] == list(range(1, 21))


@pytest.mark.parametrize("value", [True, False, 1.0, "1.0", "", "-1", " 1", "１２", 0, 2**63])
def test_message_identity_rejects_imprecise_coordinates(value):
    with pytest.raises(ValueError):
        integer_id(value)
