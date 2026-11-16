"""Independent pools atomically claim inbox items on all three databases."""

import asyncio
import dataclasses

import pytest

from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

@pytest.fixture
async def stores(registry_store):
    store, connect = registry_store
    yield [store] + [await connect(schema="verify") for _ in range(7)]

@pytest.mark.asyncio
async def test_concurrent_drains_deliver_each_item_exactly_once(stores) -> None:
    """The regression that SQLite could not see: steering must never be replayed."""
    sid = await stores[0].create_session(dataclasses.replace(STORE_DESCRIPTOR, config={"system_prompt": "S"}), actor=STORE_ACTOR)
    await stores[0].inbox_put(sid, [{"item_id": f"m{i}", "content": f"m{i}"} for i in range(50)])

    batches = await asyncio.gather(*(s.inbox_claim(sid) for s in stores))
    seen = [r["item_id"] for batch in batches for r in batch]

    assert len(seen) == len(set(seen)), "an item was delivered to more than one drain"
    assert sorted(seen) == sorted(f"m{i}" for i in range(50)), "items lost or duplicated"

    for batch in batches:
        if batch:
            await stores[0].append_message(sid,role="user",content="delivered",inbox={
                "claim_token":batch[0]['claim_token'],"ids":[row['id'] for row in batch]})
    await stores[0].close_session(sid)
