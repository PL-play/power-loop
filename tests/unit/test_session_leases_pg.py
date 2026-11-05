"""Independent pools atomically claim inbox items on all three databases."""
import asyncio

import pytest

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

@pytest.fixture
async def stores(registry_store):
    store, connect = registry_store
    yield [store] + [await connect(schema="compatible") for _ in range(7)]

@pytest.mark.asyncio
async def test_concurrent_drains_deliver_each_item_exactly_once(stores) -> None:
    """The regression that SQLite could not see: steering must never be replayed."""
    sid = await stores[0].create_session(system_prompt="S")
    await stores[0].inbox_put(sid, [{"item_id": f"m{i}", "content": f"m{i}"} for i in range(50)])

    batches = await asyncio.gather(*(s.inbox_claim(sid) for s in stores))
    seen = [r["item_id"] for batch in batches for r in batch]

    assert len(seen) == len(set(seen)), "an item was delivered to more than one drain"
    assert sorted(seen) == sorted(f"m{i}" for i in range(50)), "items lost or duplicated"

    await stores[0].close_session(sid)
