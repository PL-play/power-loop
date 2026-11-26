"""Independent pools atomically claim queued inputs on all three databases."""

import asyncio
import dataclasses

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import TargetRef
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

@pytest.fixture
async def stores(registry_store):
    store, connect = registry_store
    yield [store] + [await connect(schema="verify") for _ in range(7)]

@pytest.mark.asyncio
async def test_concurrent_drains_deliver_each_item_exactly_once(stores) -> None:
    """The regression that SQLite could not see: concurrent claims must never replay items."""
    sid = await stores[0].create_session(dataclasses.replace(STORE_DESCRIPTOR, config={"system_prompt": "S"}), actor=STORE_ACTOR)
    target = TargetRef(stores[0].store_id, sid)
    ports = [RuntimeInputStore(s) for s in stores]
    for i in range(50):
        await ports[0].accept(target, f"m{i}", message_id=f"m{i}", actor=STORE_ACTOR)

    batches = await asyncio.gather(*(
        p.claim(target, node_id=f"node-{idx}", boot_id=f"boot-{idx}")
        for idx, p in enumerate(ports)
    ))
    claimed = [b for b in batches if b is not None]
    assert len(claimed) == 1, f"expected exactly one winner, got {len(claimed)}"
    seen = [ref.message_id for b in claimed for ref in b.input_refs]

    assert len(seen) == len(set(seen)), "an item was delivered to more than one drain"
    assert sorted(seen) == sorted(f"m{i}" for i in range(50)), "items lost or duplicated"

    await ports[0].final(claimed[0], status="completed", final_text="delivered")
    await stores[0].close_session(sid)
