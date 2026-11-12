"""P1 raw-message coordinates; model-loop history behavior belongs to P2."""
from __future__ import annotations

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore

from .session_native_fixture import ACTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_next_seq_fence_and_gaps(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "before", message_id="before", actor=ACTOR)
    first = await port.claim(root, node_id="a", boot_id="a")
    await port.final(first, final_text="before result")
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET next_seq=next_seq+3 WHERE session_id=?",
        (root.session_id,),
    )
    await port.accept(root, "after", message_id="after", actor=ACTOR)
    second = await port.claim(root, node_id="b", boot_id="b")
    assert second.message_seq == 6 and second.send_index == 2
    await port.final(second, final_text="after result")
    rows = (await port.messages(root, actor=ACTOR)).items
    assert [row.seq for row in rows] == [1, 2, 6, 7]
    tail = await port.messages(root, actor=ACTOR, after_seq=2)
    assert [row.seq for row in tail.items] == [6, 7]
    assert (await store.message_ref(root.session_id, 6)).role == "user"
    cold = RuntimeInputStore(await connect(schema="verify"))
    assert (await cold.get_result(second.input_refs[0], actor=ACTOR)).result_seq == 7
