"""P1 storage-only SQL and row profile for accept + one batch claim + final."""
from __future__ import annotations

import json
import os

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from tests.write_meter import WriteMeter

from .session_native_fixture import ACTOR, target
from .test_send_sql_profile import Profiled, _counts

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_session_storage_sql_profile(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix)
    await meter.reset()
    profiled = Profiled(original, meter)
    store._db = profiled
    try:
        await port.accept(root, "one", message_id="one", actor=ACTOR)
        batch = await port.claim(root, node_id="a", boot_id="a")
        # Explicit consumed (the worker passes original + boundary follow-up refs): final
        # stamps exactly these inputs without a resolution read, keeping the fixed budget.
        await port.final(batch, consumed=batch.input_refs, final_text="done", usage={"total_tokens": 1})
        counts = _counts(profiled.records, profiled.tx_sites, profiled.tx_categories)
        writes = await meter.report()
        assert counts["legacy_job_control"] == 0
        assert counts["billing"] == 0 and counts["periodic"] == 0
        assert profiled.transactions == 3
        assert counts["session_control_tx"] == 3
        ceiling = 24 if original.dialect.name == "mysql" else 22
        assert len(profiled.records) <= ceiling, (len(profiled.records), counts)
        if output := os.environ.get("P215_PROFILE_OUT"):
            with open(output, "a") as stream:
                stream.write(json.dumps({
                    "db": original.dialect.name, "sql": len(profiled.records),
                    "counts": counts, "transactions": profiled.transactions,
                    "writes": writes,
                }) + "\n")
    finally:
        store._db = original
        await meter.close()
