"""Design 203 workflow behavior through the public runtime."""

import pytest

from power_loop import AgentLoopConfig
from power_loop.agent.runtime_store import now_ms
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_explicit_continuation_reuses_sibling_and_recomputes_tail(registry_store):
    store, _ = registry_store
    model = Model()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(compactor=None)).start()
    try:
        root = {"type": "sequence", "steps": [{"type": "parallel", "branches": [leaf("A"), leaf("B")]}, leaf("C")]}
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "continue", "root": root}, message_id="run", actor=runner.actor
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed"
        await runner.drain()
        continued = await handle.resume(generation=1, node_id="A", message="improve A", message_id="more")
        second = await continued.wait(deadline_ms=now_ms() + 15000)
        assert second.status == "completed", second
        rows = await store._db.fetchall(
            f"SELECT title,generation,session_id FROM {store.table_prefix}jobs WHERE kind='workflow_node'"
        )
        assert len([r for r in rows if r["title"] == "A"]) == 1
        assert next(r for r in rows if r["title"] == "A")["generation"] == 2
        assert len([r for r in rows if r["title"] == "B"]) == 1
        assert len([r for r in rows if r["title"] == "C"]) == 2
    finally:
        await runner.aclose()
