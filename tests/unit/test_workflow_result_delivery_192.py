"""Frozen admission and exact saved result identity; no callback authority."""

from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.workflow import journal
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_admission_retry_keeps_one_driver_and_original_result(registry_store):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()
    spec = {"name": "once", "root": leaf("A")}
    try:
        first = await runner.runtime.client.workflow(runner.target, spec, message_id="run", actor=runner.actor)
        result = await first.wait(deadline_ms=now_ms() + 10000)
        again = await runner.runtime.client.workflow(runner.target, spec, message_id="run", actor=runner.actor)
        assert first.input_ref == again.input_ref and first.job_id == again.job_id
        assert (await again.wait(deadline_ms=now_ms() + 10000)).to_dict() == result.to_dict()
        with pytest.raises(RuntimeRequestError, match="duplicate_input_conflict"):
            await runner.runtime.client.workflow(
                runner.target, {**spec, "root": leaf("B")}, message_id="run", actor=runner.actor
            )
        with pytest.raises(RuntimeRequestError, match="forbidden"):
            await runner.runtime.client.wait(
                first.input_ref, actor=replace(runner.actor, tenant_id="foreign"), deadline_ms=None
            )
        saved = await journal.read(store, runner.target.session_id, first.run_id)
        assert len(saved["nodes"]) == 1 and saved["nodes"][0]["result_ref"]["generation"] == 1
        events = await store._db.fetchall(
            f"SELECT event_type FROM {store.table_prefix}job_events WHERE job_id=?", (first.job_id,)
        )
        assert [e["event_type"] for e in events].count("result") == 1
    finally:
        await runner.aclose()
