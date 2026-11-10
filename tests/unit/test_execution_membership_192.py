"""Ordinary send usage stays on the result. It is not a task result event."""

import pytest

from power_loop import AgentLoopConfig
from power_loop.runtime.jobs import JobManager
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_send_has_one_result_and_retains_original_usage_after_reopen(registry_store):
    store, connect = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None, system_prompt="main")).start()
    try:
        result = await runner.send("run")
        assert result.usage["total_tokens"] > 0
        manager = JobManager(await connect())
        # An ordinary turn keeps usage on the result. It writes no send task and no result event.
        assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='send'")
        assert not await manager.db.fetchall(
            f"SELECT event_id FROM {manager.events} WHERE event_type='result'")
    finally:
        await runner.aclose()
