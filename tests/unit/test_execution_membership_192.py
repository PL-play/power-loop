"""Original generation usage comes from canonical results and logical edges."""

import json

import pytest

from power_loop import AgentLoopConfig
from power_loop.runtime.jobs import JobActor, JobManager
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
        manager = JobManager(await connect())
        actor = JobActor(runner.actor.tenant_id, runner.actor.principal_id, runner.target.session_id)
        events = await manager.db.fetchall(
            f"SELECT event_type,payload_json FROM {manager.events} WHERE job_id=?", (result.job_ref.job_id,)
        )
        kinds = [e["event_type"] for e in events]
        assert kinds.count("result") == 1
        assert not {"model_closed", "supervision_closed", "execution_membership_seal"}.intersection(kinds)
        saved = json.loads(next(e["payload_json"] for e in events if e["event_type"] == "result"))
        assert saved["result"]["usage"] == result.usage
        assert saved["evidence"]["last_assistant_ref"]["session_id"] == runner.target.session_id
        assert await manager.execution_usage(actor, result.job_ref.job_id, generation=1) == result.usage
    finally:
        await runner.aclose()
