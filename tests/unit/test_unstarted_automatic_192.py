"""A failed unstarted-result write never executes or replays the external effect."""
import pytest

from power_loop.runtime.managed_effects import terminate_unstarted

from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

@pytest.mark.parametrize('stage', ['result', 'commit', 'commit_ack'])
async def test_unstarted_result_write_failure_remains_readable_without_external_replay(registry_store, stage):
    c = await scenario(registry_store, budgeted=True, claim_failure=True, unstarted_failure=stage)
    assert c.fault.fired
    assert c.result.status == 'failed'
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    assert c.model.calls == c.host.reserve_calls == 1
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert row['generation'] == 0
    original = await terminate_unstarted(c.manager, c.actor, row['job_id'])
    assert original['resolution'] == 'not_dispatched'
    assert not original['execution_authority']
    assert await terminate_unstarted(c.manager, c.actor, row['job_id']) == original
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
