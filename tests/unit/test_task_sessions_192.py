"""Persistent task authority outlives physical driver cleanup."""
from __future__ import annotations

import pytest

from power_loop.runtime import task_sessions
from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime

__all__ = ['registry_store', 'registry_runtime']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
















@pytest.mark.parametrize("value", [None, "", True, 123, "a"*256])
async def test_invalid_task_session_coordinates_fail_before_any_store_access(value):
    with pytest.raises(JobError):
        await task_sessions.authorize_session(None, None, "job-original", value)


@pytest.mark.parametrize("value", [None, "", True, 123, "a"*65])
async def test_invalid_task_id_fails_before_any_store_access(value):
    with pytest.raises(JobError):
        await task_sessions.list_sessions(None, None, value)




@pytest.mark.parametrize("kind", ["subagent", "workflow", "background_command", "timer"])
async def test_queued_or_logical_task_does_not_grant_access_to_its_callers_history(registry_runtime, kind):
    _, manager, actor, _, parent, _, _ = registry_runtime
    child_sid = await manager.store.create_session(parent_session_id=actor.session_id)
    registration = await manager.register_job(actor, kind=kind, parent=parent,
        system_key="unstarted-"+kind, session_id=child_sid)
    result = await task_sessions.list_sessions(manager, actor, registration.job.job_id)
    assert result["execution_authority"] is False
    if kind == "subagent":
        assert [entry["session_id"] for entry in result["bindings"]] == [child_sid]
        assert result["bindings"][0]["generation"] == 0
    else:
        assert result["bindings"] == [] and result["reason"] == "logical_task"
    with pytest.raises(JobError, match="access_denied"):
        await task_sessions.authorize_session(manager, actor, registration.job.job_id, actor.session_id)
