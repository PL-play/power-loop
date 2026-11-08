"""Current worker lifecycle preserves original results and prevents new work after stop."""
import asyncio

import pytest

from power_loop import JobRef, RuntimeRequestError
from power_loop.agent.runtime_store import now_ms

from .test_job_registry_192 import registry_store
from .test_subagent_resume_192 import ready
from .test_subagent_resume_192 import (
    test_original_job_and_session_resume_once_with_distinct_results_and_usage as verify_resume,
)
from .test_task_runtime_192 import managed, receipt

__all__ = ['registry_store', 'ready', 'managed']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_current_worker_resume_retains_original_task_and_per_generation_usage(ready, monkeypatch):
    await verify_resume(ready, monkeypatch)
    assert ready.loop.runtime.status()['accepting']


async def test_shutdown_stops_original_child_and_rejects_new_paid_input(managed):
    _,manager,runtime,agent,model,sid,_=managed
    parent=await agent.send('delegate')
    await asyncio.wait_for(model.entered.wait(),5)
    created=await receipt(managed)
    calls=model.root_calls,model.child_calls
    stopped=await agent.runtime.worker.shutdown(timeout_s=5,cancel_running=True)
    assert stopped.drained
    client=agent.runtime.client
    child=await client.wait(JobRef(**created['job_ref']),actor=agent.actor,deadline_ms=now_ms()+1000)
    assert child.status=='cancelled'
    with pytest.raises(RuntimeRequestError,match='session_closed'):
        await client.send(agent.target,'must not dispatch',message_id='rejected',actor=agent.actor)
    assert (model.root_calls,model.child_calls)==calls
    assert await client.wait(parent.input_ref,actor=agent.actor,deadline_ms=now_ms()+1000)==parent
    saved=await manager.get_job(runtime.actor(sid),created['job_id'])
    assert saved.outcome=='cancelled'
