"""A real main reads the original known outcome through current runtime input."""
import pytest

from power_loop import AgentLoopConfig, SessionStore, TargetRef, ToolRegistry
from power_loop.runtime.managed_delivery import enqueue_runtime
from power_loop.runtime.managed_effects import reconcile
from tests.runtime_fixture import RunningAgent
from tests.unit.test_managed_delivery_192 import consume_notice
from tests.unit.test_managed_effects_192 import scenario

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_real_main_handles_original_outcome_without_replaying_tool(tmp_path,record_property):
    store=await SessionStore.open(str(tmp_path/'known-result.db'),schema='tasks')
    provider=BoundedProvider(limit=1)
    runner=None
    try:
        c=await scenario((store,None),pending=True)
        [job]=await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
        original=await store.load_all_messages(c.sid)
        c.state.resolved=True
        await reconcile(c.runtime,c.actor,job['job_id'],c.adapter)
        runner=RunningAgent(store,provider,ToolRegistry(),task_runtime=c.runtime,
            config=AgentLoopConfig(compactor=None,max_rounds=2,max_tokens=2048))
        runner.target=TargetRef(store.store_id,c.sid)
        notice=await enqueue_runtime(c.runtime,c.actor,job['job_id'])
        assert provider.calls==0
        result=await consume_notice(runner,notice['notification_id'])
        assert result.status=='completed' and result.final_text.strip()
        assert (await c.manager.get_job(c.actor,job['job_id'])).result_status=='handled'
        assert c.state.dispatched==c.state.prepared==c.state.lookups==1
        history=await store.load_all_messages(c.sid)
        assert history[:len(original)]==original
        assert provider.calls==1
    finally:
        if runner is not None:
            await runner.aclose()
        provider.report(record_property)
        await provider.close()
        await store.close()
