"""Normal query behavior retained after retiring exact abort-proof recovery."""
import pytest

from power_loop.runtime.abort_outcomes import pending_reviews
from power_loop.runtime.jobs import JobManager

from .test_job_registry_192 import registry_store, runtime
from .test_send_sql_profile import Profiled

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

async def rows(manager):
    return await manager.db.fetchall(f"SELECT * FROM {manager.prefix}session_obligations ORDER BY session_id,kind,source_id")


async def test_completed_control_history_does_not_enter_discovery(runtime):
    store, manager, actor, _, parent, connect, _now = runtime
    from types import SimpleNamespace

    from power_loop.runtime.result_consumption import finish, observe
    from power_loop.runtime.task_runtime import Execution

    async def discover():
        cold = JobManager(await connect(), now_ms=manager.now)
        original = cold.db
        cold.db = measured = Profiled(original)
        try:
            assert await pending_reviews(cold, cold.db, actor) == []
            return [(record["sql"].replace(cold.prefix, "PREFIX_"), record["rows"]) for record in measured.records]
        finally:
            cold.db = original

    short = await discover()
    for index in range(20):
        sid = await store.create_session(parent_session_id=actor.session_id)
        registered = await manager.register_job(actor, kind="subagent", session_id=sid, parent=parent, independent=True,
            system_key=f"original-{index}")
        child = await manager.claim_job(actor, registered.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
        key = await manager.record_execution_result(actor, child, outcome="succeeded", result={"value": index}, resource_state="stopped")
        await store.inbox_put(actor.session_id, [{"item_id": key, "kind": "task_done", "content": str(index)}])
        claimed = await store.inbox_claim(actor.session_id)
        seq = await store.append_message(actor.session_id, role="user", content=str(index), send_index=index,
            inbox={"claim_token": claimed[0]["claim_token"], "ids": [item["id"] for item in claimed]})
        binding = Execution(SimpleNamespace(manager=manager), actor, parent)
        binding.resource_evidence["_control_source"] = await manager._row(manager.db, parent.job_id)
        await observe(binding, [{"seq": seq, "role": "user", "content": str(index), "item_ids": [key]}],
            round_index=0, send_index=index)
        await finish(binding)
    assert await rows(manager) == []
    assert await discover() == short
