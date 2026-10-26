"""Real model output survives a failing post-commit observer without replay."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_output_and_usage_survive_failed_closure_observer(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "events.db"), schema="tasks")
    llm, bus, events = BoundedProvider(limit=1), AgentEventBus(suppress_subscriber_errors=True), []

    def broken(event):
        raise RuntimeError("test observer unavailable")

    bus.subscribe(AgentEventType.TASK_SUPERVISION_CLOSED, broken)
    bus.subscribe(AgentEventType.TASK_SUPERVISION_CLOSED, events.append)
    manager = JobManager(store, event_bus=bus)
    runtime = TaskRuntime(manager, tenant_id="test", principal_id="host", executor_id="exec", boot_id="boot")
    loop = StatefulAgentLoop(llm=llm, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=1, max_tokens=1024))
    sid = await loop.new_session()
    try:
        async with asyncio.timeout(90):
            result = await loop.send("Reply with the number 42 only.", sid)
            await runtime.flush_control_checkpoints()
            assert result.status == "completed" and "42" in result.final_text and llm.calls == 1
            [row] = await manager.db.fetchall(f"SELECT job_id,generation FROM {manager.jobs}")
            ref = await manager.finalize_supervision(runtime.actor(sid), row["job_id"], generation=row["generation"])
            assert events and events[-1].data.event_id == ref["event_id"]
            assert (await manager.get_job(runtime.actor(sid), row["job_id"])).supervision_status == "closed"
            [message] = [m for m in await store.load_all_messages(sid) if m.role == "assistant"]
            assert message.content == result.final_text and llm.calls == 1
            assert llm.responses[0].token_usage.total_tokens > 0
    finally:
        llm.report(record_property)
        await runtime.flush_control_checkpoints()
        await llm.close()
        await store.close()
