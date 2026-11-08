"""Real model output survives a failing post-commit observer without replay."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, SessionStore
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_output_and_usage_survive_failed_closure_observer(tmp_path, record_property, runtime_agent):
    store = await SessionStore.open(str(tmp_path / "events.db"), schema="tasks")
    llm, bus, events = BoundedProvider(limit=1), AgentEventBus(suppress_subscriber_errors=True), []

    def broken(event):
        raise RuntimeError("test observer unavailable")

    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, broken)
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, events.append)
    manager = JobManager(store, event_bus=bus)
    runtime = TaskRuntime(manager, tenant_id="test", principal_id="host", executor_id="exec", boot_id="boot")
    loop = await runtime_agent(model=llm, store=store, task_runtime=runtime, event_bus=bus,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=1, max_tokens=1024))
    sid = loop.target.session_id
    try:
        async with asyncio.timeout(90):
            result = await loop.send("Reply with the number 42 only.")
            assert result.status == "completed" and "42" in result.final_text and llm.calls == 1
            from power_loop.agent.runtime_store import now_ms
            queried = await loop.runtime.client.wait(result.input_ref, actor=loop.actor, deadline_ms=now_ms()+5000)
            assert queried == result and result.usage['total_tokens'] > 0
            assert len(events) == 1 and events[0].data.model_attempt_job_id == result.job_ref.job_id
            [message] = [m for m in await store.load_all_messages(sid) if m.role == "assistant"]
            assert message.content == result.final_text and llm.calls == 1
            assert llm.responses[0].token_usage.total_tokens > 0
    finally:
        llm.report(record_property)
        await loop.aclose()
        await llm.close()
        await store.close()
