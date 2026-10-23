"""Physical-call events carry the actual durable identity across sends and children."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    CancellationToken,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    StatefulAgentLoop,
)
from power_loop.runtime import delegation
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Model(LLMService):
    def __init__(self):
        self.failure = self.spawn = self.block = False
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls, self.main_calls = 0, 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        child = "IDENTITY-CHILD" in (request.system_prompt or "")
        self.main_calls += not child
        self.entered.set()
        if self.block:
            await self.release.wait()
        if self.failure:
            raise OSError("test supplier outcome unknown")
        tools = None
        if self.spawn and not child and self.main_calls == 1:
            tools = [
                {
                    "id": "spawn-identity",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": json.dumps(
                            {
                                "name": "child",
                                "system_prompt": "IDENTITY-CHILD",
                                "task": "Return 42",
                                "tools": [],
                                "max_rounds": 2,
                                "max_tokens": 20,
                            }
                        ),
                    },
                }
            ]
        return LLMResponse(
            raw_text="" if tools else "42",
            tool_calls=tools,
            token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )

    def stream(self, request):
        raise AssertionError("complete transport only")

    async def close(self):
        pass


@pytest.fixture
async def event_case(registry_store, request):
    store, _ = registry_store
    mode = getattr(request, "param", "budget")
    model, host, bus, events = Model(), Host(), AgentEventBus(), []
    host.service = model
    bus.subscribe(None, lambda event: events.append(event))
    manager = JobManager(store)
    runtime = (
        None
        if mode == "legacy"
        else TaskRuntime(
            manager,
            tenant_id="events",
            principal_id="host",
            executor_id="executor",
            boot_id="boot",
            lease_ms=300000,
            boot_contract_hash="a" * 64,
            budget_config=BudgetConfig("events", (LocalBudgetScope("shared", max_calls=10),), host)
            if mode == "budget"
            else None,
        )
    )
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(
        llm=model,
        store=store,
        task_runtime=runtime,
        tool_registry=registry,
        event_bus=bus,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=20, model="test-model"),
    )
    before = set(delegation._TASKS)
    try:
        yield SimpleNamespace(
            store=store, manager=manager, runtime=runtime, model=model, host=host, loop=loop, events=events, mode=mode
        )
    finally:
        model.release.set()
        tasks = list(set(delegation._TASKS) - before)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        if runtime:
            await runtime.shutdown(timeout_s=2)


def paired(c):
    started = [e for e in c.events if e.type == AgentEventType.LLM_CALL_STARTED]
    completed = [e for e in c.events if e.type == AgentEventType.LLM_CALL_COMPLETED]
    assert len(started) == len(completed) == c.model.calls
    return started, completed


@pytest.mark.parametrize("event_case", ["legacy", "tracked", "budget"], indirect=True)
async def test_same_short_call_id_in_two_sends_does_not_reuse_the_durable_attempt_id(event_case):
    c = event_case
    sid = await c.loop.new_session()
    await c.loop.send("First", sid)
    await c.loop.send("Second", sid)
    started, completed = paired(c)
    assert len(started) == 2 and started[0].data.call_id == started[1].data.call_id == "r0.a1"
    if c.mode == "legacy":
        assert all(e.data.model_attempt_id is None and not e.data.budget_managed for e in started + completed)
        return
    ids = [e.data.model_attempt_id for e in started]
    assert len(set(ids)) == 2 and all(x.startswith("attempt_") for x in ids)
    for event in started + completed:
        payload = event.data
        assert payload.model_attempt_store_id == c.store.store_id
        assert payload.budget_managed == (c.mode == "budget")
        assert payload.model_attempt_generation == 1
        [saved] = (await list_attempts(c.manager, c.runtime.actor(sid), payload.model_attempt_job_id))["items"]
        assert saved["attempt_id"] == payload.model_attempt_id and saved["status"] == "returned"
    assert {e.data.model_attempt_id for e in completed} == set(ids)


@pytest.mark.parametrize("fault", ["failure", "cancel"])
async def test_failed_or_cancelled_wire_call_keeps_the_same_original_identity(event_case, fault):
    c = event_case
    c.model.failure = fault == "failure"
    c.model.block = fault == "cancel"
    token = CancellationToken()
    task = asyncio.create_task(c.loop.send("Return", await c.loop.new_session(), stop_event=token))
    await asyncio.wait_for(c.model.entered.wait(), 5)
    if fault == "cancel":
        token.cancel("test stop")
    await asyncio.wait_for(task, 10)
    started, completed = paired(c)
    assert len(started) == 1 and started[0].data.model_attempt_id == completed[0].data.model_attempt_id
    assert completed[0].data.model_attempt_id and completed[0].data.budget_managed
    assert completed[0].data.success is False and completed[0].data.estimated is True


async def test_child_events_keep_their_own_job_and_sid_without_reusing_main_identity(event_case):
    c = event_case
    c.model.spawn = True
    sid = await c.loop.new_session()
    await c.loop.send("Create child", sid)
    await asyncio.wait_for(asyncio.gather(*list(delegation._TASKS), return_exceptions=False), 10)
    started, completed = paired(c)
    assert len(completed) == 3 and len({e.data.model_attempt_id for e in completed}) == 3
    children = [e for e in completed if e.session_id != sid]
    assert len(children) == 1
    for event in completed:
        job = await c.manager.get_job(c.runtime.actor(event.session_id), event.data.model_attempt_job_id)
        assert job.session_id == event.session_id and event.data.model_attempt_store_id == c.store.store_id
        assert event.data.budget_managed


async def test_rejected_budget_has_no_fake_call_started_or_completed_event(event_case):
    c = event_case
    # The trusted test host returns a bound lower than the request output cap.
    from power_loop.runtime.budget_contracts import BudgetPlan

    async def invalid_bound(context, request, *, service):
        return BudgetPlan(1, {"attempt_id": context["attempt_id"]})

    c.host.prepare = invalid_bound
    result = await c.loop.send("Must not dispatch", await c.loop.new_session())
    assert result.status == "degraded" and c.model.calls == 0
    assert not [e for e in c.events if e.type in (AgentEventType.LLM_CALL_STARTED, AgentEventType.LLM_CALL_COMPLETED)]
