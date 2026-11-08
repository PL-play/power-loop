"""A workflow node or sub-agent session that reaches a second send folds under its own booking."""
from __future__ import annotations

import dataclasses
import json

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    JobRef,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RuntimeRequestError,
    TargetRef,
)
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.fold import LLMSummaryFold
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Summarizer(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        return LLMResponse(raw_text="FOLDED-SUMMARY", token_usage=LLMTokenUsage(
            prompt_tokens=5, completion_tokens=2, total_tokens=7))

    def stream(self, request):
        raise NotImplementedError

    async def close(self):
        pass


def folding(config, summarizer):
    return dataclasses.replace(config, representation=ProjectedRepresentation(), context_budget_tokens=1,
                               fold_strategy=LLMSummaryFold(keep_last_sends=1, summary_llm=summarizer))


class SpawnModel(LLMService):
    def __init__(self):
        self.root, self.calls = None, {"MAIN": 0, "CHILD": 0}

    async def complete(self, request, **kwargs):
        role = "MAIN" if get_session_id() == self.root else "CHILD"
        self.calls[role] += 1
        text, calls = "done", None
        if role == "MAIN" and self.calls[role] == 1:
            text, calls = "", [{"id": "spawn-1", "type": "function", "function": {"name": "spawn_agent", "arguments": json.dumps({
                "task": "answer", "system_prompt": "child", "lifecycle": "linked", "tools": [], "max_rounds": 2,
                "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}})}}]
        elif role == "CHILD":
            text = "not json at all" if self.calls[role] == 1 else '{"ok": true}'
        return LLMResponse(raw_text=text, tool_calls=calls, token_usage=LLMTokenUsage(
            prompt_tokens=7, completion_tokens=3, total_tokens=10))

    def stream(self, request):
        raise NotImplementedError

    async def close(self):
        pass


async def test_subagent_schema_repair_send_folds_under_the_child_session(registry_store, runtime_agent):
    store, _ = registry_store
    runtime = TaskRuntime(JobManager(store), tenant_id="fold-positions", principal_id="host",
                          executor_id="executor", boot_id="boot", lease_ms=2000)
    model, summarizer, bus, events = SpawnModel(), Summarizer(), AgentEventBus(), []
    bus.subscribe(AgentEventType.LLM_CALL_STARTED, events.append)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = await runtime_agent(model=model, store=store, task_runtime=runtime, registry=registry, event_bus=bus,
        config=folding(AgentLoopConfig(delegation_policy="single_level", max_rounds=4), summarizer))
    sid = model.root = loop.target.session_id
    assert (await loop.send("delegate")).status == "completed"
    await loop.drain()
    assert model.calls["CHILD"] == 2
    child_sessions = {event.session_id for event in events} - {sid}
    assert len(child_sessions) == 1
    child = child_sessions.pop()
    from power_loop.agent.runtime_store import now_ms
    with pytest.raises(RuntimeRequestError) as rejected:
        await loop.runtime.client.send(TargetRef(store.store_id, child), "wrong child entry",
            message_id="invalid-child-input", actor=loop.actor)
    assert rejected.value.code == "invalid_options"
    rows = await store.load_all_messages(sid)
    spawn = json.loads(next(row.content for row in rows if row.tool_call_id == "spawn-1"))
    original = await loop.runtime.client.wait(JobRef(**spawn["job_ref"]), actor=loop.actor, deadline_ms=now_ms()+15000)
    receipt = await loop.runtime.client.resume(original.job_ref, body="next input",
        message_id="child-second-input", actor=loop.actor)
    result = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+15000)
    assert result.status == "completed", result
    await loop.drain()
    assert model.calls["CHILD"] == 3
    attempts = [event for event in events if event.data.purpose == "fold"]
    assert len(attempts) == summarizer.calls
    children = {event.session_id for event in attempts} - {sid}
    assert len(children) == 1
    assert (await store.latest_project_compact(children.pop())) is not None
