"""A workflow node or sub-agent session that reaches a second send folds under its own booking."""
from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime.fold import LLMSummaryFold
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import managed_workflow, records, settle

__all__ = ["registry_store", "managed_workflow"]
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


async def fold_attempts(manager):
    rows = await manager.db.fetchall(
        f"SELECT payload_json FROM {manager.events} WHERE event_type='model_attempt_prepared'")
    return [p for p in (json.loads(r["payload_json"]) for r in rows) if p.get("purpose") == "fold"]


async def test_workflow_node_second_send_folds_under_the_node_session(managed_workflow):
    store, manager, _, loop, model, sid, state, _ = managed_workflow
    from power_loop.workflow.engine import InProcessExecutor
    model.hold = set()
    summarizer = Summarizer()
    loop.config = folding(loop.config, summarizer)

    class LateInputExecutor(InProcessExecutor):
        async def run_agent(self, spec, user_input, **kwargs):
            raw = await super().run_agent(spec, user_input, **kwargs)
            if spec.name == "BUILD":
                await store.append_message(raw["session_id"], role="user", content="late-input-proof")
                raw["new_input_after_pass"] = True
            return raw

    state["executor"] = LateInputExecutor()
    await loop.send("run", sid)
    await settle(managed_workflow)
    node = next(r for r in await records(managed_workflow) if r.kind == "workflow_node" and r.title == "BUILD")
    sends = [r for r in await records(managed_workflow) if r.kind == "send" and r.parent_job_id == node.job_id]
    assert len(sends) == 2 and len({r.session_id for r in sends}) == 1
    node_sid = sends[0].session_id
    attempts = await fold_attempts(manager)
    assert len(attempts) == summarizer.calls
    assert node_sid in {a["session_id"] for a in attempts}
    assert (await store.latest_project_compact(node_sid)) is not None


class SpawnModel(LLMService):
    def __init__(self):
        self.root, self.calls = None, {"MAIN": 0, "CHILD": 0}

    async def complete(self, request, **kwargs):
        role = "MAIN" if get_session_id() == self.root else "CHILD"
        self.calls[role] += 1
        text, calls = "done", None
        if role == "MAIN" and self.calls[role] == 1:
            text, calls = "", [{"id": "spawn-1", "type": "function", "function": {"name": "spawn_agent", "arguments": json.dumps({
                "task": "answer", "system_prompt": "child", "tools": [], "max_rounds": 2,
                "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}})}}]
        elif role == "CHILD":
            text = "not json at all" if self.calls[role] == 1 else '{"ok": true}'
        return LLMResponse(raw_text=text, tool_calls=calls, token_usage=LLMTokenUsage(
            prompt_tokens=7, completion_tokens=3, total_tokens=10))

    def stream(self, request):
        raise NotImplementedError

    async def close(self):
        pass


async def test_subagent_schema_repair_send_folds_under_the_child_session(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="fold-positions", principal_id="host",
                          executor_id="executor", boot_id="boot", lease_ms=2000)
    model, summarizer = SpawnModel(), Summarizer()
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
                             config=folding(AgentLoopConfig(delegation_policy="single_level", max_rounds=4), summarizer))
    sid = await loop.new_session()
    model.root = sid
    before = set(delegation._TASKS)
    try:
        assert (await loop.send("delegate", sid)).status == "completed"
        tasks = list(set(delegation._TASKS) - before)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 8)
        assert model.calls["CHILD"] == 2
        attempts = await fold_attempts(manager)
        assert len(attempts) == summarizer.calls
        children = {a["session_id"] for a in attempts} - {sid}
        assert len(children) == 1
        assert (await store.latest_project_compact(children.pop())) is not None
    finally:
        tasks = list(set(delegation._TASKS) - before)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 8)
