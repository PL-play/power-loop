"""Cumulative continuation evidence must charge actual usage exactly once."""

from __future__ import annotations

import pytest

from power_loop import LLMResponse, LLMTokenUsage, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.workflow_resources import (
    WorkflowResourceAdapter,
    WorkflowResourceBinding,
    WorkflowResourceRegistry,
    WorkflowResources,
    execution_contract,
)
from power_loop.workflow import InProcessExecutor, SharedBudget, WorkflowSpec, create_workflow
from power_loop.workflow.engine import WorkflowEngine

from .test_job_registry_192 import registry_store
from .test_workflow import _FakeLLM
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "managed_workflow"]

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("ceiling,expected_calls,expected_spend", [(1000, 2, 17), (5, 0, 5), (10, 1, 12)])
async def test_continuations_charge_once_and_observe_spend_before_next_request(ceiling, expected_calls, expected_spend):
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(llm=_FakeLLM(), store=store)
    pool = SharedBudget(ceiling)
    calls = []
    sid = await store.create_session(system_prompt="work")

    class Executor:
        async def run_agent(self, *args, **kwargs):
            return {
                "status": "hit_round_limit",
                "final_text": "partial",
                "session_id": sid,
                "rounds": 1,
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }

        async def continue_agent(self, *args, **kwargs):
            calls.append(args)
            done = len(calls) == 2
            usage = (
                {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
                if done
                else {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
            )
            return {
                "status": "completed" if done else "hit_round_limit",
                "final_text": "done" if done else "partial",
                "session_id": sid,
                "rounds": 1,
                "usage": usage,
            }

    spec = WorkflowSpec.from_json(
        {
            "name": "continued",
            "root": {
                "type": "agent",
                "id": "a",
                "spec": {"name": "a", "system_prompt": "work"},
                "continuation": {"max_continuations": 2, "extra_rounds": 1, "gate": "always"},
            },
        }
    )
    try:
        result = await WorkflowEngine(loop, executor=Executor(), budget=pool).run(spec)
        assert len(calls) == expected_calls
        assert pool.spent == expected_spend
        assert result.usage["total_tokens"] == expected_spend
        assert result.results["a"].usage["total_tokens"] == expected_spend
    finally:
        await store.close()


async def test_preexisting_shared_spend_and_failed_continuation_are_preserved():
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(llm=_FakeLLM(), store=store)
    pool = SharedBudget(1000)
    pool.commit({"total_tokens": 13})
    sid = await store.create_session(system_prompt="work")

    class Executor:
        async def run_agent(self, *args, **kwargs):
            return {
                "status": "hit_round_limit",
                "final_text": "partial",
                "session_id": sid,
                "usage": {"total_tokens": 5},
            }

        async def continue_agent(self, *args, **kwargs):
            return {"status": "failed", "final_text": "failure", "session_id": sid, "usage": {"total_tokens": 7}}

    spec = WorkflowSpec.from_json(
        {
            "name": "continued",
            "root": {
                "type": "agent",
                "id": "a",
                "spec": {"name": "a", "system_prompt": "work"},
                "continuation": {"max_continuations": 2, "extra_rounds": 1, "gate": "always"},
            },
        }
    )
    try:
        result = await WorkflowEngine(loop, executor=Executor(), budget=pool).run(spec)
        assert result.results["a"].status == "failed"
        assert pool.spent == 25 and result.usage["total_tokens"] == 12
        assert result.results["a"].usage["total_tokens"] == 12
    finally:
        await store.close()


async def test_native_automatic_continuation_keeps_pool_equal_to_canonical_usage(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    node = state["spec"]["root"]["steps"][0]
    node["spec"].update(max_rounds=1, tools=["touch"])
    node["continuation"] = {"max_continuations": 1, "extra_rounds": 2, "gate": "always"}
    original = model.complete
    first = True

    async def complete(request, **kwargs):
        nonlocal first
        if first and get_session_id() != sid and "ROLE-BUILD" in (request.system_prompt or ""):
            first = False
            model.counts["BUILD"] += 1
            return LLMResponse(
                raw_text="",
                tool_calls=[{"id": "touch-call", "type": "function", "function": {"name": "touch", "arguments": "{}"}}],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            )
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", complete)

    async def touch():
        return "done"

    loop.tool_registry.register(
        ToolDefinition(name="touch", description="One local read", input_schema={"type": "object"}), touch
    )
    spec = WorkflowSpec.from_json(state["spec"])
    binding = WorkflowResourceBinding(
        "native-continuation",
        "1.0.0",
        "original-resource",
        1,
        runtime.tenant_id,
        runtime.principal_id,
        sid,
        "a" * 64,
        execution_contract(spec, ["touch"]),
        budget_scope_id="original-pool",
    )
    pool = SharedBudget(10000)
    pool.commit({"total_tokens": 7})
    resource = WorkflowResources(binding, InProcessExecutor(), budget=pool)

    async def prepare(*args):
        return resource

    async def resolve(*args):
        return resource

    async def authorize(*args):
        return True

    registry = WorkflowResourceRegistry()
    registry.register("native-continuation", "1.0.0", WorkflowResourceAdapter(prepare, resolve, authorize))
    runtime.workflow_resources = registry

    async def launch():
        handle = await create_workflow(
            spec, parent_loop=loop, parent_session_id=sid, allowed_tools=["touch"], resource_binding=binding
        ).start(detached=True)
        state["handles"].append(handle)
        return handle.run_id

    loop.tool_registry.register(loop.tool_registry.get("launch").definition, launch, overwrite=True)
    await loop.send("launch", sid)
    await settle(managed_workflow)
    job = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert job.result["status"] == "completed", job.result
    assert model.counts["BUILD"] == 3
    assert job.result["usage"]["total_tokens"] == 40
    assert pool.spent == 47, (pool.spent, job.result["usage"], dict(model.counts))
