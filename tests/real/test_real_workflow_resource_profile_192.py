"""Real model creates a v2-bound native workflow through the library facade."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.workflow_resources import (
    WorkflowResourceAdapter,
    WorkflowResourceBinding,
    WorkflowResourceRegistry,
    WorkflowResources,
)
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import InProcessExecutor, SharedBudget
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("deny_dispatch", [False, True])
async def test_real_native_create_saves_resource_profile_and_reauthorizes_before_child(tmp_path, record_property, deny_dispatch):
    store = await SessionStore.open(str(tmp_path / "bound-create.db"), schema="tasks")
    manager = JobManager(store)
    resources = WorkflowResourceRegistry()
    runtime = TaskRuntime(manager, tenant_id="resource-profile", principal_id="host",
        executor_id="executor", boot_id="boot", workflow_resources=resources)
    provider = BoundedProvider(limit=10)
    accepted, actual = [], []
    pool = SharedBudget(50000)
    pool.commit({"total_tokens": 11})
    authorization_calls = 0

    async def prepare(scope, contract):
        binding = WorkflowResourceBinding("test-host", "1.0.0", "original-host-resource", 1,
            scope.tenant_id, scope.principal_id, scope.main_session_id, "a" * 64, contract,
            budget_scope_id="original-pool")
        value = WorkflowResources(binding, InProcessExecutor(), budget=pool)
        actual.append(value)
        return value

    async def resolve(binding):
        assert len(actual) == 1 and binding == actual[0].binding
        return actual[0]

    async def authorize(_binding, _resources, _scope):
        nonlocal authorization_calls
        authorization_calls += 1
        return not (deny_dispatch and authorization_calls >= 3)

    resources.register("test-host", "1.0.0", WorkflowResourceAdapter(prepare, resolve, authorize))
    async def observe(admission):
        accepted.append(admission)
    tools = ToolRegistry()
    register_workflow_tool(tools, ports=WorkflowToolPorts(resource_adapter=("test-host", "1.0.0"), on_accepted=observe))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=tools,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Call the requested workflow action once, then reply WAITING. Never retry, poll, resume or create substitutes. "
                "When its result arrives, report the saved arithmetic answer or the saved failure without tools."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    spec = {"name": "real-bound-create", "root": {"type": "agent", "id": "calculate", "input": "Compute 19+23.",
        "spec": {"name": "calculate", "system_prompt": "Return only the arithmetic answer.", "tools": [], "max_rounds": 2}}}
    try:
        async with asyncio.timeout(180):
            await loop.send("Call workflow action=create exactly once with this spec: " + json.dumps(spec) +
                ". After the receipt reply WAITING; do not call any other action.", sid)
            [admission] = accepted
            saved = await manager.get_workflow_profile(runtime.actor(sid), admission.handle.job_id)
            assert saved["version"] == 2 and saved["resource_binding"] == actual[0].binding.to_dict()
            assert saved["budget"]["spent_at_admission"] == 11
            assert authorization_calls == 3
            if not deny_dispatch:
                await asyncio.wait_for(provider.child_entered.wait(), 10)
            provider.release_children.set()
            await asyncio.wait_for(admission.handle.task, 90)
            job = await manager.get_job(runtime.actor(sid), admission.handle.job_id)
            if deny_dispatch:
                assert not provider.child_entered.is_set() and pool.spent == 11
                assert job.result["status"] == "failed" and "workflow_resource_access_denied" in str(job.result)
                assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
            else:
                assert job.result["status"] == "completed" and "42" in job.result["final"]["text"]
                assert pool.spent == 11 + job.result["usage"]["total_tokens"]
                final = await loop.flush_follow_ups(sid, max_rounds=3)
                assert final is not None and "42" in final.final_text
            messages = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
            assert len(messages) == 1
            receipt = json.loads(messages[0].content)
            operation = await manager.get_operation(runtime.actor(sid), receipt["operation_id"])
            assert operation.receipt_ref["id"] == str(messages[0].id)
            assert operation.call_ref["id"] != operation.receipt_ref["id"]
            assert len(actual) == 1 and len(accepted) == 1
    finally:
        provider.release_children.set()
        try:
            await asyncio.wait_for(asyncio.gather(*(item.handle.task for item in accepted), return_exceptions=True), 15)
        finally:
            provider.report(record_property)
            await provider.close()
            await store.close()
