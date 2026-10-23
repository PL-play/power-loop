"""Real provider behind a trusted registry; not a native restoration test yet."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime.jobs import JobError, _hash
from power_loop.runtime.workflow_resources import (
    WorkflowResourceAdapter,
    WorkflowResourceBinding,
    WorkflowResourceRegistry,
    WorkflowResources,
    WorkflowResourceScope,
)
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import InProcessExecutor, SharedBudget, WorkflowSpec, create_workflow

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_registered_resources_preserve_pool_and_deny_changed_original_before_provider(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "resources.db"))
    provider = BoundedProvider(limit=3)
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=ToolRegistry(),
        config=AgentLoopConfig(max_tokens=2048, max_rounds=2))
    sid = await loop.new_session()
    spec = WorkflowSpec.from_json({"name": "registered-host", "root": {
        "type": "agent", "id": "calculate", "input": "Compute 19+23.",
        "spec": {"name": "calculate", "system_prompt": "Return only the arithmetic answer.",
            "tools": [], "max_rounds": 2}}})
    contract = _hash({"spec": spec.to_dict(), "allowed_tools": []})
    scope = WorkflowResourceScope("resource-tenant", "resource-host", sid)
    original = WorkflowResourceBinding("test-host", "1.0.0", "original-host-resource", 1,
        scope.tenant_id, scope.principal_id, sid, "a" * 64, contract, budget_scope_id="original-pool")
    pool = SharedBudget(50000)
    pool.commit({"total_tokens": 7})
    resources = WorkflowResources(original, InProcessExecutor(), budget=pool)
    access, calls = True, []

    async def prepare(_scope, _contract):
        calls.append("prepare")
        return resources

    async def resolve(binding):
        calls.append("resolve")
        assert binding == original
        return resources

    async def authorize(binding, actual, actor):
        calls.append("authorize")
        return (access and binding == original and actual is resources and actor == scope)

    adapter = WorkflowResourceAdapter(prepare, resolve, authorize)
    registry = WorkflowResourceRegistry()
    registry.register("test-host", "1.0.0", adapter)
    try:
        async with asyncio.timeout(120):
            prepared = await registry.prepare("test-host", "1.0.0", scope=scope, contract_sha256=contract)
            saved = json.loads(json.dumps(prepared.binding.to_dict()))
            # A fresh host registry must register the original version explicitly.
            next_registry = WorkflowResourceRegistry()
            next_registry.register("test-host", "1.0.0", adapter)
            with pytest.raises(JobError, match="scope_mismatch"):
                await next_registry.resolve(saved, scope=replace(scope, tenant_id="foreign"), contract_sha256=contract)
            assert provider.calls == 0 and calls == ["prepare", "authorize"]
            restored = await next_registry.resolve(saved, scope=scope, contract_sha256=contract)
            assert restored.budget is pool and restored.budget.spent == 7
            workflow = create_workflow(spec, parent_loop=loop, parent_session_id=sid,
                allowed_tools=[], executor=restored.executor, budget=restored.budget)
            result = await workflow.run()
            assert result.status == "completed" and result.final is not None and "42" in result.final.text
            assert 0 < provider.calls <= 3
            assert pool.spent == 7 + result.usage["total_tokens"]
            spent, requests = pool.spent, provider.calls
            access = False
            with pytest.raises(JobError, match="access_denied"):
                await next_registry.resolve(saved, scope=scope, contract_sha256=contract)
            assert pool.spent == spent and provider.calls == requests
    finally:
        provider.report(record_property)
        await provider.close()
        await store.close()
