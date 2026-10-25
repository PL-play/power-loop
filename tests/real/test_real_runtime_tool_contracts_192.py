"""Real model uses independently named host control tools through the SDK base."""

import asyncio

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.runtime.tool_contracts import RuntimeTool, ToolDispatchPolicy, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("name", ["read_probe", "task_snapshot"])
async def test_real_host_control_base_needs_no_builtin_tool_name(tmp_path, record_property, name):
    store = await SessionStore.open(str(tmp_path / "generic-control.db"), schema="tasks")
    provider, host = BoundedProvider(limit=2), RealBudgetHost()
    host.service = provider
    runtime = TaskRuntime(
        JobManager(store),
        tenant_id="generic-control",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        boot_contract_hash="e" * 64,
        budget_config=BudgetConfig("control-v1", (LocalBudgetScope("authority", max_calls=2),), host),
    )

    class OriginalSnapshot(RuntimeTool):
        calls = 0

        @property
        def runtime_contract(self):
            return ToolRuntimeContract("example.original-snapshot", allowed_roles=("main",))

        def dispatch_policy(self, arguments, *, runtime=None):
            return ToolDispatchPolicy(native_admitted=True, control_only=True)

        async def __call__(self, **arguments):
            self.calls += 1
            assert arguments == {}
            execution = _SEND.get()
            original = await runtime.manager.get_job(execution.actor, execution.lease.job_id)
            assert original.status == "running"
            return "SNAPSHOT_OK"

    implementation, registry = OriginalSnapshot(), ToolRegistry()
    definition = ToolDefinition(
        name, "Read the original task snapshot once. No arguments.", input_schema={"type": "object", "properties": {}}
    )
    registry.register(definition, implementation)
    loop = StatefulAgentLoop(
        llm=provider,
        store=store,
        task_runtime=runtime,
        tool_registry=registry,
        config=AgentLoopConfig(
            max_rounds=3, max_tokens=2048, delegation_policy="single_level", retry_policy=LLMRetryPolicy(max_attempts=1)
        ),
    )

    async def schema(ctx):
        ctx.tools = [definition.to_openai_tool()] if provider.calls == 0 else []

    loop.hooks.register(HookPoint.LLM_BEFORE, schema, name="test.generic-control-schema")
    try:
        async with asyncio.timeout(150):
            sid = await loop.new_session()
            result = await loop.send(
                f"Call {name} exactly once with no arguments. After SNAPSHOT_OK reply only DONE. Never call other tools.",
                sid,
            )
            assert result.status == "completed" and result.final_text.strip() == "DONE"
            assert implementation.calls == 1
            assert provider.calls == host.reserve_calls == len(host.applied) == 2
            assert len(await runtime.manager.db.fetchall(f"SELECT job_id FROM {runtime.manager.jobs}")) == 1
    finally:
        provider.report(record_property)
        record_property("snapshot_queries", implementation.calls)
        try:
            await loop.aclose(drain_timeout_s=0.1)
        finally:
            await provider.close()
            await store.close()
