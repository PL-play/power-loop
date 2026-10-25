"""Real model dispatch with a generic managed implementation and budget gates.

The finite effect and budget authority are test ports. Provider responses are
real; this does not claim a platform API transaction or supplier settlement.
"""

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.managed_effects import (
    ManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
    reconcile,
    terminate_unstarted,
)
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("pending", [False, True, "unclaimed"])
async def test_real_generic_base_admission_and_query_only_resolution(tmp_path, pending, record_property):
    store = await SessionStore.open(str(tmp_path / "managed-tool.db"), schema="tasks")
    provider, host = BoundedProvider(limit=2), RealBudgetHost()
    host.service = provider
    manager = JobManager(store)
    runtime = TaskRuntime(
        manager,
        tenant_id="generic-real",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        boot_contract_hash="d" * 64,
        budget_config=BudgetConfig("generic-real-v1", (LocalBudgetScope("authority", max_calls=2),), host),
    )
    if pending == "unclaimed":
        claim = runtime.claim

        async def fail_before_claim(actor, registration, **kwargs):
            if registration.job.kind == "background_tool":
                raise JobError("test_claim_not_started")
            return await claim(actor, registration, **kwargs)

        runtime.claim = fail_before_claim

    class SetBeacon(ManagedTool):
        prepared = 0
        executed = 0
        queries = 0
        resolved = not pending

        @property
        def contract(self):
            return ManagedToolContract("example.beacon", "a" * 64)

        async def prepare(self, context, arguments):
            self.prepared += 1
            assert arguments == {"value": 42}
            return ManagedToolPreparation({"original_call_id": context.tool_call_id, "value": arguments["value"]})

        async def execute(self, context, original_reference):
            self.executed += 1
            assert original_reference["value"] == 42
            if pending:
                raise TimeoutError("Test effect committed; its immediate receipt was lost")
            return ManagedToolResult("Beacon set to 42", {"original_receipt": "beacon-42"})

        async def recover(self, context, original_reference):
            self.queries += 1
            assert original_reference["value"] == 42
            if not self.resolved:
                return None
            return ManagedToolResult("Beacon set to 42", {"original_receipt": "beacon-42"})

    implementation = SetBeacon()
    definition = ToolDefinition(
        name="set_beacon",
        description="Set the beacon to the integer value exactly once.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer", "enum": [42]}},
            "required": ["value"],
        },
    )
    registry = ToolRegistry()
    registry.register(definition, implementation.bind(runtime, definition))
    loop = StatefulAgentLoop(
        llm=provider,
        store=store,
        task_runtime=runtime,
        tool_registry=registry,
        config=AgentLoopConfig(
            max_rounds=3,
            max_tokens=2048,
            delegation_policy="single_level",
            retry_policy=LLMRetryPolicy(max_attempts=1),
            system_prompt="Follow the user's exact tool sequence. Never call a tool more than once.",
        ),
    )

    async def only_requested_tool(ctx):
        # Select the actual available schema without fabricating provider output.
        ctx.tools = [definition.to_openai_tool()] if provider.calls == 0 else []

    loop.hooks.register(HookPoint.LLM_BEFORE, only_requested_tool, name="test.generic-managed-tool")
    sid = await loop.new_session()
    try:
        async with asyncio.timeout(150):
            try:
                result = await loop.send(
                    "Call set_beacon exactly once with value=42. After its result, reply only DONE. Do not call any other tool.",
                    sid,
                )
            except JobError as exc:
                if pending != "unclaimed" or exc.code != "managed_effect_result_missing":
                    raise
                result = None
            if pending == "unclaimed":
                assert implementation.prepared == implementation.executed == implementation.queries == 0
                [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
                assert child["generation"] == 0 and child["result_json"] is None
                negative = await terminate_unstarted(manager, runtime.actor(sid), child["job_id"])
                assert negative["resolution"] == "not_dispatched" and not negative["tool_receipt_written"]
                assert (await manager.get_job(runtime.actor(sid), child["job_id"])).supervision_status == "closed"
                assert provider.calls == host.reserve_calls == len(host.applied) == 1
                return
            assert implementation.prepared == implementation.executed == 1
            [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
            original = json.loads(child["result_json"])
            assert original["usage"] == {}
            if pending:
                assert original["managed_effect_pending"] is True
                assert child["supervision_status"] == "open"
                assert provider.calls == host.reserve_calls == len(host.applied) == 1
                assert await reconcile(runtime, runtime.actor(sid), child["job_id"], implementation) is None
                implementation.resolved = True
                recovered = await reconcile(runtime, runtime.actor(sid), child["job_id"], implementation)
                assert recovered.reference == {"original_receipt": "beacon-42"}
                await manager.finalize_supervision(runtime.actor(sid), child["job_id"], generation=1)
                assert implementation.queries == 2
                assert provider.calls == 1  # Recovery added no model or reservation.
            else:
                assert result.status == "completed" and result.final_text.strip() == "DONE"
                assert child["supervision_status"] == "closed"
                assert provider.calls == host.reserve_calls == len(host.applied) == 2
                assert implementation.queries == 0
            assert implementation.executed == 1
            assert (await manager.get_job(runtime.actor(sid), child["job_id"])).supervision_status == "closed"
    finally:
        provider.report(record_property)
        record_property("effect_executions", implementation.executed)
        record_property("effect_queries", implementation.queries)
        try:
            await loop.aclose(drain_timeout_s=0.1)
        finally:
            await provider.close()
            await store.close()
