"""Independent driver/leaf process for the P4 graph contract."""

import asyncio
import json
import sys

from power_loop import (
    AgentLoopConfig,
    DatabasePollingTransport,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RegistryRuntimeFactory,
    Runtime,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.testing.db_session_ownership import DatabaseSessionOwnershipCoordinator
from power_loop.tools import create_default_tool_registry

from .test_model_budgets_192 import Host


def report(**data):
    print(json.dumps(data), flush=True)


async def main():
    dsn, prefix, coord, role, workspace, scenario = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    gate = asyncio.Event()
    auto_calls = 0

    class Coordinator(DatabaseSessionOwnershipCoordinator):
        async def acquire(self, key, **kwargs):
            metadata = await store._db.fetchone(
                f"SELECT metadata_json FROM {store.t.sessions} WHERE session_id=?", (key[1],)
            )
            descriptor = json.loads(metadata["metadata_json"])["runtime_descriptor"]
            if role.startswith("C"):
                permitted = descriptor["kind"] == "workflow_driver"
            else:
                permitted = (
                    descriptor["kind"] == "workflow_node" and descriptor["config"]["child_spec"]["name"] == role[0]
                )
            return await super().acquire(key, **kwargs) if permitted else None

    registry = create_default_tool_registry(workspace_dir=workspace)

    async def hold():
        from power_loop.runtime.task_runtime import _SEND

        allocation = _SEND.get().resource_evidence["_control_allocation"]
        from pathlib import Path

        with (Path(workspace) / (role[0] + "-actions")).open("a") as marker:
            marker.write("external action\n")
        report(event="running", calls=allocation.calls, used=allocation.used_calls)
        await gate.wait()
        return "released"

    registry.register(
        ToolDefinition(
            name="hold",
            description="hold leaf",
            input_schema={"type": "object"},
            interrupt="abort",
            pricing=Free("local"),
        ),
        hold,
    )

    async def progress():
        from pathlib import Path

        from power_loop.core.agent_context import get_session_id

        with (Path(workspace) / "A-progress").open("a") as marker:
            marker.write("progress\n")
        await store.set_runtime_state(get_session_id(), "todo", {
            "items": [{"text": "finish A", "status": "pending" if auto_calls < 3 else "completed"}]
        })
        if auto_calls == 3 and scenario == "auto_pending":
            report(event="continuing")
            await gate.wait()
        return "saved"

    registry.register(
        ToolDefinition(name="progress", description="advance todo", input_schema={"type": "object"},
                       pricing=Free("local")), progress,
    )

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal auto_calls
            usage = LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            if scenario.startswith("auto_") and role == "A":
                auto_calls += 1
                if auto_calls <= 3:
                    return LLMResponse(raw_text="", token_usage=usage, tool_calls=[{
                        "id": f"progress-{auto_calls}", "type": "function",
                        "function": {"name": "progress", "arguments": "{}"},
                    }])
                report(event="completed", text="A done")
                return LLMResponse(raw_text="A done", token_usage=usage)
            if not any(m.get("role") == "tool" for m in request.messages):
                return LLMResponse(
                    raw_text="",
                    token_usage=usage,
                    tool_calls=[{"id": "hold", "type": "function", "function": {"name": "hold", "arguments": "{}"}}],
                )
            report(event="completed", text=request.system_prompt)
            return LLMResponse(raw_text="done:" + request.system_prompt, token_usage=usage)

        async def close(self):
            pass

    class Provider(Host):
        async def prepare_tool(self, context, call, *, pricing):
            return None

        async def prepare(self, context, logical_request, *, service):
            return BudgetPlan(logical_request.max_tokens, {"attempt_id": context["attempt_id"], "origin": context})

    provider = Provider()
    config = BudgetConfig(
        "p4-shared",
        (
            LocalBudgetScope("root", max_calls=20, binding="root"),
            LocalBudgetScope("leaf", max_calls=4 if scenario.startswith("auto_") else 2,
                             binding="owner", owner_kinds=("workflow_node",)),
        ),
        provider,
        max_pending=64,
    )
    task_runtime = TaskRuntime(
        JobManager(store),
        tenant_id="tenant",
        principal_id="principal",
        executor_id=role,
        boot_id=role,
        boot_contract_hash="a" * 64,
        budget_config=config,
    )
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        return ExecutionEnvironment(
            Model(), registry, AgentLoopConfig(compactor=None, max_tokens=10), task_runtime=task_runtime
        )

    factory.register("p4.agent@1", build)
    runtime = Runtime(
        store=store,
        factory=factory,
        node_id=role,
        boot_id=role,
        capacity=2,
        coordinator=Coordinator(coord),
        transport=DatabasePollingTransport(RuntimeInputStore(store), interval_s=0.03, control_interval_s=0.03),
        # The normal live-leaf assertions must not inject an accidental lease
        # loss when the three database suites contend for the same host.
        # The parent test still kills the driver explicitly to test takeover.
        session_lease_ttl_s=5,
        recovery_interval_s=0.15,
        shared_execution=True,
    )
    try:
        await runtime.worker.start()
        report(event="ready")
        while True:
            command = (await asyncio.to_thread(sys.stdin.readline)).strip()
            if not command or command == "quit":
                break
            if command == "release":
                gate.set()
            if command == "budget":
                report(event="budget", reserved=provider.reserve_calls, settled=provider.apply_calls)
    finally:
        await task_runtime.shutdown(timeout_s=2)
        await runtime.aclose(timeout_s=2, cancel_running=True)


if __name__ == "__main__":
    asyncio.run(main())
