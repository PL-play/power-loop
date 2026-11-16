"""Real independent driver/leaf processes for the session-native P4 contract."""

import asyncio
import json
import sys

from power_loop import (
    AgentLoopConfig,
    DatabasePollingTransport,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.runtime.store.factory import open_store
from power_loop.tools.registry import ToolRegistry


def report(**data):
    print(json.dumps(data), flush=True)


class Model(LLMService):
    def __init__(self, role, gate):
        self.role, self.gate = role, gate

    async def complete(self, request, **kwargs):
        report(event="node_started", role=self.role)
        await self.gate.wait()
        return LLMResponse(raw_text="two process done")

    async def close(self):
        pass


async def main():
    dsn, prefix, coordinator_path, role = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    gate = asyncio.Event()
    factory = RegistryRuntimeFactory()

    def can_execute(descriptor):
        if role.startswith("C"):
            return descriptor.kind == "workflow_driver"
        return descriptor.kind == "agent" and "workflow_node" in descriptor.config

    async def build(descriptor, context):
        if descriptor.kind == "workflow_driver":
            report(event="driver_claimed", role=role, send_index=context.send_index)
        return ExecutionEnvironment(
            Model(role, gate), ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("p4.agent@215", build, can_execute=can_execute)
    runtime = Runtime(
        store=store,
        factory=factory,
        node_id=role,
        boot_id=role,
        capacity=2,
        transport=DatabasePollingTransport(RuntimeInputStore(store), interval_s=0.03),
        session_lease_ttl_s=1,
        recovery_interval_s=0.05,
        shared_execution=True,
        timers_enabled=False,
    )
    try:
        await runtime.worker.start()
        report(event="ready", role=role)
        while True:
            command = (await asyncio.to_thread(sys.stdin.readline)).strip()
            if not command or command == "quit":
                break
            if command == "release":
                gate.set()
                report(event="released", role=role)
    finally:
        await runtime.aclose(timeout_s=2, cancel_running=True)


if __name__ == "__main__":
    asyncio.run(main())
