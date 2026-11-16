"""Power Loop 12 single-process runtime example (no Redis or broker).

Run from a fresh directory: python design215_single_process.py
"""
import asyncio
import time
import uuid

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    TrustedActor,
)
from power_loop.runtime.store.factory import open_store
from power_loop.tools.registry import ToolRegistry


class FixedModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text="done")

    def stream(self, request):
        raise NotImplementedError("This example uses nonstream responses")

    async def close(self):
        pass


async def build_demo(descriptor, execution_context):
    # Credentials, tools and clients are resolved here on this node.
    return ExecutionEnvironment(
        llm=FixedModel(), tool_registry=ToolRegistry(),
        config=AgentLoopConfig(max_rounds=descriptor.config["max_rounds"], context_budget_tokens=10**9),
    )


async def main():
    store = await open_store(":memory:", table_prefix="demo_", schema="tasks")
    factory = RegistryRuntimeFactory()
    factory.register("demo.agent@215", build_demo)
    runtime = Runtime(store=store, factory=factory, node_id="demo", boot_id=uuid.uuid4().hex, capacity=1)
    # Host-authenticated identity, never supplied by the model.
    actor = TrustedActor(tenant_id="demo", principal_id="local-user", session_id=None)
    try:
        await runtime.worker.start()
        target = await runtime.open_session(RuntimeDescriptor(
            schema_version=2, kind="agent", definition_ref="demo.agent@215",
            model_profile_ref=None, tool_profile_ref="builtin.empty@215",
            hook_profile_ref="builtin.none@215", resource_profile_ref="builtin.none@215",
            config={"max_rounds": 2}, host_context={},
        ), actor=actor)
        deadline = int(time.time() * 1000) + 30_000
        receipt = await runtime.client.send(
            target, "hello", message_id=uuid.uuid4().hex, actor=actor,
            deadline_ms=deadline,
        )
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline)
        assert result.status == "completed"
        print(result.final_text)
    finally:
        await runtime.aclose(timeout_s=5)


if __name__ == "__main__":
    asyncio.run(main())
