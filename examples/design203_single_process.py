"""Design 203 P0 contract example. Syntax-checked in P0; runnable after P1.

These future Runtime exports are intentionally NOT implemented by P0.
Run from a fresh directory after P1: python design203_single_process.py
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
    RequestOptions,
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
        config=AgentLoopConfig(max_rounds=descriptor.config["max_rounds"], compactor=None),
    )


async def main():
    store = await open_store("design203_example_test.db", table_prefix="demo_", schema="tasks")
    factory = RegistryRuntimeFactory()
    factory.register("demo.agent@1", build_demo)
    runtime = Runtime(store=store, factory=factory, node_id="demo", boot_id=uuid.uuid4().hex, capacity=1)
    # Host-authenticated identity, never supplied by the model.
    actor = TrustedActor(tenant_id="demo", principal_id="local-user", session_id=None)
    try:
        await runtime.worker.start()
        target = await runtime.open_session(RuntimeDescriptor(
            schema_version=1, kind="agent", definition_ref="demo.agent@1",
            model_profile_ref=None, tool_profile_ref="builtin.empty@1",
            hook_profile_ref="builtin.none@1", resource_profile_ref="builtin.none@1",
            config={"max_rounds": 2}, host_context={},
        ), actor=actor)
        deadline = int(time.time() * 1000) + 30_000
        receipt = await runtime.client.send(
            target, "hello", message_id=uuid.uuid4().hex, actor=actor,
            options=RequestOptions(history=True), deadline_ms=deadline,
        )
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline)
        assert result.status == "completed"
        print(result.final_text)
    finally:
        await runtime.aclose(timeout_s=5)


if __name__ == "__main__":
    asyncio.run(main())
