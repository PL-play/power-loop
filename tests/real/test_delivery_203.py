"""Representative live-provider checks through the current client/worker API."""
import time
import uuid
from contextlib import asynccontextmanager

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMSummaryFold,
    ProjectedRepresentation,
    RegistryRuntimeFactory,
    RequestOptions,
    Runtime,
    RuntimeDescriptor,
    SessionStore,
    StructuredOutputSpec,
    ToolRegistry,
    TrustedActor,
    parse_structured,
)

from ._llm import make_llm

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@asynccontextmanager
async def live_runtime(tmp_path, *, config=None):
    store = await SessionStore.open(str(tmp_path / "real_delivery_test.db"))
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(make_llm(max_tokens=4096, temperature=0), ToolRegistry(),
                                    config or AgentLoopConfig(max_rounds=2, max_tokens=4096))
    factory.register("delivery.agent@1", build)
    runtime = Runtime(store=store, factory=factory, node_id="p7-real", recovery_interval_s=None)
    actor = TrustedActor("p7-test", "reviewer", None)
    try:
        await runtime.worker.start()
        target = await runtime.open_session(RuntimeDescriptor(
            1, "agent", "delivery.agent@1", None, "empty@1", "none@1", "none@1", {}, {}), actor=actor)
        async def send(body, **options):
            deadline = int(time.time() * 1000) + 120_000
            receipt = await runtime.client.send(target, body, message_id=uuid.uuid4().hex,
                actor=actor, deadline_ms=deadline, options=RequestOptions(**options))
            result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline)
            while runtime.worker._tasks:
                import asyncio
                await asyncio.wait(tuple(runtime.worker._tasks))
            assert result.status == "completed", result
            return result
        yield store, target, send
    finally:
        await runtime.aclose(timeout_s=5, cancel_running=True)


async def test_live_structured_output(tmp_path):
    schema = {"type": "object", "properties": {"number": {"type": "integer"}},
              "required": ["number"], "additionalProperties": False}
    spec = StructuredOutputSpec(name="Number", schema=schema)
    async with live_runtime(tmp_path) as (_, _, send):
        result = await send("Return a JSON object with number equal to 37.",
                            response_format=spec.to_openai_response_format())
        assert parse_structured(result.final_text, schema=schema) == {"number": 37}


async def test_live_repaired_tool_history(tmp_path):
    async with live_runtime(tmp_path) as (store, target, send):
        await store.append_message(target.session_id, role="assistant", tool_calls=[{
            "id": "unknown-call", "type": "function", "function": {"name": "echo", "arguments": "{}"}}])
        before = await store.load_all_messages(target.session_id)
        result = await send("The old tool result is unknown. Do not repeat it. Reply with the number 37.")
        assert "37" in result.final_text
        assert (await store.load_all_messages(target.session_id))[:len(before)] == before


async def test_live_summary_and_next_send(tmp_path):
    config = AgentLoopConfig(max_rounds=2, max_tokens=4096, context_budget_tokens=40,
        representation=ProjectedRepresentation(), fold_strategy=LLMSummaryFold(keep_last_sends=1))
    async with live_runtime(tmp_path, config=config) as (store, target, send):
        for index in range(3):
            await send(f"Remember project code BLUE-37. Acknowledge briefly. Message {index}. " + "context " * 80)
        compact = await store.latest_project_compact(target.session_id)
        assert compact is not None and compact.content.get("summary")
        result = await send("What is the project code? Reply with the code only.")
        assert "BLUE-37" in result.final_text
