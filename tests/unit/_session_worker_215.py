"""Real-process session worker; DSN is an argument and is never printed."""
from __future__ import annotations

import asyncio
import json
import sys

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RegistryRuntimeFactory,
    Runtime,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import InputRef
from power_loop.runtime.store.factory import open_store
from tests.unit.session_native_fixture import ACTOR


def emit(**payload):
    print(json.dumps(payload), flush=True)


class EchoModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text="done", tool_calls=None,
                           token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


async def main():
    dsn, prefix, session_id, role = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    try:
        if role != "worker_takeover":
            raise AssertionError(role)
        factory = RegistryRuntimeFactory()

        async def build(descriptor, context):
            return ExecutionEnvironment(EchoModel(), None,
                                        AgentLoopConfig(max_rounds=4, context_budget_tokens=10**9))

        factory.register("test.agent@215", build)
        runtime = Runtime(store=store, factory=factory, node_id="B", boot_id="B",
                          recovery_interval_s=0.2, timers_enabled=False)
        await runtime.worker.start()
        port = RuntimeInputStore(store)
        ref = InputRef(store.store_id, session_id, "new")
        result = None
        for _ in range(200):
            result = await port.get_result(ref, actor=ACTOR)
            if result is not None:
                break
            await asyncio.sleep(0.1)
        assert result is not None, "worker never completed the taken-over input"
        emit(event="completed", status=result.status, send_index=result.send_index)
        await runtime.aclose()
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
