"""Subprocess fixture: persist an effect/result, then die before receipt binding."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from power_loop import AgentLoopConfig, LLMResponse, LLMService, StatefulAgentLoop, ToolDefinition
from power_loop.runtime import task_runtime
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry


async def main(dsn, prefix, sid, evidence_path, stage):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="crash-test", principal_id="host", executor_id="executor",
                          boot_id="before-crash", lease_ms=300000, boot_contract_hash="c" * 64)
    evidence = Path(evidence_path)
    calls = []

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            calls.append("model")
            evidence.write_text(json.dumps(calls))
            return LLMResponse(raw_text="", tool_calls=[{
                "id": "external-call", "type": "function",
                "function": {"name": "external_effect", "arguments": "{}"}}])

        def stream(self, request):
            raise AssertionError("nonstream fixture")

        async def close(self):
            pass

    async def effect():
        execution = task_runtime._SEND.get()
        seq, call_id = current_invocation(sid)
        ref = await store.message_ref(sid, seq)
        registered = await manager.register_job(
            execution.actor, kind="subagent", parent=execution.lease, independent=True,
            call_ref=ref, tool_call_id=call_id)
        calls.append("effect")
        evidence.write_text(json.dumps(calls))
        return json.dumps({"operation_id": registered.operation.operation_id, "accepted": True})

    registry = ToolRegistry()
    registry.register(ToolDefinition("external_effect", "Fixture external acceptance", input_schema={"type": "object"},
                                     pricing=Free("fixture")), effect)
    original = task_runtime.bind_tool_receipt

    async def stop(*args, **kwargs):
        assert task_runtime.tool_receipt_required(store, sid, args[2], args[3])
        if stage == "before_bind":
            os._exit(73)
        await original(*args, **kwargs)
        if stage == "clear_failed":
            async def fail_clear(*args, **kwargs):
                raise OSError("pending_clear_failed")
            store.set_pending = fail_clear
            return
        # A crash after binding but before marker removal is conservative too.
        os._exit(74)

    task_runtime.bind_tool_receipt = stop
    loop = StatefulAgentLoop(llm=Model(), store=store, task_runtime=runtime, tool_registry=registry,
                             config=AgentLoopConfig(max_rounds=2, compactor=None, delegation_policy="single_level"))
    try:
        await loop.send("Accept the external operation once", sid)
    except OSError as exc:
        if stage != "clear_failed" or str(exc) != "pending_clear_failed":
            raise
        os._exit(75)
    raise AssertionError("crash boundary was not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
