"""Real-process parent/child worker for design 215 P3."""
from __future__ import annotations

import asyncio
import json
import os
import sys

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    DatabasePollingTransport,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import InputRef, TargetRef, TrustedActor
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.delegation import start
from power_loop.runtime.store.factory import open_store
from power_loop.testing.db_session_ownership import DatabaseSessionOwnershipCoordinator
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent


def emit(**payload):
    print(json.dumps(payload), flush=True)


class Model(LLMService):
    def __init__(self):
        self.spawned = False

    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith("CHILD"):
            return LLMResponse(raw_text="child result")
        latest = next((m.get("content", "") for m in reversed(request.messages) if m.get("role") == "user"), "")
        if "agent_result" in latest:
            self.spawned = True
            return LLMResponse(raw_text="parent handled")
        if not self.spawned or latest == "delegate again":
            call_id = "spawn" if not self.spawned else "spawn-again"
            self.spawned = True
            return LLMResponse(raw_text="", tool_calls=[{
                "id": call_id, "type": "function", "function": {
                    "name": "spawn_agent",
                    "arguments": json.dumps({"task": "compute", "system_prompt": "CHILD", "tools": []}),
                },
            }])
        return LLMResponse(raw_text="parent done")

    async def close(self):
        pass


async def main():
    dsn, prefix, parent_sid, coordinator_path, role = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    port = RuntimeInputStore(store)
    actor = TrustedActor("tenant", "host", None)
    parent = TargetRef(store.store_id, parent_sid)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    if role == "crash":
        async def crash_spawn(**kwargs):
            receipt = await start(
                get_current_loop(),
                AgentSpec(name="child", system_prompt="CHILD", tools=[], lifecycle="linked"),
                kwargs.get("task", "compute"),
            )
            emit(event="child_created", receipt=receipt)
            os._exit(0)
        from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract
        from power_loop.tools.spawn_agent import SPAWN_AGENT_DEFINITION
        registry.register(
            SPAWN_AGENT_DEFINITION,
            FunctionRuntimeTool(
                crash_spawn,
                ToolRuntimeContract("test.crash-spawn", background_allowed=False),
            ),
            overwrite=True,
        )
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        return ExecutionEnvironment(Model(), registry, AgentLoopConfig(max_rounds=4, compactor=None))

    factory.register(
        "test.agent@215", build,
        can_execute=lambda descriptor: ("child_spec" in descriptor.config) == (role == "B"),
    )
    runtime = Runtime(
        store=store, factory=factory, node_id=role, boot_id=role,
        coordinator=DatabaseSessionOwnershipCoordinator(coordinator_path),
        transport=DatabasePollingTransport(port, interval_s=0.05, control_interval_s=0.05),
        shared_execution=True, recovery_interval_s=0.1, session_lease_ttl_s=2,
        timers_enabled=False, capacity=2,
    )
    try:
        await runtime.worker.start()
        if role in {"A", "crash"}:
            receipt = await runtime.client.send(parent, "delegate", message_id="parent-start", actor=actor)
            result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=None)
            rows = await store._db.fetchall(
                f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='tool' AND tool_call_id='spawn'",
                (parent_sid,),
            )
            content = rows[-1]["content"] if rows else ""
            emit(event="parent_finished", status=result.status,
                 receipt=json.loads(content) if content.startswith("{") else {"error": content})
        else:
            emit(event="ready")
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line or line.strip() == "quit":
                break
            command = json.loads(line)
            action = command["action"]
            if action == "wait":
                ref = InputRef(**command["input_ref"])
                result = await runtime.client.wait(ref, actor=actor, deadline_ms=None)
                emit(event="waited", status=result.status, final_text=result.final_text,
                     send_index=result.send_index, input_ref=command["input_ref"])
            elif action == "send":
                receipt = await runtime.client.send(
                    parent, command["body"], message_id=command["message_id"], actor=actor,
                )
                result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=None)
                emit(event="sent", status=result.status, final_text=result.final_text,
                     send_index=result.send_index, message_id=command["message_id"])
            elif action == "cancel_subtree":
                receipt = await runtime.client.cancel(parent, actor=actor, scope="subtree", reason="stop tree")
                emit(event="cancelled", state=receipt.state)
            elif action == "read_repair":
                rows = []
                for _ in range(200):
                    rows = await store._db.fetchall(
                        f"SELECT content,meta_json FROM {store.t.messages} "
                        "WHERE session_id=? AND role='tool' AND tool_call_id='spawn' ORDER BY seq",
                        (parent_sid,),
                    )
                    if rows:
                        break
                    await asyncio.sleep(0.05)
                emit(event="repair", content=json.loads(rows[-1]["content"]),
                     meta=json.loads(rows[-1]["meta_json"] or "{}"))
    finally:
        await runtime.aclose(timeout_s=2, cancel_running=True)


if __name__ == "__main__":
    asyncio.run(main())
