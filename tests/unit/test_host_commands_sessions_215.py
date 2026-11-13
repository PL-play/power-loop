"""Host commands and inline children use ordinary sessions without an assistant call."""
from __future__ import annotations

import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_types import InputRef
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.spec import AgentSpec, run_agent_spec
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class ChildOnly(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        assert request.system_prompt.startswith("CHILD")
        return LLMResponse(raw_text="inline child result")

    async def close(self):
        pass


async def test_inline_child_create_send_and_consume_without_parent_assistant(registry_store):
    store, _ = registry_store
    registry = ToolRegistry()
    model = ChildOnly()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="child", system_prompt="CHILD", tools=[], lifecycle="linked"),
            "compute",
            parent_loop=get_current_loop(),
        )

    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.pricing import Free
    registry.register(ToolDefinition("delegate", "inline child", pricing=Free("test")), delegate)
    runner = await RunningAgent(
        store, model, registry, AgentLoopConfig(compactor=None), command=("delegate", {}),
    ).start()
    try:
        result = await runner.send("run inline child")
        assert result.status == "completed"
        payload = json.loads(result.final_text)
        assert payload["final_text"] == "inline child result"
        assert payload["input_ref"]["message_id"]
        assert model.calls == 1
        children = await runner.runtime.client.list_children(runner.target, actor=runner.actor)
        assert len(children.items) == 1
        start = InputRef(store.store_id, children.items[0].session_id, payload["input_ref"]["message_id"])
        stored = await runner.runtime.worker.inputs.get_result(start, actor=runner.actor)
        assert stored.status == "completed" and stored.final_text == "inline child result"
        rows = await store._db.fetchall(
            f"SELECT kind,status,terminal_status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='agent_result'",
            (runner.target.session_id,),
        )
        assert rows == [{"kind": "agent_result", "status": "delivered", "terminal_status": "completed"}]
    finally:
        await runner.aclose()
