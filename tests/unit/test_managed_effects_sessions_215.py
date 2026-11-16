"""Managed effects keep original-call state and recover without another execute."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, TargetRef, result_message_id
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.managed_effects import (
    FunctionManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
    reconcile,
)
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class ManagedParent(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(raw_text="", tool_calls=[{
                "id": "original-call", "type": "function", "function": {
                    "name": "post_once", "arguments": json.dumps({"text": "hello"}),
                },
            }])
        return LLMResponse(raw_text="parent done")

    async def close(self):
        pass


def _adapter(state, *, pending=False):
    async def prepare(context, arguments):
        state.prepared += 1
        assert context.tool_call_id == "original-call"
        return ManagedToolPreparation({
            "original_call": context.tool_call_id,
            "text": arguments["text"],
        })

    async def execute(context, reference):
        state.executed += 1
        if pending:
            raise TimeoutError("remote decision unknown")
        return ManagedToolResult("effect completed", {"receipt": "original"})

    async def recover(context, reference):
        state.recovered += 1
        if not state.resolved:
            return None
        return ManagedToolResult("effect recovered", {"receipt": "original"})

    return FunctionManagedTool(
        ManagedToolContract("test.original-effect", "a" * 64),
        prepare,
        execute,
        recover,
    )


def _registry(adapter):
    registry = ToolRegistry()
    definition = ToolDefinition(
        name="post_once",
        description="one managed effect",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        pricing=Free("test"),
    )
    registry.register(definition, adapter.bind(None, definition))
    return registry


def _receipt(rows):
    return json.loads(next(m.content for m in rows if m.tool_call_id == "original-call"))


async def test_prepare_execute_and_original_call_state(registry_store):
    store, _ = registry_store
    state = SimpleNamespace(prepared=0, executed=0, recovered=0, resolved=True)
    runner = await RunningAgent(
        store, ManagedParent(), _registry(_adapter(state)), AgentLoopConfig(context_budget_tokens=10**9),
    ).start()
    try:
        assert (await runner.send("post once")).status == "completed"
        start = InputRef(**_receipt(
            await store.load_all_messages(runner.target.session_id)
        )["input_ref"])
        result = await runner.runtime.client.wait(
            start, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert result.status == "completed" and result.final_text == "effect completed", result.structured_output
        assert (state.prepared, state.executed, state.recovered) == (1, 1, 0)
        rows = await store._db.fetchall(
            f"SELECT state_key,value_json FROM {store.t.session_runtime_state} "
            "WHERE session_id=?",
            (start.session_id,),
        )
        assert len(rows) == 1 and "dispatched" not in rows[0]["value_json"]
        assert '"phase":"completed"' in rows[0]["value_json"]
    finally:
        await runner.aclose()


async def test_unknown_recover_is_read_only_and_late_result_is_new_input(registry_store):
    store, _ = registry_store
    state = SimpleNamespace(prepared=0, executed=0, recovered=0, resolved=False)
    adapter = _adapter(state, pending=True)
    runner = await RunningAgent(
        store, ManagedParent(), _registry(adapter), AgentLoopConfig(context_budget_tokens=10**9),
    ).start()
    try:
        await runner.send("post once")
        start = InputRef(**_receipt(
            await store.load_all_messages(runner.target.session_id)
        )["input_ref"])
        target = TargetRef(start.store_id, start.session_id)
        unknown = await runner.runtime.client.wait(
            start, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert unknown.status == "interrupted"
        assert (state.prepared, state.executed, state.recovered) == (1, 1, 0)
        assert await reconcile(store, runner.actor, start, adapter) is None
        assert state.recovered == 1 and state.executed == 1

        state.resolved = True
        recovered = await reconcile(
            store, runner.actor, start, adapter, client=runner.runtime.client,
        )
        assert recovered.display == "effect recovered"
        assert state.recovered == 2 and state.executed == 1
        late = await runner.runtime.client.inspect(start, actor=runner.actor)
        assert late.terminal_status == "interrupted"
        parent_late = InputRef(
            store.store_id,
            runner.target.session_id,
            result_message_id(target, start, 2, "managed_effect_result"),
        )
        await runner.drain(after_input=parent_late)
        routed = await store._db.fetchone(
            f"SELECT kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (runner.target.session_id, parent_late.message_id),
        )
        assert routed["kind"] == "managed_effect_result"
        assert "effect recovered" in routed["content_json"]
    finally:
        await runner.aclose()
