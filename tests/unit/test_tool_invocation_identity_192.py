"""Actual call identities survive batching, retry and resume without context leaks."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    get_tool_runtime_context,
)
from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime.invocation import invocation_context
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Model(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(raw_text="", tool_calls=[{
                "id": f"call-{i}", "type": "function", "function": {"name": "inspect", "arguments": json.dumps({"number": i})}}
                for i in (1, 2)])
        return LLMResponse(raw_text="done")

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


@pytest.mark.parametrize("concurrency", [1, 2])
async def test_real_persisted_seq_and_call_ids_are_isolated_per_invocation(tmp_path, concurrency):
    store = await SessionStore.open(str(tmp_path / "invocation.db"), schema="tasks")
    registry = ToolRegistry()
    contexts = []
    first, second = asyncio.Event(), asyncio.Event()

    async def inspect(number):
        ctx = get_tool_runtime_context(required=True)
        if concurrency == 2:
            (first if number == 1 else second).set()
            await (second if number == 1 else first).wait()
        ref = await ctx.store.message_ref(ctx.session_id, ctx.assistant_seq)
        contexts.append((number, ctx.tool_call_id, ctx.assistant_seq, ref.id))
        return "inspected"

    registry.register(ToolDefinition(name="inspect", description="Inspect invocation", async_capable=True,
        input_schema={"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}, pricing=Free("test_fixture")), inspect)
    loop = StatefulAgentLoop(llm=Model(), store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=4, tool_batch_concurrency=concurrency))
    sid = await loop.new_session()
    try:
        # Existing rows make DB seq differ from a fresh send's in-memory indices.
        for _ in range(3):
            await store.append_message(sid, role="user", content="earlier")
        result = await asyncio.wait_for(loop.send("inspect", sid), 5)
        assert result.status == "completed"
        assert sorted((n, call_id) for n, call_id, _, _ in contexts) == [(1, "call-1"), (2, "call-2")]
        assert len({seq for _, _, seq, _ in contexts}) == 1
        assert contexts[0][2] > 3
        assert len({identity for _, _, _, identity in contexts}) == 1
        outside = get_tool_runtime_context()
        assert outside.assistant_seq is None and outside.tool_call_id is None
    finally:
        await store.close()


async def test_resume_replays_the_original_persisted_call_identity(tmp_path):
    store = await SessionStore.open(str(tmp_path / "resume.db"), schema="tasks")
    registry = ToolRegistry()
    observed = []

    async def inspect(number):
        ctx = get_tool_runtime_context(required=True)
        observed.append((ctx.assistant_seq, ctx.tool_call_id))
        return "replayed"

    registry.register(ToolDefinition(name="inspect", description="Resume inspector", pricing=Free("test_fixture")), inspect)
    model = Model()
    model.calls = 1  # The resumed provider round only returns its final text.
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=registry, config=AgentLoopConfig(max_rounds=2))
    sid = await loop.new_session()
    calls = [{"id": "original-call", "type": "function", "function": {"name": "inspect", "arguments": '{"number":1}'}}]
    seq = await store.append_message(sid, role="assistant", tool_calls=calls)
    await store.set_pending(sid, {"assistant_seq": seq, "round_index": 0, "tool_call_ids": ["original-call"], "tool_calls": calls})
    try:
        result = await loop.resume(sid)
        assert result.status == "completed" and observed == [(seq, "original-call")]
        assert get_tool_runtime_context().tool_call_id is None
    finally:
        await store.close()


async def test_child_session_does_not_inherit_its_parent_tool_identity():
    token = set_session_id("parent")
    try:
        with invocation_context("parent", 12, "parent-call"):
            assert get_tool_runtime_context().tool_call_id == "parent-call"
            child = set_session_id("child")
            try:
                ctx = get_tool_runtime_context()
                assert ctx.assistant_seq is None and ctx.tool_call_id is None
                with invocation_context("child", 3, "child-call"):
                    assert get_tool_runtime_context().tool_call_id == "child-call"
            finally:
                reset_session_id(child)
            assert get_tool_runtime_context().tool_call_id == "parent-call"
    finally:
        reset_session_id(token)
