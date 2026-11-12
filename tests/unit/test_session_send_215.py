"""P2 session-native main-agent send: the whole loop on accept/claim/final, no job."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class EchoModel(LLMService):
    """One assistant turn, optionally a tool round first."""

    def __init__(self, *, tool_rounds=0, text="done"):
        self.calls = 0
        self.tool_rounds = tool_rounds
        self.text = text
        self.requests = []

    async def complete(self, request, **kwargs):
        self.calls += 1
        self.requests.append(request)
        tools = None
        if self.calls <= self.tool_rounds:
            tools = [{"id": f"call-{self.calls}", "type": "function",
                      "function": {"name": "echo", "arguments": '{"text":"ok"}'}}]
        return LLMResponse(raw_text="" if tools else self.text, tool_calls=tools,
                           token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


def echo_registry(seen):
    registry = ToolRegistry()

    async def echo(text):
        seen.append(text)
        return text

    registry.register(ToolDefinition(
        "echo", "Return input",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        pricing=Free("fixture")), echo)
    return registry


async def test_idle_send_completes_without_job(registry_store):
    store, _ = registry_store
    model = EchoModel()
    agent = await RunningAgent(store, model, config=AgentLoopConfig(max_rounds=4, compactor=None)).start()
    try:
        result = await agent.send("hello")
        assert result.status == "completed"
        assert result.final_text == "done"
        assert result.send_index == 1
        # No job/task table exists in schema 17; the result is an ordinary assistant row.
        rows = await store.load_all_messages(agent.target.session_id)
        assert [r.role for r in rows] == ["user", "assistant"]
        assert rows[-1].content == "done"
        assert rows[-1].send_index == 1
        # Input terminal state is durable and cold-readable.
        cold = await agent.runtime.client.wait(result.input_ref, actor=agent.actor, deadline_ms=None)
        assert cold.status == "completed" and cold.final_text == "done"
        status = await agent.runtime.client.inspect(result.input_ref, actor=agent.actor)
        assert status.terminal_status == "completed" and status.result_available
    finally:
        await agent.aclose()


async def test_tool_loop_runs_and_records_pending_atomically(registry_store):
    store, _ = registry_store
    seen = []
    model = EchoModel(tool_rounds=2)
    agent = await RunningAgent(store, model, echo_registry(seen),
                               config=AgentLoopConfig(max_rounds=6, compactor=None)).start()
    try:
        result = await agent.send("run tools")
        assert result.status == "completed"
        assert seen == ["ok", "ok"]
        assert model.calls == 3
        rows = await store.load_all_messages(agent.target.session_id)
        assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
        # pending cleared after the last tool result.
        state = await store.get_state(agent.target.session_id)
        assert state.pending is None
        assert result.usage["total_tokens"] == 15
    finally:
        await agent.aclose()


class GatedToolModel(LLMService):
    """Round 1 calls a gating tool; round 2 records the request and completes."""

    def __init__(self):
        self.calls = 0
        self.requests = []

    async def complete(self, request, **kwargs):
        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            return LLMResponse(raw_text="", tool_calls=[{
                "id": "gate-1", "type": "function",
                "function": {"name": "gate", "arguments": "{}"}}],
                token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        return LLMResponse(raw_text="final", tool_calls=None,
                           token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


async def test_render_batch_text_single_item_markers():
    """P2-R1-3: a single urgent or non-user item keeps the deterministic ID/source/priority
    header; a single ordinary user input stays plain text (no regression)."""
    from power_loop.agent.follow_up import render_batch_text

    def batch(*items):
        return {"version": 1, "items": list(items)}

    normal = {"message_id": "m1", "priority": "normal", "kind": "user", "content": "hello"}
    urgent = {"message_id": "m2", "priority": "urgent", "kind": "user", "content": "stop now"}
    result = {"message_id": "m3", "priority": "normal", "kind": "agent_result", "content": "child done"}
    # Single ordinary user: plain body, unchanged.
    assert render_batch_text(batch(normal)) == "hello"
    # Single urgent: header carries message_id and priority.
    assert render_batch_text(batch(urgent)) == "[user · m2 · urgent]\n\nstop now"
    # Single non-user kind: header carries source kind.
    assert render_batch_text(batch(result)) == "[agent_result · m3 · normal]\n\nchild done"
    # Multi-item: every item keeps its header (unchanged).
    multi = render_batch_text(batch(normal, urgent))
    assert "[user · m1 · normal]" in multi and "[user · m2 · urgent]" in multi


async def test_busy_followup_is_read_at_next_boundary_not_one_round_late(registry_store):
    """design 215 §7.4: the boundary reads the CURRENT inbox version, so an input
    committed after the assistant write is seen by the very next model call."""
    store, _ = registry_store
    entered, release = asyncio.Event(), asyncio.Event()
    registry = ToolRegistry()

    async def gate():
        entered.set()
        await release.wait()
        return "tool done"

    registry.register(ToolDefinition("gate", "gate", input_schema={"type": "object", "properties": {}},
                                     pricing=Free("fixture")), gate)
    model = GatedToolModel()
    agent = await RunningAgent(store, model, registry,
                               config=AgentLoopConfig(max_rounds=6, compactor=None)).start()
    try:
        sending = asyncio.create_task(agent.send("first"))
        await entered.wait()
        # Another client commits a follow-up while the tool runs (inbox_version advances).
        receipt = await agent.runtime.client.send(
            agent.target, "urgent redirect", message_id="follow", actor=agent.actor, priority="urgent")
        assert receipt.state == "accepted"
        release.set()
        result = await sending
        assert result.status == "completed"
        # The second model request must already contain the follow-up (not one round late),
        # AND a single urgent item keeps its deterministic ID/priority header (P2-R1-3).
        second = model.requests[1]
        texts = [m.get("content") for m in second.messages if m.get("role") == "user"]
        joined = "\n".join(t if isinstance(t, str) else str(t) for t in texts)
        assert "urgent redirect" in joined, texts
        assert "follow" in joined and "urgent" in joined, texts  # message_id + priority mark
        assert "[user · follow · urgent]" in joined, texts
        # One send_index, one merged follow-up user row, both inputs terminal completed.
        rows = await store.load_all_messages(agent.target.session_id)
        assert all(r.send_index == 1 for r in rows)
        follow = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=None)
        assert follow.status == "completed" and follow.send_index == 1
    finally:
        await agent.aclose()


async def test_twenty_backlog_inputs_merge_into_one_send(registry_store):
    store, _ = registry_store
    model = EchoModel()
    agent = await RunningAgent(store, model, config=AgentLoopConfig(max_rounds=4, compactor=None)).start()
    try:
        # Hold notifications so the worker cannot claim until all 20 are accepted; then
        # one discovery scan claims the whole backlog as a single batch (one run).
        from power_loop.agent.runtime_transport import PublishReceipt

        async def held_publish(packet):
            return PublishReceipt(False)

        agent.runtime.transport.publish = held_publish
        refs = []
        for i in range(20):
            receipt = await agent.runtime.client.send(
                agent.target, f"item {i}", message_id=f"b{i}", actor=agent.actor)
            refs.append(receipt.input_ref)
        await agent.runtime.worker.recover()
        result = await agent.runtime.client.wait(refs[0], actor=agent.actor, deadline_ms=None)
        assert result.status == "completed" and result.send_index == 1
        assert model.calls == 1  # one model run for the whole batch
        rows = await store.load_all_messages(agent.target.session_id)
        user_rows = [r for r in rows if r.role == "user"]
        assert len(user_rows) == 1  # one merged user message
        for ref in refs:
            assert (await agent.runtime.client.inspect(ref, actor=agent.actor)).terminal_status == "completed"
    finally:
        await agent.aclose()


async def test_notification_loss_still_completes_via_scan(registry_store):
    """A lost activation notification must not lose the input: the periodic unfinished
    scan picks it up (design 215 §9.2)."""
    store, _ = registry_store
    model = EchoModel()
    agent = await RunningAgent(store, model, config=AgentLoopConfig(max_rounds=4, compactor=None)).start()
    try:
        # Drop every notification by failing the transport publish.
        async def lost_publish(packet):
            from power_loop.agent.runtime_transport import PublishReceipt
            return PublishReceipt(False)

        agent.runtime.transport.publish = lost_publish
        receipt = await agent.runtime.client.send(agent.target, "hello", message_id="lost", actor=agent.actor)
        assert receipt.notification == "failed"
        # Force the discovery path the worker uses on its periodic scan.
        await agent.runtime.worker.recover()
        result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=None)
        assert result.status == "completed" and result.final_text == "done"
    finally:
        await agent.aclose()


async def test_cancel_active_run_stops_at_checkpoint(registry_store):
    store, _ = registry_store
    entered, release = asyncio.Event(), asyncio.Event()
    registry = ToolRegistry()

    async def gate():
        entered.set()
        await release.wait()
        return "tool done"

    registry.register(ToolDefinition("gate", "gate", input_schema={"type": "object", "properties": {}},
                                     pricing=Free("fixture")), gate)
    model = GatedToolModel()
    agent = await RunningAgent(store, model, registry,
                               config=AgentLoopConfig(max_rounds=6, compactor=None)).start()
    try:
        sending = asyncio.create_task(agent.send("first"))
        await entered.wait()
        receipt = await agent.runtime.client.cancel(agent.target, actor=agent.actor, reason="stop now")
        assert receipt.state == "cancel_requested"
        release.set()
        result = await sending
        assert result.status == "cancelled"
        # The second model call never happens: cancel took effect at the boundary.
        assert model.calls == 1
        status = await agent.runtime.client.inspect(result.input_ref, actor=agent.actor)
        assert status.terminal_status == "cancelled"
    finally:
        await agent.aclose()


async def test_continue_completed_session_is_a_new_send(registry_store):
    store, _ = registry_store
    model = EchoModel()
    agent = await RunningAgent(store, model, config=AgentLoopConfig(max_rounds=4, compactor=None)).start()
    try:
        first = await agent.send("one")
        second = await agent.send("two")
        assert first.send_index == 1 and second.send_index == 2
        assert second.status == "completed"
        # The second send sees the first send's history (entry read), no resume needed.
        rows = await store.load_all_messages(agent.target.session_id)
        assert [r.send_index for r in rows] == [1, 1, 2, 2]
        assert model.calls == 2
    finally:
        await agent.aclose()


async def test_detached_client_waits_without_in_memory_result(registry_store):
    """A brand-new client/store connection waits on the durable result (design 215 §5.3)."""
    store, connect = registry_store
    model = EchoModel()
    agent = await RunningAgent(store, model, config=AgentLoopConfig(max_rounds=4, compactor=None)).start()
    try:
        receipt = await agent.runtime.client.send(agent.target, "hello", message_id="m1", actor=agent.actor)
        # A detached client over a fresh connection to the same database.
        other_store = await connect(schema="verify")
        from power_loop.agent.runtime_client import RuntimeClient
        from power_loop.agent.runtime_transport import MemoryMessageTransport
        detached = RuntimeClient(store=other_store, transport=MemoryMessageTransport(),
                                 namespace=agent.runtime.namespace)
        result = await detached.wait(receipt.input_ref, actor=agent.actor, deadline_ms=None)
        assert result.status == "completed" and result.final_text == "done"
        assert result.usage["total_tokens"] == 5
        await detached.aclose()
        await other_store.close()
    finally:
        await agent.aclose()
