"""The session inbox (design/124 §6): durable, deduplicated, delivered exactly once.

Covers the loop-level contract on top of the store tests in test_session_leases*.py:
- a cancelled run neither loses nor double-delivers what was waiting (the Z4 incident);
- what a person said never shares a message with a system wake-up (Z2);
- the idle path, the in-flight path and the terminal-window flush all go through the inbox;
- nothing private (the ``_inbox`` marker) ever reaches the provider;
- ordinary opening refuses old formats without changing queued originals.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from power_loop import SessionStore
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.agent.follow_up import FOLLOW_UP_MESSAGE_NAME
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]


class _Gate(LLMService):
    """Scripted LLM; call #n blocks until ``gates[n]`` is set (if one is registered)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.gates: dict[int, asyncio.Event] = {}
        self.system_prompts = []

    def gate(self, n: int) -> asyncio.Event:
        self.gates[n] = asyncio.Event()
        return self.gates[n]

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        n = len(self.calls)
        self.calls.append([dict(m) for m in request.messages])
        self.system_prompts.append(request.system_prompt)
        if n in self.gates:
            await self.gates[n].wait()
        return self.responses.pop(0) if self.responses else LLMResponse(raw_text="done")

    async def close(self) -> None:
        return None


def _tool_call(call_id: str) -> LLMResponse:
    return LLMResponse(raw_text="", tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": "echo", "arguments": '{"text":"x"}'},
    }])


def _registry(executed=None) -> ToolRegistry:
    reg = ToolRegistry()
    def echo(**kw):
        value = str(kw.get("text") or "")
        if executed is not None:
            executed.append(value)
        return value
    reg.register(
        ToolDefinition(name="echo", description="Echo",
                       input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                       required_params=("text",), pricing=Free("test_fixture")),
        echo,
    )
    return reg


async def _wait_calls(llm: _Gate, n: int) -> None:
    for _ in range(400):
        if len(llm.calls) >= n:
            return
        await asyncio.sleep(0.005)
    pytest.fail(f"expected ≥{n} LLM calls, got {len(llm.calls)}")


def _user_texts(rows: list[Any]) -> list[str]:
    return [str(r.content or "") for r in rows if r.role == "user"]


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def make_agent(store):
    agents = []
    async def make(model):
        agent = await RunningAgent(store, model, _registry()).start()
        agents.append(agent)
        return agent
    yield make
    for agent in agents:
        await agent.aclose()


async def test_duplicate_delivery_returns_original_result_without_new_model_call(store, make_agent):
    model = _Gate([LLMResponse(raw_text='first')])
    agent = await make_agent(model)
    original = await agent.send('hello', message_id='one')
    duplicate = await agent.runtime.client.send(agent.target, 'hello', message_id='one', actor=agent.actor,
        deadline_ms=(await agent.runtime.worker.inputs.input(store._db,
            original.input_ref)).row['expires_at'])
    assert not duplicate.newly_accepted and duplicate.state == 'delivered'
    result = await agent.runtime.client.wait(duplicate.input_ref, actor=agent.actor, deadline_ms=now_ms()+5000)
    assert result == original and len(model.calls) == 1
    with pytest.raises(RuntimeRequestError, match='duplicate_input_conflict'):
        await agent.runtime.client.send(agent.target, 'changed', message_id='one', actor=agent.actor)


async def test_inflight_append_preserves_image_and_hides_delivery_markers(store, make_agent):
    model = _Gate([_tool_call('c1'), LLMResponse(raw_text='done')])
    gate = model.gate(0)
    agent = await make_agent(model)
    run = asyncio.create_task(agent.send('start'))
    await _wait_calls(model, 1)
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}
    receipt = await agent.loop.follow_up(agent.target,
        {'role': 'user', 'content': [{'type': 'text', 'text': 'look'}, image]},
        message_id='append', actor=agent.actor)
    # Notification handler runs separately from the gated model.
    for _ in range(100):
        if agent.runtime.worker._active[agent.target.session_id].append_refs:
            break
        await asyncio.sleep(.001)
    gate.set()
    result = await run
    joined = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+5000)
    assert joined.job_ref == result.job_ref
    assert any(image in m.get('content', []) for m in model.calls[1] if isinstance(m.get('content'), list))
    assert not any('_inbox' in m or 'inbox' in m for call in model.calls for m in call)
    rows = await store.load_active_messages(agent.target.session_id)
    assert sum(r.name == FOLLOW_UP_MESSAGE_NAME for r in rows) == 1
    assert not any(r['status'] == 'claimed' for r in await store.inbox_items(agent.target.session_id))


async def test_idle_append_starts_one_turn_and_retry_does_not_repeat(store, make_agent):
    model = _Gate([LLMResponse(raw_text='done')])
    agent = await make_agent(model)
    first = await agent.loop.follow_up(agent.target, 'idle append', message_id='a', actor=agent.actor)
    result = await agent.runtime.client.wait(first.input_ref, actor=agent.actor, deadline_ms=now_ms()+5000)
    await agent.drain()
    second = await agent.loop.follow_up(agent.target, 'idle append', message_id='a', actor=agent.actor)
    assert not second.newly_accepted and second.job_ref == result.job_ref
    assert len(model.calls) == 1


async def test_steer_text_is_persisted_once_and_bound_to_original_job(store, make_agent):
    model = _Gate([_tool_call('c1'), LLMResponse(raw_text='done')])
    gate = model.gate(0)
    agent = await make_agent(model)
    run = asyncio.create_task(agent.send('start'))
    await _wait_calls(model, 1)
    ctx = agent.runtime.worker._active[agent.target.session_id]
    receipt = await agent.runtime.client.steer(ctx.job_ref, 'new direction', message_id='steer', actor=agent.actor)
    await asyncio.wait_for(ctx.loop.steer_event(ctx.session_id).wait(), 2)
    gate.set()
    result = await run
    assert (await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+5000)).job_ref == result.job_ref
    rows = await store.load_active_messages(agent.target.session_id)
    assert sum('new direction' in (r.content or '') for r in rows) == 1
    assert (await store.inbox_counts(agent.target.session_id))['pending'] == 0


@pytest.mark.parametrize("notification", [
    "first_only", "duplicate_delivered", "delayed_void", "received_void", "received_void_new_steer",
    "claim_first_only", "duplicate_claim", "duplicate_claim_new_steer",
])
async def test_steer_notifications_do_not_skip_later_tools(store, monkeypatch, notification):
    await _check_steer_notifications(store, monkeypatch, notification)


@pytest.mark.parametrize("registry_store", ["postgres"], indirect=True)
@pytest.mark.parametrize("notification", [
    "claim_first_only", "duplicate_read", "duplicate_read_before_claim",
    "new_read", "duplicate_read_new_steer",
])
async def test_steer_control_read_crosses_claim_completion(registry_store, monkeypatch, notification):
    store, _ = registry_store
    await _check_steer_notifications(store, monkeypatch, notification, postgres=True)


async def _check_steer_notifications(store, monkeypatch, notification, *, postgres=False):
    from power_loop import TransportPacket

    read_window = "read" in notification
    concurrent = notification in {"received_void_new_steer", "duplicate_claim_new_steer",
                                   "new_read", "duplicate_read_new_steer"}
    claim_gate = concurrent or read_window or notification in {"claim_first_only", "duplicate_claim"}
    void = notification in {"delayed_void", "received_void", "received_void_new_steer"}
    calls = [_tool_call("c1"), _tool_call("c2")]
    model = _Gate([*calls, *([_tool_call("c3")] if concurrent else []), LLMResponse(raw_text="done")])
    first, second = model.gate(0), model.gate(1)
    executed = []
    agent = await RunningAgent(store, model, _registry(executed)).start()
    worker = agent.runtime.worker
    received, release, processed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    draining, resume_drain = asyncio.Event(), asyncio.Event()
    read_waiting, resume_read = asyncio.Event(), asyncio.Event()
    reading_task = None
    delay_read = False
    in_drain = armed = False
    original = worker._control
    original_drain, transaction = worker.drain_append, worker.inputs.db.transaction

    async def drain(ctx):
        nonlocal in_drain
        in_drain = True
        try:
            return await original_drain(ctx)
        finally:
            in_drain = False

    @asynccontextmanager
    async def pause_transaction():
        # Yield after drain snapshots its refs, before acquiring any DB lock.
        if armed and in_drain and not draining.is_set():
            draining.set()
            await resume_drain.wait()
        async with transaction() as tx:
            yield tx

    async def control(ctx, packet):
        nonlocal reading_task
        if packet.input_ref and packet.input_ref.message_id in {"steer", "keep"}:
            received.set()
            if notification == "delayed_void":
                await release.wait()
            if delay_read:
                reading_task = asyncio.current_task()
            await original(ctx, packet)
            processed.set()
        else:
            await original(ctx, packet)

    if postgres:
        pool = store._db._pool
        pool_release = type(pool).release

        async def pause_release(self, connection, *, timeout=None):
            # The real SELECT has completed; only delay this control task's
            # existing connection-release await, leaving other connections free.
            if self is pool and asyncio.current_task() is reading_task and not read_waiting.is_set():
                read_waiting.set()
                await resume_read.wait()
            return await pool_release(self, connection, timeout=timeout)

        monkeypatch.setattr(type(pool), "release", pause_release)

    async def repeat_control(receipt):
        processed.clear()
        await agent.runtime.transport.publish(TransportPacket(42, agent.runtime.namespace,
            "duplicate", receipt.target, "control", receipt.input_ref))

    monkeypatch.setattr(worker, "_control", control)
    if claim_gate:
        monkeypatch.setattr(worker, "drain_append", drain)
        monkeypatch.setattr(worker.inputs.db, "transaction", pause_transaction)
    run = asyncio.create_task(agent.send("start"))
    try:
        async with asyncio.timeout(10):
            await _wait_calls(model, 1)
            ctx = worker._active[agent.target.session_id]
            receipt = await agent.runtime.client.steer(ctx.job_ref, "new direction", message_id="steer", actor=agent.actor)
            await received.wait()
            if notification == "delayed_void":
                assert (await agent.runtime.client.cancel(receipt.input_ref, actor=agent.actor)).job_ref is None
                release.set()
            await processed.wait()
            assert ctx.loop.steer_event(ctx.session_id).is_set() == (notification != "delayed_void")
            if notification.startswith("received_void"):
                cancelled = await agent.runtime.client.cancel(receipt.input_ref, actor=agent.actor)
                assert cancelled.reason == "cancelled" and cancelled.job_ref is None
            armed = claim_gate
            if notification == "duplicate_read_before_claim":
                delay_read = True
                await repeat_control(receipt)
                await read_waiting.wait()
            first.set()
            if claim_gate:
                await draining.wait()
                if concurrent:
                    processed.clear()
                    delay_read = notification == "new_read"
                    keep = await agent.runtime.client.steer(ctx.job_ref, "kept direction", message_id="keep", actor=agent.actor)
                    if delay_read:
                        await read_waiting.wait()
                    else:
                        await processed.wait()
                        assert ctx.loop.steer_event(ctx.session_id).is_set()
                if notification.startswith("duplicate") and notification != "duplicate_read_before_claim":
                    delay_read = read_window
                    await repeat_control(receipt)
                    if read_window:
                        await read_waiting.wait()
                    else:
                        await processed.wait()
                resume_drain.set()
            await _wait_calls(model, 2)
            original_state = await worker.inputs.input(worker.inputs.db, receipt.input_ref)
            assert original_state.row["status"] == ("void" if void else "delivered")
            if not void:
                assert (original_state.row["delivered_job_id"], original_state.row["delivered_job_generation"]) == (
                    ctx.job_ref.job_id, ctx.job_ref.generation)
            if concurrent:
                assert (await worker.inputs.input(worker.inputs.db, keep.input_ref)).row["status"] == "pending"
            if read_window:
                assert not ctx.claiming_refs
                assert ctx.loop.steer_event(ctx.session_id).is_set() == (notification == "duplicate_read_new_steer")
                resume_read.set()
                await processed.wait()
            second_steer = ctx.loop.steer_event(ctx.session_id).is_set()
            if notification == "duplicate_delivered":
                processed.clear()
                await agent.runtime.transport.publish(TransportPacket(42, agent.runtime.namespace,
                    "duplicate", receipt.target, "control", receipt.input_ref))
                await processed.wait()
            second.set()
            result = await run
        assert result.status == "completed"
        assert result.job_ref == ctx.job_ref
        assert not ctx.claiming_refs
        assert ctx.control_read is None
        assert executed == (["x", "x"] if notification == "delayed_void" else ["x"])
        assert second_steer == concurrent
        rows = await store.load_all_messages(agent.target.session_id)
        assert sum("new direction" in (r.content or "") for r in rows) == (not void)
        assert sum("new direction" in str(m) for m in model.calls[1]) == (not void)
        if void:
            assert "new direction" not in str(model.calls)
        if concurrent:
            assert sum("kept direction" in (r.content or "") for r in rows) == 1
            assert "kept direction" in str(model.calls[2])
            assert sum("kept direction" in str(m) for m in model.calls[2]) == 1
            kept = await worker.inputs.input(worker.inputs.db, keep.input_ref)
            assert kept.row["status"] == "delivered"
            assert (kept.row["delivered_job_id"], kept.row["delivered_job_generation"]) == (
                result.job_ref.job_id, result.job_ref.generation)
            assert [r.content for r in rows if r.role == "tool" and r.tool_call_id == "c2"][0].startswith("[skipped:")
        last_call = "c3" if concurrent else "c2"
        assert [r.content for r in rows if r.role == "tool" and r.tool_call_id == last_call] == ["x"]
    finally:
        first.set()
        second.set()
        release.set()
        resume_drain.set()
        resume_read.set()
        await agent.aclose()
        await asyncio.gather(run, return_exceptions=True)
