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


class _Gate(LLMService):
    """Scripted LLM; call #n blocks until ``gates[n]`` is set (if one is registered)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.gates: dict[int, asyncio.Event] = {}

    def gate(self, n: int) -> asyncio.Event:
        self.gates[n] = asyncio.Event()
        return self.gates[n]

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        n = len(self.calls)
        self.calls.append([dict(m) for m in request.messages])
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


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="echo", description="Echo",
                       input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                       required_params=("text",), pricing=Free("test_fixture")),
        lambda **kw: str(kw.get("text") or ""),
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
