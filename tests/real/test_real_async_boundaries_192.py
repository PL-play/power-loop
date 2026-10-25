"""Bounded real-provider acceptance for async ownership and cancellation.

Uses the existing .env and make_llm client. Only local evidence tools are enabled;
there is no paid image generation, business database or production conversation.
The wrapper bounds actual requests and records usage without a judge-model call.
"""

from __future__ import annotations

import asyncio
import json
import secrets

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    AgentSpec,
    CancellationToken,
    LLMService,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    create_default_tool_registry,
)
from power_loop.agent.follow_up import InboxItem
from power_loop.core.agent_context import get_current_loop, get_session_id, reset_session_id, set_session_id
from power_loop.core.pass_turn import request_pass
from power_loop.runtime import delegation
from power_loop.runtime.background import record_key
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools import default_tools
from power_loop.tools.spawn_agent import register_spawn_agent

from ._llm import make_llm

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


class BoundedProvider(LLMService):
    """Keep the provider real; gates only control when a child may start its request."""

    def __init__(self, *, limit=8):
        self.inner = make_llm(max_tokens=2048, temperature=0)
        self.limit = limit
        self.calls = 0
        self.refused_calls = 0
        self.responses = []
        self.root_sid = None
        self.hold_children = False
        self.child_entered = asyncio.Event()
        self.release_children = asyncio.Event()

    async def complete(self, request, **kwargs):
        if self.hold_children and get_session_id() != self.root_sid:
            self.child_entered.set()
            await self.release_children.wait()
        if self.calls >= self.limit:
            self.refused_calls += 1
            raise AssertionError(f"real-provider test exceeded {self.limit} requests")
        self.calls += 1
        response = await asyncio.wait_for(self.inner.complete(request, **kwargs), 60)
        self.responses.append(response)
        return response

    def stream(self, request):
        return self.inner.stream(request)

    async def close(self):
        await self.inner.close()

    def report(self, record_property):
        record_property("real_provider_requests", self.calls)
        record_property("local_provider_calls_refused", self.refused_calls)
        record_property("reported_tokens", sum(
            getattr(getattr(r, "token_usage", None), "total_tokens", 0) or 0
            for r in self.responses))


async def settle_children():
    tasks = list(delegation._TASKS)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=False), 120)


@pytest.mark.parametrize("ending", ["natural", "pass_turn"])
async def test_real_catalog_child_result_returns_after_parent_yields(
        tmp_path, ending, record_property):
    """Real parent chooses spawn, yields, then actually consumes its real child's output."""
    store = await SessionStore.open(str(tmp_path / "child.db"))
    llm = BoundedProvider(limit=7)
    registry = create_default_tool_registry(include=["recall_send"], workspace_dir=tmp_path)
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry, config=AgentLoopConfig(
        system_prompt=("Follow the requested tool sequence exactly. Upon receiving a subagent_result "
                       "completion event, report its answer and do not start or nudge another task."),
        max_rounds=4, max_tokens=2048, representation=ProjectedRepresentation()))
    sid = await loop.new_session()
    llm.root_sid, llm.hold_children = sid, True
    instruction = ("call pass_turn with reason waiting for child" if ending == "pass_turn"
                   else "finish with a short text and no further tool calls; do not call pass_turn")
    try:
        async with asyncio.timeout(180):
            first = await loop.send(
                "Use spawn_agent exactly once to calculate 19+23. Give the child tools=[] and "
                "instruct it to reply with only the number. After the acceptance receipt, "
                + instruction + ". The acceptance receipt is not the calculation result.", sid)
            await asyncio.wait_for(llm.child_entered.wait(), 10)
            assert first.status == "completed"
            assert first.passed == (ending == "pass_turn")
            children = await store.list_children(sid)
            assert len(children) == 1
            active = delegation.running_jobs(store, sid)
            assert len(active) == 1 and active[0]["session_id"] == children[0].session_id
            assert (await store.inbox_counts(sid))["pending"] == 0
            llm.release_children.set()
            await settle_children()
            assert (await store.inbox_counts(sid))["pending"] == 1
            continued = await loop.flush_follow_ups(sid, max_rounds=3)
            assert continued and continued.status == "completed"
            assert "42" in continued.final_text
            assert len(await store.list_children(sid)) == 1
            assert not delegation.running_jobs(store, sid)
            assert (await store.inbox_counts(sid))["pending"] == 0
            child_rows = await store.load_all_messages(children[0].session_id)
            assert any(r.role == "assistant" and "42" in str(r.content) for r in child_rows)
            parent_rows = await store.load_all_messages(sid)
            receipts = [json.loads(r.content) for r in parent_rows
                        if r.role == "tool" and r.name == "spawn_agent"]
            assert len(receipts) == 1 and receipts[0]["status"] == "queued"
            result = (await store.get_runtime_state(
                sid, "subagent:run:" + receipts[0]["run_id"]))["result"]
            assert result["usage"]["total_tokens"] > 0
            assert result["messages"] and result["projection"]
    finally:
        llm.release_children.set()
        await settle_children()
        llm.report(record_property)
        await llm.close()
        await store.close()


async def test_real_catalog_background_result_is_consumed_after_pass(
        tmp_path, monkeypatch, record_property):
    store = await SessionStore.open(str(tmp_path / "background.db"))
    llm = BoundedProvider(limit=6)
    manager = default_tools.BackgroundManager()
    monkeypatch.setattr(default_tools, "BG", manager)
    monkeypatch.setattr(default_tools, "_TOOL_TASK_ON_COMPLETE", None)
    entered, release = asyncio.Event(), asyncio.Event()
    proof = "LOCAL-PROOF-" + secrets.token_hex(6)
    registry = create_default_tool_registry(include=["background_run"], workspace_dir=tmp_path)

    async def evidence():
        entered.set()
        await release.wait()
        return proof

    registry.register(ToolDefinition(
        name="evidence", description="Read a local evidence code; may run in background.",
        input_schema={"type": "object", "properties": {}}, async_capable=True), evidence)
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry, config=AgentLoopConfig(
        system_prompt="Follow instructions precisely. Use the background result when it arrives.",
        max_rounds=4, max_tokens=2048, representation=ProjectedRepresentation()))
    sid = await loop.new_session()
    try:
        async with asyncio.timeout(180):
            first = await loop.send(
                "Start evidence with background_run action=tool, tool=evidence, args={}. "
                "Do not call evidence directly. After the receipt call pass_turn; do not wait or poll.", sid)
            await asyncio.wait_for(entered.wait(), 10)
            assert first.passed
            rows = await store.list_background_tasks(sid)
            assert len(rows) == 1 and rows[0].status == "running"
            release.set()
            await asyncio.wait_for(asyncio.gather(*list(manager._workers.values())), 10)
            record = await store.get_runtime_state(sid, record_key(rows[0].task_id))
            assert record["status"] == "completed" and record["result"] == proof
            result = await loop.send(
                "The local background task has finished. Read its supplied result and reply "
                "with the exact local evidence code. Do not call any tools.", sid)
            assert proof in result.final_text
            transcript = await store.load_all_messages(sid)
            assert any(proof in str(r.content) for r in transcript if r.role != "assistant")
            assert not manager._workers
    finally:
        release.set()
        if manager._workers:
            await asyncio.wait_for(asyncio.gather(
                *list(manager._workers.values()), return_exceptions=True), 10)
        llm.report(record_property)
        await llm.close()
        await store.close()


async def test_real_standalone_child_consumes_input_arriving_inside_pass(
        tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "standalone-pass.db"))
    llm = BoundedProvider(limit=4)
    registry = create_default_tool_registry(include=[], workspace_dir=tmp_path)
    proof = "CORRECTION-" + secrets.token_hex(6)

    async def boundary_pass():
        request_pass("waiting for correction")
        await get_current_loop().deliver(InboxItem(proof), get_session_id(), enqueue_only=True)
        return "Passed control."

    registry.register(ToolDefinition(name="boundary_pass", description="Yield for the correction.",
        input_schema={"type": "object", "properties": {}}), boundary_pass)
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry,
                             config=AgentLoopConfig(max_rounds=4, max_tokens=2048))
    parent = await loop.new_session()
    context = set_session_id(parent)
    try:
        async with asyncio.timeout(120):
            receipt = await delegation.start(loop, AgentSpec(name="reader", tools=["boundary_pass"],
                system_prompt=("First call boundary_pass exactly once. After receiving the later "
                               "correction code, reply with that exact code and no tools."),
                max_rounds=4, lifecycle="linked"), "Follow the procedure.")
            await settle_children()
            saved = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
            assert saved["status"] == "completed" and proof in saved["final_text"]
            assert not saved["new_input_after_pass"]
            rows = await store.load_all_messages(receipt["session_id"])
            assert len([r for r in rows if r.role == "tool" and r.name == "boundary_pass"]) == 1
            assert (await store.inbox_counts(parent))["pending"] == 1
    finally:
        await settle_children()
        reset_session_id(context)
        llm.report(record_property)
        await llm.close()
        await store.close()


async def test_real_stop_mid_generation_preserves_usage_and_can_continue(
        tmp_path, record_property):
    """Stop a real request after streamed output, then reuse the session protocol."""
    store = await SessionStore.open(str(tmp_path / "cancel.db"))
    llm = BoundedProvider(limit=3)
    bus = AgentEventBus()
    started = asyncio.Event()
    usage_events = []

    def stream_seen(event):
        if (event.payload or {}).get("text"):
            started.set()

    bus.subscribe(AgentEventType.STREAM_DELTA, stream_seen)
    bus.subscribe(AgentEventType.STREAM_THINK_DELTA, stream_seen)
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda event: usage_events.append(event.data))
    loop = StatefulAgentLoop(llm=llm, store=store, event_bus=bus, config=AgentLoopConfig(
        system_prompt="Follow the user's formatting request.", max_rounds=2, max_tokens=2048))
    sid = await loop.new_session()
    stop = CancellationToken()
    task = asyncio.create_task(loop.send(
        "Write all integers from 1 through 10000, one integer per line. Do not summarize.",
        sid, stop_event=stop))
    try:
        async with asyncio.timeout(120):
            await asyncio.wait_for(started.wait(), 60)
            stop.cancel("real-test cancellation")
            result = await asyncio.wait_for(task, 15)
            assert result.status == "cancelled"
            assert usage_events
            cancelled = usage_events[0]
            assert not cancelled.success and cancelled.estimated
            assert cancelled.total_tokens > 0
            follow = await loop.send("Stop listing. Reply only OK.", sid)
            assert follow.status == "completed" and "OK" in follow.final_text.upper()
            record_property("cancelled_attempt_estimated_tokens", cancelled.total_tokens)
    finally:
        stop.cancel("test cleanup")
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        llm.report(record_property)
        await llm.close()
        await store.close()
