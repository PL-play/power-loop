"""Real-provider cover for design/124 §7 steering: a new message reaches the live model while its
output is streaming or a tool is running, and the transcript stays valid for the provider.

What only a live provider proves: after an aborted stream and a synthetic tool result the next
request is still ACCEPTED (tool-call pairing intact, reasoning_content rules satisfied), and the
model actually acts on the steer instead of resuming the old task.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    InboxItem,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.default_tools import register_tool_task_callback
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm
from .judge import assert_passes

pytestmark = pytest.mark.skipif(
    not os.environ.get("POWER_LOOP_API_KEY"), reason="needs the real provider in .env")


def _collect(bus: AgentEventBus, kind: AgentEventType) -> list[dict]:
    out: list[dict] = []
    bus.subscribe(kind, lambda e: out.append(dict(e.payload or {})))
    return out


@pytest.mark.asyncio
async def test_real_steer_cuts_a_long_generation() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        interrupted = _collect(bus, AgentEventType.STEER_INTERRUPTED)
        calls = _collect(bus, AgentEventType.LLM_CALL_COMPLETED)
        chars = [0]

        def _count(e: Any) -> None:
            chars[0] += len((e.payload or {}).get("text", ""))

        bus.subscribe(AgentEventType.STREAM_DELTA, _count)
        bus.subscribe(AgentEventType.STREAM_THINK_DELTA, _count)
        loop = StatefulAgentLoop(llm=make_llm(max_tokens=6000, temperature=0.7), store=store,
                                 event_bus=bus, config=AgentLoopConfig(
                                     system_prompt="你是一个作家助手。", max_rounds=3,
                                     max_tokens=6000, compactor=None))
        sid = await loop.new_session()
        run = asyncio.create_task(loop.send("写一篇至少 5000 字的连续小说，一直写下去，不要停。", sid))
        for _ in range(6000):
            if chars[0] >= 300:
                break
            await asyncio.sleep(0.01)
        assert chars[0] >= 300, "the model never started streaming"
        await loop.deliver(InboxItem("别写小说了。只回答：2+2 等于几？只回一个数字。",
                                     mode="steer", item_id="steer-1"), sid)
        res = await asyncio.wait_for(run, 180)
        assert res.status == "completed", res.status
        assert interrupted and interrupted[0]["where"] == "llm"
        aborted = [c for c in calls if c.get("outcome") == "aborted"]
        assert aborted and aborted[0]["estimated"] and aborted[0]["prompt_tokens"] > 0
        assert "4" in (res.final_text or ""), res.final_text
        rows = await store.load_active_messages(sid)
        assistants = [r for r in rows if r.role == "assistant"]
        assert len(assistants) == 1, "the aborted generation left a row"
        assert len(assistants[0].content or "") < 200, "the novel leaked into history"
    finally:
        await store.close()


def _slow_registry(interrupt: str, state: dict[str, Any]) -> ToolRegistry:
    reg = ToolRegistry()

    async def research(topic: str = "", **kw: Any) -> str:
        state["calls"] = state.get("calls", 0) + 1
        state["started"].set()
        try:
            await asyncio.sleep(4)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        state["finished"] = True
        return f"Research on {topic}: the market grew 12% in 2025."

    reg.register(ToolDefinition(
        name="research", description="Slow web research on a topic (takes a while).",
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}},
                      "required": ["topic"]},
        interrupt=interrupt), research)
    return reg


@pytest.mark.asyncio
async def test_real_steer_moves_a_running_tool_to_background() -> None:
    settled: list[tuple[str, str]] = []

    async def on_done(sid: str, task_id: str, status: str) -> None:
        settled.append((task_id, status))

    register_tool_task_callback(on_done)
    store = await SessionStore.open(":memory:")
    try:
        state: dict[str, Any] = {"started": asyncio.Event()}
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0), store=store,
            tool_registry=_slow_registry("background", state),
            config=AgentLoopConfig(system_prompt="Use the research tool when asked to research. "
                                                 "Answer in English.",
                                   max_rounds=4, max_tokens=2048, compactor=None))
        sid = await loop.new_session()
        run = asyncio.create_task(loop.send("Research the electric bike market.", sid))
        await asyncio.wait_for(state["started"].wait(), 120)
        await loop.deliver(InboxItem("Quick question first: what is the capital of France?",
                                     mode="steer", item_id="steer-2"), sid)
        res = await asyncio.wait_for(run, 180)
        assert res.status == "completed", res.status
        assert not state.get("cancelled"), "background mode must not cancel the tool"
        assert state["calls"] == 1, (
            f"the model re-ran a tool that was still running: {state} / {res.final_text!r}")
        await assert_passes(
            question="(steer while researching) what is the capital of France?",
            answer=res.final_text or "",
            rubric="The reply says the capital of France is Paris.")
        for _ in range(600):
            if settled:
                break
            await asyncio.sleep(0.02)
        assert settled and settled[0][1] == "completed", (settled, state)
        assert state.get("finished"), state
    finally:
        register_tool_task_callback(None)
        await store.close()


@pytest.mark.asyncio
async def test_real_steer_aborts_an_abortable_tool_and_the_provider_accepts_the_turn() -> None:
    store = await SessionStore.open(":memory:")
    try:
        state: dict[str, Any] = {"started": asyncio.Event()}
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0), store=store,
            tool_registry=_slow_registry("abort", state),
            config=AgentLoopConfig(system_prompt="Use the research tool when asked to research. "
                                                 "Answer in English.",
                                   max_rounds=4, max_tokens=2048, compactor=None))
        sid = await loop.new_session()
        run = asyncio.create_task(loop.send("Research the drone delivery market.", sid))
        await asyncio.wait_for(state["started"].wait(), 120)
        await loop.deliver(InboxItem("Never mind the research — don't run it again. "
                                     "Just reply with the single word: OK",
                                     mode="steer", item_id="steer-3"), sid)
        res = await asyncio.wait_for(run, 180)
        assert res.status == "completed", res.status   # the provider accepted the repaired turn
        assert state.get("cancelled") and not state.get("finished")
        assert "ok" in (res.final_text or "").lower(), res.final_text
        rows = await store.load_active_messages(sid)
        tool_row = next(r for r in rows if r.role == "tool")
        assert "interrupted" in (tool_row.content or "")
    finally:
        await store.close()
