"""Real-provider cover for design/124 §8 stopping: a stop lands fast even mid-generation and
mid-sub-agent, and the transcript it leaves is one the provider accepts on the next send."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    AgentSpec,
    CancellationToken,
    SessionStore,
    StatefulAgentLoop,
    StopPolicy,
    run_agent_spec,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm

pytestmark = pytest.mark.skipif(
    not os.environ.get("POWER_LOOP_API_KEY"), reason="needs the real provider in .env")


@pytest.mark.asyncio
async def test_real_stop_mid_generation_is_fast_and_leaves_a_valid_transcript() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        chars = [0]

        def _count(e: Any) -> None:
            chars[0] += len((e.payload or {}).get("text", ""))

        bus.subscribe(AgentEventType.STREAM_DELTA, _count)
        bus.subscribe(AgentEventType.STREAM_THINK_DELTA, _count)
        loop = StatefulAgentLoop(llm=make_llm(max_tokens=6000, temperature=0.7), store=store,
                                 event_bus=bus, config=AgentLoopConfig(
                                     system_prompt="你是作家。", max_rounds=3, max_tokens=6000,
                                     compactor=None))
        sid = await loop.new_session()
        tok = CancellationToken()
        run = asyncio.create_task(loop.send("写一篇至少 5000 字的小说。", sid, stop_event=tok))
        for _ in range(6000):
            if chars[0] >= 200:
                break
            await asyncio.sleep(0.01)
        assert chars[0] >= 200
        t0 = time.monotonic()
        tok.cancel("user stop")
        res = await asyncio.wait_for(run, 30)
        took = time.monotonic() - t0
        assert res.status == "cancelled" and took < 3, took
        rows = await store.load_active_messages(sid)
        assert not any(r.role == "assistant" for r in rows)
        nxt = await loop.send("好了，不写了。只回一个字：好", sid)
        assert nxt.status == "completed", nxt.status
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_real_stop_reaches_a_running_sub_agent() -> None:
    store = await SessionStore.open(":memory:")
    try:
        child_state: dict[str, Any] = {"started": asyncio.Event()}
        reg = ToolRegistry()

        async def slow_lookup(topic: str = "", **kw: Any) -> str:
            child_state["started"].set()
            await asyncio.sleep(30)
            return f"{topic}: done"

        reg.register(ToolDefinition(
            name="slow_lookup", description="Look something up (slow).",
            input_schema={"type": "object", "properties": {"topic": {"type": "string"}},
                          "required": ["topic"]}, interrupt="abort"), slow_lookup)

        async def delegate(task: str = "", **kw: Any) -> str:
            out = await run_agent_spec(
                AgentSpec(name="researcher",
                          system_prompt="Always call slow_lookup first, then answer."),
                task, parent_loop=get_current_loop())
            child_state["status"] = out.get("status")
            return f"sub-agent {out.get('status')}: {out.get('final_text')}"

        reg.register(ToolDefinition(
            name="delegate", description="Hand a research task to a sub-agent.",
            input_schema={"type": "object", "properties": {"task": {"type": "string"}},
                          "required": ["task"]}, interrupt="background"), delegate)
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0), store=store, tool_registry=reg,
            config=AgentLoopConfig(system_prompt="Delegate research tasks with the delegate tool.",
                                   max_rounds=4, max_tokens=2048, compactor=None,
                                   stop_policy=StopPolicy(subtask_s=10)))
        sid = await loop.new_session()
        tok = CancellationToken()
        run = asyncio.create_task(loop.send("Research solar panel prices in 2026.", sid,
                                            stop_event=tok))
        await asyncio.wait_for(child_state["started"].wait(), 120)
        t0 = time.monotonic()
        tok.cancel("user stop")
        res = await asyncio.wait_for(run, 60)
        took = time.monotonic() - t0
        assert res.status == "cancelled", res.status
        assert child_state.get("status") == "cancelled", child_state
        assert took < 12, took
        await loop.abort_pending(sid, reason="cancelled by user")
        nxt = await loop.send("Never mind. Reply with just: OK", sid)
        assert nxt.status == "completed"
        assert "ok" in (nxt.final_text or "").lower()
    finally:
        await store.close()
