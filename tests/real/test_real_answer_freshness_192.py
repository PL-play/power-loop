"""A real empty correction pass requires a fresh answer in the original node."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_corrected_node_does_not_deliver_old_text_after_empty_pass(tmp_path, record_property, monkeypatch):
    store = await SessionStore.open(str(tmp_path / "answer-freshness.db"), schema="tasks")
    provider = BoundedProvider(limit=12)
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-freshness", principal_id="host", executor_id="executor", boot_id="boot")
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Follow the requested tool sequence exactly. After launch acceptance reply WAITING, no more tools or pass_turn. "
                "After a workflow result is delivered, report its final numeric answer without tools."))
    parent = await loop.new_session()
    handles, node_sids = [], []
    complete = provider.complete
    async def corrected_after_real_response(request, **kwargs):
        response = await complete(request, **kwargs)
        if get_session_id() != parent and not node_sids:
            # Keep the real response intact. Only queue the next user input
            # before its already-produced old answer closes this live send.
            assert response.raw_text.strip() == "42" and not response.get_tool_calls()
            sid = get_session_id()
            node_sids.append(sid)
            await get_current_loop().follow_up(
                "Correction: the current answer is 43. For this test, call pass_turn exactly once with reason='Computed 43'. "
                "Include no assistant text. A later final-answer request will ask you to write 43.", sid)
        return response
    monkeypatch.setattr(provider, "complete", corrected_after_real_response)
    async def launch():
        spec = {"name": "answer-freshness", "root": {"type": "agent", "id": "answer",
            "input": "Return exactly 42 as plain assistant text without tools or pass_turn. Follow any later user correction when it arrives.",
            "spec": {"name": "answer", "system_prompt": "Follow each new user instruction. The latest numeric answer replaces the earlier one. "
                "If a final-answer request says not to call pass_turn, write the latest numeric answer as assistant text only. "
                "Never repeat completed actions.", "tools": [], "max_rounds": 4}}}
        handle = await create_workflow(spec, parent_loop=get_current_loop(), parent_session_id=parent).start(detached=True)
        handles.append(handle)
        return json.dumps({"job_id": handle.job_id, "status": "accepted"})
    for name, fn in (("launch", launch),):
        registry.register(ToolDefinition(name=name, description=name, input_schema={"type": "object"}, pricing=Free("test_fixture")), fn)
    try:
        async with asyncio.timeout(180):
            await loop.send("Call launch exactly once. After acceptance reply WAITING without tools or pass_turn.", parent)
            assert len(handles) == 1
            await handles[0].task
            assert len(node_sids) == 1
            rows = await store.load_all_messages(node_sids[0])
            corrected = next(i for i, m in enumerate(rows) if m.role == "user" and m.name == "follow_up" and "Correction:" in str(m.content))
            assert any(m.role == "assistant" and "42" in str(m.content) for m in rows[:corrected])
            first = next(m for m in rows[corrected + 1:] if m.role == "assistant")
            assert not (first.content or "").strip()
            assert [c["function"]["name"] for c in first.tool_calls or []] == ["pass_turn"]
            assert any(m.role == "assistant" and (m.content or "").strip() == "43" for m in rows[corrected + 1:])
            [node] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
            result = await manager.get_job(runtime.actor(parent), node["job_id"])
            assert result.generation == 1 and result.outcome == "succeeded" and result.result["text"].strip() == "43"
            assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='node_session'")) == 1
            final = await loop.flush_follow_ups(parent, max_rounds=3)
            assert final.status == "completed" and "43" in final.final_text
    finally:
        try:
            for h in handles:
                if not h.task.done():
                    h.task.cancel()
            if handles:
                await asyncio.gather(*(h.task for h in handles), return_exceptions=True)
            for sid in node_sids:
                record_property("node_transcript", json.dumps([{"role": m.role, "content": m.content, "name": m.name,
                    "tool_calls": m.tool_calls} for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
        finally:
            try:
                await provider.close()
            finally:
                await store.close()
