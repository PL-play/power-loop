"""Real create -> yield -> nudge the original active child -> consume its result."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("fault", ["none", "lost_ack"])
async def test_real_nudge_is_durable_and_processed_in_the_original_active_child(tmp_path, record_property, monkeypatch, fault):
    store = await SessionStore.open(str(tmp_path / "real-nudge.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-nudge-test", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=12)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_tokens=2048, max_rounds=4,
            system_prompt="Follow the requested sequence exactly. Acceptance is not a result. Never create replacement tasks. "
                "When a child completion arrives, report its final answer without tools or pass_turn."))
    parent = await loop.new_session()
    provider.root_sid, provider.hold_children = parent, True
    before = set(delegation._TASKS)
    admissions = []
    accept = manager.accept_nudge

    async def lost_ack(*args, **kwargs):
        admissions.append(1)
        result = await accept(*args, **kwargs)
        if fault == "lost_ack":
            raise RuntimeError("test: nudge committed but ack lost")
        return result

    monkeypatch.setattr(manager, "accept_nudge", lost_ack)
    try:
        async with asyncio.timeout(180):
            first = await loop.send("Call spawn_agent create exactly once with tools=[], max_rounds=3, "
                "system_prompt='Calculate the requested arithmetic. Follow the latest correction in your inbox. "
                "Reply with only the final number in assistant text. Do not call pass_turn.', task='Compute 19+23'. "
                "After acceptance reply WAITING without more tools or pass_turn.", parent)
            assert first.status == "completed"
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            rows = await store.load_all_messages(parent)
            [created] = [json.loads(row.content) for row in rows if row.role == "tool" and row.name == "spawn_agent"]
            original = await manager.get_job(runtime.actor(parent), created["job_id"])
            assert original.result is None
            second = await loop.send(f"Use spawn_agent action=nudge exactly once, sub_session='{created['session_id']}', "
                "message='Correction: compute 20+23 instead of the original task. Return only the resulting number.'. "
                "After acceptance reply WAITING with no more tools or pass_turn. Do not create another child.", parent)
            assert second.status == "completed"
            rows = await store.load_all_messages(parent)
            tool_rows = [row for row in rows if row.role == "tool" and row.name == "spawn_agent"]
            assert len(tool_rows) == 2
            steered = json.loads(tool_rows[-1].content)
            assert steered["status"] == "accepted" and steered["job_id"] == created["job_id"]
            assert steered["run_id"] == created["run_id"] and steered["session_id"] == created["session_id"]
            assert admissions == [1]
            operation = await manager.get_operation(runtime.actor(parent), steered["operation_id"])
            assert operation.receipt_ref["tool_call_id"] == tool_rows[-1].tool_call_id
            assert operation.call_ref["id"] != operation.receipt_ref["id"]
            provider.release_children.set()
            await asyncio.gather(*(set(delegation._TASKS) - before))
            saved = await manager.get_job(runtime.actor(parent), created["job_id"])
            assert saved.outcome == "succeeded" and saved.result["final_text"].strip() == "43", saved.result
            [event] = await manager.db.fetchall(f"SELECT delivery_status FROM {manager.events} WHERE event_id=?", (steered["operation_id"],))
            assert event["delivery_status"] == "handled" and saved.result_status == "pending_delivery"
            [decision] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} "
                "WHERE job_id=? AND event_type='processing_decision'", (created["job_id"],))
            decision = json.loads(decision["payload_json"])
            assert decision["event"]["event_id"] == steered["operation_id"]
            assert decision["response_ref"]["session_id"] == created["session_id"]
            final = await loop.flush_follow_ups(parent, max_rounds=3)
            assert final.status == "completed" and "43" in final.final_text
            assert (await manager.get_job(runtime.actor(parent), created["job_id"])).result_status == "handled"
            assert len(await store.list_children(parent)) == 1
    finally:
        provider.release_children.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
