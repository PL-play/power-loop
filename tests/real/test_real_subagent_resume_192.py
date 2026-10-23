"""Real finished child -> handled result -> same SID/new generation -> result."""
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


async def _receipts(store, manager, parent):
    """Exactly-once admission, including real-model invalid-argument recovery."""
    receipts = []
    for row in await store.load_all_messages(parent):
        if row.role != "tool" or row.name != "spawn_agent":
            continue
        try:
            value = json.loads(row.content)
        except json.JSONDecodeError:
            # Validation failure is not a receipt. It must have registered no
            # async operation; never filter out a lost/ambiguous accepted task.
            assert row.content.startswith("Error:") and row.tool_call_id
            assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='operation' "
                "AND caller_session_id=? AND tool_call_id=?", (parent, row.tool_call_id))
        else:
            assert isinstance(value, dict) and all(k in value for k in ("job_id", "operation_id", "session_id"))
            receipts.append(value)
    return receipts


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack"])
async def test_real_finished_subagent_resumes_original_session_without_overwriting_result(tmp_path, record_property, monkeypatch, fault):
    store = await SessionStore.open(str(tmp_path / "real-resume.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-resume", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=14)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_tokens=2048, max_rounds=4,
            system_prompt="Follow the requested tool sequence exactly. An acceptance receipt is not a result. Never create replacement tasks. "
                "When a subagent_result arrives report its final number in assistant text without tools or pass_turn."))
    parent = await loop.new_session()
    provider.root_sid, provider.hold_children = parent, True
    before = set(delegation._TASKS)
    counts = {"admission": 0, "dispatch": 0}
    for name, key in (("resume_subagent", "admission"), ("claim_resume_dispatch", "dispatch")):
        original = getattr(manager, name)
        async def lost_ack(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            value = await _original(*args, **kwargs)
            if fault == ("admission_ack" if _key == "admission" else "dispatch_ack"):
                raise RuntimeError("committed; lost acknowledgement")
            return value
        monkeypatch.setattr(manager, name, lost_ack)
    try:
        async with asyncio.timeout(180):
            first = await loop.send("Call spawn_agent create exactly once with tools=[], max_rounds=3, "
                "system_prompt='Calculate the requested arithmetic. Follow the newest correction or follow-up request in your conversation. "
                "Reply only with the final number. Do not call pass_turn.', task='Compute 19+23'. "
                "After acceptance reply WAITING without more tools or pass_turn.", parent)
            assert first.status == "completed"
            [create] = await _receipts(store, manager, parent)
            assert create["mode"] == "create"
            provider.release_children.set()
            await asyncio.gather(*(set(delegation._TASKS)-before))
            result = await loop.flush_follow_ups(parent, max_rounds=3)
            assert result.status == "completed" and "42" in result.final_text
            old = await manager.get_result_revision(runtime.actor(parent), create["job_id"], 1)
            assert old["payload"]["result"]["final_text"].strip() == "42" and old["delivery_status"] == "handled"
            first_closed = await manager.get_job(runtime.actor(parent), create["job_id"])
            creator = await manager.get_job(runtime.actor(parent), first_closed.parent_job_id)
            assert first_closed.supervision_status == creator.supervision_status == "closed"
            provider.release_children.clear()
            provider.child_entered.clear()
            second = await loop.send(f"Use spawn_agent action=nudge exactly once with sub_session='{create['session_id']}', "
                "message='New follow-up: compute 20+23 instead of 19+23. Reply only with the resulting number.'. "
                "After acceptance reply WAITING without more tools or pass_turn. Do not create a new agent.", parent)
            assert second.status == "completed"
            reopened = await manager.get_job(runtime.actor(parent), creator.job_id)
            assert reopened.supervision_status == "open" and reopened.status == "model_closed"
            assert reopened.generation == creator.generation
            [first_receipt, resume] = await _receipts(store, manager, parent)
            assert first_receipt == create
            assert resume["mode"] == "continue" and resume["generation"] == 2
            assert resume["session_id"] == create["session_id"] and resume["job_id"] == create["job_id"]
            assert resume["run_id"] != create["run_id"] and counts == {"admission": 1, "dispatch": 1}
            provider.release_children.set()
            await asyncio.gather(*(set(delegation._TASKS)-before))
            current = await manager.get_job(runtime.actor(parent), create["job_id"])
            assert current.generation == 2 and current.outcome == "succeeded" and current.result["final_text"].strip() == "43", current.result
            assert await manager.get_result_revision(runtime.actor(parent), current.job_id, 1) == old
            revised = await manager.get_result_revision(runtime.actor(parent), current.job_id, 2)
            assert revised["payload"]["result"]["run_id"] == resume["run_id"]
            assert revised["event_id"] == resume["result_event_id"]
            op = await manager.get_operation(runtime.actor(parent), resume["operation_id"])
            assert op.call_ref["id"] != op.receipt_ref["id"]
            final = await loop.flush_follow_ups(parent, max_rounds=3)
            assert final.status == "completed" and "43" in final.final_text
            assert (await manager.get_job(runtime.actor(parent), current.job_id)).result_status == "handled"
            assert (await manager.get_job(runtime.actor(parent), current.job_id)).supervision_status == "closed"
            assert (await manager.get_job(runtime.actor(parent), creator.job_id)).supervision_status == "closed"
            assert len(await store.list_children(parent)) == 1
            per_execution = await manager.execution_usage(runtime.actor(parent), current.job_id, generation=2)
            assert per_execution == current.result["usage"] and per_execution["total_tokens"] > 0
            lifetime = await manager.execution_usage(runtime.actor(parent), current.job_id)
            assert lifetime["total_tokens"] == old["payload"]["result"]["usage"]["total_tokens"]+per_execution["total_tokens"]
    finally:
        provider.release_children.set()
        tasks = set(delegation._TASKS)-before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
