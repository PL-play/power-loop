"""Configured real model uses the library jobs tool to cancel original work."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition, create_default_tool_registry
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import delegation, task_control
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import create_workflow

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("native", [True, False])
async def test_real_jobs_backend_valueerror_stays_private_and_is_not_an_empty_task_list(tmp_path, record_property, monkeypatch, native):
    from power_loop.runtime import task_query

    store = await SessionStore.open(str(tmp_path/"jobs-errors.db"), **({"schema": "tasks"} if native else {}))
    runtime = TaskRuntime(JobManager(store), tenant_id="real-jobs-errors", principal_id="host",
        executor_id="executor", boot_id="boot") if native else None
    provider = BoundedProvider(limit=4)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime,
        tool_registry=create_default_tool_registry(include=["jobs"]),
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=2048,
            system_prompt="Call jobs action=list exactly once when asked. On any error, do not retry, create work or infer an empty task list. "
                "Give an actual assistant response explaining that the directory is unavailable. Do not use pass_turn."))
    sid = await loop.new_session()
    async def unavailable(*args, **kwargs):
        raise ValueError("PRIVATE-CONNECTION-SECRET")
    if native:
        monkeypatch.setattr(task_query, "list_jobs", unavailable)
    else:
        monkeypatch.setattr(loop, "list_jobs", unavailable)
    try:
        async with asyncio.timeout(90):
            result = await loop.send("Inspect the task directory with jobs action=list once; then report the actual outcome in assistant text.", sid)
            assert result.status == "completed" and result.final_text.strip()
            messages = await store.load_all_messages(sid)
            tools = [m for m in messages if m.role == "tool" and m.name == "jobs"]
            assert len(tools) == 1
            error = json.loads(tools[0].content)
            assert error == {"version": 1, "error": {"code": "jobs_backend_unavailable"}}
            assert "PRIVATE-CONNECTION-SECRET" not in str([(m.content, m.tool_calls) for m in messages])
            if native:
                assert not await runtime.manager.db.fetchall(f"SELECT job_id FROM {runtime.manager.jobs} WHERE kind<>'send'")
    finally:
        record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
            name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
            for m in await store.load_all_messages(sid)], ensure_ascii=False))
        provider.report(record_property)
        await provider.close()
        await store.close()


@pytest.mark.parametrize(("kind", "fault"), [("subagent", "none"), ("subagent", "stop_ack"), ("workflow", "none")])
async def test_real_jobs_list_get_stop_preserves_identity_receipt_and_pending_control(tmp_path, record_property, monkeypatch, kind, fault):
    store = await SessionStore.open(str(tmp_path / "real-jobs.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-jobs", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=12)
    registry = create_default_tool_registry(include=["jobs"])
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=6, max_tokens=2048,
            system_prompt="Follow the requested tool order exactly. Accepted work is asynchronous. "
                "For cancellation use jobs on the original task only; do not create replacement work or poll. "
                "A durable cancellation receipt does not confirm physical stop or billing settlement."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    before, handles = set(delegation._TASKS), []
    callbacks = []
    async def notified(*args):
        callbacks.append(args)
    original_start = delegation.start
    async def start(*args, **kwargs):
        return await original_start(*args, **kwargs, on_complete=notified)
    monkeypatch.setattr(delegation, "start", start)
    async def launch():
        spec = {"name": "jobs-cancel-original", "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "calculate", "input": "Compute 19+23.", "spec": {"name": "calculate", "system_prompt": "Compute the requested arithmetic and return the number.", "tools": [], "max_rounds": 2}},
            {"type": "agent", "id": "review", "input": "Review the previous result.", "inputs_from": ["calculate"], "spec": {"name": "review", "system_prompt": "Review the supplied result and return its number.", "tools": [], "max_rounds": 2}},
        ]}}
        handle = await create_workflow(spec, parent_loop=get_current_loop(), parent_session_id=get_session_id()).start(detached=True, on_complete=notified)
        handles.append(handle)
        return json.dumps({"job_id": handle.job_id, "operation_id": handle.operation_id, "run_id": handle.run_id, "status": "accepted"})
    registry.register(ToolDefinition(name="launch_workflow", description="Start the predefined calculation/review workflow asynchronously once.", input_schema={"type": "object", "properties": {}}), launch)
    original = task_control.request_stop
    if fault == "stop_ack":
        async def lost(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("stop committed, ACK lost")
        monkeypatch.setattr(task_control, "request_stop", lost)
    try:
        async with asyncio.timeout(180):
            instruction = ("Call spawn_agent exactly once, task='Compute 19+23 and return only the final number.', tools=[], max_rounds=2. "
                if kind == "subagent" else "Call launch_workflow exactly once. ")
            await loop.send(instruction + "Then reply WAITING without any more tools or pass_turn. Do not use jobs yet.", sid)
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            creations = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == ("spawn_agent" if kind == "subagent" else "launch_workflow")]
            assert len(creations) == 1
            accepted = json.loads(creations[0].content)
            actor = runtime.actor(sid)
            original_job = await manager.get_job(actor, accepted["job_id"])
            assert original_job.status == "running" and original_job.generation == 1
            result = await loop.send("The user has now cancelled only the original task " + accepted["job_id"] + ". "
                "Perform exactly these three jobs calls in order: first action=list with kinds=['"+kind+"']; "
                "second action=get with job_id='"+accepted["job_id"]+"'; third action=stop with that same job_id, expected_generation=1, reason='User cancelled this task'. "
                "Then acknowledge the cancellation request without tools or pass_turn. Do not claim all resources have stopped. "
                "Do not call spawn_agent or launch_workflow again and do not poll.", sid)
            assert result.status == "completed"
            receipts = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "jobs"]
            assert len(receipts) == 3
            listed, detail, stopped = [json.loads(m.content) for m in receipts]
            assert listed["matched_count"] == 1 and detail["node"]["job_id"] == accepted["job_id"]
            assert stopped["job_id"] == accepted["job_id"] and stopped["generation"] == 1
            assert stopped["cancel_requested"] and stopped["control_pending"] and stopped["resource_stop_confirmed"] is False
            assert stopped["local_signals"] >= 1 and stopped["replayed"] == (fault == "stop_ack")
            op = await manager.get_operation(actor, stopped["operation_id"])
            assert op.call_ref["id"] != op.receipt_ref["id"] and op.receipt_ref["id"] == str(receipts[-1].id)
            assert (await store.message_by_id(sid, op.receipt_ref["id"])).content == receipts[-1].content
            tasks = [h.task for h in handles] if handles else list(set(delegation._TASKS)-before)
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
            current = await manager.get_job(actor, accepted["job_id"])
            assert current.outcome == "cancelled" and current.supervision_status == "open", current
            assert current.parent_job_id == original_job.parent_job_id and current.created_by_session_id == original_job.created_by_session_id
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind=?", (kind,))) == 1
            assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")) == 1
            if kind == "workflow":
                assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 1
            assert not runtime._local_executions
            assert callbacks == []
    finally:
        record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role, name=m.name,
            content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id) for m in await store.load_all_messages(sid)], ensure_ascii=False))
        provider.release_children.set()
        try:
            pending = [h.task for h in handles] + list(set(delegation._TASKS)-before)
            if pending:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 15)
        finally:
            provider.report(record_property)
            await provider.close()
            await store.close()


async def test_real_legacy_jobs_bridge_stops_one_real_timer_and_reports_its_limits(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "legacy-jobs.db"))
    provider = BoundedProvider(limit=6)
    registry = create_default_tool_registry(include=["jobs"])
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=5, max_tokens=2048, system_prompt="Follow the requested jobs sequence. Do not schedule replacement work or poll."))
    sid = await loop.new_session()
    timer = await loop.schedule_timer(sid, delay_s=600, note="Original reminder")
    try:
        async with asyncio.timeout(120):
            result = await loop.send("Cancel the existing reminder. Use jobs action=list first. Then jobs action=get with its listed id, "
                "then jobs action=stop with that same job_id. Make exactly these three jobs calls and acknowledge without further tools.", sid)
            assert result.status == "completed"
            messages = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "jobs"]
            assert len(messages) == 3
            listed, detail, stopped = [json.loads(m.content) for m in messages]
            assert listed["backend"] == detail["backend"] == stopped["backend"] == "legacy"
            assert not listed["history_available"] and not listed["hierarchy_available"]
            assert detail["node"]["id"] == "timer:"+str(timer.timer_id)
            assert stopped["results"] == [{"id": "timer:"+str(timer.timer_id), "result": "stopped"}]
            assert await loop.list_timers(sid) == []
    finally:
        provider.report(record_property)
        await provider.close()
        await store.close()
