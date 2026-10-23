"""Real models cancel original workflows through the shared library facade."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime import task_control
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import journal
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize(("native", "fault"), [(True, "none"), (True, "commit_ack"), (False, "none")])
async def test_real_workflow_cancel_uses_original_admission_without_replacement_or_false_stop_claim(tmp_path, record_property, monkeypatch, native, fault):
    store = await SessionStore.open(str(tmp_path / "cancel.db"), **({"schema": "tasks"} if native else {}))
    runtime = TaskRuntime(JobManager(store), tenant_id="real-workflow-cancel", principal_id="host",
        executor_id="executor", boot_id="boot") if native else None
    provider = BoundedProvider(limit=10)
    accepted, completed = [], []
    async def accepted_view(admission):
        accepted.append(admission)
    async def completion(value):
        completed.append(value)
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=accepted_view, on_complete=completion))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Follow the requested workflow sequence exactly. Create only once. "
                "A cancel acceptance is not confirmation of resource stop, supervision closure or billing settlement. "
                "Do not create replacement work or poll."))
    loop.config.stop_policy = replace(loop.config.stop_policy, workflow_s=.02)
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    spec = {"name": "cancel-original", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "calculate", "input": "19+23", "spec": {"name": "calculate", "system_prompt": "Compute the arithmetic and return the number.", "tools": [], "max_rounds": 2}},
        {"type": "agent", "id": "review", "input": "Review previous result.", "inputs_from": ["calculate"], "spec": {"name": "review", "system_prompt": "Review supplied result.", "tools": [], "max_rounds": 2}},
    ]}}
    if fault == "commit_ack":
        original = task_control.request_stop
        async def lost(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("durable stop committed, ACK lost")
        monkeypatch.setattr(task_control, "request_stop", lost)
    try:
        async with asyncio.timeout(180):
            await loop.send("Call workflow action=create exactly once with spec="+json.dumps(spec)+
                ". After acceptance finish with WAITING and no further tools.", sid)
            assert len(accepted) == 1
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            handle = accepted[0].handle
            original_job = await runtime.manager.get_job(runtime.actor(sid), handle.job_id) if native else None
            result = await loop.send("The user now cancels only the original workflow. Call workflow action=cancel exactly once "
                "with run_id='"+handle.run_id+"'. Then acknowledge the request in assistant text without more tools or pass_turn. "
                "Do not claim physical stop or create/resume any task.", sid)
            assert result.status == "completed"
            receipts = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
            assert len(receipts) == 2
            creation, stopped = [json.loads(m.content) for m in receipts]
            assert stopped["run_id"] == creation["run_id"] == handle.run_id
            assert stopped["status"] == "accepted" and stopped["control_pending"] and not stopped["resource_stop_confirmed"]
            if native:
                assert stopped["job_id"] == original_job.job_id and stopped["generation"] == 1
                assert stopped["replayed"] == (fault == "commit_ack")
                operation = await runtime.manager.get_operation(runtime.actor(sid), stopped["operation_id"])
                assert operation.call_ref["id"] != operation.receipt_ref["id"]
                assert operation.receipt_ref["id"] == str(receipts[-1].id)
                await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), 15)
                current = await runtime.manager.get_job(runtime.actor(sid), handle.job_id)
                assert current.outcome == "cancelled" and current.supervision_status == "open"
                assert len(await runtime.manager.db.fetchall(f"SELECT job_id FROM {runtime.manager.jobs} WHERE kind='workflow'")) == 1
                assert len(await runtime.manager.db.fetchall(f"SELECT job_id FROM {runtime.manager.jobs} WHERE kind='workflow_node'")) == 1
            else:
                assert stopped["backend"] == "legacy" and stopped["durable_intent"] is False
                await asyncio.wait_for(asyncio.gather(handle._local_stop_task, return_exceptions=True), 15)
                await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), 15)
                assert (await journal.read(store, sid, handle.run_id))["status"] == "cancelled"
            assert not completed and len(accepted) == 1
    finally:
        provider.release_children.set()
        try:
            tasks = [a.handle.task for a in accepted] + [a.handle._local_stop_task for a in accepted if hasattr(a.handle, "_local_stop_task")]
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        finally:
            record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
                name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
