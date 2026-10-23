"""Actual model uses the shared workflow tool to correct its original live node."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("action", ["nudge", "steer"])
@pytest.mark.parametrize("invalid_first", [False, True])
async def test_real_shared_workflow_control_preserves_original_node_and_call_refs(tmp_path, record_property, action, invalid_first):
    store = await SessionStore.open(str(tmp_path / "node-facade.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-node-facade", principal_id="host",
        executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=18)
    accepted = []
    async def on_accepted(admission):
        accepted.append(admission)
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=on_accepted))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Follow the requested workflow tool action exactly once in each turn. "
                "An accepted receipt is not a result. Never create replacement work, extend rounds or poll. "
                "After an accepted receipt reply WAITING with no more calls. "
                "When the workflow result is delivered, report its computed number in assistant text without tools."))
    parent = await loop.new_session()
    provider.root_sid, provider.hold_children = parent, True
    spec = {"name": "real-node-control", "root": {"type": "agent", "id": "calculate", "input": "Compute 19+23.",
        "spec": {"name": "calculate", "system_prompt": "Compute the latest supplied arithmetic correction. Return only the number.",
            "tools": [], "max_rounds": 3}}}
    node_sid = None
    try:
        async with asyncio.timeout(240):
            await loop.send("Call workflow action=create exactly once with spec " + json.dumps(spec) +
                ". After acceptance reply WAITING and stop. Do not call any other workflow action yet.", parent)
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            [admission] = accepted
            handle = admission.handle
            [binding] = await manager.db.fetchall(f"SELECT job_id,payload_json FROM {manager.events} WHERE event_type='node_session'")
            descriptor = json.loads(binding["payload_json"])
            node_sid = descriptor["session_id"]
            if invalid_first:
                invalid = {"action": action, "run_id": handle.run_id, "sid": "sess_"+"f"*24, "message": "Do not dispatch a substitute node."}
                await loop.send("Call workflow exactly once with these arguments, deliberately selecting an unknown saved SID: " +
                    json.dumps(invalid) + ". The expected result is an error. After this error reply WAITING and stop; do not correct, retry or replace anything yet.", parent)
                errors = [m for m in await store.load_all_messages(parent) if m.role == "tool" and m.name == "workflow"]
                assert len(errors) == 2 and json.loads(errors[-1].content)["error"]["code"] == "workflow_node_not_found", errors
                assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
                assert not handle.task.done() and (await manager.get_job(runtime.actor(parent), binding["job_id"])).result is None
            message = "Correction: compute 20+23 instead of 19+23. Return only the resulting number."
            selector = {"node_job_id": binding["job_id"]} if action == "nudge" else {"sid": node_sid}
            options = {"action": action, "run_id": handle.run_id, "message": message, **selector}
            await loop.send("Call workflow exactly once with these arguments: " + json.dumps(options) +
                ". Then reply WAITING. Do not create, resume, poll or extend anything.", parent)
            messages = [m for m in await store.load_all_messages(parent) if m.role == "tool" and m.name == "workflow"]
            assert len(messages) == (3 if invalid_first else 2), messages
            receipt = json.loads(messages[-1].content)
            assert receipt["status"] == "accepted" and receipt["mode"] == "steer", receipt
            assert receipt["session_id"] == node_sid and receipt["job_id"] == binding["job_id"]
            operation = await manager.get_operation(runtime.actor(parent), receipt["operation_id"])
            assert operation.receipt_ref["id"] == str(messages[-1].id)
            assert operation.call_ref["id"] != operation.receipt_ref["id"]
            assert operation.result["request"]["message"] == message
            assert not handle.task.done()
            assert (await manager.get_job(runtime.actor(parent), binding["job_id"])).result is None
            provider.release_children.set()
            await asyncio.wait_for(handle.task, 100)
            result = await manager.get_job(runtime.actor(parent), binding["job_id"])
            assert result.generation == 1 and result.outcome == "succeeded" and "43" in result.result["text"], result
            final = await loop.flush_follow_ups(parent, max_rounds=3)
            assert final and "43" in final.final_text
            assert (await store.inbox_counts(parent))["pending"] == 0
            assert len(accepted) == 1
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1
            [same] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
            assert json.loads(same["payload_json"]) == descriptor
    finally:
        provider.release_children.set()
        try:
            if accepted:
                await asyncio.wait_for(asyncio.gather(*(a.handle.task for a in accepted), return_exceptions=True), 20)
        finally:
            for name, sid in (("main", parent), ("node", node_sid)):
                if sid:
                    record_property(name + "_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
                        name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                        for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
