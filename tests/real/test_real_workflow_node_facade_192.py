"""Actual model uses the shared workflow tool to correct its original live node."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("action", ["nudge", "steer"])
@pytest.mark.parametrize("invalid_first", [False, True])
async def test_real_shared_workflow_control_preserves_original_node_and_call_refs(tmp_path, record_property, action, invalid_first, runtime_agent):
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
    loop = await runtime_agent(model=provider, store=store, task_runtime=runtime, registry=registry, capacity=2,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Follow the requested workflow tool action exactly once in each turn. "
                "An accepted receipt is not a result. Never create replacement work, extend rounds or poll. "
                "After an accepted receipt reply WAITING with no more calls. "
                "When the workflow result is delivered, report its computed number in assistant text without tools."))
    parent = loop.target.session_id
    provider.root_sid, provider.hold_children = parent, True
    spec = {"name": "real-node-control", "root": {"type": "agent", "id": "calculate", "input": "Compute 19+23.",
        "spec": {"name": "calculate", "system_prompt": "Compute the latest supplied arithmetic correction. Return only the number.",
            "tools": [], "max_rounds": 3}}}
    node_sid = None
    try:
        async with asyncio.timeout(240):
            await loop.send("Call workflow action=create exactly once with spec " + json.dumps(spec) +
                ". After acceptance reply WAITING and stop. Do not call any other workflow action yet.")
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            [admission] = accepted
            handle = admission.handle
            [binding] = await manager.db.fetchall(f"SELECT job_id,payload_json FROM {manager.events} WHERE event_type='node_session'")
            descriptor = json.loads(binding["payload_json"])
            node_sid = descriptor["session_id"]
            if invalid_first:
                invalid = {"action": action, "run_id": handle.run_id, "sid": "sess_"+"f"*24, "message": "Do not dispatch a substitute node."}
                await loop.send("Call workflow exactly once with these arguments, deliberately selecting an unknown saved SID: " +
                    json.dumps(invalid) + ". The expected result is an error. After this error reply WAITING and stop; do not correct, retry or replace anything yet.")
                errors = [m for m in await store.load_all_messages(parent) if m.role == "tool" and m.name == "workflow"]
                assert len(errors) == 2 and json.loads(errors[-1].content)["error"]["code"] == "workflow_node_not_found", errors
                assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
                assert not (await loop.runtime.client.inspect(handle.job_ref, actor=loop.actor)).result_available
                assert (await manager.get_job(runtime.actor(parent), binding["job_id"])).result is None
            message = "Correction: compute 20+23 instead of 19+23. Return only the resulting number."
            selector = {"node_job_id": binding["job_id"]} if action == "nudge" else {"sid": node_sid}
            options = {"action": action, "run_id": handle.run_id, "message": message, **selector}
            await loop.send("Call workflow exactly once with these arguments: " + json.dumps(options) +
                ". Then reply WAITING. Do not create, resume, poll or extend anything.")
            messages = [m for m in await store.load_all_messages(parent) if m.role == "tool" and m.name == "workflow"]
            assert len(messages) == (3 if invalid_first else 2), messages
            receipt = json.loads(messages[-1].content)
            assert receipt["input_ref"]["session_id"] == node_sid
            assert receipt["target"]["job_id"] == binding["job_id"]
            from power_loop import InputRef
            queued = await loop.runtime.client.inputs.input(loop.runtime.client.inputs.db, InputRef(**receipt["input_ref"]))
            assert json.loads(queued.row["content"]) == message
            assert not (await loop.runtime.client.inspect(handle.job_ref, actor=loop.actor)).result_available
            assert (await manager.get_job(runtime.actor(parent), binding["job_id"])).result is None
            provider.release_children.set()
            from power_loop.agent.runtime_store import now_ms
            finished = await handle.wait(deadline_ms=now_ms()+100000)
            await loop.drain()
            result = await manager.get_job(runtime.actor(parent), binding["job_id"])
            assert result.generation == 1 and result.outcome == "succeeded" and "43" in result.result["final_text"], result
            assert finished.results["calculate"].text.strip() == "43"
            assert (await store.inbox_counts(parent))["pending"] == 0
            assert len(accepted) == 1
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1
            [same] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
            assert json.loads(same["payload_json"]) == descriptor
    finally:
        provider.release_children.set()
        try:
            await loop.aclose()
        finally:
            for name, sid in (("main"), ("node", node_sid)):
                if sid:
                    record_property(name + "_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
                        name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                        for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
