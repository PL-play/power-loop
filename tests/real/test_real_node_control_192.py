"""Real models stop one original node while its sibling and review continue."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.jobs import register_jobs_tools
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import journal
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


class NodeProvider(BoundedProvider):
    def __init__(self):
        super().__init__(limit=14)
        self.entered = defaultdict(asyncio.Event)

    async def complete(self, request, **kwargs):
        if get_session_id() != self.root_sid:
            for role in ("CALCULATE", "AUDIT", "REVIEW"):
                if "ROLE-"+role in (request.system_prompt or ""):
                    self.entered[role].set()
        return await super().complete(request, **kwargs)


@pytest.mark.parametrize(("native", "tool"), [(True, "workflow"), (True, "jobs"), (False, "workflow")])
async def test_real_node_cancel_preserves_original_parallel_sibling_review_and_completion(tmp_path, record_property, native, tool):
    store = await SessionStore.open(str(tmp_path/"nodes.db"), **({"schema": "tasks"} if native else {}))
    runtime = TaskRuntime(JobManager(store), tenant_id="real-node-control", principal_id="host",
        executor_id="executor", boot_id="boot") if native else None
    provider = NodeProvider()
    accepted, completed = [], []
    async def admission(value):
        accepted.append(value)
    async def completion(value):
        completed.append(value)
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=admission, on_complete=completion))
    register_jobs_tools(registry)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Follow the exact requested workflow or jobs call once, then answer in assistant text. "
                "Create only once; do not poll, repeat, resume, or replace any work. "
                "A cancellation receipt is not proof of physical resource stop or billing settlement. "
                "When processing completion, give an actual assistant answer, not only pass_turn.reason."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    def leaf(role, task, **extra):
        return {"type": "agent", "id": role, "input": task, "spec": {"name": role,
            "system_prompt": "ROLE-"+role+". Compute or verify the supplied arithmetic. Return the numeric answer in assistant text.",
            "tools": [], "max_rounds": 2}, **extra}
    spec = {"name": "one-node-stop", "root": {"type": "sequence", "steps": [
        {"type": "parallel", "branches": [leaf("CALCULATE", "19+23"), leaf("AUDIT", "7+5")]},
        leaf("REVIEW", "Verify the AUDIT arithmetic and return its number.", inputs_from=["AUDIT"])]}}
    try:
        async with asyncio.timeout(180):
            await loop.send("Call workflow action=create exactly once with spec="+json.dumps(spec)+
                ". After acceptance finish with WAITING and no more tools.", sid)
            assert len(accepted) == 1
            handle = accepted[0].handle
            await asyncio.wait_for(asyncio.gather(provider.entered["CALCULATE"].wait(), provider.entered["AUDIT"].wait()), 10)
            if native:
                manager, actor = runtime.manager, runtime.actor(sid)
                [node] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node' AND title='CALCULATE'")
                args = {"action": "cancel", "run_id": handle.run_id, "node_job_id": node["job_id"]} if tool == "workflow" else {
                    "action": "stop", "job_id": node["job_id"]}
                # Give the model a real, authorized directory receipt before
                # asking it to control an ID absent from the creation receipt.
                await loop.send("Call jobs exactly once with action=get and job_id='"+node["job_id"]+
                    "' to inspect the CALCULATE node. Then finish in assistant text with no further tools.", sid)
                lookup = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "jobs"][-1]
                assert json.loads(lookup.content)["node"]["job_id"] == node["job_id"]
            else:
                args = {"action": "cancel", "run_id": handle.run_id, "node": "CALCULATE"}
            await loop.send("Stop only CALCULATE, preserving AUDIT and the original workflow. Call "+tool+
                " exactly once with arguments="+json.dumps(args)+". Acknowledge in assistant text with no further tools.", sid)
            messages = await store.load_all_messages(sid)
            raw = [m for m in messages if m.role == "tool" and m.name == tool][-1]
            receipt = json.loads(raw.content)
            assert receipt["status"] == "accepted" and receipt["control_pending"] and not receipt["resource_stop_confirmed"]
            assert receipt["local_signals"] > 0 and not handle.task.done() and not handle.task.cancelling()
            if native:
                assert receipt["job_id"] == node["job_id"]
                operation = await manager.get_operation(actor, receipt["operation_id"])
                assert operation.receipt_ref["id"] == str(raw.id) and operation.call_ref["id"] != operation.receipt_ref["id"]
                assert (await manager.get_job(actor, handle.job_id)).cancel_requested_at is None
                assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")) == 1
            else:
                assert not receipt["durable_intent"] and receipt["node"] == "CALCULATE"
            provider.release_children.set()
            await asyncio.wait_for(handle.task, 45)
            assert provider.entered["REVIEW"].is_set() and len(completed) == 1 and len(accepted) == 1
            if native:
                workflow = await manager.get_job(actor, handle.job_id)
                assert workflow.outcome == "succeeded" and workflow.result["results"]["CALCULATE"]["status"] == "cancelled"
                assert workflow.result["results"]["AUDIT"]["status"] == "completed"
                assert workflow.result["results"]["REVIEW"]["status"] == "completed"
                answer = await loop.flush_follow_ups(sid)
                assert answer is not None and answer.status == "completed" and answer.final_text.strip()
            else:
                view = await journal.read(store, sid, handle.run_id)
                assert view["status"] == "completed"
                assert next(step for step in view["steps"] if step["node_id"] == "CALCULATE")["status"] == "cancelled"
    finally:
        provider.release_children.set()
        try:
            await asyncio.wait_for(asyncio.gather(*(a.handle.task for a in accepted), return_exceptions=True), 15)
        finally:
            record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
                name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
