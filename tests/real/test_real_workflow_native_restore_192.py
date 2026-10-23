"""Actual 42 -> same workflow/SIDs -> 43 -> fresh review -> main consumption."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import workflow_restore_execution
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.workflow_resources import (
    WorkflowResourceAdapter,
    WorkflowResourceBinding,
    WorkflowResourceRegistry,
    WorkflowResources,
    execution_contract,
)
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import InProcessExecutor, SharedBudget, WorkflowSpec, create_workflow, journal
from power_loop.workflow.resume import resume_detached

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("fault", ["none", "node_ack"])
@pytest.mark.parametrize("resource_mode", ["native", "bound"])
async def test_real_native_restoration_recomputes_original_nodes_and_main_consumes_new_result(tmp_path, record_property, monkeypatch, fault, resource_mode):
    store = await SessionStore.open(str(tmp_path / "native-restore.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="native-restore", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=16)
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=2048,
            system_prompt="Follow the requested tool sequence exactly. An acceptance receipt is not a result. "
                "Never create replacement tasks. When a task_done result arrives, report its final number in assistant text without tools or pass_turn."))
    parent = await loop.new_session()
    provider.root_sid, provider.hold_children = parent, True
    handles = []
    spec = {"name": "native-calculation-repair", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "calculate", "input": "Compute 19+23.", "spec": {"name": "calculate", "tools": [], "max_rounds": 2,
            "system_prompt": "Calculate the newest arithmetic requested by the user. Reply only with the final number in assistant text. Do not use pass_turn."}},
        {"type": "agent", "id": "review", "inputs_from": ["calculate"], "input": "Check the supplied calculation result and report its verified number.",
         "spec": {"name": "review", "tools": [], "max_rounds": 2, "system_prompt": "Review the newest calculation result supplied by the user. Return its number only. Do not use pass_turn."}},
    ]}}
    pool = SharedBudget(1000000)
    pool.commit({"total_tokens": 13})
    resource_binding = None
    if resource_mode == "bound":
        resource_registry = WorkflowResourceRegistry()
        runtime.workflow_resources = resource_registry
        resource_binding = WorkflowResourceBinding("real-restore", "1.0.0", "original-resources", 1,
            runtime.tenant_id, runtime.principal_id, parent, "a"*64,
            execution_contract(WorkflowSpec.from_json(spec), None), budget_scope_id="original-shared-pool")
        resources = WorkflowResources(resource_binding, InProcessExecutor(), budget=pool)
        async def prepared(*args):
            return resources
        async def authorized(*args):
            return True
        resource_registry.register("real-restore", "1.0.0", WorkflowResourceAdapter(prepared, prepared, authorized))
    async def launch():
        handle = await create_workflow(spec, parent_loop=get_current_loop(), parent_session_id=get_session_id(),
            resource_binding=resource_binding).start(detached=True)
        handles.append(handle)
        return json.dumps({"job_id": handle.job_id, "operation_id": handle.operation_id, "run_id": handle.run_id, "status": "accepted"})
    async def repair():
        handle = await resume_detached(get_current_loop(), parent, handles[0].run_id,
            continuation=("calculate", "Correction: compute 20+23 instead of 19+23. Return the new number only.", 2))
        handles.append(handle)
        return json.dumps({"job_id": handle.job_id, "operation_id": handle.operation_id, "run_id": handle.run_id, "generation": 2, "status": "accepted"})
    for name, fn in (("launch_workflow", launch), ("repair_workflow", repair)):
        description = "Start the predefined 19+23 workflow once." if name == "launch_workflow" else "Continue the completed original workflow with the user's new correction: compute 20+23, re-review, and deliver a new result. Reuses original task and sessions."
        registry.register(ToolDefinition(name=name, description=description, input_schema={"type": "object", "properties": {}}), fn)
    claims = []
    original = workflow_restore_execution.claim_node
    async def claim(*args, **kwargs):
        value = await original(*args, **kwargs)
        claims.append(args[2])
        if fault == "node_ack" and args[2] == "calculate":
            raise RuntimeError("node claim committed, ACK lost")
        return value
    monkeypatch.setattr(workflow_restore_execution, "claim_node", claim)
    try:
        async with asyncio.timeout(180):
            await loop.send("Call launch_workflow exactly once, then reply WAITING without tools or pass_turn.", parent)
            assert len(handles) == 1
            provider.release_children.set()
            await handles[0].task
            actor = runtime.actor(parent)
            first = await manager.get_job(actor, handles[0].job_id)
            assert first.outcome == "succeeded" and first.result["final"]["text"].strip() == "42", first.result
            if resource_mode == "bound":
                assert pool.spent == 13 + first.result["usage"]["total_tokens"]
                pool.commit({"total_tokens": 7})  # another scope's spend cannot disappear
            before_restore = pool.spent
            consumed = await loop.flush_follow_ups(parent)
            assert "42" in consumed.final_text
            first_closed = await manager.get_job(actor, first.job_id)
            creator = await manager.get_job(actor, first.parent_job_id)
            assert first_closed.supervision_status == creator.supervision_status == "closed"
            old = await manager.get_result_revision(actor, first.job_id, 1)
            old_nodes = {n["job_id"]: n for n in await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node'")}
            provider.release_children.clear()
            provider.child_entered.clear()
            accepted = await loop.send("New correction: the requested arithmetic is now 20+23, replacing 19+23. The old 42 result does not answer this new request. "
                "Call repair_workflow exactly once to continue the completed original workflow with this correction. Then reply WAITING without other tools or pass_turn. "
                "Do not call launch_workflow; wait for the new generation's completion result.", parent)
            assert accepted.status == "completed" and len(handles) == 2
            reopened_creator = await manager.get_job(actor, creator.job_id)
            assert reopened_creator.supervision_status == "open" and reopened_creator.status == "model_closed"
            assert reopened_creator.generation == creator.generation
            new = handles[1]
            assert new.run_id == handles[0].run_id and new.job_id == handles[0].job_id and new.operation_id != handles[0].operation_id
            provider.release_children.set()
            await new.task
            current = await manager.get_job(actor, new.job_id)
            assert current.outcome == "succeeded" and current.generation == 2, current.result
            assert current.result["results"]["calculate"]["text"].strip() == "43" and current.result["final"]["text"].strip() == "43", current.result
            assert await manager.get_result_revision(actor, first.job_id, 1) == old
            new_nodes = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node'")
            assert len(new_nodes) == 2 and set(n["job_id"] for n in new_nodes) == set(old_nodes)
            for row in new_nodes:
                assert row["generation"] == 2 and row["created_by_session_id"] == old_nodes[row["job_id"]]["created_by_session_id"]
            for nid, value in current.result["results"].items():
                assert value["session_id"] == first.result["results"][nid]["session_id"]
                assert (await store.get_session(value["session_id"])).parent_session_id is None
            op = await manager.get_operation(actor, new.operation_id)
            assert op.call_ref["id"] != op.receipt_ref["id"] and sorted(claims) == ["calculate", "review"]
            assert current.result["usage"] == await manager.execution_usage(actor, first.job_id, generation=2)
            if resource_mode == "bound":
                assert pool.spent == before_restore + current.result["usage"]["total_tokens"]
                assert (await manager.get_workflow_profile(actor, first.job_id))["resource_binding"] == resource_binding.to_dict()
                assert op.result["request"]["resource_evidence"]["budget_spent"] == before_restore
            lifetime = await manager.execution_usage(actor, first.job_id)
            assert lifetime["total_tokens"] == first.result["usage"]["total_tokens"]+current.result["usage"]["total_tokens"]
            final = await loop.flush_follow_ups(parent)
            assert final.status == "completed" and "43" in final.final_text
            assert (await manager.get_job(actor, first.job_id)).result_status == "handled"
            assert (await manager.get_job(actor, first.job_id)).supervision_status == "closed"
            assert (await manager.get_job(actor, creator.job_id)).supervision_status == "closed"
            assert (await journal.read(store, parent, new.run_id))["task_generation"] == 2
    finally:
        provider.release_children.set()
        if handles:
            await asyncio.wait_for(asyncio.gather(*(h.task for h in handles), return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
