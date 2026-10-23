"""Actual model delivery/consumption followed by a non-executing recovery plan."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager, _hash
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_native_profile_and_consumed_result_can_plan_retained_nodes_without_dispatch(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "restore-plan.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="restore-plan", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=10)
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=2048,
            system_prompt="Follow the requested tool sequence. A queued receipt is not a result. Never create duplicate tasks. "
                "When the task_done summary arrives later, report the final number in assistant text without tools or pass_turn."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    handles = []
    spec = {"name": "native-restore-plan", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "calculate", "input": "Compute 19+23.",
         "spec": {"name": "calculate", "tools": [], "max_rounds": 2,
                  "system_prompt": "Calculate the requested arithmetic. Reply only with its final number as assistant text. Do not use pass_turn."}},
        {"type": "agent", "id": "review", "inputs_from": ["calculate"], "input": "Check the supplied calculation and report its number.",
         "spec": {"name": "review", "tools": [], "max_rounds": 2,
                  "system_prompt": "Read the supplied result. Return its verified number only in assistant text. Do not use pass_turn."}},
    ]}}

    async def launch():
        workflow = create_workflow(spec, parent_loop=get_current_loop(), parent_session_id=get_session_id())
        handle = await workflow.start(detached=True)
        handles.append(handle)
        return json.dumps({"job_id": handle.job_id, "operation_id": handle.operation_id, "run_id": handle.run_id, "status": "queued"})

    registry.register(ToolDefinition(name="launch_workflow", description="Start the predefined calculation and review asynchronously once.",
        input_schema={"type": "object", "properties": {}}), launch)
    try:
        async with asyncio.timeout(120):
            result = await loop.send("Call launch_workflow exactly once, then reply WAITING without other tools or pass_turn.", sid)
            assert result.status == "completed"
            [handle] = handles
            actor = runtime.actor(sid)
            provider.release_children.set()
            await handle.task
            old = await manager.get_job(actor, handle.job_id)
            assert old.outcome == "succeeded" and old.result["final"]["text"].strip() == "42"
            consumed = await loop.flush_follow_ups(sid)
            assert consumed.status == "completed" and "42" in consumed.final_text
            assert (await manager.get_job(actor, handle.job_id)).result_status == "handled"
            assert await store.get_session(old.session_id) is None
            prior_result = await manager.get_result_revision(actor, handle.job_id, 1)
            before = provider.calls
            plan = await manager.get_workflow_restore_plan(actor, handle.job_id,
                generation=1, node_id="calculate", message="Compute 20+23 instead.")
            assert plan["dependency"]["invalidated"] == ["calculate", "review"] and not plan["replay"]
            assert plan["profile"]["notification"] == {"kind": "durable_inbox"}
            for node in plan["nodes"].values():
                assert (await store.get_session(node["binding"]["session_id"])).parent_session_id is None
            second = await manager.get_workflow_restore_plan(actor, handle.job_id,
                generation=1, node_id="review", message="Review once more.")
            assert second["replay"] == ["calculate"]
            assert provider.calls == before
            assert (await manager.get_job(actor, handle.job_id)).generation == 1
            assert await manager.get_result_revision(actor, handle.job_id, 1) == prior_result
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 2
            from power_loop.runtime.workflow_resume import admit
            admissions = []
            async def resume_intent():
                source = _SEND.get()
                seq, call_id = current_invocation(sid)
                ref = await store.message_ref(sid, seq)
                operation, repeated = await admit(manager, actor, handle.job_id, caller=source.lease, call_ref=ref,
                    tool_call_id=call_id, generation=1, node_id="calculate", message="Compute 20+23 instead.",
                    executor_id=runtime.executor_id, boot_id=runtime.boot_id, lease_ms=runtime.lease_ms, plan_hash=_hash(plan))
                assert not repeated
                admissions.append(operation)
                return json.dumps({"operation_id": operation.operation_id, "job_id": handle.job_id,
                    "run_id": handle.run_id, "generation": 2, "status": "accepted", "dispatch": "pending"})
            registry.register(ToolDefinition(name="resume_workflow", description="Register one new execution intent for the original workflow. Does not execute it yet.",
                input_schema={"type": "object", "properties": {}}), resume_intent)
            accepted = await loop.send("Call resume_workflow exactly once, then reply WAITING without tools or pass_turn. "
                "This registers an intent only; no new calculation or result is available yet. Do not call launch_workflow.", sid)
            assert accepted.status == "completed" and len(admissions) == 1
            [operation] = admissions
            stored_op = await manager.get_operation(actor, operation.operation_id)
            assert stored_op.call_ref["id"] != stored_op.receipt_ref["id"]
            current = await manager.get_job(actor, handle.job_id)
            assert current.generation == 2 and current.result is None and current.session_id is None
            assert await manager.get_result_revision(actor, handle.job_id, 1) == prior_result
            assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='workflow_dispatch'")
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 2
            # Storage admission is deliberately not a running restoration adapter.
            # Preserve cancel responsibility; no child task/model was dispatched.
            await manager.request_cancel(actor, handle.job_id, system_key="test-finish-undispatched", reason="test cleanup")
    finally:
        provider.release_children.set()
        if handles:
            await asyncio.wait_for(asyncio.gather(*(h.task for h in handles), return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
