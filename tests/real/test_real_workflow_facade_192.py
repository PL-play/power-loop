"""Real model submits through the library facade and consumes original work."""
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


async def test_real_facade_validation_admission_and_actual_result_after_caller_yields(tmp_path, record_property, runtime_agent):
    store = await SessionStore.open(str(tmp_path / "facade.db"), schema="tasks")
    runtime = TaskRuntime(JobManager(store), tenant_id="real-facade", principal_id="host",
        executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=12)
    accepted = []
    async def on_accepted(admission):
        accepted.append(admission)
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=on_accepted))
    loop = await runtime_agent(model=provider, store=store, task_runtime=runtime, registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=6, max_tokens=2048,
            system_prompt="Follow the specified workflow tool order exactly. An accepted receipt is not the result. "
                "Do not start replacement work or poll. After asynchronous completion, consume its result and "
                "report the computed answer in assistant text. Do not call pass_turn in the completion turn: "
                "its reason is not a user-visible answer."))
    sid = loop.target.session_id
    provider.root_sid, provider.hold_children = sid, True
    spec = {"name": "real-facade", "root": {"type": "agent", "id": "calculate", "input": "Compute 19+23.",
        "spec": {"name": "calculate", "system_prompt": "Compute the supplied arithmetic. Return only the number.", "tools": [], "max_rounds": 2}}}
    try:
        async with asyncio.timeout(180):
            initial = await loop.send("Use workflow exactly twice in this order: action=validate with the following spec, "
                "then action=create with that SAME spec. Spec: " + json.dumps(spec) + ". "
                "After the accepted receipt, finish with text WAITING and no further tool calls. Do not use status yet.")
            assert initial.status == "completed"
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            assert len(accepted) == 1
            assert not (await loop.runtime.client.inspect(accepted[0].handle.job_ref, actor=loop.actor)).result_available
            rows = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
            assert len(rows) == 2
            validation, receipt = [json.loads(m.content) for m in rows]
            assert validation["status"] == "valid" and receipt["status"] == "accepted"
            assert receipt["run_id"] == accepted[0].handle.run_id
            operation = await runtime.manager.get_operation(runtime.actor(sid), receipt["operation_id"])
            assert operation.receipt_ref["id"] == str(rows[-1].id)
            assert operation.call_ref["id"] != operation.receipt_ref["id"]
            provider.release_children.set()
            from power_loop.agent.runtime_store import now_ms
            result = await accepted[0].handle.wait(deadline_ms=now_ms()+90000)
            assert result.status == "completed" and result.results["calculate"].text.strip() == "42"
            await loop.drain()
            assert (await store.inbox_counts(sid))["pending"] == 0
            assert any(m.role=="assistant" and "42" in (m.content or "") for m in await store.load_all_messages(sid))
            assert len(accepted) == 1
    finally:
        provider.release_children.set()
        try:
            await loop.aclose()
        finally:
            record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
                name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
