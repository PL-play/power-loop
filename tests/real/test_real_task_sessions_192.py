"""Real parent audits the retained physical node through persistent authority."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import task_sessions
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from ._bounded import BoundedProvider
from ._required_response import consume_completed_result, require_visible_response

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_parent_reads_original_retained_node_binding_without_replacement_work(tmp_path, record_property, runtime_agent):
    store = await SessionStore.open(str(tmp_path / "sessions.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-task-sessions", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=10)
    admitted, reads = [], []
    async def admitted_view(admission):
        admitted.append(admission)
    async def completed(_):
        pass
    async def inspect_bindings(job_id):
        before = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} ORDER BY event_id")
        result = await task_sessions.list_sessions(manager, runtime.actor(get_session_id()), job_id)
        assert before == await manager.db.fetchall(f"SELECT event_id FROM {manager.events} ORDER BY event_id")
        reads.append(result)
        return json.dumps(result)
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=admitted_view, on_complete=completed))
    registry.register(ToolDefinition(name="inspect_task_sessions", description="Read the existing authorized task's original physical-session bindings. Read-only; no execution authority.",
        input_schema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}, pricing=Free("test_fixture")), inspect_bindings)
    loop = await runtime_agent(model=provider, store=store, task_runtime=runtime, registry=registry, config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Follow the requested workflow and audit tool order. Create only when explicitly asked. A read binding is not execution permission."))
    sid = loop.target.session_id
    require_visible_response(registry, store, sid)
    provider.root_sid, provider.hold_children = sid, True
    spec = {"name": "session-audit", "root": {"type": "agent", "id": "calculate", "input": "19+23",
        "spec": {"name": "calculate", "system_prompt": "Compute the supplied arithmetic. Return only the number.", "tools": [], "max_rounds": 2}}}
    try:
        async with asyncio.timeout(180):
            await loop.send("Call workflow action=create exactly once with spec="+json.dumps(spec)+
                ". After acceptance finish this turn with WAITING, without any additional tools.")
            assert len(admitted) == 1
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            provider.release_children.set()
            await asyncio.wait_for(admitted[0].handle.wait(deadline_ms=now_ms()+60000), 90)
            consumed = await consume_completed_result(loop, sid, max_rounds=2)
            assert consumed and "42" in consumed.final_text
            [node] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
            [event] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
            physical_sid = json.loads(event["payload_json"])["session_id"]
            assert (await store.get_session(physical_sid)).parent_session_id is None
            result = await loop.send("Only audit existing work. Call inspect_task_sessions exactly once with job_id='"+node["job_id"]+
                "'. Then report the node_id and its exact original session_id from that response. Do not create, resume, nudge or execute any workflow.")
            assert len(reads) == 1 and len(admitted) == 1
            [binding] = reads[0]["bindings"]
            assert binding["session_id"] == physical_sid and binding["available"] and not binding["execution_authority"]
            assert physical_sid in result.final_text and "calculate" in result.final_text
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 1
    finally:
        provider.release_children.set()
        try:
            if admitted:
                await asyncio.wait_for(asyncio.gather(*(a.handle.wait(deadline_ms=now_ms()+60000) for a in admitted), return_exceptions=True), 15)
        finally:
            record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
                name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await loop.aclose()
            await store.close()
