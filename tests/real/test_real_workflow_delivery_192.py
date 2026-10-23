"""Real provider evidence survives failed workflow projection without reexecution."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import workflow_results
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow, journal
from power_loop.workflow.engine import WorkflowEngine

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("fault", ["journal", "inbox_ack"])
async def test_real_workflow_projection_repair_preserves_usage_and_never_reexecutes(tmp_path, record_property, monkeypatch, fault):
    store = await SessionStore.open(str(tmp_path / "delivery.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-workflow-delivery", principal_id="host", executor_id="test", boot_id="boot")
    provider = BoundedProvider(limit=10)
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=3, max_tokens=2048, delegation_policy="single_level",
            system_prompt="Follow the requested tool sequence. A queued receipt is not an answer. "
                          "When task_done arrives, reply with the final number in assistant text without using tools or pass_turn."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    handles = []
    spec = {"name": "delivery-proof", "root": {"type": "agent", "id": "calculate", "input": "Compute 19+23.",
        "spec": {"name": "calculate", "tools": [], "max_rounds": 2,
                 "system_prompt": "Compute the requested arithmetic. Return only its number in assistant text. Do not call pass_turn."}}}
    async def launch():
        workflow = create_workflow(spec, parent_loop=get_current_loop(), parent_session_id=get_session_id())
        handle = await workflow.start(detached=True)
        handles.append(handle)
        return json.dumps({"status": "queued", "run_id": handle.run_id, "job_id": handle.job_id})
    registry.register(ToolDefinition(name="launch_workflow", description="Launch the predefined arithmetic task once.",
        input_schema={"type": "object", "properties": {}}), launch)
    original_build = WorkflowEngine._build_result
    def inflated_rollup(self, *args, **kwargs):
        result = original_build(self, *args, **kwargs)
        result.usage = {"total_tokens": 999999999}
        return result
    monkeypatch.setattr(WorkflowEngine, "_build_result", inflated_rollup)
    original_write = manager._write
    @asynccontextmanager
    async def fault_transaction():
        projected = False
        async with original_write() as tx:
            async def execute(sql, params=()):
                nonlocal projected
                if sql.startswith(f"UPDATE {store.t.session_runtime_state} SET value_json="):
                    projected = True
                    if fault == "journal":
                        raise RuntimeError("test: result journal unavailable")
                return await tx.execute(sql, params)
            yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
        if fault == "inbox_ack" and projected:
            raise RuntimeError("test: atomic projection committed before lost ACK")
    try:
        async with asyncio.timeout(150):
            with monkeypatch.context() as patch:
                patch.setattr(manager, "_write", fault_transaction)
                result = await loop.send("Call launch_workflow exactly once. Then reply WAITING without pass_turn. "
                                         "The answer will be delivered later.", sid)
                assert result.status == "completed" and len(handles) == 1
                provider.release_children.set()
                await handles[0].task
            job_id, run_id = handles[0].job_id, handles[0].run_id
            actor = runtime.actor(sid)
            canonical = await manager.get_result_revision(actor, job_id, 1)
            saved = await manager.get_job(actor, job_id)
            assert saved.outcome == "succeeded" and "42" in saved.result["final"]["text"], saved.result
            assert saved.result["usage"] == await manager.execution_usage(actor, job_id, generation=1)
            assert 0 < saved.result["usage"]["total_tokens"] < 999999999
            assert saved.result_status == "pending_delivery" and await store.get_session(saved.session_id) is None
            pending = (await store.inbox_counts(sid))["pending"]
            j = await journal.read(store, sid, run_id)
            assert pending == (0 if fault == "journal" else 1)
            assert j["status"] == ("running" if fault == "journal" else "completed")
            before = provider.calls
            for _ in range(2):
                envelope = await workflow_results.project(loop, actor, job_id, 1)
            assert provider.calls == before and (await store.inbox_counts(sid))["pending"] == 1
            assert envelope["generation"] == 1 and envelope["result_event_id"] == canonical["event_id"]
            assert (await journal.read(store, sid, run_id))["result"]["usage"] == saved.result["usage"]
            final = await loop.flush_follow_ups(sid)
            assert final and final.status == "completed" and "42" in final.final_text
            assert (await manager.get_job(actor, job_id)).result_status == "handled"
            assert (await manager.get_result_revision(actor, job_id, 1))["payload"] == canonical["payload"]
            before = provider.calls
            await workflow_results.project(loop, actor, job_id, 1)
            assert provider.calls == before and (await store.inbox_counts(sid))["pending"] == 0
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 1
    finally:
        provider.release_children.set()
        if handles:
            await asyncio.wait_for(asyncio.gather(*(h.task for h in handles), return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
