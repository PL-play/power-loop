"""Real parent recalls and exports retained node originals or projections."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, create_default_tool_registry
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("source", ["messages", "projection"])
async def test_real_parent_reads_and_exports_existing_node_history_without_reexecution(tmp_path, record_property, source):
    store = await SessionStore.open(str(tmp_path/"history.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-history", principal_id="host", executor_id="executor", boot_id="boot")
    provider, accepted = BoundedProvider(limit=22), []
    async def admission(value):
        accepted.append(value)
    async def completed(_):
        pass
    registry = create_default_tool_registry(include=["jobs", "recall_send"], workspace_dir=tmp_path)
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=admission, on_complete=completed))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            representation=ProjectedRepresentation(),
            system_prompt="Follow the exact requested workflow and read-only audit sequence. Create only when explicitly asked. "
                "Reading or exporting history never resumes, repeats or executes the recorded task. "
                "Report an actual assistant answer after each requested tool sequence, not just pass_turn.reason."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    spec = {"name": "retained-audit", "root": {"type": "agent", "id": "calculate", "input": "19+23",
        "spec": {"name": "calculate", "system_prompt": "Compute the requested arithmetic and return its number in assistant text.",
            "tools": [], "max_rounds": 2}}}
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            async with asyncio.timeout(180):
                await loop.send("Call workflow action=create exactly once with spec="+json.dumps(spec)+
                    ". After acceptance finish with WAITING and no further tools.", sid)
                assert len(accepted) == 1
                await asyncio.wait_for(provider.child_entered.wait(), 10)
                provider.release_children.set()
                await asyncio.wait_for(accepted[0].handle.task, 45)
                await loop.flush_follow_ups(sid)
                [node] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
                [event] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
                physical = json.loads(event["payload_json"])["session_id"]
                assert (await store.get_session(physical)).parent_session_id is None
                history_before = await store.load_all_messages(physical)
                prior_views = len([m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"])
                await loop.send("Inspect the saved workflow without executing anything. Call workflow action=leaves with run_id='"+
                    accepted[0].handle.run_id+"'. Then call workflow action=logs with the same run_id, node='calculate', source='"+source+
                    "'. Report its saved arithmetic result in assistant text. No other tools.", sid)
                views = [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"][prior_views:]
                assert len(views) == 2 and views[0]["nodes"][0]["job_id"] == node["job_id"]
                assert views[1]["status"] == "ok" and views[1]["session_id"] == physical and views[1]["source"] == source
                args = {"job_id": node["job_id"], "source": source, "limit": 20}
                answer = await loop.send("Audit only the existing completed node. First call jobs action=get with job_id='"+node["job_id"]+
                    "'. Then call recall_send exactly once with "+json.dumps(args)+
                    ". Report the saved arithmetic answer from its history. Do not create, resume or nudge any task.", sid)
                assert answer.status == "completed" and "42" in answer.final_text
                raw = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "recall_send"]
                assert len(raw) == 1
                page = json.loads(raw[0].content)
                assert page["status"] == "ok" and page["session_id"] == physical and not page["execution_authority"]
                assert page["source"] == source and "42" in str(page["records"])
                if source == "messages":
                    target = next(r for r in history_before if r.role == "assistant" and "42" in (r.content or ""))
                    exact = {"job_id": node["job_id"], "message_id": str(target.id), "char_limit": 8000}
                    await loop.send("Read only the exact saved original. Call recall_send once with "+json.dumps(exact)+
                        ". Then give its arithmetic result in assistant text. No other tools.", sid)
                    records = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "recall_send"]
                    exact_receipt = json.loads(records[-1].content)
                    assert exact_receipt["truncated"] is False
                    value = json.loads(exact_receipt["text"])
                    assert value["message_ref"]["id"] == str(target.id) and value["seq"] == str(target.seq)
                    origin = await manager.get_operation(runtime.actor(sid), accepted[0].handle.operation_id)
                    await loop.send("Read only the existing workflow creation receipt. Call recall_send once with "+json.dumps({
                        "job_id": accepted[0].handle.job_id, "message_id": origin.receipt_ref["id"], "char_limit": 8000})+
                        ". Then acknowledge in assistant text. Do not create or execute anything.", sid)
                    raw = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "recall_send"]
                    assert json.loads(json.loads(raw[-1].content)["text"])["message_ref"] == origin.receipt_ref
                output = "audit/"+source+".jsonl"
                exported = await loop.send("Export this existing selection only. Call recall_send exactly once with "+json.dumps({
                    "job_id": node["job_id"], "source": source, "output_file": output})+
                    ". Omit char_limit/cursor/char_offset. Report the returned path and record count in assistant text; no other tools.", sid)
                assert exported.status == "completed"
                records = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "recall_send"]
                receipt = json.loads(records[-1].content)
                assert receipt["status"] == "exported" and receipt["path"] == output and receipt["records"] > 0
                lines = [json.loads(line) for line in (tmp_path/output).read_text().splitlines()]
                assert len(lines) == receipt["records"] and "42" in str(lines)
                assert all(isinstance(line["id"], str) and line["session_id"] == physical for line in lines)
                assert await store.load_all_messages(physical) == history_before
                assert len(accepted) == 1 and len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 1
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
