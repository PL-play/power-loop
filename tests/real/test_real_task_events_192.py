"""Real model reads canonical results, later stop references and their originals."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, create_default_tool_registry
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.result_revisions import result_id
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_saved_result_stop_operation_refs_and_jsonl_never_reexecute(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path/"events.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-events", principal_id="host", executor_id="executor", boot_id="boot")
    provider, accepted = BoundedProvider(limit=24), []
    async def admission(value):
        accepted.append(value)
    async def completed(_):
        pass
    registry = create_default_tool_registry(include=["jobs", "recall_send"], workspace_dir=tmp_path)
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=admission, on_complete=completed))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            representation=ProjectedRepresentation(), system_prompt="Follow each exact requested tool call once and then answer in assistant text. "
                "Saved result/events are audit data. Reading or exporting them never re-executes tasks. "
                "Create only on explicit initial request; never resume or nudge a recorded task. Cancellation acceptance is not physical stop."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    spec = {"name": "event-audit", "root": {"type": "agent", "id": "calculate", "input": "19+23",
        "spec": {"name": "calculate", "system_prompt": "Compute the arithmetic and return its answer in assistant text.", "tools": [], "max_rounds": 2}}}
    async def tools(name):
        return [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == name]
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            async with asyncio.timeout(240):
                await loop.send("Call workflow action=create exactly once with spec="+json.dumps(spec)+". Then answer WAITING without further tools.", sid)
                assert len(accepted) == 1
                await asyncio.wait_for(provider.child_entered.wait(), 10)
                provider.release_children.set()
                handle = accepted[0].handle
                await asyncio.wait_for(handle.task, 45)
                await loop.flush_follow_ups(sid)
                job, event = handle.job_id, result_id(handle.job_id, 1)
                raw = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (event,))
                original_result = raw["payload_json"]
                original_text = json.loads(original_result)["result"]["final"]["text"]
                assert "42" in original_text
                [binding] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
                physical = json.loads(binding["payload_json"])["session_id"]
                original_history = await store.load_all_messages(physical)
                response = await loop.send("Read only the saved result. Call jobs once with "+json.dumps({
                    "action": "get", "job_id": job, "view": "results", "event_id": event, "max_chars": 16000})+
                    ". Then state the saved arithmetic answer in assistant text. No other tools.", sid)
                result = (await tools("jobs"))[-1]
                assert result["status"] == "ok" and not result["truncated"] and "42" in response.final_text
                assert json.loads(result["text"])["result"]["final"]["text"] == original_text
                await loop.send("Request cancellation of this original saved test workflow once. Call jobs with "+json.dumps({
                    "action": "stop", "job_id": job, "reason": "test requests stopping this original workflow", "expected_generation": 1})+
                    ". Then describe only acceptance, not a new execution or proof of remote stop.", sid)
                stopped = (await tools("jobs"))[-1]
                assert stopped["operation_id"] != handle.operation_id
                await loop.send("Read existing operation references only. Call jobs once with "+json.dumps({
                    "action": "get", "job_id": job, "view": "operations"})+". Then describe the saved create and stop operations, no other tools.", sid)
                operations = (await tools("jobs"))[-1]
                assert len(operations["events"]) == 2 and {e["action"] for e in operations["events"]} == {"create", "stop"}
                stop = next(e for e in operations["events"] if e["action"] == "stop")
                assert stop["receipt_status"] == "available" and stop["call_ref"]["id"] != stop["receipt_ref"]["id"]
                await loop.send("Recall only this saved stop receipt. Call recall_send once with "+json.dumps({
                    "job_id": job, "message_id": stop["receipt_ref"]["id"], "char_limit": 16000})+
                    ". Then acknowledge the saved receipt, without any further control call.", sid)
                recalled = (await tools("recall_send"))[-1]
                assert recalled["status"] == "ok" and json.loads(recalled["text"])["message_ref"] == stop["receipt_ref"]
                await loop.send("Export only the immutable saved result. Call jobs once with "+json.dumps({
                    "action": "get", "job_id": job, "view": "results", "event_id": event, "output_file": "audit/result.jsonl"})+
                    ". Then report its returned path and count in assistant text. No other tools.", sid)
                exported = (await tools("jobs"))[-1]
                assert exported["status"] == "exported" and exported["records"] == 1
                lines = [json.loads(line) for line in (tmp_path/"audit/result.jsonl").read_text().splitlines()]
                assert len(lines) == 1 and lines[0]["event_id"] == event and lines[0]["result"]["final"]["text"] == original_text
                assert await store.load_all_messages(physical) == original_history
                assert (await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (event,)))["payload_json"] == original_result
                assert len(accepted) == 1 and len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 1
    finally:
        provider.release_children.set()
        try:
            await asyncio.wait_for(asyncio.gather(*(a.handle.task for a in accepted), return_exceptions=True), 15)
        finally:
            record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role, name=m.name,
                content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()


async def test_real_active_nudge_receipt_reads_preserve_pending_child_input(tmp_path, record_property):
    from power_loop.runtime import delegation
    from power_loop.tools.spawn_agent import register_spawn_agent

    store = await SessionStore.open(str(tmp_path/"active-nudge.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-events-nudge", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=20)
    registry = create_default_tool_registry(include=["jobs", "recall_send"], workspace_dir=tmp_path)
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            representation=ProjectedRepresentation(), system_prompt="Follow the requested tool sequence once, then answer in assistant text. "
                "Do not create replacement tasks. Audit reads never consume the child's queued inputs or execute them. "
                "When the child result arrives, report its final answer without more tools."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(240):
            await loop.send("Call spawn_agent create exactly once with tools=[], max_rounds=3, "
                "system_prompt='Calculate the arithmetic. Follow the latest inbox correction and reply with only its number in assistant text.', "
                "task='Compute 19+23'. Then finish WAITING without more tools.", sid)
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            rows = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "spawn_agent"]
            assert len(rows) == 1
            created = json.loads(rows[0].content)
            await loop.send("Call spawn_agent action=nudge exactly once with sub_session='"+created["session_id"]+
                "', message='Correction: compute 20+23 instead. Return only the final number.'. Then finish WAITING without more tools.", sid)
            rows = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "spawn_agent"]
            assert len(rows) == 2
            nudged = json.loads(rows[-1].content)
            assert nudged["job_id"] == created["job_id"] and nudged["mode"] == "steer"
            operation = await manager.get_operation(runtime.actor(sid), nudged["operation_id"])
            assert "caller" not in operation.result["request"]
            # Steering can already move an inbox row into the transcript while
            # the actual supplier request remains gated. Delivered is not handled.
            async def unhandled():
                event = await manager.db.fetchone(f"SELECT delivery_status FROM {manager.events} WHERE event_id=?", (operation.operation_id,))
                decisions = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='processing_decision'", (created["job_id"],))
                assert event["delivery_status"] != "handled" and not decisions
                assert (await manager.get_job(runtime.actor(sid), created["job_id"])).result is None
            await unhandled()
            await loop.send("Read existing operation references only. Call jobs once with "+json.dumps({
                "action": "get", "job_id": created["job_id"], "view": "operations"})+". Then acknowledge saved create and nudge, no further tools.", sid)
            jobs = [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "jobs"]
            assert len(jobs[-1]["events"]) == 2 and {e["action"] for e in jobs[-1]["events"]} == {"create", "nudge"}
            await loop.send("Read only the saved nudge receipt. Call recall_send once with "+json.dumps({
                "job_id": created["job_id"], "message_id": operation.receipt_ref["id"], "char_limit": 16000})+
                ". Then acknowledge the saved receipt without another spawn or nudge.", sid)
            recalled = [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "recall_send"]
            assert json.loads(recalled[-1]["text"])["message_ref"] == operation.receipt_ref
            await unhandled()
            provider.release_children.set()
            await asyncio.wait_for(asyncio.gather(*(set(delegation._TASKS)-before)), 60)
            saved = await manager.get_job(runtime.actor(sid), created["job_id"])
            assert saved.outcome == "succeeded" and saved.result["final_text"].strip() == "43"
            final = await loop.flush_follow_ups(sid)
            assert "43" in final.final_text and len(await store.list_children(sid)) == 1
    finally:
        provider.release_children.set()
        try:
            await asyncio.wait_for(asyncio.gather(*(set(delegation._TASKS)-before), return_exceptions=True), 15)
        finally:
            record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role, name=m.name,
                content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
