"""Live provider tests on the explicit v10 store, bounded and isolated."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition, get_tool_runtime_context
from power_loop.agent.follow_up import InboxItem
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_tool_call_and_receipt_have_separate_recallable_message_ids(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "real-ids.db"), schema="tasks")
    llm = BoundedProvider(limit=4)
    registry = ToolRegistry()

    async def add(a, b):
        return json.dumps({"answer": a + b})

    registry.register(ToolDefinition(name="local_add", description="Add two integers locally.",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                      "required": ["a", "b"]}), add)
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=3, max_tokens=2048))
    sid = await loop.new_session()
    try:
        async with asyncio.timeout(120):
            result = await loop.send("Call local_add exactly once for 19+23, then reply with its answer.", sid)
            assert result.status == "completed" and "42" in result.final_text
            rows = await store.load_all_messages(sid)
            [receipt] = [r for r in rows if r.role == "tool" and r.name == "local_add"]
            [call] = [r for r in rows if any(t["id"] == receipt.tool_call_id for t in r.tool_calls or [])]
            call_ref, receipt_ref = await store.message_ref(sid, call.seq), await store.message_ref(sid, receipt.seq)
            assert call_ref.id != receipt_ref.id and call_ref.role == "assistant" and receipt_ref.role == "tool"
            recalled = await store.message_by_id(sid, receipt_ref.to_dict()["id"])
            assert recalled.content == receipt.content and "42" in recalled.content
            assert receipt_ref.tool_call_id == receipt.tool_call_id
    finally:
        llm.report(record_property)
        await llm.close()
        await store.close()


async def test_real_model_admission_receipt_result_and_reopen_use_durable_registry(tmp_path, record_property):
    path = str(tmp_path / "real-registry.db")
    store = await SessionStore.open(path, schema="tasks")
    llm = BoundedProvider(limit=6)
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=4, max_tokens=2048,
            system_prompt="Follow the tool sequence. A queued receipt is not a result. Never start the same task twice. "
                          "When the task_done answer arrives later, report it in final assistant text; do not call pass_turn instead of replying."))
    sid = await loop.new_session()
    manager = JobManager(store)
    actor = JobActor("isolated-real-test", "host", sid)
    root = await manager.register_job(actor, kind="send", session_id=sid, system_key="persistent-real-root")
    proof = await manager.claim_job(actor, root.job.job_id, executor_id="test-host", boot_id="test-boot",
                                    expected_version=1, lease_ms=300000)
    admissions = []

    async def start_add(a, b):
        ctx = get_tool_runtime_context(required=True)
        accepted = await manager.register_job(actor, kind="background_tool", parent=proof, independent=True,
            call_ref=await store.message_ref(sid, ctx.assistant_seq), tool_call_id=ctx.tool_call_id,
            request={"a": a, "b": b})
        admissions.append((accepted, a, b))
        return json.dumps({"job_id": accepted.job.job_id, "operation_id": accepted.operation.operation_id,
                           "status": "queued"})

    registry.register(ToolDefinition(name="start_add", description="Accept a local addition task; result arrives later.",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                      "required": ["a", "b"]}), start_add)
    recovered = None
    try:
        async with asyncio.timeout(120):
            first = await loop.send("Use start_add exactly once for 19+23. After its receipt, end this turn. Do not poll.", sid)
            assert first.status == "completed" and len(admissions) == 1
            accepted, a, b = admissions[0]
            [receipt] = [r for r in await store.load_all_messages(sid) if r.role == "tool" and r.name == "start_add"]
            await manager.bind_receipt(actor, accepted.operation.operation_id,
                                       await store.message_ref(sid, receipt.seq), caller=proof)
            await manager.close_model_execution(actor, proof)
            assert (await manager.get_job(actor, root.job.job_id)).status == "model_closed"
            worker = await manager.claim_job(actor, accepted.job.job_id, executor_id="local-worker", boot_id="test-boot", expected_version=1)
            event = await manager.record_result(actor, worker, outcome="succeeded", result={"answer": a + b},
                                                resource_state="stopped")
            recovered = await SessionStore.open(path, schema="compatible")
            persisted = JobManager(recovered)
            op = await persisted.get_operation(actor, accepted.operation.operation_id)
            saved = await persisted.get_job(actor, accepted.job.job_id)
            assert op.receipt_ref["id"] == str(receipt.id) and saved.result == {"answer": 42}
            assert saved.status == "settling" and saved.supervision_status == "open"
            await loop.deliver(InboxItem(json.dumps({"event_id": event, "result": saved.result}),
                                          kind="task_done", item_id=event), sid, enqueue_only=True)
            final = await loop.flush_follow_ups(sid)
            assert final and final.status == "completed" and "42" in final.final_text
            assert len(admissions) == 1
    finally:
        if recovered:
            await recovered.close()
        llm.report(record_property)
        await llm.close()
        await store.close()


@pytest.mark.parametrize("ending", ["natural", "pass_turn"])
async def test_real_runtime_automatically_tracks_parent_and_child_with_actual_receipt(tmp_path, record_property, ending):
    from power_loop.runtime import delegation
    store = await SessionStore.open(str(tmp_path / "real-adapter.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-adapter-test", principal_id="host",
                          executor_id="test-executor", boot_id="test-boot")
    llm = BoundedProvider(limit=7)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=llm, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Follow the requested tool sequence exactly. Never duplicate an accepted task. Results arrive later. "
                          "When a task_done result arrives, report its answer in final assistant text. Do not call pass_turn instead of replying."))
    sid = await loop.new_session()
    llm.root_sid, llm.hold_children = sid, True
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(150):
            instruction = "Then call pass_turn with reason waiting for child." if ending == "pass_turn" else "Then reply WAITING and end this turn without calling pass_turn."
            first = await loop.send("Call spawn_agent exactly once, tools=[], max_rounds=2, "
                                    "system_prompt='Compute the requested arithmetic and return the number as final assistant text. Do not call pass_turn.', "
                                    "task='Compute 19+23 and reply with only the answer.' " + instruction +
                                    " Do not wait or poll. When its result later arrives, report the answer.", sid)
            assert first.status == "completed" and first.passed == (ending == "pass_turn")
            await asyncio.wait_for(llm.child_entered.wait(), 5)
            [tool] = [r for r in await store.load_all_messages(sid) if r.role == "tool" and r.name == "spawn_agent"]
            accepted = json.loads(tool.content)
            actor = runtime.actor(sid)
            child = await manager.get_job(actor, accepted["job_id"])
            parent = await manager.get_job(actor, child.parent_job_id)
            assert parent.status == "model_closed" and child.status == "running"
            operation = await manager.get_operation(actor, accepted["operation_id"])
            assert operation.receipt_ref["id"] == str(tool.id)
            llm.release_children.set()
            await asyncio.gather(*(set(delegation._TASKS) - before))
            child = await manager.get_job(actor, child.job_id)
            assert child.resource_state == "stopped" and child.status == "settling"
            assert "42" in child.result["final_text"] and child.result["usage"]["total_tokens"] > 0
            final = await loop.flush_follow_ups(sid)
            assert final and final.status == "completed" and "42" in final.final_text
            rows = await manager.db.fetchall(f"SELECT kind,session_id FROM {manager.jobs}")
            assert len(rows) == 4 and sum(r["kind"] == "subagent" for r in rows) == 1
    finally:
        llm.release_children.set()
        pending = set(delegation._TASKS) - before
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 10)
        llm.report(record_property)
        await llm.close()
        await store.close()
