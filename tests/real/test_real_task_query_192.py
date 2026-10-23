"""A configured real model reads saved task history and its actual result pointer."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import task_query
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from ._required_response import require_visible_response
from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_model_queries_reopened_task_history_and_reads_actual_saved_output(tmp_path, record_property):
    path = str(tmp_path / "task-directory.db")
    store = await SessionStore.open(path, schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="task-directory", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=5)
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Use the directory tools in the requested order. Read saved outputs; do not start replacement work or guess output."))
    sid = await loop.new_session()
    require_visible_response(registry, store, sid)
    actor = runtime.actor(sid)
    root = await manager.register_job(actor, kind="send", session_id=sid, system_key="original-send")
    root_lease = await manager.claim_job(actor, root.job.job_id, executor_id="old-executor", boot_id="old-boot", expected_version=1)
    saved = await manager.register_job(actor, kind="background_command", parent=root_lease, independent=True,
        title="Completed library inventory", system_key="original-inventory")
    child_lease = await manager.claim_job(actor, saved.job.job_id, executor_id="old-executor", boot_id="old-boot", expected_version=1)
    await manager.record_execution_result(actor, child_lease, outcome="succeeded", result={"final_text": "Inventory contains 47 books."}, resource_state="stopped")
    await manager.close_model_execution(actor, root_lease)
    reopened = await SessionStore.open(path, schema="compatible")
    reader = JobManager(reopened)
    calls = []
    async def inspect_tasks(action="list", job_id=None):
        assert get_session_id() == sid
        calls.append((action, job_id))
        if action == "list":
            result = await task_query.list_jobs(reader, actor, status="terminal", kinds=["background_command"])
        elif action == "get":
            result = await task_query.get_job(reader, actor, job_id)
        else:
            raise ValueError("only list/get")
        return json.dumps(result, ensure_ascii=False)
    registry.register(ToolDefinition(name="inspect_tasks", description="Read persistent task history. First list completed tasks; get a listed job_id to read its saved output and canonical pointer.",
        input_schema={"type": "object", "properties": {"action": {"type": "string", "enum": ["list", "get"]}, "job_id": {"type": "string"}}, "required": ["action"]}), inspect_tasks)
    try:
        async with asyncio.timeout(120):
            before = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? ORDER BY event_id", (saved.job.job_id,))
            result = await loop.send("Call inspect_tasks action=list, then action=get with the listed completed inventory job_id. "
                "Report the number of books and its canonical result event_id from that saved output. Do not launch work.", sid)
            assert result.status == "completed" and "47" in result.final_text
            assert calls == [("list", None), ("get", saved.job.job_id)], calls
            assert "result_"+saved.job.job_id in result.final_text
            messages = await store.load_all_messages(sid)
            receipts = [m for m in messages if m.role == "tool" and m.name == "inspect_tasks"]
            assert len(receipts) == 2
            for receipt in receipts:
                origin = next(m for m in messages if any(t["id"] == receipt.tool_call_id for t in m.tool_calls or []))
                assert origin.id != receipt.id and await store.message_by_id(sid, str(receipt.id)) == receipt
            detail = json.loads(receipts[-1].content)["node"]
            assert detail["billing_status"] == "unavailable" and detail["result_ref"]["event_id"] == "result_"+saved.job.job_id
            assert await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? ORDER BY event_id", (saved.job.job_id,)) == before
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='background_command'")) == 1
    finally:
        record_property("main_transcript", json.dumps([{"id": str(m.id), "seq": m.seq, "role": m.role,
            "content": m.content, "name": m.name, "tool_calls": m.tool_calls}
            for m in await store.load_all_messages(sid)], ensure_ascii=False))
        provider.report(record_property)
        await provider.close()
        await reopened.close()
        await store.close()
