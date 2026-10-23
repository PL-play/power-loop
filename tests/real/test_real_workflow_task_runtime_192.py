"""Actual LLM parent -> workflow node -> subagent -> original node -> review."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import create_workflow

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("ending", ["natural", "pass_turn", "decision_write_failure", "empty_child_pass", "node_answer_write_failure"])
async def test_real_registered_workflow_consumes_nested_child_before_review(tmp_path, record_property, ending, monkeypatch):
    store = await SessionStore.open(str(tmp_path / "real-workflow-runtime.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-workflow-runtime-test", principal_id="host",
                          executor_id="test-executor", boot_id="test-boot")
    provider = BoundedProvider(limit=12)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    handles, completions = [], []
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Follow the requested tool sequence. A queued receipt is not a result. Never duplicate a workflow. "
                          "When its task_done summary arrives, report the final answer in final assistant text; do not call pass_turn instead of replying."))
    sid = await loop.new_session()
    if ending == "decision_write_failure":
        original_commit = manager.commit_processing_decision

        async def fail_decision(actor, *args, **kwargs):
            if actor.session_id != sid:
                raise RuntimeError("test: processing decision write unavailable")
            return await original_commit(actor, *args, **kwargs)

        monkeypatch.setattr(manager, "commit_processing_decision", fail_decision)
    provider.root_sid, provider.hold_children = sid, True
    spec = {"name": "registered-nested-calculation", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "calculate", "input": "Calculate 19+23 by delegating exactly once, then report the returned answer.",
         "spec": {"name": "calculate", "tools": ["spawn_agent"], "max_rounds": 4,
                  "system_prompt": "Use spawn_agent exactly once with tools=[], max_rounds=2, "
                    "system_prompt='Calculate the requested arithmetic. Return only the number in final assistant text. Do not call pass_turn.', "
                    "task='Compute 19+23'. After acceptance call pass_turn to wait. "
                    "When subagent_result arrives, report the answer in final assistant text without calling any more tools."}},
        {"type": "agent", "id": "review", "inputs_from": ["calculate"], "input": "Verify the prior result and report the final number.",
         "spec": {"name": "review", "tools": [], "max_rounds": 2,
                  "system_prompt": "Read the supplied calculation result. Reply with its correct number only in final assistant text. Do not call pass_turn."}},
    ]}}
    if ending == "decision_write_failure":
        # A lost processing decision is not permission to restart business
        # work, even when the original leaf explicitly permits failed retries.
        spec["root"]["steps"][0]["retry"] = {"max_attempts": 3, "on": ["failed"], "backoff_s": 0}
    if ending == "node_answer_write_failure":
        node = spec["root"]["steps"][0]
        node["spec"].update(tools=[], system_prompt="On your FIRST turn use pass_turn once and leave assistant text empty. "
            "When later asked for the final answer, return exactly 42 as assistant text without tools or pass_turn.")
        node["input"] = "The result code is 42. Follow your first-turn/final-answer instructions."
        node["fallback"] = {"type": "agent", "id": "rescue", "input": "Return 42.", "spec": {
            "name": "rescue", "tools": [], "max_rounds": 2, "system_prompt": "Return exactly 42."}}
        original_complete = provider.inner.complete
        calc_requests, calc_sid = 0, None
        async def first_node_pass(request, **kwargs):
            nonlocal calc_requests, calc_sid
            current = await store.get_session(get_session_id())
            if current is not None and current.metadata.get("workflow_node_id") == "calculate":
                calc_requests += 1
                calc_sid = get_session_id()
                if calc_requests == 1:
                    extra = dict(request.extra or {})
                    extra["extra_body"] = {**extra.get("extra_body", {}), "thinking": {"type": "disabled"}}
                    request = replace(request, tool_choice={"type": "function", "function": {"name": "pass_turn"}},
                        extra=extra, system_prompt=request.system_prompt + " On this first call use pass_turn and no assistant text.")
            return await original_complete(request, **kwargs)
        monkeypatch.setattr(provider.inner, "complete", first_node_pass)
        original_append = store.append_message
        async def lose_final_answer(session_id, **kwargs):
            if session_id == calc_sid and calc_requests >= 2 and kwargs.get("role") == "assistant":
                raise RuntimeError("test: final answer persistence unavailable")
            return await original_append(session_id, **kwargs)
        monkeypatch.setattr(store, "append_message", lose_final_answer)


    if ending == "empty_child_pass":
        spec["root"]["steps"][0]["spec"]["system_prompt"] = (
            "Use spawn_agent exactly once with tools=[], max_rounds=2, task='Report result code 42', "
            "system_prompt='The exact result code is 42. On your FIRST turn call pass_turn exactly once and do not write the answer. "
            "When later asked for a final answer, return exactly 42 in assistant text without any tools or pass_turn.'. "
            "After acceptance call pass_turn to wait. When subagent_result arrives, report its answer in final assistant text. "
            "Do not create or nudge any other task.")
        original_complete = provider.inner.complete
        forced = False
        async def first_child_pass(request, **kwargs):
            nonlocal forced
            current = await store.get_session(get_session_id())
            if not forced and current is not None and current.metadata.get("execution_role") == "subagent":
                forced = True
                # Only force the fault boundary, never the model's response.
                # DeepSeek requires thinking disabled for named tool choice.
                # https://api-docs.deepseek.com/api/create-chat-completion/
                extra = dict(request.extra or {})
                extra["extra_body"] = {**extra.get("extra_body", {}), "thinking": {"type": "disabled"}}
                request = replace(request, tool_choice={"type": "function", "function": {"name": "pass_turn"}},
                    extra=extra, system_prompt=request.system_prompt + " Use only pass_turn arguments on this first call; leave assistant text empty.")
            return await original_complete(request, **kwargs)
        monkeypatch.setattr(provider.inner, "complete", first_child_pass)

    async def completed(completion):
        saved = await manager.get_job(runtime.actor(sid), handles[0].job_id)
        assert saved.result and (await store.inbox_counts(sid))["pending"] == 1
        completions.append(completion)

    async def launch():
        workflow = create_workflow(spec, parent_loop=get_current_loop(), parent_session_id=get_session_id())
        handle = await workflow.start(detached=True, on_complete=completed)
        handles.append(handle)
        return json.dumps({"status": "queued", "job_id": handle.job_id, "operation_id": handle.operation_id, "run_id": handle.run_id})

    registry.register(ToolDefinition(name="launch_workflow", description="Start the predefined calculation/review workflow asynchronously, once.",
                                     input_schema={"type": "object", "properties": {}}), launch)
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(180):
            finish = "Then call pass_turn with reason waiting for workflow." if ending == "pass_turn" else "Then reply WAITING without calling pass_turn."
            result = await loop.send("Call launch_workflow exactly once. Do not call spawn_agent yourself. " + finish +
                                     " When its completion summary arrives later, report the answer.", sid)
            assert result.status == "completed" and result.passed == (ending == "pass_turn")
            await asyncio.wait_for(provider.child_entered.wait(), 5)
            [handle] = handles
            actor = runtime.actor(sid)
            wf = await manager.get_job(actor, handle.job_id)
            original_profile = await manager.get_workflow_profile(actor, handle.job_id)
            assert original_profile["executor"] == {"kind": "native_in_process"}
            assert original_profile["budget"] == {"kind": "unbounded"}
            assert original_profile["file_io"] == {"kind": "none"}
            assert original_profile["notification"] == {"kind": "opaque_callback", "settlement": "library_exit_only"}
            assert (await manager.get_job(actor, wf.parent_job_id)).status == "model_closed" and wf.status == "running"
            provider.release_children.set()
            await handle.task
            [callback_exit] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='workflow_callback_exit'", (handle.job_id,))
            assert json.loads(callback_exit["payload_json"])["status"] == "completed"
            assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='workflow_projected'", (handle.job_id,))) == 1
            wf = await manager.get_job(actor, handle.job_id)
            assert await manager.get_workflow_profile(actor, handle.job_id) == original_profile
            if ending in ("decision_write_failure", "node_answer_write_failure"):
                assert wf.result["status"] == "failed" and wf.outcome == "failed", wf.result
                nodes = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node'")
                assert len(nodes) == 1 and nodes[0]["outcome"] == "failed"
                decisions = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='processing_decision'")
                assert not decisions and len(completions) == 1
                assert (await store.inbox_counts(sid))["pending"] == 1
                if ending == "node_answer_write_failure":
                    assert calc_requests == 2
                    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='node_session'")) == 1
                    record_property("node_answer_provider_requests", calc_requests)
                return
            assert wf.result["status"] == "completed" and "42" in wf.result["final"]["text"], wf.result
            assert wf.result["results"]["calculate"]["usage"]["total_tokens"] > 0
            [audit] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='subagent'")
            audit = await manager.get_job(actor, audit["job_id"])
            assert audit.outcome == "succeeded" and "42" in audit.result["final_text"]
            assert audit.result_status == "handled"
            assert audit.supervision_status == "closed"
            if ending == "empty_child_pass":
                rows = await store.load_all_messages(audit.session_id)
                first_assistant = next(r for r in rows if r.role == "assistant")
                assert forced and not (first_assistant.content or "").strip()
                assert any(r.role == "tool" and r.name == "pass_turn" for r in rows)
                assert audit.result["usage"]["calls"] >= 2

            [decision] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} "
                "WHERE job_id=? AND event_type='processing_decision'", (audit.job_id,))
            decision = json.loads(decision["payload_json"])
            assert decision["decision"] == "consumed" and decision["response_ref"]["session_id"] == audit.created_by_session_id
            assert decision["event"]["event_id"] == "result_" + audit.job_id
            [calculate] = [r for r in await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node'")
                           if json.loads(r["result_json"] or "{}").get("node_id") == "calculate"]
            assert audit.control_owner_job_id == calculate["job_id"]
            assert len(completions) == 1
            final = await loop.flush_follow_ups(sid)
            assert final and final.status == "completed" and "42" in final.final_text
            assert (await manager.get_job(actor, handle.job_id)).result_status == "handled"
            assert (await manager.get_job(actor, handle.job_id)).supervision_status == "open"  # opaque callback has no business ACK
            nodes = await manager.db.fetchall(f"SELECT supervision_status FROM {manager.jobs} WHERE kind='workflow_node'")
            assert nodes and all(node["supervision_status"] == "closed" for node in nodes)
            assert len(handles) == 1
    finally:
        provider.release_children.set()
        tasks = [h.task for h in handles] + list(set(delegation._TASKS) - before)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
