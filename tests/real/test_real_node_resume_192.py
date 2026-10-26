"""Native YAML compilation and real original-workflow leaf continuation."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import WorkflowSpec, create_workflow, runner

from ._required_response import consume_completed_result, require_visible_response
from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]

DOCUMENT = """name: native-yaml-leaf-resume
jobs:
  calculate:
    prompt: >-
      ROLE-CALCULATE. VERSION-42 is a literal test identifier, not program source.
      Return that literal identifier verbatim. A later user correction replaces
      the identifier; output only the latest identifier.
      Use plain text without Markdown, backticks, quotes, labels or code fences.
      Never use tools or pass_turn.
    tools: []
    with:
      request: Return exactly the literal text VERSION-42. This is the entire task.
  sibling:
    prompt: ROLE-SIBLING. Output only READY. Never use tools or pass_turn.
    tools: []
    with:
      request: Return exactly READY.
  review:
    needs: [calculate, sibling]
    prompt: >-
      ROLE-REVIEW. Read the calculate result in your supplied context.
      Output only its latest VERSION code. Ignore sibling READY.
      Use plain text without Markdown, backticks, quotes, labels or code fences.
      Never use tools or pass_turn.
    tools: []
    with:
      request: Verify and return the latest literal VERSION identifier supplied in context.
"""


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack", "membership_relabel", "whole_restore", "bind_error", "result_ack"])
async def test_real_yaml_workflow_resumes_original_finished_leaf_before_join(tmp_path, monkeypatch, record_property, fault):
    store = await SessionStore.open(str(tmp_path / "native-yaml-resume.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-native-yaml", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=16)
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Follow the requested tool sequence exactly. Accepted is not completed. Never launch replacement tasks. "
                          "When completion arrives output the final VERSION code without tools or pass_turn."))
    parent = await loop.new_session()
    require_visible_response(registry, store, parent)
    sibling_entered, sibling_release = asyncio.Event(), asyncio.Event()
    resume_entered, resume_release = asyncio.Event(), asyncio.Event()
    complete = provider.complete
    async def gated(request, **kwargs):
        if "ROLE-SIBLING" in (request.system_prompt or ""):
            sibling_entered.set()
            await sibling_release.wait()
        if "ROLE-CALCULATE" in (request.system_prompt or "") and "VERSION-43" in str(request.messages):
            resume_entered.set()
            await resume_release.wait()
        return await complete(request, **kwargs)
    monkeypatch.setattr(provider, "complete", gated)
    handles, replies, targets = [], [], []
    async def launch():
        spec = WorkflowSpec.from_yaml(DOCUMENT)
        handle = await create_workflow(spec.to_dict(), parent_loop=get_current_loop(), parent_session_id=get_session_id()).start(detached=True)
        handles.append(handle)
        return json.dumps({"run_id": handle.run_id, "job_id": handle.job_id, "status": "queued"})
    async def correct():
        reply = await runner.steer_node(handles[0].run_id, "calculate", targets[0]["session_id"],
            "Correction: the new code is VERSION-43. Output only VERSION-43. Do not repeat completed actions or call pass_turn.")
        replies.append(reply)
        return json.dumps(reply)
    for name, fn in (("launch_workflow", launch), ("correct_node", correct)):
        description = ("Start the predefined original workflow once; this tool takes no arguments."
            if name == "launch_workflow" else "Correct the calculate node of the existing workflow after launch_workflow returns its run_id/job_id. A queued receipt is durable acceptance and authorizes this correction; it does not need a separate status=accepted receipt. Takes no arguments.")
        registry.register(ToolDefinition(name=name, description=description,
            input_schema={"type": "object", "properties": {}, "additionalProperties": False}), fn)
    if fault == "whole_restore":
        from power_loop.workflow.resume import resume_detached
        async def restore_whole():
            handle = await resume_detached(get_current_loop(), parent, handles[0].run_id,
                continuation=("calculate", "Correction: the newest code is VERSION-44. Output only VERSION-44.", 3))
            handles.append(handle)
            return json.dumps({"job_id": handle.job_id, "run_id": handle.run_id, "status": "accepted"})
    counts = {"admit": 0, "dispatch": 0}
    for name, key in (("resume_workflow_node", "admit"), ("claim_resume_dispatch", "dispatch")):
        original = getattr(manager, name)
        async def wrapped(*args, _fn=original, _key=key, **kwargs):
            counts[_key] += 1
            value = await _fn(*args, **kwargs)
            if fault == ("admission_ack" if _key == "admit" else "dispatch_ack"):
                raise RuntimeError("committed, ACK lost")
            return value
        monkeypatch.setattr(manager, name, wrapped)
    try:
        async with asyncio.timeout(180):
            await loop.send("Call launch_workflow once. Then reply WAITING without tools or pass_turn.", parent)
            assert len(handles) == 1
            await asyncio.wait_for(sibling_entered.wait(), 10)
            for _ in range(1500):
                rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='node_session'")
                targets[:] = [json.loads(row["payload_json"]) | {"job_id": row["job_id"]} for row in rows if json.loads(row["payload_json"])["node_id"] == "calculate"]
                if targets:
                    job = await manager.get_job(runtime.actor(parent), targets[0]["job_id"])
                    engine = runner._ENGINES[handles[0].run_id]
                    if job.resource_state == "stopped" and any(not f["active"] and f["result"] and f["result"].node_id == "calculate" for f in engine._frames.values()):
                        break
                await asyncio.sleep(.02)
            else:
                pytest.fail("calculate did not settle")
            assert job.result["text"].strip() == "VERSION-42", job.result
            old = await manager.get_result_revision(runtime.actor(parent), job.job_id, 1)
            committed = []
            if fault == "bind_error":
                bind = manager.bind_node_session
                async def fail_binding(actor, proof, sid):
                    if proof.job_id == job.job_id and proof.generation == 2:
                        raise RuntimeError("original resumed binding unavailable")
                    return await bind(actor, proof, sid)
                monkeypatch.setattr(manager, "bind_node_session", fail_binding)
            elif fault == "result_ack":
                record = runtime.record_scope
                async def lose_result_ack(binding, result):
                    saved = await record(binding, result)
                    if binding.lease.job_id == job.job_id and binding.lease.generation == 2:
                        committed.append(saved)
                        raise RuntimeError("original result committed; response lost")
                    return saved
                monkeypatch.setattr(runtime, "record_scope", lose_result_ack)
            await loop.send("Call correct_node once. Then reply WAITING without more tools or pass_turn. Never launch another workflow.", parent)
            if fault == "bind_error":
                assert not replies and counts == {"admit": 1, "dispatch": 1}
                sibling_release.set()
                await handles[0].task
                new = await manager.get_job(runtime.actor(parent), job.job_id)
                assert new.generation == 2 and new.outcome == "failed" and new.resource_state == "stopped"
                assert new.result["usage"] == {} and not resume_entered.is_set()
                assert await manager.get_result_revision(runtime.actor(parent), job.job_id, 1) == old
                wf = await manager.get_job(runtime.actor(parent), handles[0].job_id)
                assert wf.outcome == "failed" and "review" not in wf.result["results"]
                return
            assert len(replies) == 1 and replies[0]["mode"] == "continue" and counts == {"admit": 1, "dispatch": 1}
            await asyncio.wait_for(resume_entered.wait(), 10)
            sibling_release.set()
            await asyncio.sleep(.1)
            assert not handles[0].task.done()
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 2
            resume_release.set()
            await handles[0].task
            new = await manager.get_job(runtime.actor(parent), job.job_id)
            assert new.generation == 2 and new.result["text"].strip() == "VERSION-43" and new.outcome == "succeeded", new
            assert (await manager.get_result_revision(runtime.actor(parent), job.job_id, 1)) == old
            wf = await manager.get_job(runtime.actor(parent), handles[0].job_id)
            if fault == "result_ack":
                assert committed == [new.result]
                assert wf.outcome == "failed" and "review" not in wf.result["results"]
                assert new.result["usage"]["total_tokens"] > 0
                assert await manager.execution_usage(runtime.actor(parent), job.job_id, generation=2) == new.result["usage"]
                return
            assert wf.outcome == "succeeded" and wf.result["final"]["text"].strip() == "VERSION-43", wf.result
            actor = runtime.actor(parent)
            snapshots = [await manager.get_execution_membership(actor, job.job_id, g) for g in (1, 2)]
            assert all(s["sealed"] and s["scope"]["owner"]["job_id"] == wf.job_id and len(s["members"]) == 1 for s in snapshots)
            versions = [await manager.execution_usage(actor, job.job_id, generation=g) for g in (1, 2)]
            assert all(v["total_tokens"] > 0 for v in versions)
            assert versions[0] == old["payload"]["result"]["usage"] and versions[1] == new.result["usage"]
            frozen_usage = await manager.execution_usage(actor, wf.job_id, generation=1)
            assert frozen_usage == wf.result["usage"]
            if fault == "membership_relabel":
                # Storage fault simulation, not permission to restart a finished workflow.
                await manager.db.execute(f"UPDATE {manager.jobs} SET control_owner_generation=99,control_owner_executor_id='future' WHERE job_id=?", (job.job_id,))
                assert await manager.execution_usage(actor, wf.job_id, generation=1) == frozen_usage
                assert [await manager.get_execution_membership(actor, job.job_id, g) for g in (1, 2)] == snapshots
                assert await manager.get_result_revision(actor, job.job_id, 1) == old
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1
            op = await manager.get_operation(runtime.actor(parent), replies[0]["operation_id"])
            assert op.call_ref["id"] != op.receipt_ref["id"]
            row = await manager.db.fetchone(f"SELECT delivery_status FROM {manager.events} WHERE event_id=?", (op.operation_id,))
            assert row["delivery_status"] == "handled"
            final = await consume_completed_result(loop, parent, max_rounds=3, result_node="review")
            assert final.status == "completed" and "VERSION-43" in final.final_text
            if fault == "whole_restore":
                registry.register(ToolDefinition(name="restore_workflow", description="Continue the completed original workflow for the user's next correction, reusing its tasks and sessions.", input_schema={"type": "object"}), restore_whole)
                saved_wf = await manager.get_result_revision(actor, wf.job_id, 1)
                assert (await manager.get_job(actor, wf.job_id)).supervision_status == "closed"
                await loop.send("New request: the newest code is VERSION-44. Call restore_workflow exactly once to continue the completed original workflow. "
                    "Do not launch a replacement workflow or call correct_node. Then reply WAITING without any more tools or pass_turn.", parent)
                assert len(handles) == 2 and handles[1].job_id == wf.job_id and handles[1].run_id == handles[0].run_id
                await handles[1].task
                current = await manager.get_job(actor, wf.job_id)
                assert current.generation == 2 and current.outcome == "succeeded", current.result
                assert current.result["final"]["text"].strip() == "VERSION-44", current.result
                assert (await manager.get_job(actor, job.job_id)).generation == 3
                assert current.result["results"]["calculate"]["session_id"] == new.result["session_id"]
                assert await manager.get_result_revision(actor, wf.job_id, 1) == saved_wf
                assert await manager.get_result_revision(actor, job.job_id, 1) == old
                final = await consume_completed_result(loop, parent, max_rounds=3, result_node="review")
                assert final.status == "completed" and "VERSION-44" in final.final_text
                assert (await manager.get_job(actor, wf.job_id)).supervision_status == "closed"
    finally:
        sibling_release.set()
        resume_release.set()
        if handles:
            await asyncio.wait_for(asyncio.gather(*(h.task for h in handles), return_exceptions=True), 30)
        async with asyncio.timeout(5):
            while runtime._adapter_tasks:
                await asyncio.gather(*tuple(runtime._adapter_tasks), return_exceptions=True)
            assert await runtime.flush_control_checkpoints() == {
                "controller_retirement_pending": 0, "controller_task_exit_pending": 0,
            }
        for saved in await store._db.fetchall(f"SELECT session_id FROM {store.t.sessions}"):
            rows = await store.load_all_messages(saved["session_id"])
            record_property("retained_transcript", json.dumps({"session_id": saved["session_id"], "messages": [
                {"id": str(m.id), "seq": m.seq, "role": m.role, "content": m.content, "name": m.name,
                 "tool_calls": m.tool_calls} for m in rows]}, ensure_ascii=False))
        provider.report(record_property)
        await provider.close()
        await store.close()
